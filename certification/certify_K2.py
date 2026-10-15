#!/usr/bin/env python3
"""
Rigorous ball-arithmetic evaluation of

    K2(v, T0, k) = 2 * sum_{0 < gamma < T0} exp(-k gamma^2) Re( e^{i gamma v} / (rho zeta'(rho)) )

over *all* zeros below T0, as required by Lemma H (residue theorem) in THEOREM.md.

Every zero is recomputed from scratch with Arb's certified zero isolation
(acb.zeta_zeros -> acb_dirichlet_zeta_zeros, Turing-method certified: the n-th returned
ball contains the n-th zero, which is simple and on the critical line), and the zero count
below T0 is the rigorous N(T0) from arb.zeta_nzeros. Nothing from ../chunks is used.

Zeros 1..N_HI are computed at HI_DPS digits (phases gamma*v resolved to ~1e-20 rad);
zeros N_HI+1..N(T0) at LO_DPS digits, where the phase ball is [0, 2pi] and each term
contributes only its rigorous magnitude bound 2 e^{-k gamma^2} / |rho zeta'(rho)| (StR (8.7)).

Usage: certify_K2.py [workers]
"""
import os, sys, time, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
from multiprocessing import Pool
import flint
from flint import arb, acb, acb_series

K_STR = "1.5e-10"
V_STR = "3484971332733132396.5383341489440461"
T0_STR = "1300000"
N_HI = 600000
HI_DPS, LO_DPS = 45, 20
BATCH = 250
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "k2_chunks")


def work(args):
    n0, num, dps = args
    path = os.path.join(OUT, f"{n0:08d}_{num}_{dps}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    flint.ctx.dps = dps
    k, v = arb(K_STR), arb(V_STR)
    S = arb(0)
    absS = arb(0)
    min_abs_zp, max_rad = None, 0.0
    zs = acb.zeta_zeros(n0, num)
    for rho in zs:
        assert rho.real == arb("0.5")
        g = rho.imag
        zp = acb_series([rho, 1], prec=2).zeta()[1]
        c = 1 / (rho * zp)                               # 1/(rho zeta'(rho))
        w = (-k * g * g).exp()
        term = 2 * w * (c * acb(0, g * v).exp()).real
        S += term
        absS += 2 * w * abs(c)
        a = abs(zp).lower()
        min_abs_zp = a if min_abs_zp is None or a < min_abs_zp else min_abs_zp
        max_rad = max(max_rad, float(term.rad()))
    res = {
        "n0": n0, "num": num, "dps": dps,
        "S": S.str(40, radius=True), "absS": absS.str(20, radius=True),
        "g_first": zs[0].imag.str(45, radius=True), "g_last": zs[-1].imag.str(45, radius=True),
        "min_abs_zeta_prime": float(min_abs_zp), "max_term_rad": max_rad,
    }
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(res, f)
    os.replace(tmp, path)
    return res


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    os.makedirs(OUT, exist_ok=True)
    flint.ctx.dps = 30
    T0 = arb(T0_STR)
    NT0 = T0.zeta_nzeros()
    assert NT0.rad() == 0, "N(T0) not determined exactly"
    N = int(NT0.mid().floor().str(30, radius=False).split(".")[0])
    nb = acb.zeta_zeros(N, 2)
    assert nb[0].imag < T0 < nb[1].imag, "T0 not strictly between gamma_N and gamma_{N+1}"
    delta = min((T0 - nb[0].imag).lower(), (nb[1].imag - T0).lower())
    print(f"N(T0={T0_STR}) = {N}; gamma_N = {nb[0].imag.str(15)}, gamma_N+1 = {nb[1].imag.str(15)}; delta >= {delta.str(8)}", flush=True)

    tasks = []
    for n0 in range(1, N + 1, BATCH):
        num = min(BATCH, N + 1 - n0)
        if n0 <= N_HI < n0 + num - 1:
            raise SystemExit("BATCH must divide N_HI")
        tasks.append((n0, num, HI_DPS if n0 <= N_HI else LO_DPS))
    tasks.sort(key=lambda t: -t[0])                      # heavy (high) zeros first for load balance

    t0 = time.time()
    results = []
    with Pool(workers) as p:
        for i, r in enumerate(p.imap_unordered(work, tasks), 1):
            results.append(r)
            if i % 200 == 0 or i == len(tasks):
                print(f"  [{i}/{len(tasks)}] {time.time()-t0:.0f}s", flush=True)

    results.sort(key=lambda r: r["n0"])
    flint.ctx.dps = 50
    # contiguity: batches cover 1..N exactly and are ordered by height
    assert results[0]["n0"] == 1 and sum(r["num"] for r in results) == N
    for a, b in zip(results, results[1:]):
        assert a["n0"] + a["num"] == b["n0"]
        assert arb(a["g_last"]) < arb(b["g_first"])
    S_hi, S_lo, A_lo = arb(0), arb(0), arb(0)
    for r in results:
        if r["dps"] == HI_DPS:
            S_hi += arb(r["S"])
        else:
            S_lo += arb(r["S"]); A_lo += arb(r["absS"])
    K2 = S_hi + S_lo
    summary = {
        "k": K_STR, "v": V_STR, "T0": T0_STR, "N_T0": N, "delta_lower": delta.str(10),
        "K2_main_(n<=%d)" % N_HI: S_hi.str(35, radius=True),
        "K2_tail_ball": S_lo.str(10, radius=True),
        "K2_tail_abs_bound": A_lo.upper().str(10, radius=False),
        "K2": K2.str(35, radius=True),
        "K2_lower": K2.lower().str(20, radius=False),
        "min_|zeta'(rho)|": min(r["min_abs_zeta_prime"] for r in results),
        "elapsed_s_this_run": round(time.time() - t0, 1),
    }
    print(json.dumps(summary, indent=2))
    with open(os.path.join(os.path.dirname(OUT), "k2_certificate.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
