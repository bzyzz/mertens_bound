#!/usr/bin/env python3
"""
Rigorous ball-arithmetic evaluation of

    K2(v, T0, k) = 2 * sum_{0 < gamma < T0} exp(-k gamma^2) Re( e^{i gamma v} / (rho zeta'(rho)) )

over *all* zeros below T0, as required by Lemma H (residue theorem) in THEOREM.md.

Every zero is computed with Arb's certified zero isolation (acb.zeta_zeros ->
acb_dirichlet_zeta_zeros: the n-th returned ball contains the n-th zero, which is simple and on
the critical line), and the zero count below T0 is the rigorous N(T0) from arb.zeta_nzeros.

Zeros 1..N_HI are computed at HI_DPS digits (phases gamma*v resolved to ~1e-20 rad);
zeros N_HI+1..N(T0) at LO_DPS digits, where the phase ball is [0, 2pi] and each term
contributes only its rigorous magnitude bound 2 e^{-k gamma^2} / |rho zeta'(rho)| (StR (8.7)).

Caching (both resumable, both safe to delete):
  zero_cache/   certified balls for gamma and c = 1/(rho zeta'(rho)), independent of (k, v).
                Written the first time a batch is computed; afterwards certifying a new point v
                costs only ball arithmetic (minutes instead of ~45 min).
  k2_chunks/<tag>/  per-batch partial sums; the tag binds (k, v, T0, precisions, batch size), the code and
                the library versions, and every cached payload is checked against its key when read.
  The zero cache carries MANIFEST.json (library versions); a mismatch stops the run.
  Set CERT_CACHE=<dir> to place both caches elsewhere (e.g. for a run from empty caches).

All certified quantities are written as enclosing Arb balls; every failed check exits nonzero.

Usage: certify_K2.py [workers] [--v V] [--k K] [--out CERT.json] [--cache-only]
       (defaults: Peak2, k = 1.5e-10, k2_certificate.json)
"""
import argparse, json, os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
from multiprocessing import Pool
import flint
from flint import arb, acb, acb_series

from certio import (CertificationError, ball, cache_root, cache_tag, exact_int, fail, manifest,
                    provenance, require, versions)

K_STR = "1.5e-10"
V_STR = "3484971332733132396.5383341489440461"
T0_STR = "1300000"
N_HI = 600000
HI_DPS, LO_DPS = 45, 20
BATCH = 250
HERE = os.path.dirname(os.path.abspath(__file__))
ZCACHE = os.path.join(cache_root(HERE), "zero_cache")
KCACHE = os.path.join(cache_root(HERE), "k2_chunks")
CODE = [os.path.abspath(__file__), os.path.join(HERE, "certio.py")]


def check_zero_cache_manifest():
    """The zero cache holds certified balls that depend only on the library; bind it to its versions."""
    os.makedirs(ZCACHE, exist_ok=True)
    mpath = os.path.join(ZCACHE, "MANIFEST.json")
    want = {"python_flint": versions()["python_flint"], "flint": versions()["flint"], "format": 1}
    if os.path.exists(mpath):
        have = json.load(open(mpath))
        require(have == want, f"zero cache {ZCACHE} was built with {have}, running {want}; use a new CERT_CACHE")
    else:
        with open(mpath, "w") as f:
            json.dump(want, f)


def zeros_and_coeffs(n0, num, dps):
    """Certified (gamma, c = 1/(rho zeta'(rho)), lower bound of |zeta'(rho)|) for zeros n0..n0+num-1."""
    path = os.path.join(ZCACHE, f"{n0:08d}_{num}_{dps}.txt")
    flint.ctx.dps = dps
    if os.path.exists(path):
        out = []
        with open(path) as f:
            for line in f:
                g, cr, ci, zl = line.rstrip("\n").split("\t")
                out.append((arb(g), acb(arb(cr), arb(ci)), float(zl)))
        require(len(out) == num, f"zero cache {path} has {len(out)} entries, expected {num}")
        for (g1, c1, _), (g2, _, _) in zip(out, out[1:]):
            require(g1 < g2, f"zero cache {path} not strictly increasing")
        require(all(c.is_finite() for _, c, _ in out), f"zero cache {path} has a non-finite coefficient")
        return out
    out, lines = [], []
    for rho in acb.zeta_zeros(n0, num):
        require(rho.real == arb("0.5"), f"zero near index {n0} not certified on the critical line")
        g = rho.imag
        zp = acb_series([rho, 1], prec=2).zeta()[1]
        require(abs(zp).lower() > 0, f"zeta'(rho) enclosure contains 0 near index {n0}")
        c = 1 / (rho * zp)
        require(c.is_finite(), f"non-finite 1/(rho zeta'(rho)) near index {n0}")
        zl = float(abs(zp).lower())
        out.append((g, c, zl))
        # arb.str(radius=True) output is guaranteed to enclose the ball, so reloading is rigorous
        lines.append(f"{g.str(dps + 5, radius=True)}\t{c.real.str(30, radius=True)}\t{c.imag.str(30, radius=True)}\t{zl!r}\n")
    os.makedirs(ZCACHE, exist_ok=True)
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "w") as f:
        f.writelines(lines)
    os.replace(tmp, path)
    return out


def work(args):
    n0, num, dps, k_str, v_str, outdir, tag, cache_only = args
    path = os.path.join(outdir, f"{n0:08d}_{num}_{dps}.json")
    if not cache_only and os.path.exists(path):
        with open(path) as f:
            res = json.load(f)
        require(res.get("tag") == tag and res["n0"] == n0 and res["num"] == num and res["dps"] == dps
                and res.get("k") == k_str and res.get("v") == v_str, f"cache payload {path} does not match its key")
        return res
    zs = zeros_and_coeffs(n0, num, dps)
    if cache_only:
        return {"n0": n0, "num": num}
    flint.ctx.dps = dps
    k, v = arb(k_str), arb(v_str)
    S, absS = arb(0), arb(0)
    max_rad = 0.0
    for g, c, _ in zs:
        w = (-k * g * g).exp()
        term = 2 * w * (c * acb(0, g * v).exp()).real
        S += term
        absS += 2 * w * abs(c)
        max_rad = max(max_rad, float(term.rad()))
    require(S.is_finite() and absS.is_finite(), f"non-finite partial sum at batch {n0}")
    res = {
        "tag": tag, "k": k_str, "v": v_str, "n0": n0, "num": num, "dps": dps,
        "S": S.str(40, radius=True), "absS": absS.str(20, radius=True),
        "g_first": zs[0][0].str(45, radius=True), "g_last": zs[-1][0].str(45, radius=True),
        "min_abs_zeta_prime": min(z[2] for z in zs), "max_term_rad": max_rad,
    }
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(res, f)
    os.replace(tmp, path)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workers", nargs="?", type=int, default=24)
    ap.add_argument("--v", default=V_STR)
    ap.add_argument("--k", default=K_STR)
    ap.add_argument("--out", default="k2_certificate.json")
    ap.add_argument("--cache-only", action="store_true", help="only populate zero_cache/")
    a = ap.parse_args()

    params = {"k": a.k, "v": a.v, "T0": T0_STR, "N_HI": N_HI, "HI_DPS": HI_DPS, "LO_DPS": LO_DPS, "BATCH": BATCH}
    tag = cache_tag(params, CODE)
    outdir = os.path.join(KCACHE, tag)
    os.makedirs(outdir, exist_ok=True)
    check_zero_cache_manifest()
    flint.ctx.dps = 30
    T0 = arb(T0_STR)
    N = exact_int(T0.zeta_nzeros(), "N(T0)")
    nb = acb.zeta_zeros(N, 2)
    require(nb[0].imag < T0 < nb[1].imag, "T0 not strictly between gamma_N and gamma_{N+1}")
    delta = min((T0 - nb[0].imag).lower(), (nb[1].imag - T0).lower())
    require(delta > 0, "cutoff gap not certified positive")
    print(f"v = {a.v}, k = {a.k} (cache tag {tag})", flush=True)
    print(f"N(T0={T0_STR}) = {N}; gamma_N = {nb[0].imag.str(15)}, gamma_N+1 = {nb[1].imag.str(15)}; delta >= {delta.str(8)}", flush=True)

    tasks = []
    for n0 in range(1, N + 1, BATCH):
        num = min(BATCH, N + 1 - n0)
        if n0 <= N_HI < n0 + num - 1:
            raise SystemExit("BATCH must divide N_HI")
        tasks.append((n0, num, HI_DPS if n0 <= N_HI else LO_DPS, a.k, a.v, outdir, tag, a.cache_only))
    tasks.sort(key=lambda t: -t[0])                      # heavy (high) zeros first for load balance

    t0 = time.time()
    results = []
    with Pool(a.workers) as p:
        for i, r in enumerate(p.imap_unordered(work, tasks), 1):
            results.append(r)
            if i % 500 == 0 or i == len(tasks):
                print(f"  [{i}/{len(tasks)}] {time.time()-t0:.0f}s", flush=True)
    if a.cache_only:
        print(f"zero cache complete: {len(results)} batches in {time.time()-t0:.0f}s")
        return

    results.sort(key=lambda r: r["n0"])
    flint.ctx.dps = 50
    # contiguity: batches cover 1..N exactly and are ordered by height
    require(results[0]["n0"] == 1 and sum(r["num"] for r in results) == N, "batches do not cover 1..N(T0)")
    for r1, r2 in zip(results, results[1:]):
        require(r1["n0"] + r1["num"] == r2["n0"], f"gap between batches at n={r2['n0']}")
        require(arb(r1["g_last"]) < arb(r2["g_first"]), f"batches not ordered by height at n={r2['n0']}")
    S_hi, S_lo, A_lo = arb(0), arb(0), arb(0)
    for r in results:
        if r["dps"] == HI_DPS:
            S_hi += arb(r["S"])
        else:
            S_lo += arb(r["S"]); A_lo += arb(r["absS"])
    K2 = S_hi + S_lo
    summary = provenance(params, CODE)
    summary.update({
        "k": a.k, "v": a.v, "T0": T0_STR, "N_T0": N, "delta_lower": ball(delta, 15),
        "K2_main_(n<=%d)" % N_HI: ball(S_hi, 35),
        "K2_tail_ball": ball(S_lo, 15),
        "K2_tail_abs_bound": ball(A_lo, 15),
        "K2": ball(K2, 35),
        "K2_lower_ball": ball(K2.lower(), 25),
        "min_|zeta'(rho)|_diagnostic": min(r["min_abs_zeta_prime"] for r in results),
        "chunk_manifest_sha256": manifest(results),
        "elapsed_s_this_run": round(time.time() - t0, 1),
    })
    print(json.dumps(summary, indent=2))
    with open(os.path.join(HERE, a.out), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    try:
        main()
    except CertificationError as e:
        fail(str(e))
