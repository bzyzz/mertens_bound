#!/usr/bin/env python3
"""
Independent recomputation of the Saouter-te Riele "Table 1" bound on the range needed by J3:

    for s = 1/2 + 1/u + it,  T0 <= t <= T1:   log|1/zeta(s)| <= log u + 11 log t + M,   M = 14 - min_nu N_nu

using Pintz Lemma 4 (log zeta(s) = sum_{|gamma-t|<=c0} log(s-rho) + theta(11 log t + 14)) and certified
Arb zeros. For t in I_nu = [theta_{nu-1}, theta_nu] (theta = midpoints of consecutive zeros):
  * the zero rho_nu contributes log|s - rho_nu| >= -log u          (real parts differ by 1/u);
  * every other zero contributes log|s - rho| >= log dist(gamma, I_nu); only negative values are kept
    (a zero at distance >= 1 contributes >= 0, so dropping it keeps a valid lower bound, and keeping a
    negative bound for a zero that is actually outside the c0-window only lowers the bound further).
  N_nu := sum of those negative logs.  The zero list is complete (consecutive Arb indices).

Usage: verify_table1.py T0 T1 [workers]
"""
import os, sys, json, time
from multiprocessing import Pool
import flint
from flint import arb, acb

BATCH = 2000
DPS = 15
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "table1_chunks")


def zeros(n0, num):
    flint.ctx.dps = DPS
    return [z.imag for z in acb.zeta_zeros(n0, num)]


def work(args):
    n0, num, lo_idx, hi_idx = args            # evaluate intervals I_nu for nu in [lo_idx, hi_idx)
    path = os.path.join(OUT, f"{lo_idx:09d}_{hi_idx}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    g = zeros(n0, num)                         # g[i] = gamma_{n0+i}; padded so windows of width 1 are covered
    worst, worst_nu = None, None
    for nu in range(lo_idx, hi_idx):
        i = nu - n0
        a = (g[i - 1] + g[i]) / 2              # theta_{nu-1}
        b = (g[i] + g[i + 1]) / 2              # theta_nu
        N = arb(0)
        j = i - 1
        while True:                            # zeros below
            d = (a - g[j]).lower()
            if not d < 1:
                break
            assert d > 0
            N += d.log()
            j -= 1
            assert j >= 0, "padding too small"
        j = i + 1
        while True:                            # zeros above
            d = (g[j] - b).lower()
            if not d < 1:
                break
            assert d > 0
            N += d.log()
            j += 1
            assert j < len(g), "padding too small"
        Nl = N.lower()
        if worst is None or Nl < worst:
            worst, worst_nu = Nl, nu
    res = {"lo": lo_idx, "hi": hi_idx, "min_N": worst.str(12, radius=False), "at_nu": worst_nu,
           "theta_lo": ((g[lo_idx - n0 - 1] + g[lo_idx - n0]) / 2).str(15),
           "theta_hi": ((g[hi_idx - 1 - n0] + g[hi_idx - n0]) / 2).str(15)}
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(res, f)
    os.replace(tmp, path)
    return res


def main():
    T0, T1 = arb(sys.argv[1]), arb(sys.argv[2])
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    os.makedirs(OUT, exist_ok=True)
    flint.ctx.dps = 30
    # intervals I_nu covering [T0, T1]: nu from N(T0) (I_nu contains T0 or starts below) to N(T1)+1
    nu_lo = int(T0.zeta_nzeros().mid().floor().str(30, radius=False).split(".")[0])
    nu_hi = int(T1.zeta_nzeros().mid().floor().str(30, radius=False).split(".")[0]) + 2
    PAD = 20                                   # zero gaps near 4e6 average ~0.35, max window needs < 1 unit
    tasks = []
    for lo in range(nu_lo, nu_hi, BATCH):
        hi = min(lo + BATCH, nu_hi)
        tasks.append((lo - PAD, hi - lo + 2 * PAD, lo, hi))
    print(f"intervals nu = {nu_lo} .. {nu_hi - 1}  ({len(tasks)} tasks)", flush=True)
    t0 = time.time()
    res = []
    with Pool(workers) as p:
        for i, r in enumerate(p.imap_unordered(work, tasks), 1):
            res.append(r)
            if i % 100 == 0 or i == len(tasks):
                print(f"  [{i}/{len(tasks)}] {time.time()-t0:.0f}s", flush=True)
    res.sort(key=lambda r: r["lo"])
    for a, b in zip(res, res[1:]):
        assert a["hi"] == b["lo"]
    assert arb(res[0]["theta_lo"]) < T0 and arb(res[-1]["theta_hi"]) > T1
    w = min(res, key=lambda r: float(r["min_N"]))
    M = 14 - arb(w["min_N"])
    summary = {"T0": sys.argv[1], "T1": sys.argv[2], "intervals": nu_hi - nu_lo,
               "min_N": w["min_N"], "at_nu": w["at_nu"], "M = 14 - min_N": M.upper().str(10, radius=False),
               "StR_Table1_M": "23.05", "elapsed_s_this_run": round(time.time() - t0, 1)}
    print(json.dumps(summary, indent=2))
    with open(os.path.join(os.path.dirname(OUT), "table1_certificate.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
