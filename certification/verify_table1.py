#!/usr/bin/env python3
"""
Independent recomputation of the Saouter-te Riele "Table 1" bound on the range needed by J3:

    for s = 1/2 + 1/u + it,  T0 <= t <= T1:   log|1/zeta(s)| <= log u + 11 log t + M,   M = 14 - min_nu N_nu

using Pintz Lemma 4 (in real-part form: |log|zeta(s)| - sum_{|gamma-t|<c0} log|s-rho|| <= 11 log t + 14)
and certified Arb zeros. For t in I_nu = [theta_{nu-1}, theta_nu] (theta = midpoints of consecutive zeros):
  * the zero rho_nu contributes log|s - rho_nu| >= -log u          (real parts differ by 1/u);
  * every other zero contributes log|s - rho| >= log dist(gamma, I_nu); only negative values are kept
    (a zero at distance >= 1 contributes >= 0, so dropping it keeps a valid lower bound, and keeping a
    negative bound for a zero that is actually outside the c0-window only lowers the bound further).
  N_nu := sum of those negative logs.  The zero list is complete (consecutive Arb indices).

Bounds are serialised as enclosing Arb balls (certio.ball); chunk caches live in
table1_chunks/<tag>/, where the tag binds parameters, code and library versions, and each payload
is checked against its parameters when read. Any failed check exits nonzero.

Usage: verify_table1.py T0 T1 [workers] [--out table1_certificate.json]
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import flint
from flint import acb, arb

from certio import (CertificationError, ball, cache_root, cache_tag, exact_int, fail, manifest,
                    provenance, require)

BATCH = 2000
DPS = 15
PAD = 20                                   # zero gaps near 4e6 average ~0.35, max window needs < 1 unit
HERE = os.path.dirname(os.path.abspath(__file__))
CODE = [os.path.abspath(__file__), os.path.join(HERE, "certio.py")]


def zeros(n0, num):
    flint.ctx.dps = DPS
    return [z.imag for z in acb.zeta_zeros(n0, num)]


def work(args):
    n0, num, lo_idx, hi_idx, outdir, tag = args     # evaluate intervals I_nu for nu in [lo_idx, hi_idx)
    path = os.path.join(outdir, f"{lo_idx:09d}_{hi_idx}.json")
    if os.path.exists(path):
        with open(path) as f:
            res = json.load(f)
        require(res.get("tag") == tag and res["lo"] == lo_idx and res["hi"] == hi_idx
                and res["n0"] == n0 and res["num"] == num, f"cache payload {path} does not match its key")
        return res
    g = zeros(n0, num)                         # g[i] = gamma_{n0+i}; padded so windows of width 1 are covered
    for x, y in zip(g, g[1:]):
        require(x < y, f"zeros not strictly increasing near index {n0}")
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
            require(d > 0, f"distance not positive at nu={nu}")
            N += d.log()
            j -= 1
            require(j >= 0, "padding too small")
        j = i + 1
        while True:                            # zeros above
            d = (g[j] - b).lower()
            if not d < 1:
                break
            require(d > 0, f"distance not positive at nu={nu}")
            N += d.log()
            j += 1
            require(j < len(g), "padding too small")
        Nl = N.lower()
        require(Nl.is_finite(), f"non-finite N at nu={nu}")
        if worst is None or Nl < worst:
            worst, worst_nu = Nl, nu
    res = {"tag": tag, "n0": n0, "num": num, "lo": lo_idx, "hi": hi_idx,
           "min_N_lower": ball(worst, 25), "at_nu": worst_nu,
           "theta_lo": ball((g[lo_idx - n0 - 1] + g[lo_idx - n0]) / 2, 20),
           "theta_hi": ball((g[hi_idx - 1 - n0] + g[hi_idx - n0]) / 2, 20)}
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(res, f)
    os.replace(tmp, path)
    return res


def main():
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    out = "table1_certificate.json"
    if "--out" in sys.argv:
        out = sys.argv[sys.argv.index("--out") + 1]
        args.remove(out)
    T0s, T1s = args[0], args[1]
    T0, T1 = arb(T0s), arb(T1s)
    workers = int(args[2]) if len(args) > 2 else 24
    params = {"T0": T0s, "T1": T1s, "BATCH": BATCH, "DPS": DPS, "PAD": PAD}
    tag = cache_tag(params, CODE)
    outdir = os.path.join(cache_root(HERE), "table1_chunks", tag)
    os.makedirs(outdir, exist_ok=True)
    flint.ctx.dps = 30
    # intervals I_nu covering [T0, T1]: nu from N(T0) (I_nu contains T0 or starts below) to N(T1)+1
    nu_lo = exact_int(T0.zeta_nzeros(), "N(T0)")
    nu_hi = exact_int(T1.zeta_nzeros(), "N(T1)") + 2
    tasks = []
    for lo in range(nu_lo, nu_hi, BATCH):
        hi = min(lo + BATCH, nu_hi)
        tasks.append((lo - PAD, hi - lo + 2 * PAD, lo, hi, outdir, tag))
    print(f"intervals nu = {nu_lo} .. {nu_hi - 1}  ({len(tasks)} tasks), cache tag {tag}", flush=True)
    t0 = time.time()
    res = []
    with Pool(workers) as p:
        for i, r in enumerate(p.imap_unordered(work, tasks), 1):
            res.append(r)
            if i % 100 == 0 or i == len(tasks):
                print(f"  [{i}/{len(tasks)}] {time.time()-t0:.0f}s", flush=True)
    res.sort(key=lambda r: r["lo"])
    require(res[0]["lo"] == nu_lo and res[-1]["hi"] == nu_hi, "chunks do not cover all intervals")
    for a, b in zip(res, res[1:]):
        require(a["hi"] == b["lo"], f"gap between chunks at nu={a['hi']}")
    require(arb(res[0]["theta_lo"]) < T0 and arb(res[-1]["theta_hi"]) > T1, "intervals do not cover [T0, T1]")
    w = min(res, key=lambda r: arb(r["min_N_lower"]).lower())
    minN = arb(w["min_N_lower"]).lower()
    M = 14 - minN
    summary = provenance(params, CODE)
    summary.update({"T0": T0s, "T1": T1s, "intervals": nu_hi - nu_lo, "N(T0)": nu_lo, "N(T1)": nu_hi - 2,
                    "min_N_lower": ball(minN, 25), "at_nu": w["at_nu"],
                    "M_ball": ball(M, 25), "M_upper_display": M.upper().str(12, radius=False) + " (display only)",
                    "StR_Table1_M": "23.05", "chunk_manifest_sha256": manifest(res),
                    "elapsed_s_this_run": round(time.time() - t0, 1)})
    print(json.dumps(summary, indent=2))
    with open(os.path.join(HERE, out), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    try:
        main()
    except CertificationError as e:
        fail(str(e))
