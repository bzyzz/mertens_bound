"""Enumerate the Kim-Nguyen quotient at a Gaussian-heuristic radius.

The lattice is nearly orthogonal, so a ball of radius gamma * GH has a
search tree dominated by one very short Gram-Schmidt direction. Collapsing
that direction leaves a few tens of thousands of genuinely different y.
"""
import math
import os
import time
from multiprocessing import Pool

import gmpy2
import numpy as np
from fpylll import BKZ, GSO, LLL
from fpylll.fplll.svpcvp import CVP

from lattice_search import build_lattice, gs_target, load_items, partial_h
from run_enum import enumerate_cpp

ROOT = os.path.dirname(os.path.abspath(__file__))
ITEMS, PI = load_items()
NFAST = 2500
K = gmpy2.mpfr("3e-9")
G = [ITEMS[i][1] for i in range(NFAST)]
WRE = np.empty(NFAST)
WIM = np.empty(NFAST)
for i in range(NFAST):
    astar, g, alpha, psi = ITEMS[i]
    w = float(alpha * gmpy2.exp(-K * g * g))
    WRE[i] = w * float(gmpy2.cos(psi))
    WIM[i] = w * float(gmpy2.sin(psi))
TWOPI = 2 * gmpy2.const_pi()
print(f"loaded {len(ITEMS)}", flush=True)


def fast_h(y):
    y = gmpy2.mpfr(y)
    ang = np.empty(NFAST)
    for i, g in enumerate(G):
        ang[i] = float((g * y) % TWOPI)
    return float(2.0 * np.sum(WRE * np.cos(ang) + WIM * np.sin(ang)))


def gh_radius(M):
    n = M.d
    logdet = 0.0
    for i in range(n):
        logdet += 0.5 * math.log(M.get_r(i, i))
    return math.exp(logdet / n) / math.sqrt(2 * math.pi * math.e)


def one(job):
    N, nu, nu_y, nu_t, negative, gamma = job
    t0 = time.time()
    B, target = build_lattice(ITEMS, N, nu, nu_y, nu_t, PI, negative)
    LLL.reduction(B)
    BKZ.reduction(B, BKZ.Param(32, max_loops=1))
    M = GSO.Mat(B, float_type="dpe")
    M.update_gso()
    n = N + 1
    gh = gh_radius(M)
    sigma = gs_target(B, target, M)
    mu = [[0.0] * n for _ in range(n)]
    r = [0.0] * n
    for i in range(n):
        r[i] = M.get_r(i, i)
        for j in range(i):
            mu[i][j] = M.get_mu(i, j)
    radius2 = (gamma * gh) ** 2
    step = 1 << nu_t
    xcol = []
    for i in range(n):
        last = int(B[i, n - 1])
        if last % step != 0:
            return ("bad-col", 0, time.time() - t0, job, "")
        xcol.append(last // step)
    # Bucket about 0.02 in y. y = x * 2^{nu_y-nu}, so dx = 0.02 * 2^{nu-nu_y}.
    bucket = max(1, int(0.02 * 2 ** (nu - nu_y)))
    sols, err = enumerate_cpp(
        mu, r, sigma, radius2, [1.0] * n, 250000, -1, bucket, xcol
    )
    best = None
    best_small = None
    seen = set()
    nun = 0
    for dist, coeff in sols:
        u = B.multiply_left(list(coeff))
        last = int(u[-1])
        if last % step != 0 or last <= 0:
            continue
        y = gmpy2.mpfr(last // step) * gmpy2.exp2(nu_y - nu)
        if not (gmpy2.mpfr("1e10") < y < gmpy2.mpfr("1e24")):
            continue
        key = int(gmpy2.floor(y * 20))
        if key in seen:
            continue
        seen.add(key)
        nun += 1
        h = fast_h(y)
        if best is None or abs(h) > abs(best[0]):
            best = (h, str(y))
        if y < gmpy2.mpfr("1.95e19") and (best_small is None or abs(h) > abs(best_small[0])):
            best_small = (h, str(y))
    tail = err.strip().splitlines()[-1] if err.strip() else ""
    return best, best_small, nun, len(sols), time.time() - t0, job, tail


def main():
    jobs = []
    for N, nu, nu_y, nu_t in (
        (140, 130, 100, 30),
        (140, 130, 98, 30),
        (140, 130, 96, 30),
        (140, 130, 94, 30),
        (120, 130, 100, 25),
        (140, 140, 108, 30),
        (140, 150, 116, 32),
        (130, 130, 97, 28),
    ):
        for negative in (True, False):
            for gamma in (1.25, 1.45):
                jobs.append((N, nu, nu_y, nu_t, negative, gamma))
    print(f"jobs {len(jobs)}", flush=True)
    best_small = None
    hits = os.path.join(ROOT, "gh_hits.txt")
    with Pool(8) as pool:
        for i, res in enumerate(pool.imap_unordered(one, jobs), 1):
            if res[0] == "bad-col":
                print(f"i={i} bad column {res}", flush=True)
                continue
            best, small, nun, ns, dt, job, tail = res
            line = f"i={i}/{len(jobs)} uniq={nun} sols={ns} t={dt:.0f}s {tail} job={job}"
            if best:
                line += f" |h|max={best[0]:.4f} y={best[1]}"
            if small:
                line += f" | below={small[0]:.4f} y={small[1]}"
            print(line, flush=True)
            with open(hits, "a") as f:
                f.write(line + "\n")
            if small and (best_small is None or abs(small[0]) > abs(best_small[0])):
                best_small = (small[0], small[1], job)
                print(f"NEW-SMALL {small[0]:.5f} y={small[1]}", flush=True)
                if abs(small[0]) > 0.99:
                    full = partial_h(gmpy2.mpfr(small[1]), ITEMS, len(ITEMS))
                    msg = f"FULL {full} y={small[1]} job={job}"
                    print(msg, flush=True)
                    with open(hits, "a") as f:
                        f.write(msg + "\n")
            if best and abs(best[0]) > 1.0:
                full = partial_h(gmpy2.mpfr(best[1]), ITEMS, len(ITEMS))
                msg = f"FULL-ANY {full} y={best[1]} job={job}"
                print(msg, flush=True)
                with open(hits, "a") as f:
                    f.write(msg + "\n")
    print("DONE", best_small, flush=True)


if __name__ == "__main__":
    main()
