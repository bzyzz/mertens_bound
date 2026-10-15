"""Enumerate distinct y-buckets inside a CVP ball and score them quickly."""
import math
import os
import time
from multiprocessing import Pool

import gmpy2
import numpy as np
from fpylll import GSO
from fpylll.fplll.svpcvp import CVP

from fpylll import BKZ, LLL

from lattice_search import build_lattice, gs_target, load_items, partial_h
from run_enum import enumerate_cpp

ROOT = os.path.dirname(os.path.abspath(__file__))
ITEMS, PI = load_items()
NFAST = 5000
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
print(f"loaded {len(ITEMS)} zeros", flush=True)


def fast_h(y):
    y = gmpy2.mpfr(y)
    ang = np.empty(NFAST)
    for i, g in enumerate(G):
        ang[i] = float((g * y) % TWOPI)
    return float(2.0 * np.sum(WRE * np.cos(ang) + WIM * np.sin(ang)))


def one(job):
    N, nu, nu_y, nu_t, rho, negative, max_sols = job
    t0 = time.time()
    B, target = build_lattice(ITEMS, N, nu, nu_y, nu_t, PI, negative)
    LLL.reduction(B)
    BKZ.reduction(B, BKZ.Param(20, max_loops=1))
    v = CVP.babai(B, target)
    err2 = sum((int(v[i]) - target[i]) ** 2 for i in range(N + 1))
    M = GSO.Mat(B, float_type="dpe")
    M.update_gso()
    n = N + 1
    sigma = gs_target(B, target, M)
    mu = [[0.0] * n for _ in range(n)]
    r = [0.0] * n
    for i in range(n):
        r[i] = M.get_r(i, i)
        for j in range(i):
            mu[i][j] = M.get_mu(i, j)
    radius2 = float(gmpy2.mpfr(err2) * (gmpy2.mpfr(rho) ** 2))
    step = 1 << nu_t
    xcol = []
    for i in range(n):
        last = int(B[i, n - 1])
        if last % step != 0:
            return ("bad-col", job, None)
        xcol.append(last // step)
    bucket = max(1, int(0.02 * 2 ** (nu - nu_y)))
    sols, err = enumerate_cpp(
        mu, r, sigma, radius2, [1.0] * n, max_sols, short_level=-1, bucket=bucket, xcol=xcol
    )
    best = None
    nunique = 0
    seen = set()
    for dist, coeff in sols:
        u = B.multiply_left(list(coeff))
        last = int(u[-1])
        if last % step != 0 or last <= 0:
            continue
        y = gmpy2.mpfr(last // step) * gmpy2.exp2(nu_y - nu)
        if not (gmpy2.mpfr("1e14") < y < gmpy2.mpfr("1.95e19")):
            continue
        key = int(gmpy2.floor(y * 20))
        if key in seen:
            continue
        seen.add(key)
        nunique += 1
        h = fast_h(y)
        if best is None or abs(h) > abs(best[0]):
            best = (h, y)
    dt = time.time() - t0
    return (best, nunique, len(sols), dt, job, err.strip().splitlines()[-1] if err.strip() else "")


def main():
    jobs = []
    for N, rhos, sols in (
        (44, (1.22, 1.36), 800),
        (50, (1.18, 1.28), 700),
        (58, (1.16, 1.24), 600),
        (66, (1.12, 1.20), 500),
        (74, (1.10, 1.16), 400),
    ):
        for nu_t in (14, 16, 18):
            for extra in (58, 64, 70):
                nu_y = nu_t + extra
                nu = nu_y + 12
                for negative in (True, False):
                    for rho in rhos:
                        jobs.append((N, nu, nu_y, nu_t, rho, negative, sols))
    print(f"jobs {len(jobs)}", flush=True)
    best_global = None
    t0 = time.time()
    hits = os.path.join(ROOT, "diverse_hits.txt")
    with Pool(16) as pool:
        for i, res in enumerate(pool.imap_unordered(one, jobs), 1):
            best, nunique, nsols, dt, job, tail = res
            N, nu, nu_y, nu_t, rho, negative, max_sols = job
            line = (
                f"i={i}/{len(jobs)} N={N} nu_t={nu_t} extra={nu_y - nu_t} "
                f"rho={rho} {'neg' if negative else 'pos'} uniq={nunique} "
                f"sols={nsols} t={dt:.0f}s {tail}"
            )
            if best is not None:
                h, y = best
                line += f" h={h:.4f} y={y}"
                if best_global is None or abs(h) > abs(best_global[0]):
                    best_global = (h, y, job)
                    print("NEW " + line, flush=True)
                    with open(hits, "a") as f:
                        f.write(line + "\n")
                    if abs(h) > 0.97:
                        full = partial_h(y, ITEMS, len(ITEMS))
                        msg = f"FULL {full} y={y} job={job}"
                        print(msg, flush=True)
                        with open(hits, "a") as f:
                            f.write(msg + "\n")
            if i % 8 == 0:
                bg = None if best_global is None else f"{best_global[0]:.4f}"
                print(f"{line} | best={bg} elapsed={time.time() - t0:.0f}", flush=True)
    print("DONE", best_global, flush=True)


if __name__ == "__main__":
    main()
