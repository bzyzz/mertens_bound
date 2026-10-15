"""High-radius CVP on random subsets of the heaviest zeros."""
import os
import random
import time
from multiprocessing import Pool

import gmpy2
import numpy as np
from fpylll import BKZ, Enumeration, GSO, IntegerMatrix, LLL
from fpylll.fplll.svpcvp import CVP

from lattice_search import load_items, partial_h

ROOT = os.path.dirname(os.path.abspath(__file__))
ITEMS, PI = load_items()
K = gmpy2.mpfr("3e-9")
TWOPI = 2 * gmpy2.const_pi()
NFAST = 6000
G, WRE, WIM = [], [], []
for _, g, alpha, psi in ITEMS[:NFAST]:
    w = float(alpha * gmpy2.exp(-K * g * g))
    G.append(g)
    WRE.append(w * float(gmpy2.cos(psi)))
    WIM.append(w * float(gmpy2.sin(psi)))
WRE = np.asarray(WRE)
WIM = np.asarray(WIM)


def fast_h(y):
    y = gmpy2.mpfr(y)
    ang = np.empty(NFAST)
    for i, g in enumerate(G):
        ang[i] = float((g * y) % TWOPI)
    return float(2.0 * np.sum(WRE * np.cos(ang) + WIM * np.sin(ang)))


def trial(args):
    seed, N, nu_y, nu_t, rho, negative, nsols = args
    rng = random.Random(seed)
    nu = nu_y + 12
    # Always keep the heaviest 18; fill the rest from the next 80.
    rest = list(range(18, 100))
    rng.shuffle(rest)
    subset = [ITEMS[i] for i in list(range(18)) + rest[: N - 18]]
    subset.sort(key=lambda z: -z[0])
    scale_nu = gmpy2.exp2(nu)
    scale_y = gmpy2.exp2(nu_y)
    B = IntegerMatrix(N + 1, N + 1)
    target = [0] * (N + 1)
    for i, (astar, g, alpha, psi) in enumerate(subset):
        s = gmpy2.sqrt(astar)
        phase = psi + (PI if negative else 0)
        B[0, i] = int(gmpy2.floor(s * g * scale_y))
        B[i + 1, i] = int(gmpy2.floor(s * 2 * PI * scale_nu))
        target[i] = int(gmpy2.floor(s * phase * scale_nu))
    B[0, N] = 1 << nu_t
    LLL.reduction(B)
    BKZ.reduction(B, BKZ.Param(18, max_loops=1))
    v = CVP.babai(B, target)
    err2 = sum((int(v[j]) - target[j]) ** 2 for j in range(N + 1))
    M = GSO.Mat(B, float_type="dpe")
    M.update_gso()
    n = N + 1
    sigma = [0.0] * n
    for i in range(n):
        ip = sum(target[k] * int(B[i, k]) for k in range(n))
        s = float(ip)
        for j in range(i):
            s -= sigma[j] * M.get_r(j, j) * M.get_mu(i, j)
        sigma[i] = s / M.get_r(i, i)
    bound = int(gmpy2.mpfr(err2) * gmpy2.mpfr(rho) ** 2) + 1
    expo = bound.bit_length() - 1
    mant = float(bound) / 2**expo
    sols = Enumeration(M, nr_solutions=nsols).enumerate(0, n, mant, expo, target=sigma)
    step = 1 << nu_t
    best = None
    seen = set()
    for dist, coeff in sols:
        c = [int(round(x)) for x in coeff]
        u = B.multiply_left(c)
        last = int(u[-1])
        if last % step != 0 or last <= 0:
            continue
        y = gmpy2.mpfr(last // step) * gmpy2.exp2(nu_y - nu)
        if y < gmpy2.mpfr("1e14") or y > gmpy2.mpfr("1.95e19"):
            continue
        key = int(gmpy2.floor(y * 8))
        if key in seen:
            continue
        seen.add(key)
        h = fast_h(y)
        if best is None or abs(h) > abs(best[0]):
            best = (h, str(y))
    return best, len(seen), seed


def main():
    jobs = []
    sid = 0
    for N, rho, nsols in ((42, 1.26, 6000), (44, 1.22, 5000)):
        for nu_y, nu_t in ((80, 14), (82, 14), (84, 16), (78, 14)):
            for neg in (True, False):
                for _ in range(4):
                    jobs.append((sid, N, nu_y, nu_t, rho, neg, nsols))
                    sid += 1
    print(f"jobs {len(jobs)}", flush=True)
    best = None
    t0 = time.time()
    with Pool(16) as pool:
        for i, (b, nuniq, seed) in enumerate(pool.imap_unordered(trial, jobs), 1):
            if b and (best is None or abs(b[0]) > abs(best[0])):
                best = b
                print(f"NEW {b[0]:.6f} y={b[1]} uniq={nuniq} i={i}", flush=True)
                with open(os.path.join(ROOT, "subset_hits.txt"), "a") as f:
                    f.write(f"{b[0]} {b[1]}\n")
            elif i % 6 == 0:
                print(
                    f"i={i}/{len(jobs)} best={None if best is None else round(best[0], 4)} "
                    f"t={time.time()-t0:.0f}",
                    flush=True,
                )
    print("DONE", best, flush=True)
    if best and abs(best[0]) >= 0.97:
        h = partial_h(best[1], ITEMS, len(ITEMS))
        print("FULL", float(h), best[1], flush=True)


if __name__ == "__main__":
    main()
