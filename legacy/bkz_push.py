"""Progressive BKZ on Kim-Nguyen lattices, then distinct-y enumeration."""
import time
from multiprocessing import Pool

import gmpy2
from fpylll import BKZ, GSO, LLL
from fpylll.fplll.svpcvp import CVP

from diverse_enum import ITEMS, PI, fast_h
from lattice_search import build_lattice, gs_target, partial_h
from run_enum import enumerate_cpp


def one(job):
    N, nu, nu_y, nu_t, negative, max_block = job
    t0 = time.time()
    B, target = build_lattice(ITEMS, N, nu, nu_y, nu_t, PI, negative)
    LLL.reduction(B)
    notes = []
    beta = 20
    while beta <= max_block:
        BKZ.reduction(B, BKZ.Param(beta, max_loops=1))
        if beta % 8 == 0 or beta == max_block:
            v = CVP.babai(B, target)
            err2 = sum((int(v[i]) - target[i]) ** 2 for i in range(N + 1))
            step = 1 << nu_t
            last = int(v[-1])
            h = None
            y = None
            if last > 0 and last % step == 0:
                y = gmpy2.mpfr(last // step) * gmpy2.exp2(nu_y - nu)
                if gmpy2.mpfr("1e8") < y < gmpy2.mpfr("1e30"):
                    h = fast_h(y)
            notes.append(
                f"b{beta} t={time.time()-t0:.0f}s log2err={err2.bit_length()} "
                f"h={None if h is None else round(h, 4)} y={y}"
            )
        beta += 4
    # Enumerate around the reduced basis.
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
    radius2 = float(gmpy2.mpfr(err2) * (gmpy2.mpfr("1.12") ** 2))
    step = 1 << nu_t
    xcol = []
    ok = True
    for i in range(n):
        last = int(B[i, n - 1])
        if last % step != 0:
            ok = False
            break
        xcol.append(last // step)
    best = None
    nun = 0
    tail = "no-enum"
    if ok:
        bucket = max(1, int(0.05 * 2 ** max(0, nu - nu_y)))
        sols, err = enumerate_cpp(
            mu, r, sigma, radius2, [1.0] * n, 800, -1, bucket, xcol
        )
        seen = set()
        for dist, coeff in sols:
            u = B.multiply_left(list(coeff))
            last = int(u[-1])
            if last % step != 0 or last <= 0:
                continue
            y = gmpy2.mpfr(last // step) * gmpy2.exp2(nu_y - nu)
            if not (gmpy2.mpfr("1e12") < y < gmpy2.mpfr("1.95e19")):
                continue
            key = int(gmpy2.floor(y * 10))
            if key in seen:
                continue
            seen.add(key)
            nun += 1
            h = fast_h(y)
            if best is None or abs(h) > abs(best[0]):
                best = (h, str(y))
        tail = err.strip().splitlines()[-1] if err.strip() else ""
    return notes, best, nun, time.time() - t0, job, tail


def main():
    jobs = [
        (140, 130, 100, 30, True, 44),
        (140, 130, 97, 30, True, 44),
        (140, 130, 94, 30, True, 44),
        (120, 120, 92, 24, True, 48),
        (130, 125, 96, 28, True, 44),
        (140, 145, 112, 32, True, 40),
    ]
    print(f"jobs {len(jobs)}", flush=True)
    bestg = None
    with Pool(6) as pool:
        for i, (notes, best, nun, dt, job, tail) in enumerate(pool.imap_unordered(one, jobs), 1):
            print(f"--- job {job} t={dt:.0f}s uniq={nun} {tail}", flush=True)
            for line in notes:
                print(" ", line, flush=True)
            if best:
                print(f"  BEST {best[0]:.4f} y={best[1]}", flush=True)
                if bestg is None or abs(best[0]) > abs(bestg[0]):
                    bestg = (best[0], best[1], job)
                if abs(best[0]) > 0.98:
                    full = partial_h(gmpy2.mpfr(best[1]), ITEMS, len(ITEMS))
                    print(f"  FULL {full}", flush=True)
    print("DONE", bestg, flush=True)


if __name__ == "__main__":
    main()
