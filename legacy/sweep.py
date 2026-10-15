"""Many short CVP enumerations, scored on the top weights, then certified.

Prints every candidate with |h| large. Appends them to hits.txt.
"""
import os
import time
from multiprocessing import Pool

import gmpy2
import numpy as np
from fpylll import BKZ, GSO, LLL
from fpylll.fplll.svpcvp import CVP

from lattice_search import build_lattice, gs_target, load_items, partial_h

ROOT = os.path.dirname(os.path.abspath(__file__))
ITEMS, PI = load_items()
# Precompute a fast scorer on the heaviest terms.
G = []
WRE = []
WIM = []
K = gmpy2.mpfr("3e-9")
for astar, g, alpha, psi in ITEMS[:4000]:
    # coeff reconstructed as alpha * exp(-i psi)
    re = alpha * gmpy2.cos(-psi)  # not used
    G.append(g)
WRE = np.zeros(4000)
WIM = np.zeros(4000)
for i, (astar, g, alpha, psi) in enumerate(ITEMS[:4000]):
    w = float(alpha * gmpy2.exp(-K * g * g))
    # cos(g y - psi) = cos(gy)cos psi + sin(gy)sin psi
    WRE[i] = w * float(gmpy2.cos(psi))
    WIM[i] = w * float(gmpy2.sin(psi))
TWOPI = 2 * gmpy2.const_pi()
print(f"loaded {len(ITEMS)} zeros, fast terms 4000", flush=True)


def fast_h(y):
    y = gmpy2.mpfr(y)
    # reduce angles for 4000 terms
    ang = np.empty(4000)
    for i, g in enumerate(G):
        ang[i] = float((g * y) % TWOPI)
    return float(2.0 * np.sum(WRE * np.cos(ang) + WIM * np.sin(ang)))


def one(job):
    N, nu, nu_y, nu_t, rho, negative, nsols = job
    t0 = time.time()
    B, target = build_lattice(ITEMS, N, nu, nu_y, nu_t, PI, negative)
    LLL.reduction(B)
    BKZ.reduction(B, BKZ.Param(20, max_loops=1))
    v = CVP.babai(B, target)
    err2 = sum((int(v[i]) - target[i]) ** 2 for i in range(N + 1))
    M = GSO.Mat(B, float_type="dpe")
    M.update_gso()
    sigma = gs_target(B, target, M)
    bound = int(gmpy2.mpfr(err2) * (gmpy2.mpfr(rho) ** 2)) + 1
    expo = bound.bit_length() - 1
    mant = float(bound) / 2**expo
    from fpylll import Enumeration

    sols = Enumeration(M, nr_solutions=nsols).enumerate(0, N + 1, mant, expo, target=sigma)
    step = 1 << nu_t
    best = None
    seen = set()
    nunique = 0
    for dist, coeff in sols:
        c = [int(round(x)) for x in coeff]
        u = B.multiply_left(c)
        last = int(u[-1])
        if last % step != 0 or last <= 0:
            continue
        y = gmpy2.mpfr(last // step) * gmpy2.exp2(nu_y - nu)
        if y < gmpy2.mpfr("1e12") or y > gmpy2.mpfr("1e22"):
            continue
        key = int(gmpy2.floor(y * 10))
        if key in seen:
            continue
        seen.add(key)
        nunique += 1
        h = fast_h(y)
        if best is None or abs(h) > abs(best[0]):
            best = (h, y)
    dt = time.time() - t0
    return {
        "job": job,
        "best": None if best is None else (best[0], str(best[1])),
        "nunique": nunique,
        "nsols": len(sols),
        "dt": dt,
    }


def main():
    jobs = []
    for N, rho, nsols in ((40, 1.18, 4000), (44, 1.14, 3000), (36, 1.22, 5000)):
        for nu_t in (14, 16, 18, 20):
            for nu_y in range(nu_t + 50, nu_t + 76, 2):
                for neg in (True, False):
                    jobs.append((N, nu_y + 12, nu_y, nu_t, rho, neg, nsols))
    print(f"jobs {len(jobs)}", flush=True)
    hits = []
    t0 = time.time()
    with Pool(28) as pool:
        for i, res in enumerate(pool.imap_unordered(one, jobs), 1):
            best = res["best"]
            mark = ""
            if best and abs(best[0]) >= 0.97:
                hits.append(res)
                mark = " HIT"
                with open(os.path.join(ROOT, "hits.txt"), "a") as f:
                    f.write(repr(res) + "\n")
            if best and (abs(best[0]) >= 0.93 or i % 10 == 0):
                print(
                    f"{i}/{len(jobs)} {mark} h={best[0] if best else None} "
                    f"y={best[1] if best else None} uniq={res['nunique']} "
                    f"dt={res['dt']:.1f} job={res['job']}",
                    flush=True,
                )
    print("sweep done", round(time.time() - t0, 1), "hits", len(hits), flush=True)
    # certify hits
    for res in sorted(hits, key=lambda r: -abs(r["best"][0])):
        h_fast, ys = res["best"]
        h = partial_h(gmpy2.mpfr(ys), ITEMS, min(len(ITEMS), 20000))
        print(f"CERT fast={h_fast} full={h} y={ys}", flush=True)


if __name__ == "__main__":
    main()
