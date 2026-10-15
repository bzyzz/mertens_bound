"""Enumerate lattices aimed at the zeros that are still misaligned.

Each trial builds a Kim-Nguyen lattice on a chosen subset, enumerates a
modest CVP ball, and scores h_StR. Subsets mix the heaviest zeros with
those whose phases are worst at the current best y.
"""
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
# Fast score on the heaviest 5000 terms. Full certification is done later.
NFAST = 5000
G, WRE, WIM = [], [], []
for astar, g, alpha, psi in ITEMS[:NFAST]:
    w = float(alpha * gmpy2.exp(-K * g * g))
    G.append(g)
    WRE.append(w * float(gmpy2.cos(psi)))
    WIM.append(w * float(gmpy2.sin(psi)))
WRE = np.asarray(WRE)
WIM = np.asarray(WIM)
print(f"loaded {len(ITEMS)}", flush=True)


def fast_h(y):
    y = gmpy2.mpfr(y)
    ang = np.empty(NFAST)
    for i, g in enumerate(G):
        ang[i] = float((g * y) % TWOPI)
    return float(2.0 * np.sum(WRE * np.cos(ang) + WIM * np.sin(ang)))


def defects(y, m=200):
    y = gmpy2.mpfr(y)
    out = []
    for i, (astar, g, alpha, psi) in enumerate(ITEMS[:m]):
        w = alpha * gmpy2.exp(-K * g * g)
        c = gmpy2.cos(g * y - psi)
        # positive defect means this term is hurting a large negative h
        out.append((float(w * (c + 1)), i))
    out.sort(reverse=True)
    return out


def enumerate_subset(indices, nu, nu_y, nu_t, rho, nsols, negative, block=20):
    idxs = list(indices)
    subset = [ITEMS[i] for i in idxs]
    subset.sort(key=lambda z: -z[0])
    N = len(subset)
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
    if block:
        BKZ.reduction(B, BKZ.Param(block, max_loops=1))
    v = CVP.babai(B, target)
    err2 = sum((int(v[j]) - target[j]) ** 2 for j in range(N + 1))
    M = GSO.Mat(B, float_type="dpe")
    M.update_gso()
    n = N + 1
    sigma = [0.0] * n
    for i in range(n):
        ip = 0
        for k in range(n):
            ip += target[k] * int(B[i, k])
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
    nunique = 0
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
        nunique += 1
        h = fast_h(y)
        if best is None or abs(h) > abs(best[0]):
            best = (h, y)
    return best, nunique


def trial(job):
    kind, seed, nu, nu_y, nu_t, rho, nsols, negative = job
    rng = random.Random(seed)
    t0 = time.time()
    if kind == "top":
        idxs = list(range(48))
    elif kind == "mix":
        # heaviest 28, plus 20 drawn from the next tier
        rest = list(range(28, 160))
        rng.shuffle(rest)
        idxs = list(range(28)) + rest[:20]
    else:
        # defect-focused around a strong published / found seed
        seed_y = gmpy2.mpfr(kind)
        worst = [i for _, i in defects(seed_y, 180)[:24]]
        idxs = list(range(26)) + [i for i in worst if i >= 26]
        idxs = sorted(set(idxs))[:50]
    try:
        best, nunique = enumerate_subset(
            idxs, nu, nu_y, nu_t, rho, nsols, negative, block=20
        )
    except Exception as exc:
        return {"err": str(exc), "job": job, "dt": time.time() - t0}
    return {
        "job": job,
        "best": None if best is None else (best[0], str(best[1])),
        "nunique": nunique,
        "dt": time.time() - t0,
    }


def main():
    seeds_y = [
        "11836211697137399925.7255859375",
        "154222634625982481.66259765625",
        "13859539710197847064.062257",
        "8895437864289868028.044074",
    ]
    jobs = []
    sid = 0
    for nu_t, nu_y in ((14, 80), (14, 82), (16, 82), (16, 84), (18, 86)):
        nu = nu_y + 14
        for neg in (True, False):
            jobs.append(("top", sid, nu, nu_y, nu_t, 1.10, 3500, neg))
            sid += 1
            for _ in range(3):
                jobs.append(("mix", sid, nu, nu_y, nu_t, 1.10, 3500, neg))
                sid += 1
            for sy in seeds_y:
                jobs.append((sy, sid, nu, nu_y, nu_t, 1.10, 3000, neg))
                sid += 1
    print(f"jobs {len(jobs)}", flush=True)
    best_global = None
    t0 = time.time()
    with Pool(20) as pool:
        for i, res in enumerate(pool.imap_unordered(trial, jobs), 1):
            b = res.get("best")
            if b and (best_global is None or abs(b[0]) > abs(best_global[0])):
                best_global = b
                print(
                    f"NEW {b[0]:.6f} y={b[1]} uniq={res.get('nunique')} "
                    f"dt={res.get('dt', 0):.1f} i={i}",
                    flush=True,
                )
                with open(os.path.join(ROOT, "focus_hits.txt"), "a") as f:
                    f.write(f"{b[0]} {b[1]} {res.get('job')}\n")
            elif i % 8 == 0:
                print(
                    f"i={i}/{len(jobs)} best={None if best_global is None else best_global[0]} "
                    f"elapsed={time.time()-t0:.0f}",
                    flush=True,
                )
    print("DONE", best_global, flush=True)
    if best_global and abs(best_global[0]) > 0.97:
        h = partial_h(best_global[1], ITEMS, len(ITEMS))
        print("FULL", h, "y", best_global[1], flush=True)


if __name__ == "__main__":
    main()
