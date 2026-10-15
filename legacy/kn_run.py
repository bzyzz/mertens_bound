"""Kim–Nguyen extreme-pruning search, aimed below the record y.

The enumerator is their NTL program. Lattices are BKZ-reduced here, then
searched with the linear pruning they used for h_StR (|h| > 1). A reweighted
lattice pulls in the medium-sized zeros that carry the gap between 0.96 and 1.
"""
import os
import subprocess
import threading
import time
from multiprocessing import Pool

import gmpy2
from fpylll import BKZ, LLL

from diverse_enum import ITEMS, PI, fast_h
from lattice_search import build_lattice, partial_h

ROOT = os.path.dirname(os.path.abspath(__file__))
ENUM = os.path.join(ROOT, "Mertens_Enum")
RECORD = gmpy2.mpfr("1.957187885e19")


def reorder(items, mode):
    if mode == "std":
        return items
    boosted = []
    for astar, g, alpha, psi in items:
        boost = gmpy2.mpfr(6) if 120 < g < 5000 else gmpy2.mpfr(1)
        boosted.append((astar * boost, g, alpha, psi))
    boosted.sort(key=lambda z: -z[0])
    return boosted


def write_ntl(path, B, target):
    n = B.nrows
    with open(path, "w") as f:
        f.write("[\n")
        for i in range(n):
            vals = " ".join(str(int(B[i, j])) for j in range(n))
            f.write(f"[ {vals} ]\n")
        f.write("]\n[ ")
        f.write(" ".join(str(int(t)) for t in target))
        f.write(" ]\n")


def reduce_basis(B, max_block):
    LLL.reduction(B)
    beta = 20
    while beta <= max_block:
        BKZ.reduction(B, BKZ.Param(beta, max_loops=1, max_time=90))
        beta += 8


def one(job):
    N, nu, nu_y, nu_t, negative, gh, mode, seconds = job
    tag = f"N{N}_nu{nu}_y{nu_y}_t{nu_t}_{'n' if negative else 'p'}_{mode}_g{gh}"
    t0 = time.time()
    items = reorder(ITEMS, mode)
    B, target = build_lattice(items, N, nu, nu_y, nu_t, PI, negative)
    reduce_basis(B, 44 if N >= 160 else 52)
    path = os.path.join(ROOT, "bases", tag + ".ntl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_ntl(path, B, target)
    print(f"reduced {tag} in {time.time()-t0:.0f}s", flush=True)
    scale = gmpy2.exp2(nu_y - nu - nu_t)
    err_path = os.path.join(ROOT, "bases", tag + ".err")
    cmd = [ENUM, "-prec", "120", "-gh", str(gh), "-lin", "3"]
    best = None
    small = None
    nsol = 0
    seen = set()
    with open(path) as fin, open(err_path, "w") as ferr:
        proc = subprocess.Popen(cmd, stdin=fin, stdout=subprocess.PIPE, stderr=ferr)

        def _kill():
            time.sleep(seconds)
            proc.kill()

        threading.Thread(target=_kill, daemon=True).start()
        while True:
            raw = proc.stdout.readline()
            if not raw:
                break
            parts = raw.split()
            if len(parts) < 1:
                continue
            try:
                x = int(parts[0])
            except ValueError:
                continue
            if x <= 0:
                continue
            y = gmpy2.mpfr(x) * scale
            if not (gmpy2.mpfr("1e12") < y < gmpy2.mpfr("1e24")):
                continue
            key = int(gmpy2.floor(y))
            if key in seen:
                continue
            seen.add(key)
            nsol += 1
            h = fast_h(y)
            improved = best is None or abs(h) > abs(best[0])
            if improved:
                best = (h, str(y))
                print(f"HIT {h:.5f} y={float(y):.6e} {tag} n={nsol}", flush=True)
            if y < RECORD and (small is None or abs(h) > abs(small[0])):
                small = (h, str(y))
                print(f"SMALL {h:.5f} y={float(y):.6e} {tag} n={nsol}", flush=True)
                if abs(h) > 0.99:
                    full = partial_h(y, ITEMS, len(ITEMS))
                    msg = f"FULL {full} y={y} {tag}"
                    print(msg, flush=True)
                    with open(os.path.join(ROOT, "kn_hits.txt"), "a") as f:
                        f.write(msg + "\n")
                    if abs(full) > gmpy2.mpfr("1.00000006"):
                        proc.kill()
                        return msg
        proc.wait()
    return f"done {tag} n={nsol} best={best} small={small} t={time.time()-t0:.0f}"


def main():
    seconds = int(os.environ.get("KN_SECONDS", "2100"))
    jobs = [
        (140, 130, 97, 30, True, 1.19, "std", seconds),
        (140, 130, 96, 30, True, 1.19, "std", seconds),
        (140, 130, 95, 30, True, 1.19, "std", seconds),
        (140, 130, 98, 30, True, 1.25, "std", seconds),
        (160, 150, 101, 34, True, 1.19, "std", seconds),
        (140, 130, 97, 30, True, 1.19, "mid", seconds),
    ]
    print(f"jobs {len(jobs)} seconds {seconds}", flush=True)
    with Pool(len(jobs)) as pool:
        for msg in pool.imap_unordered(one, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
