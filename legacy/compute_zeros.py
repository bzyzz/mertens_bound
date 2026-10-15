"""Refine Odlyzko's first 100k zeta zeros and store 1/(rho * zeta'(rho)).

Each output line:
    gamma  re(1/(rho*zeta'))  im(1/(rho*zeta'))
with gamma < 74000, at 40 decimal digits.
"""
import gzip
import os
import sys
from multiprocessing import Pool

import mpmath as mp

OUTDIR = os.path.dirname(os.path.abspath(__file__))
SEEDS = os.path.join(OUTDIR, "zeros1.gz")
DPS = 40
GAMMA_MAX = mp.mpf("74000")


def load_seeds():
    with gzip.open(SEEDS, "rt") as f:
        gammas = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            gammas.append(line)
    return gammas


def refine_chunk(args):
    start, seeds = args
    mp.mp.dps = DPS
    half = mp.mpf("0.5")
    lines = []
    for i, g0 in enumerate(seeds):
        t = mp.mpf(g0)
        if t >= GAMMA_MAX:
            break
        zp = None
        for _ in range(3):
            s = half + 1j * t
            z = mp.zeta(s)
            zp = mp.zeta(s, derivative=1)
            corr = z / (1j * zp)
            t = t - corr.real
            if abs(corr) < mp.mpf("1e-28"):
                break
        s = half + 1j * t
        # one extra derivative at the refined zero
        zp = mp.zeta(s, derivative=1)
        coeff = 1 / (s * zp)
        lines.append(
            f"{mp.nstr(t, 42, strip_zeros=False)} "
            f"{mp.nstr(coeff.real, 30, strip_zeros=False)} "
            f"{mp.nstr(coeff.imag, 30, strip_zeros=False)}"
        )
    path = os.path.join(OUTDIR, "chunks", f"chunk_{start:06d}.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")
    return start, len(lines)


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    chunk = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    os.makedirs(os.path.join(OUTDIR, "chunks"), exist_ok=True)
    seeds = load_seeds()
    jobs = []
    for start in range(0, len(seeds), chunk):
        out = os.path.join(OUTDIR, "chunks", f"chunk_{start:06d}.txt")
        if os.path.exists(out) and os.path.getsize(out) > 0:
            continue
        jobs.append((start, seeds[start : start + chunk]))
    print(f"seeds={len(seeds)} pending_chunks={len(jobs)} workers={workers}", flush=True)
    if not jobs:
        print("nothing to do", flush=True)
        return
    with Pool(workers) as pool:
        done = 0
        for start, n in pool.imap_unordered(refine_chunk, jobs):
            done += 1
            if done % 10 == 0 or done == len(jobs):
                print(f"done {done}/{len(jobs)} last_start={start} n={n}", flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
