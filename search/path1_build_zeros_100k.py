#!/usr/bin/env python3
"""
Builds zeros_100k.bin (all 100,000 zeros up to gamma = 74920.83)
and zeros_98625.bin (all 98,625 zeros with gamma < 74000)
and path1_zeros_100k.npz from chunks/chunk_*.txt.
Never modifies zeros.bin!
"""
import glob
import os
import struct
import time
import gmpy2
from gmpy2 import mpfr
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
CHUNK_DIR = os.path.join(ROOT, "chunks")


def wait_for_chunks(expected_chunks=400):
    while True:
        files = sorted(glob.glob(os.path.join(CHUNK_DIR, "chunk_*.txt")))
        valid = 0
        for p in files:
            if os.path.getsize(p) > 0:
                valid += 1
        if valid >= expected_chunks:
            print(f"All {valid} chunks ready!", flush=True)
            return files
        print(f"Waiting for chunks: {valid}/{expected_chunks} ready...", flush=True)
        time.sleep(5)


def write_bin(out_path, n, two_astar, gamma_f64, ghi_lo, ghi_hi, glo_lo, glo_hi, psi_lo, psi_hi):
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "wb") as f:
        f.write(struct.pack("<i", n))
        f.write(two_astar[:n].tobytes())
        f.write(gamma_f64[:n].tobytes())
        f.write(ghi_lo[:n].tobytes())
        f.write(ghi_hi[:n].tobytes())
        f.write(glo_lo[:n].tobytes())
        f.write(glo_hi[:n].tobytes())
        f.write(psi_lo[:n].tobytes())
        f.write(psi_hi[:n].tobytes())
    os.replace(tmp_path, out_path)
    print(f"Wrote {out_path} ({os.path.getsize(out_path)} bytes, n={n})", flush=True)


def main():
    t0 = time.time()
    files = wait_for_chunks(400)
    gmpy2.get_context().precision = 256
    twopi = 2 * gmpy2.const_pi()
    k_str = mpfr("3e-9")
    scale_glo = mpfr(2) ** 98
    scale_ghi = mpfr(2) ** 146
    scale_psi = mpfr(2) ** 128
    mask64 = (1 << 64) - 1
    mask128 = (1 << 128) - 1

    lines = []
    for p in files:
        with open(p, "r") as f:
            for line in f:
                s = line.strip()
                if s:
                    lines.append(s)

    n_total = len(lines)
    print(f"Loaded {n_total} zero lines from {len(files)} chunk files in {time.time()-t0:.2f}s", flush=True)

    gamma_f64 = np.zeros(n_total, dtype=np.float64)
    re_f64 = np.zeros(n_total, dtype=np.float64)
    im_f64 = np.zeros(n_total, dtype=np.float64)
    alpha_f64 = np.zeros(n_total, dtype=np.float64)
    two_astar = np.zeros(n_total, dtype=np.float64)
    ghi_lo = np.zeros(n_total, dtype=np.uint64)
    ghi_hi = np.zeros(n_total, dtype=np.uint64)
    glo_lo = np.zeros(n_total, dtype=np.uint64)
    glo_hi = np.zeros(n_total, dtype=np.uint64)
    psi_lo = np.zeros(n_total, dtype=np.uint64)
    psi_hi = np.zeros(n_total, dtype=np.uint64)

    n_74000 = 0
    for i, line in enumerate(lines):
        gs, rs, ims = line.split()
        g = mpfr(gs)
        re_c = mpfr(rs)
        im_c = mpfr(ims)
        if g < 74000:
            n_74000 = i + 1
        alpha = gmpy2.hypot(re_c, im_c)
        astar = alpha * gmpy2.exp(-k_str * g * g)

        gamma_f64[i] = float(g)
        re_f64[i] = float(re_c)
        im_f64[i] = float(im_c)
        alpha_f64[i] = float(alpha)
        two_astar[i] = float(2 * astar)

        glo_128 = int(gmpy2.rint((g / twopi) * scale_glo)) & mask128
        ghi_128 = int(gmpy2.rint((g / twopi) * scale_ghi)) & mask128
        psi_128 = int(gmpy2.rint(((-gmpy2.atan2(im_c, re_c) / twopi) % 1) * scale_psi)) & mask128

        glo_lo[i] = glo_128 & mask64
        glo_hi[i] = (glo_128 >> 64) & mask64
        ghi_lo[i] = ghi_128 & mask64
        ghi_hi[i] = (ghi_128 >> 64) & mask64
        psi_lo[i] = psi_128 & mask64
        psi_hi[i] = (psi_128 >> 64) & mask64

    assert np.all(np.diff(gamma_f64) > 0), "Gammas are not strictly increasing!"
    print(
        f"n_total={n_total} (gamma_max={gamma_f64[-1]:.8f}), "
        f"n_74000={n_74000} (gamma[{n_74000-1}]={gamma_f64[n_74000-1]:.8f}, "
        f"gamma[{n_74000}]={gamma_f64[n_74000]:.8f})",
        flush=True,
    )

    write_bin(
        os.path.join(ROOT, "zeros_100k.bin"),
        n_total,
        two_astar,
        gamma_f64,
        ghi_lo,
        ghi_hi,
        glo_lo,
        glo_hi,
        psi_lo,
        psi_hi,
    )
    write_bin(
        os.path.join(ROOT, "zeros_98625.bin"),
        n_74000,
        two_astar,
        gamma_f64,
        ghi_lo,
        ghi_hi,
        glo_lo,
        glo_hi,
        psi_lo,
        psi_hi,
    )
    npz_path = os.path.join(ROOT, "path1_zeros_100k.npz")
    np.savez(
        npz_path,
        gamma=gamma_f64,
        re_c=re_f64,
        im_c=im_f64,
        alpha=alpha_f64,
        two_astar=two_astar,
        ghi_lo=ghi_lo,
        ghi_hi=ghi_hi,
        glo_lo=glo_lo,
        glo_hi=glo_hi,
        psi_lo=psi_lo,
        psi_hi=psi_hi,
    )
    print(f"Saved {npz_path} in {time.time()-t0:.2f}s", flush=True)


if __name__ == "__main__":
    main()
