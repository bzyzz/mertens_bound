"""BKZ-reduce a Mertens lattice and enumerate CVP solutions with enum_cvp.

The shortest Gram-Schmidt line is collapsed, so the returned y values are
distinct alignments rather than tiny shifts of one y.
"""
import math
import os
import struct
import subprocess
import sys

import gmpy2
from fpylll import BKZ, GSO, LLL

from lattice_search import build_lattice, gs_target, load_items, partial_h, progressive_bkz

ROOT = os.path.dirname(os.path.abspath(__file__))
ENUM = os.path.join(ROOT, "enum_cvp")


def beta_prune(n):
    out = []
    for k in range(n):
        i = n - k  # projection dimension of coordinates k..n-1
        mean = i / n
        var = mean * (1.0 - mean) / (n + 1.0)
        f = min(1.0, mean + 2.0 * math.sqrt(var))
        out.append(f)
    return out


def enumerate_cpp(mu, r, sigma, radius2, prune, max_sols, short_level=-1, bucket=0, xcol=None):
    n = len(r)
    blob = struct.pack("<iii", n, short_level, max_sols)
    blob += struct.pack("<d", float(radius2))
    blob += struct.pack("<" + "d" * n, *[float(z) for z in r])
    blob += struct.pack("<" + "d" * n, *[float(z) for z in sigma])
    flat = []
    for j in range(n):
        flat.extend(float(mu[j][i]) for i in range(n))
    blob += struct.pack("<" + "d" * (n * n), *flat)
    blob += struct.pack("<" + "d" * n, *prune)
    blob += struct.pack("<q", int(bucket))
    if bucket:
        lo, hi = [], []
        for z in xcol:
            z = int(z)
            if abs(z).bit_length() > 126:
                raise RuntimeError(f"xcol width {z.bit_length()} bits")
            zmod = z & ((1 << 128) - 1)
            lo.append(zmod & ((1 << 64) - 1))
            hi.append((zmod >> 64) & ((1 << 64) - 1))
        blob += struct.pack("<" + "Q" * n, *lo)
        blob += struct.pack("<" + "Q" * n, *hi)
    proc = subprocess.run([ENUM], input=blob, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(
            f"enum_cvp failed with code {proc.returncode}: {proc.stderr.decode()[-500:]}"
        )
    err = proc.stderr.decode()
    data = proc.stdout
    ns = struct.unpack_from("<i", data, 0)[0]
    off = 4
    sols = []
    for _ in range(ns):
        dist = struct.unpack_from("<d", data, off)[0]
        off += 8
        xs = struct.unpack_from("<" + "q" * n, data, off)
        off += 8 * n
        sols.append((dist, xs))
    return sols, err


def main():
    N = int(sys.argv[1])
    nu = int(sys.argv[2])
    nu_y = int(sys.argv[3])
    nu_t = int(sys.argv[4])
    block = int(sys.argv[5])
    rho = float(sys.argv[6])
    negative = sys.argv[7] != "pos"
    max_sols = int(sys.argv[8]) if len(sys.argv) > 8 else 4000
    items, pi = load_items()
    print(
        f"zeros={len(items)} N={N} nu={nu} nu_y={nu_y} nu_t={nu_t} "
        f"block={block} rho={rho} sign={'neg' if negative else 'pos'}",
        flush=True,
    )
    B, target = build_lattice(items, N, nu, nu_y, nu_t, pi, negative)
    progressive_bkz(B, block)
    from fpylll.fplll.svpcvp import CVP

    v = CVP.babai(B, target)
    err2 = sum((int(v[i]) - target[i]) ** 2 for i in range(N + 1))
    print(f"babai log2||e||^2={err2.bit_length()}", flush=True)
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
    short = min(range(n), key=lambda i: r[i])
    print(
        f"short level {short} log2||b*||={math.log2(r[short]) / 2:.2f} "
        f"median {math.log2(sorted(r)[n // 2]) / 2:.2f}",
        flush=True,
    )
    radius2 = float(err2) * (rho ** 2)
    # float(err2) may overflow; use gmpy
    radius2 = float(gmpy2.mpfr(err2) * gmpy2.mpfr(rho) ** 2)
    prune = [1.0] * n
    step = 1 << nu_t
    xcol = []
    for i in range(n):
        last = int(B[i, n - 1])
        if last % step != 0:
            raise RuntimeError("last column not divisible by 2^nu_t")
        xcol.append(last // step)
    bucket = max(1, int(0.02 * 2 ** (nu - nu_y)))
    sols, err = enumerate_cpp(
        mu, r, sigma, radius2, prune, max_sols, short_level=-1, bucket=bucket, xcol=xcol
    )
    print(err, f"returned {len(sols)}", flush=True)
    best = []
    seen = set()
    step = 1 << nu_t
    for dist, coeff in sols:
        u = B.multiply_left(list(coeff))
        last = int(u[-1])
        if last % step != 0 or last <= 0:
            continue
        y = gmpy2.mpfr(last // step) * gmpy2.exp2(nu_y - nu)
        if y < gmpy2.mpfr("1e10") or y > gmpy2.mpfr("1e30"):
            continue
        key = int(gmpy2.floor(y * 50))
        if key in seen:
            continue
        seen.add(key)
        h = partial_h(y, items, min(len(items), 12000))
        best.append((abs(h), h, y))
    best.sort(key=lambda z: -z[0])
    out = os.path.join(
        ROOT,
        f"enum_N{N}_nu{nu}_y{nu_y}_t{nu_t}_b{block}_r{rho}_{'n' if negative else 'p'}.txt",
    )
    with open(out, "w") as f:
        for ah, h, y in best[:30]:
            line = f"{h} y={y}"
            print(line, flush=True)
            f.write(line + "\n")
    print(f"unique {len(best)} wrote {out}", flush=True)


if __name__ == "__main__":
    main()
