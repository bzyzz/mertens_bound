"""Search for y with large |h_StR| via the Kim-Nguyen lattice.

The lattice is the one in their equation (3.2). Closest-vector enumeration
uses Gram-Schmidt target coordinates, which fplll's enumerator expects.
"""
import glob
import math
import os
import sys

import gmpy2
from fpylll import BKZ, Enumeration, FPLLL, GSO, IntegerMatrix, LLL

ROOT = os.path.dirname(os.path.abspath(__file__))


def load_items(k="3e-9"):
    gmpy2.get_context().precision = 200
    kk = gmpy2.mpfr(k)
    pi = gmpy2.const_pi()
    items = []
    for path in sorted(glob.glob(os.path.join(ROOT, "chunks", "chunk_*.txt"))):
        with open(path) as f:
            for line in f:
                gs, rs, ims = line.split()
                g = gmpy2.mpfr(gs)
                if g >= 74000:
                    continue
                re = gmpy2.mpfr(rs)
                im = gmpy2.mpfr(ims)
                alpha = gmpy2.sqrt(re * re + im * im)
                # coeff = alpha * exp(-i psi) = 1/(rho zeta')
                psi = -gmpy2.atan2(im, re)
                astar = alpha * gmpy2.exp(-kk * g * g)
                items.append((astar, g, alpha, psi))
    items.sort(key=lambda z: -z[0])
    return items, pi


def build_lattice(items, N, nu, nu_y, nu_t, pi, negative=True):
    scale_nu = gmpy2.exp2(nu)
    scale_y = gmpy2.exp2(nu_y)
    B = IntegerMatrix(N + 1, N + 1)
    target = [0] * (N + 1)
    rows_info = []
    for i in range(N):
        astar, g, alpha, psi = items[i]
        s = gmpy2.sqrt(astar)
        a = int(gmpy2.floor(s * g * scale_y))
        p = int(gmpy2.floor(s * (2 * pi) * scale_nu))
        phase = psi + (pi if negative else 0)
        t = int(gmpy2.floor(s * phase * scale_nu))
        B[0, i] = a
        B[i + 1, i] = p
        target[i] = t
        rows_info.append((s, g, psi))
    B[0, N] = 1 << nu_t
    target[N] = 0
    return B, target


def progressive_bkz(B, block):
    LLL.reduction(B)
    beta = 20
    while beta < block:
        BKZ.reduction(B, BKZ.Param(beta, max_loops=1))
        print(f"  bkz-{beta} done", flush=True)
        beta += 4
    BKZ.reduction(B, BKZ.Param(block, max_loops=1))
    print(f"  bkz-{block} done", flush=True)


def gs_target(B, target, M):
    n = B.nrows
    sigma = [0.0] * n
    for i in range(n):
        ip = 0
        for k in range(n):
            ip += target[k] * int(B[i, k])
        s = float(ip)
        for j in range(i):
            s -= sigma[j] * M.get_r(j, j) * M.get_mu(i, j)
        r = M.get_r(i, i)
        sigma[i] = s / r
    return sigma


def y_from_coeff(B, coeff, nu, nu_y, nu_t):
    n = B.nrows
    c = [int(round(x)) for x in coeff]
    last = 0
    for i in range(n):
        last += c[i] * int(B[i, n - 1])
    step = 1 << nu_t
    if last % step != 0:
        return None
    x = last // step
    if x <= 0:
        return None
    y = gmpy2.mpfr(x) * gmpy2.exp2(nu_y - nu)
    return y


def partial_h(y, items, nterms, k="3e-9"):
    gmpy2.get_context().precision = 180
    kk = gmpy2.mpfr(k)
    y = gmpy2.mpfr(y)
    s = gmpy2.mpfr(0)
    for astar, g, alpha, psi in items[:nterms]:
        w = alpha * gmpy2.exp(-kk * g * g)
        ang = g * y - psi
        s += w * gmpy2.cos(ang)
    return 2 * s


def main():
    N = int(sys.argv[1])
    nu = int(sys.argv[2])
    nu_y = int(sys.argv[3])
    nu_t = int(sys.argv[4])
    block = int(sys.argv[5])
    # Linear radius factor over the Babai distance. The ball volume scales as rho**dim.
    rho = float(sys.argv[6])
    negative = sys.argv[7] != "pos"
    nsols = int(sys.argv[8]) if len(sys.argv) > 8 else 8000
    items, pi = load_items()
    print(
        f"loaded {len(items)} zeros "
        f"N={N} nu={nu} nu_y={nu_y} nu_t={nu_t} block={block} rho={rho} "
        f"sign={'neg' if negative else 'pos'} nsols={nsols}",
        flush=True,
    )
    if len(items) < N:
        raise SystemExit("not enough zeros")
    B, target = build_lattice(items, N, nu, nu_y, nu_t, pi, negative=negative)
    print("reducing", flush=True)
    progressive_bkz(B, block)
    from fpylll.fplll.svpcvp import CVP

    v = CVP.babai(B, target)
    err2 = sum((int(v[i]) - target[i]) ** 2 for i in range(N + 1))
    print(f"babai log2||e||^2={err2.bit_length()}", flush=True)
    M = GSO.Mat(B, float_type="dpe")
    M.update_gso()
    sigma = gs_target(B, target, M)
    sq = rho * rho
    # bound = sq * err2, written as mant * 2^expo
    bound = int(err2 * int(sq * 10000) // 10000) + 1
    expo = bound.bit_length() - 1
    mant = float(bound) / 2**expo
    print(f"enumerating mant={mant:.4f} * 2^{expo}", flush=True)
    enum = Enumeration(M, nr_solutions=nsols)
    sols = enum.enumerate(0, N + 1, mant, expo, target=sigma)
    print(f"got {len(sols)} solutions", flush=True)
    # Dedup by y rounded to roughly 0.05% , then score the distinct ones.
    best = []
    seen = set()
    nterms = min(len(items), 8000)
    for dist, coeff in sols:
        y = y_from_coeff(B, coeff, nu, nu_y, nu_t)
        if y is None:
            continue
        if y < gmpy2.mpfr("1e8") or y > gmpy2.mpfr("1e40"):
            continue
        # Collapse neighbors generated by the very short lattice vector.
        key = int(gmpy2.floor(y * 2000))
        if key in seen:
            continue
        seen.add(key)
        h = partial_h(y, items, nterms)
        best.append((abs(h), h, y, dist))
    best.sort(key=lambda z: -z[0])
    tag = "n" if negative else "p"
    out = os.path.join(
        ROOT, f"cands_N{N}_nu{nu}_y{nu_y}_t{nu_t}_b{block}_r{rho}_{tag}.txt"
    )
    with open(out, "w") as f:
        for ah, h, y, dist in best[:40]:
            line = f"{h} y={y} dist={dist}"
            print(line, flush=True)
            f.write(line + "\n")
    print(f"unique={len(best)} wrote {out}", flush=True)


if __name__ == "__main__":
    main()
