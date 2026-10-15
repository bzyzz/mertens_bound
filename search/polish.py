"""Fine search for a larger |h_StR| near a published seed y.

h_StR(y) = 2 * sum_{gamma<74000} exp(-3e-9 gamma^2) * Re(e^{i gamma y} / (rho zeta'(rho))).
Phases are reduced with mpfr; the grid is then scored in float64.
"""
import glob
import os
import sys

import gmpy2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
K = "3e-9"


def load():
    gmpy2.get_context().precision = 256
    k = gmpy2.mpfr(K)
    twopi = 2 * gmpy2.const_pi()
    gammas = []
    gre = []
    gim = []
    for path in sorted(glob.glob(os.path.join(ROOT, "chunks", "chunk_*.txt"))):
        with open(path) as f:
            for line in f:
                gs, rs, ims = line.split()
                g = gmpy2.mpfr(gs)
                if g >= 74000:
                    continue
                re = gmpy2.mpfr(rs)
                im = gmpy2.mpfr(ims)
                w = gmpy2.exp(-k * g * g)
                gammas.append(g)
                gre.append(w * re)
                gim.append(w * im)
    return gammas, np.array(gre, dtype=np.float64), np.array(gim, dtype=np.float64), twopi


def h_mpfr(y, gammas, gre, gim):
    gmpy2.get_context().precision = 256
    y = gmpy2.mpfr(y)
    s = gmpy2.mpfr(0)
    for g, gr, gi in zip(gammas, gre, gim):
        ang = g * y
        s += gmpy2.mpfr(gr) * gmpy2.cos(ang) - gmpy2.mpfr(gi) * gmpy2.sin(ang)
    return 2 * s


def scan(y0, half_width, step, gammas, gre, gim, twopi):
    y0 = gmpy2.mpfr(y0)
    step_m = gmpy2.mpfr(step)
    phi = np.empty(len(gammas), dtype=np.float64)
    dphi = np.empty(len(gammas), dtype=np.float64)
    for i, g in enumerate(gammas):
        phi[i] = float((g * y0) % twopi)
        dphi[i] = float((g * step_m) % twopi)
    m_half = int(round(half_width / step))
    best = None
    chunk = 4000
    for m0 in range(-m_half, m_half + 1, chunk):
        m1 = min(m0 + chunk, m_half + 1)
        m = np.arange(m0, m1, dtype=np.float64)[:, None]
        ang = phi[None, :] + m * dphi[None, :]
        h = 2.0 * np.sum(gre * np.cos(ang) - gim * np.sin(ang), axis=1)
        j = int(np.argmax(np.abs(h)))
        val = float(h[j])
        yy = y0 + gmpy2.mpfr(m0 + j) * step_m
        if best is None or abs(val) > abs(best[0]):
            best = (val, yy, m0 + j)
        print(f"  chunk m={m0}..{m1-1} best_so_far {best[0]:.8f} at {best[1]}", flush=True)
    return best


def main():
    y0 = sys.argv[1]
    half = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
    step = float(sys.argv[3]) if len(sys.argv) > 3 else 2e-5
    gammas, gre, gim, twopi = load()
    print(f"terms={len(gammas)} g_last={gammas[-1]} half={half} step={step}", flush=True)
    base = h_mpfr(y0, gammas, gre, gim)
    print(f"h({y0}) = {base}", flush=True)
    val, yy, m = scan(y0, half, step, gammas, gre, gim, twopi)
    print(f"GRID best |h| candidate h={val:.10f} y={yy} m={m}", flush=True)
    certified = h_mpfr(yy, gammas, gre, gim)
    print(f"CERTIFIED h={certified}", flush=True)
    # also check a few neighbors at +/- one grid step in mpfr
    step_m = gmpy2.mpfr(step)
    for d in (-2, -1, 1, 2):
        h = h_mpfr(yy + d * step_m, gammas, gre, gim)
        print(f"  neighbor {d} h={h}", flush=True)


if __name__ == "__main__":
    main()
