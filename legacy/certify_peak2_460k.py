#!/usr/bin/env python3
import glob, os, time, math
from multiprocessing import Pool
import numpy as np
import gmpy2
from gmpy2 import mpfr
import flint
from decimal import Decimal, getcontext

getcontext().prec = 60

ROOT = os.path.dirname(os.path.abspath(__file__))
CHUNK_DIR = os.path.join(ROOT, "chunks") if os.path.isdir(os.path.join(ROOT, "chunks")) else os.path.join(os.path.dirname(ROOT), "chunks")
MASK128 = (1 << 128) - 1


def parse_chunk_file(path):
    gmpy2.get_context().precision = 256
    twopi = 2 * gmpy2.const_pi()
    scale_glo = mpfr(2) ** 98
    scale_ghi = mpfr(2) ** 146
    scale_psi = mpfr(2) ** 128

    gammas = []
    two_alphas = []
    ghis = []
    glos = []
    psis = []
    raw_lines = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            raw_lines.append(s)
            gs, rs, ims = s.split()
            g = mpfr(gs)
            re_c = mpfr(rs)
            im_c = mpfr(ims)
            alpha = gmpy2.hypot(re_c, im_c)
            gammas.append(float(g))
            two_alphas.append(float(2 * alpha))
            glos.append(int(gmpy2.rint((g / twopi) * scale_glo)) & MASK128)
            ghis.append(int(gmpy2.rint((g / twopi) * scale_ghi)) & MASK128)
            psis.append(int(gmpy2.rint(((-gmpy2.atan2(im_c, re_c) / twopi) % 1) * scale_psi)) & MASK128)
    return gammas, two_alphas, ghis, glos, psis, raw_lines


def arb_eval_chunk(args):
    raw_lines, y_str, k_str, T_str = args
    flint.ctx.dps = 80
    y = flint.arb(y_str)
    k = flint.arb(k_str)
    T = flint.arb(T_str)
    pi = flint.arb.pi()

    sum_gauss = flint.arb(0)
    sum_jp = flint.arb(0)
    for line in raw_lines:
        gs, rs, ims = line.split()
        g = flint.arb(gs)
        re_c = flint.arb(rs)
        im_c = flint.arb(ims)
        # Re((re_c + i*im_c) * exp(i * g * y)) = re_c * cos(g*y) - im_c * sin(g*y)
        gy = g * y
        s_gy, c_gy = gy.sin_cos()
        osc = 2 * (re_c * c_gy - im_c * s_gy)

        # Gaussian weight exp(-k * g^2)
        w_g = (-k * g * g).exp()
        sum_gauss += osc * w_g

        # Jurkat-Peyerimhoff weight on [0, T]
        if float(g.mid()) < float(T.mid()):
            u = g / T
            pi_u = pi * u
            s_pu, c_pu = pi_u.sin_cos()
            w_jp = (1 - u) * c_pu + s_pu / pi
            sum_jp += osc * w_jp
    return sum_gauss.str(50, radius=True), sum_jp.str(50, radius=True)


def main():
    t0 = time.time()
    files = sorted(glob.glob(os.path.join(CHUNK_DIR, "chunk_*.txt")))
    with Pool(64) as p:
        results = p.map(parse_chunk_file, files)

    gamma_list, ta_list, ghi_list, glo_list, psi_list, raw_chunks = [], [], [], [], [], []
    for g, ta, ghi, glo, psi, raw in results:
        gamma_list.extend(g)
        ta_list.extend(ta)
        ghi_list.extend(ghi)
        glo_list.extend(glo)
        psi_list.extend(psi)
        raw_chunks.append(raw)

    gamma = np.array(gamma_list, dtype=np.float64)
    two_alpha = np.array(ta_list, dtype=np.float64)
    g2 = gamma * gamma
    n_total = len(gamma)
    print(f"Loaded {n_total} zeros in {time.time()-t0:.2f}s | gamma_max = {gamma[-1]:.8f}", flush=True)

    # Evaluate Peak2 across N = 100k..460k
    ys = "3484971332733132396.5383310879"
    y_dec = Decimal(ys)
    x_int = int(y_dec * Decimal(1 << 30))
    dy0 = float(y_dec - Decimal(x_int) / Decimal(1 << 30))
    x_hi = x_int >> 48
    x_lo = x_int & ((1 << 48) - 1)
    SCALE = (2.0 * math.pi) / float(1 << 64)

    base_ang = np.empty(n_total, dtype=np.float64)
    for i in range(n_total):
        ph = (x_hi * ghi_list[i] + x_lo * glo_list[i] - psi_list[i]) & MASK128
        hi64 = ph >> 64
        if hi64 >= (1 << 63):
            hi64 -= (1 << 64)
        base_ang[i] = float(hi64) * SCALE + gamma[i] * dy0

    print("\n=== PEAK2 (y0 = 3484971332733132396.5383310879) ACROSS N = 100,000 .. 460,000 ===")
    ks = [3.0e-9, 3.0e-10, 2.5e-10, 2.2e-10, 2.0e-10, 1.5e-10, 1.0e-10, 5.0e-11, 0.0]
    hdr = "   N_zeros (    T_max) |   JP_kernel (    ddy*) | " + " | ".join(f"k={k:.1e}" for k in ks)
    print(hdr)
    print("-" * len(hdr))

    dd_grid = np.linspace(-1.0e-5, 1.0e-5, 401)
    best_jp_overall = (0.0, 0, 0.0, 0.0)

    for N_cut in [100000, 150000, 200000, 240000, 260000, 300000, 350000, 400000, 460000]:
        if N_cut > n_total:
            continue
        T_cut = gamma[N_cut - 1]
        ba = base_ang[:N_cut]
        ga = gamma[:N_cut]
        ta = two_alpha[:N_cut]
        g2a = g2[:N_cut]

        # JP kernel
        u = ga / T_cut
        w_jp = ta * ((1.0 - u) * np.cos(np.pi * u) + np.sin(np.pi * u) / np.pi)
        jp_vals = [float(np.sum(w_jp * np.cos(ba + ga * dd))) for dd in dd_grid]
        idx_jp = int(np.argmax(np.abs(jp_vals)))
        v_jp = abs(jp_vals[idx_jp])
        dd_jp = dd_grid[idx_jp]
        # 2 steps of Newton on JP
        for _ in range(2):
            ang = ba + ga * dd_jp
            d1 = -float(np.sum(w_jp * ga * np.sin(ang)))
            d2 = -float(np.sum(w_jp * g2a * np.cos(ang)))
            dd_jp -= d1 / d2
        v_jp = abs(float(np.sum(w_jp * np.cos(ba + ga * dd_jp))))
        if v_jp > best_jp_overall[0]:
            best_jp_overall = (v_jp, N_cut, T_cut, dd_jp)

        k_strs = []
        for k in ks:
            w = ta * np.exp(-k * g2a)
            vals = [float(np.sum(w * np.cos(ba + ga * dd))) for dd in dd_grid]
            idx_k = int(np.argmax(np.abs(vals)))
            dd_k = dd_grid[idx_k]
            for _ in range(2):
                ang = ba + ga * dd_k
                d1 = -float(np.sum(w * ga * np.sin(ang)))
                d2 = -float(np.sum(w * g2a * np.cos(ang)))
                dd_k -= d1 / d2
            vk = abs(float(np.sum(w * np.cos(ba + ga * dd_k))))
            k_strs.append(f"{vk:9.6f}")
        print(f"{N_cut:10d} ({T_cut:9.2f}) | {v_jp:11.7f} ({dd_jp:+8.2e}) | " + " | ".join(k_strs), flush=True)

    # Now perform rigorous 80-digit (266-bit) flint.arb ball-arithmetic verification at N = 460,000!
    N_cert = n_total
    T_cert_str = "296152.66"
    # Find optimal ddy at k = 1.5e-10 and for JP at T = 296152.66
    u = gamma / float(T_cert_str)
    w_jp = two_alpha * ((1.0 - u) * np.cos(np.pi * u) + np.sin(np.pi * u) / np.pi)
    dd_jp = best_jp_overall[3]
    for _ in range(3):
        ang = base_ang + gamma * dd_jp
        d1 = -float(np.sum(w_jp * gamma * np.sin(ang)))
        d2 = -float(np.sum(w_jp * g2 * np.cos(ang)))
        dd_jp -= d1 / d2

    y_cert_dec = y_dec + Decimal(str(dd_jp))
    y_cert_str = f"{y_cert_dec:.16f}"
    k_cert_str = "0.00000000015"  # 1.5e-10

    print(f"\nRunning rigorous 80-digit Arb interval verification at y* = {y_cert_str}, T = {T_cert_str}, k = {k_cert_str}...", flush=True)
    with Pool(64) as p:
        arb_res = p.map(arb_eval_chunk, [(ch, y_cert_str, k_cert_str, T_cert_str) for ch in raw_chunks])

    flint.ctx.dps = 80
    tot_gauss = flint.arb(0)
    tot_jp = flint.arb(0)
    for sg, sjp in arb_res:
        tot_gauss += flint.arb(sg)
        tot_jp += flint.arb(sjp)

    print(f"Rigorous 80-digit Arb enclosure (N={n_total}, T={T_cert_str}):")
    print(f"  h_JP(y*; T={T_cert_str})        = {tot_jp.str(30, radius=True)}")
    print(f"  h_Gauss(y*; k={k_cert_str}) = {tot_gauss.str(30, radius=True)}", flush=True)


if __name__ == "__main__":
    main()
