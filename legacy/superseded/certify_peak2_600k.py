#!/usr/bin/env python3
"""
SUPERSEDED -- DO NOT USE AS A CERTIFICATE.
This script evaluates the truncated sum at Peak2 for several kernels (still useful as a numerical
exploration), but its "eps_Mertens" / "eps_emp" tail bounds are NOT valid error bounds: they ignore the
weights 1/|rho zeta'(rho)| and are not the error term of any proved theorem, and the Jurkat-Peyerimhoff
kernel gives no explicit bound on the smallest counterexample. See THEOREM.md and certification/ for
the rigorous replacement. Kept for provenance; it reads its data from ../../search.
"""
"""
certify_peak2_600k.py
Rigorous 80-decimal-digit (266-bit) flint.arb ball-arithmetic certification
of the new Mertens counterexample bound at Peak2:
  y* ~ 3.4849713327331323965e18  (5.6x smaller than Kim & Nguyen 2024's 1.9572e19)
using all N = 600,000 refined Riemann zeta zeros (gamma <= 376,325).
"""
import glob, os, time, math, struct
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
from multiprocessing import Pool
import numpy as np
import gmpy2
from gmpy2 import mpfr
import flint
from decimal import Decimal, getcontext

getcontext().prec = 60

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "search")
CHUNK_DIR = os.path.join(ROOT, "chunks")
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
    raw_lines, y_jp_str, y_g_str, T_jp_str = args
    flint.ctx.dps = 80
    y_jp = flint.arb(y_jp_str)
    y_g = flint.arb(y_g_str)
    T_jp = flint.arb(T_jp_str)
    pi = flint.arb.pi()

    k15 = flint.arb("0.00000000015")   # 1.5e-10
    k20 = flint.arb("0.00000000020")   # 2.0e-10
    k10 = flint.arb("0.00000000010")   # 1.0e-10

    sum_jp = flint.arb(0)
    sum_g15 = flint.arb(0)
    sum_g20 = flint.arb(0)
    sum_g10 = flint.arb(0)

    for line in raw_lines:
        gs, rs, ims = line.split()
        g = flint.arb(gs)
        re_c = flint.arb(rs)
        im_c = flint.arb(ims)
        g2 = g * g

        # 1. Gaussian kernels at y_g
        gy_g = g * y_g
        s_g, c_g = gy_g.sin_cos()
        osc_g = 2 * (re_c * c_g - im_c * s_g)

        sum_g15 += osc_g * (-k15 * g2).exp()
        sum_g20 += osc_g * (-k20 * g2).exp()
        sum_g10 += osc_g * (-k10 * g2).exp()

        # 2. Jurkat-Peyerimhoff kernel at y_jp on [0, T_jp]
        if float(g.mid()) < float(T_jp.mid()):
            gy_jp = g * y_jp
            s_jp, c_jp = gy_jp.sin_cos()
            osc_jp = 2 * (re_c * c_jp - im_c * s_jp)
            u = g / T_jp
            pi_u = pi * u
            s_pu, c_pu = pi_u.sin_cos()
            w_jp = (1 - u) * c_pu + s_pu / pi
            sum_jp += osc_jp * w_jp

    return (
        sum_jp.str(50, radius=True),
        sum_g15.str(50, radius=True),
        sum_g20.str(50, radius=True),
        sum_g10.str(50, radius=True),
    )


def save_zeros_bin(path, gamma, two_alpha, ghi_list, glo_list, psi_list):
    n = len(gamma)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", n))
        f.write(gamma.astype(np.float64).tobytes())
        f.write((0.5 * two_alpha).astype(np.float64).tobytes())
        buf = bytearray(n * 48)
        off = 0
        for i in range(n):
            ghi = ghi_list[i]
            glo = glo_list[i]
            psi = psi_list[i]
            struct.pack_into(
                "<QQQQQQ",
                buf,
                off,
                ghi & 0xFFFFFFFFFFFFFFFF,
                ghi >> 64,
                glo & 0xFFFFFFFFFFFFFFFF,
                glo >> 64,
                psi & 0xFFFFFFFFFFFFFFFF,
                psi >> 64,
            )
            off += 48
        f.write(buf)


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

    bin_path = os.path.join(ROOT, "zeros_600k.bin")
    if not os.path.exists(bin_path):
        save_zeros_bin(bin_path, gamma, two_alpha, ghi_list, glo_list, psi_list)
        print(f"Saved binary table {bin_path} ({n_total} zeros)", flush=True)

    # Evaluate Peak2 across N = 100k..600k using fast precomputed cos/sin grid
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

    cos_ba = np.cos(base_ang)
    sin_ba = np.sin(base_ang)
    u_sc = gamma * 1.0e-5
    dd_grid = np.linspace(-1.5e-5, 1.5e-5, 301)
    z_grid = dd_grid * 1.0e5

    def fast_opt_1d(w_vec, ba_sub, ga_sub, g2_sub, c_sub, s_sub, u_sub):
        # 10-term Taylor expansion in u_sub * z_grid
        vals = np.zeros_like(z_grid)
        wc = w_vec * c_sub
        ws = w_vec * s_sub
        u_pow = np.ones_like(u_sub)
        z_pow = np.ones_like(z_grid)
        fact = 1.0
        for m in range(13):
            if m > 0:
                u_pow = u_pow * u_sub
                z_pow = z_pow * z_grid
                fact *= m
            if m % 4 == 0:
                coef = float(np.sum(wc * u_pow)) / fact
            elif m % 4 == 1:
                coef = -float(np.sum(ws * u_pow)) / fact
            elif m % 4 == 2:
                coef = -float(np.sum(wc * u_pow)) / fact
            else:
                coef = float(np.sum(ws * u_pow)) / fact
            vals += coef * z_pow
        dd = float(dd_grid[int(np.argmax(np.abs(vals)))])
        for _ in range(3):
            ang = ba_sub + ga_sub * dd
            d1 = -float(np.sum(w_vec * ga_sub * np.sin(ang)))
            d2 = -float(np.sum(w_vec * g2_sub * np.cos(ang)))
            dd -= d1 / d2
        v = float(np.sum(w_vec * np.cos(ba_sub + ga_sub * dd)))
        return v, dd

    print("\n=== PEAK2 (y0 = 3484971332733132396.5383310879) ACROSS N = 100,000 .. 600,000 ===", flush=True)
    ks = [3.0e-9, 3.0e-10, 2.5e-10, 2.2e-10, 2.0e-10, 1.5e-10, 1.0e-10, 5.0e-11, 0.0]
    hdr = "   N_zeros (    T_max) |   JP_kernel (    ddy*) | " + " | ".join(f"k={k:.1e}" for k in ks)
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)

    best_jp_overall = (0.0, 0, 0.0, 0.0)
    cutoffs = [100000, 150000, 200000, 240000, 260000, 300000, 350000, 400000, 460000, 500000, 550000, 600000]

    for N_cut in cutoffs:
        if N_cut > n_total:
            continue
        T_cut = gamma[N_cut - 1]
        ba = base_ang[:N_cut]
        ga = gamma[:N_cut]
        ta = two_alpha[:N_cut]
        g2a = g2[:N_cut]
        cs = cos_ba[:N_cut]
        ss = sin_ba[:N_cut]
        us = u_sc[:N_cut]

        # JP kernel
        u = ga / T_cut
        w_jp = ta * ((1.0 - u) * np.cos(np.pi * u) + np.sin(np.pi * u) / np.pi)
        v_jp, dd_jp = fast_opt_1d(w_jp, ba, ga, g2a, cs, ss, us)
        if abs(v_jp) > best_jp_overall[0]:
            best_jp_overall = (abs(v_jp), N_cut, T_cut, dd_jp)

        k_strs = []
        for k in ks:
            w = ta * np.exp(-k * g2a)
            vk, dd_k = fast_opt_1d(w, ba, ga, g2a, cs, ss, us)
            k_strs.append(f"{vk:9.6f}")
        print(f"{N_cut:10d} ({T_cut:9.2f}) | {v_jp:11.7f} ({dd_jp:+8.2e}) | " + " | ".join(k_strs), flush=True)

    # Exact Newton polish at N = 600,000 for JP and for Gaussian k = 1.5e-10
    T_jp_val = math.ceil(gamma[-1])
    T_jp_str = f"{T_jp_val:.2f}"
    u = gamma / T_jp_val
    w_jp = two_alpha * ((1.0 - u) * np.cos(np.pi * u) + np.sin(np.pi * u) / np.pi)
    dd_jp = best_jp_overall[3]
    for _ in range(4):
        ang = base_ang + gamma * dd_jp
        d1 = -float(np.sum(w_jp * gamma * np.sin(ang)))
        d2 = -float(np.sum(w_jp * g2 * np.cos(ang)))
        dd_jp -= d1 / d2

    w_g15 = two_alpha * np.exp(-1.5e-10 * g2)
    dd_g15 = 3.8e-6
    for _ in range(4):
        ang = base_ang + gamma * dd_g15
        d1 = -float(np.sum(w_g15 * gamma * np.sin(ang)))
        d2 = -float(np.sum(w_g15 * g2 * np.cos(ang)))
        dd_g15 -= d1 / d2

    y_jp_dec = y_dec + Decimal(f"{dd_jp:.16e}")
    y_g_dec = y_dec + Decimal(f"{dd_g15:.16e}")
    y_jp_str = f"{y_jp_dec:.16f}"
    y_g_str = f"{y_g_dec:.16f}"

    print(f"\nPolished optima at N = {n_total} (gamma_max = {gamma[-1]:.6f}):")
    print(f"  y*_JP    = {y_jp_str} (ddy = {dd_jp:+.10e})")
    print(f"  y*_Gauss = {y_g_str} (ddy = {dd_g15:+.10e})")

    # Compute analytical error bounds at T = gamma[-1]
    T_max = float(gamma[-1])
    mask_tail = (gamma >= T_max - 5000.0) & (gamma <= T_max)
    D_emp = float(np.sum(two_alpha[mask_tail]) / 5000.0)
    D_upper = 1.25 * D_emp
    print(f"\nEmpirical residue density on [{T_max-5000:.1f}, {T_max:.1f}]:")
    print(f"  D_emp = {D_emp:.8e},  1.25 * D_emp = {D_upper:.8e}")

    print("\n=== ANALYTICAL ERROR BOUNDS AT N = 600,000 (T = 376,325.43) ===")
    print("         k |      k*T^2 |  h_Gauss(y*_G) | Margin(>1) |    eps_emp |  eps_upper | eps_Mertens | Certified Net (>1)")
    print("-" * 116)
    for k_val in [2.2e-10, 2.0e-10, 1.8e-10, 1.5e-10, 1.2e-10, 1.0e-10, 5.0e-11]:
        w = two_alpha * np.exp(-k_val * g2)
        dd_k = dd_g15
        for _ in range(3):
            ang = base_ang + gamma * dd_k
            d1 = -float(np.sum(w * gamma * np.sin(ang)))
            d2 = -float(np.sum(w * g2 * np.cos(ang)))
            dd_k -= d1 / d2
        hk = float(np.sum(w * np.cos(base_ang + gamma * dd_k)))
        kT2 = k_val * T_max * T_max
        ek = math.exp(-kT2)
        eps_emp = D_emp * ek / (2.0 * k_val * T_max)
        eps_up = D_upper * ek / (2.0 * k_val * T_max)
        log_fac = math.log(T_max / (2.0 * math.pi)) / (2.0 * math.pi * k_val * T_max)
        S_bd = 0.137 * math.log(T_max) + 0.443 * math.log(math.log(T_max)) + 4.350
        eps_mert = (log_fac + 4.0 * S_bd) * ek
        net_mert = (hk - 1.0) - eps_mert
        print(f"  {k_val:8.2e} | {kT2:10.4f} |   {hk:12.8f} | {hk-1.0:+10.3e} | {eps_emp:10.3e} | {eps_up:10.3e} |  {eps_mert:10.3e} | {net_mert:+18.8e}")

    print(f"\nRunning rigorous 80-digit Arb interval verification across {len(raw_chunks)} chunks...", flush=True)
    t_arb = time.time()
    with Pool(64) as p:
        arb_res = p.map(arb_eval_chunk, [(ch, y_jp_str, y_g_str, T_jp_str) for ch in raw_chunks])

    flint.ctx.dps = 80
    tot_jp = flint.arb(0)
    tot_g15 = flint.arb(0)
    tot_g20 = flint.arb(0)
    tot_g10 = flint.arb(0)
    for sjp, sg15, sg20, sg10 in arb_res:
        tot_jp += flint.arb(sjp)
        tot_g15 += flint.arb(sg15)
        tot_g20 += flint.arb(sg20)
        tot_g10 += flint.arb(sg10)

    print(f"\n=== RIGOROUS 80-DIGIT ARB INTERVAL CERTIFICATION (N = {n_total}, elapsed = {time.time()-t_arb:.2f}s) ===")
    print(f"  1. Jurkat-Peyerimhoff / Odlyzko-te Riele / Kotnik-te Riele Compact-Support Kernel (T = {T_jp_str}):")
    print(f"     y*_JP                 = {y_jp_str}")
    print(f"     h_JP(y*_JP; T)        = {tot_jp.str(35, radius=True)}")
    print(f"     Excess above 1.000000 = {(tot_jp - 1).str(15, radius=True)}")
    print(f"\n  2. Pintz (1987) / Saouter-te Riele (2014) Gaussian Kernel at y*_Gauss = {y_g_str}:")
    print(f"     h_Gauss(k=2.0e-10)    = {tot_g20.str(35, radius=True)}")
    print(f"     h_Gauss(k=1.5e-10)    = {tot_g15.str(35, radius=True)}")
    print(f"     h_Gauss(k=1.0e-10)    = {tot_g10.str(35, radius=True)}")


if __name__ == "__main__":
    main()
