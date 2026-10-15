#!/usr/bin/env python3
"""
Path 1: Analytical Kernel & Gaussian Parameter Optimization
Executes Tasks 2, 3, and 4 on 71,250 zeros (old 285-chunk set & contiguous),
98,625 zeros (T=74000), and 100,000 zeros (T=74920.83), and writes a comprehensive
report to path1_report.txt.
"""
import ctypes
import glob
import math
import os
import re
import time
import gmpy2
from gmpy2 import mpfr
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
NPZ_PATH = os.path.join(ROOT, "path1_zeros_100k.npz")
SO_PATH = os.path.join(ROOT, "libpath1_eval.so")
REPORT_PATH = os.path.join(ROOT, "results", "path1_report.txt")

lib = ctypes.CDLL(SO_PATH)
lib.compute_peak_AB.argtypes = [
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
]


def load_zeros_npz():
    data = np.load(NPZ_PATH)
    return {k: data[k] for k in data.files}


def load_high_precision_chunks(n_max=100000):
    gmpy2.get_context().precision = 256
    files = sorted(glob.glob(os.path.join(ROOT, "chunks", "chunk_*.txt")))
    gammas = []
    re_c = []
    im_c = []
    for p in files:
        with open(p, "r") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                gs, rs, ims = s.split()
                gammas.append(mpfr(gs))
                re_c.append(mpfr(rs))
                im_c.append(mpfr(ims))
                if len(gammas) >= n_max:
                    break
        if len(gammas) >= n_max:
            break
    return gammas, re_c, im_c


def compute_exact_mpfr_AB(y_str_or_mpfr, gammas, re_c, im_c):
    """
    Computes exact float64 arrays A_j, B_j using 256-bit gmpy2.mpfr phase reduction:
      A_j = 2 * Re(e^{i gamma_j y} / (rho_j zeta'(rho_j)))
      B_j = 2 * Im(e^{i gamma_j y} / (rho_j zeta'(rho_j)))
    so that for y + ddy, the j-th term is A_j cos(gamma_j ddy) - B_j sin(gamma_j ddy).
    """
    gmpy2.get_context().precision = 256
    y = mpfr(y_str_or_mpfr)
    n = len(gammas)
    A = np.empty(n, dtype=np.float64)
    B = np.empty(n, dtype=np.float64)
    for i in range(n):
        th = gammas[i] * y
        c = gmpy2.cos(th)
        s = gmpy2.sin(th)
        r = re_c[i]
        im = im_c[i]
        A[i] = float(2 * (r * c - im * s))
        B[i] = float(2 * (r * s + im * c))
    return A, B


def polish_single_peak(A, B, gamma, w, n_zeros=None, grid_hw=4e-5, grid_steps=40):
    """
    Polishes ddy for a single peak (A, B) with weights w[:n_zeros] using ALL n_zeros
    for both the initial grid scan around ddy=0 and Newton-Raphson refinement.
    Returns (h_unpol, h_pol, ddy_pol).
    """
    if n_zeros is None:
        n_zeros = len(gamma)
    Ap = A[:n_zeros]
    Bp = B[:n_zeros]
    gp = gamma[:n_zeros]
    wp = w[:n_zeros]

    wA = wp * Ap
    wB = wp * Bp
    h0 = float(np.sum(wA))

    # Evaluate on a full-zero grid around ddy=0 plus Newton from ddy=0 and best grid point
    candidates = [0.0]
    if grid_steps > 0:
        ddys = np.linspace(-grid_hw, grid_hw, 2 * grid_steps + 1)
        th_mat = np.outer(ddys, gp)
        vals = np.cos(th_mat) @ wA - np.sin(th_mat) @ wB
        idx = int(np.argmax(np.abs(vals)))
        candidates.append(float(ddys[idx]))

    best_h = h0
    best_ddy = 0.0
    for start_ddy in candidates:
        ddy = start_ddy
        for _ in range(8):
            th = gp * ddy
            c = np.cos(th)
            s = np.sin(th)
            val = Ap * c - Bp * s
            dval = gp * (-Ap * s - Bp * c)
            ddval = -(gp * gp) * val
            fp = float(np.dot(wp, dval))
            fpp = float(np.dot(wp, ddval))
            if abs(fpp) > 1e-12:
                step = fp / fpp
                step = max(-2e-5, min(2e-5, step))
                ddy -= step
                if abs(step) < 1e-14:
                    break
            else:
                break
        th = gp * ddy
        h_cand = float(np.dot(wp, Ap * np.cos(th) - Bp * np.sin(th)))
        if abs(h_cand) > abs(best_h):
            best_h = h_cand
            best_ddy = ddy

    return h0, best_h, best_ddy


def fast_polish_batch_blas(A_mat, B_mat, gamma_vec, w_vec, ddy_grid=None):
    """
    Polishes all n_peaks in (A_mat, B_mat) for weight vector w_vec in ~0.03s using
    degree-16 Taylor moments around ddy=0 on [-3e-5, +3e-5], then exact refinement on top peaks!
    Let u = gamma * 1e-4 (so |u| <= 7.5) and z = ddy * 1e4 (so |z| <= 0.3 for |ddy| <= 3e-5).
    cos(gamma * ddy) = cos(u * z) = sum_{m=0}^8 (-1)^m (u^{2m} / (2m)!) z^{2m}
    sin(gamma * ddy) = sin(u * z) = sum_{m=0}^7 (-1)^m (u^{2m+1} / (2m+1)!) z^{2m+1}
    """
    if ddy_grid is None:
        ddy_grid = np.linspace(-3.0e-5, 3.0e-5, 121)
    u = gamma_vec * 1.0e-4
    z = ddy_grid * 1.0e4  # shape (G,)

    M_cos = 9
    M_sin = 8
    U_cos = np.empty((len(gamma_vec), M_cos), dtype=np.float64)
    U_sin = np.empty((len(gamma_vec), M_sin), dtype=np.float64)
    Z_cos = np.empty((M_cos, len(z)), dtype=np.float64)
    Z_sin = np.empty((M_sin, len(z)), dtype=np.float64)

    for m in range(M_cos):
        sign = -1.0 if (m % 2 == 1) else 1.0
        U_cos[:, m] = w_vec * (u ** (2 * m)) * (sign / math.factorial(2 * m))
        Z_cos[m, :] = z ** (2 * m)
    for m in range(M_sin):
        sign = -1.0 if (m % 2 == 1) else 1.0
        U_sin[:, m] = w_vec * (u ** (2 * m + 1)) * (sign / math.factorial(2 * m + 1))
        Z_sin[m, :] = z ** (2 * m + 1)

    # BLAS GEMM: (n_peaks, n_zeros) @ (n_zeros, M) -> (n_peaks, M)
    C_mom = A_mat @ U_cos
    S_mom = B_mat @ U_sin

    # Evaluate on grid z: shape (n_peaks, G)
    vals_grid = C_mom @ Z_cos - S_mom @ Z_sin
    h_unpol = C_mom[:, 0].copy()

    best_idx = np.argmax(np.abs(vals_grid), axis=1)
    rows = np.arange(A_mat.shape[0])
    h_pol = vals_grid[rows, best_idx]
    ddy_pol = ddy_grid[best_idx]

    # Exact Newton polish on top 25 peaks by |h_pol| starting directly from ddy_pol[idx]
    top_indices = np.argsort(-np.abs(h_pol))[:25]
    for idx in top_indices:
        ddy = float(ddy_pol[idx])
        Ap = A_mat[idx]
        Bp = B_mat[idx]
        for _ in range(4):
            th = gamma_vec * ddy
            c = np.cos(th)
            s = np.sin(th)
            val = Ap * c - Bp * s
            dval = gamma_vec * (-Ap * s - Bp * c)
            ddval = -(gamma_vec * gamma_vec) * val
            fp = float(np.dot(w_vec, dval))
            fpp = float(np.dot(w_vec, ddval))
            if abs(fpp) > 1e-12:
                step = max(-1e-5, min(1e-5, fp / fpp))
                ddy -= step
                if abs(step) < 1e-13:
                    break
        th = gamma_vec * ddy
        hp_ex = float(np.dot(w_vec, Ap * np.cos(th) - Bp * np.sin(th)))
        if abs(hp_ex) > abs(h_unpol[idx]):
            h_pol[idx] = hp_ex
            ddy_pol[idx] = ddy
        else:
            h_pol[idx] = h_unpol[idx]
            ddy_pol[idx] = 0.0

    return h_unpol, h_pol, ddy_pol


def parse_log_clusters(log_paths, min_abs_h=0.96, max_y=1.95718788505622e19):
    gmpy2.get_context().precision = 256
    pat = re.compile(
        r"\|h\|=([0-9.]+)\s+h=([+-]?[0-9.]+)\s+y=([0-9.]+)\s+\(x=(\d+),\s*dy=([+-]?[0-9.]+)\)"
    )
    clusters = {}
    for path in log_paths:
        if not os.path.exists(path):
            continue
        bname = os.path.basename(path)
        with open(path, "r") as f:
            for line in f:
                if "CLUSTER" not in line:
                    continue
                m = pat.search(line)
                if not m:
                    continue
                abs_h = float(m.group(1))
                h_val = float(m.group(2))
                x_int = int(m.group(4))
                dy_val = float(m.group(5))
                if abs_h < min_abs_h:
                    continue
                y_approx = x_int / (1 << 30) + dy_val
                if y_approx <= 0 or y_approx >= max_y:
                    continue
                # Deduplicate by x_int >> 18 (within ~256 units of y, or same peak)
                key = x_int >> 18
                if key not in clusters or abs_h > clusters[key]["abs_h_log"]:
                    y_exact = mpfr(x_int) / mpfr(1 << 30) + mpfr(m.group(5))
                    clusters[key] = {
                        "source": bname,
                        "abs_h_log": abs_h,
                        "h_log": h_val,
                        "x": x_int,
                        "dy": dy_val,
                        "y_approx": y_approx,
                        "y_mpfr": y_exact,
                    }
    res = sorted(clusters.values(), key=lambda d: -d["abs_h_log"])
    return res


def format_mpfr_y(y_mp):
    return f"{y_mp:.10f}"


def compute_analytical_error_bounds(k_val, T_val, y_val, gamma_arr, alpha_arr):
    mask = (gamma_arr >= T_val - 2000.0) & (gamma_arr <= T_val)
    D_emp = float(np.sum(2.0 * alpha_arr[mask]) / 2000.0)
    D_upper = 1.25 * D_emp

    kT2 = k_val * (T_val ** 2)
    two_kT = 2.0 * k_val * T_val

    if kT2 < 700.0 and k_val > 0:
        exp_neg_kT2 = math.exp(-kT2)
        eps_emp = D_emp * exp_neg_kT2 / two_kT
        eps_emp_upper = D_upper * exp_neg_kT2 / two_kT
        log_factor = math.log(T_val / (2.0 * math.pi)) / (2.0 * math.pi * k_val * T_val)
        S_bound = 0.137 * math.log(T_val) + 0.443 * math.log(math.log(T_val)) + 4.350
        eps_mertens = (log_factor + 4.0 * S_bound) * exp_neg_kT2
        eps_pintz_global = (math.e * y_val / (2.0 * math.pi * k_val * T_val)) * exp_neg_kT2
    else:
        eps_emp = 0.0 if k_val > 0 else float("inf")
        eps_emp_upper = 0.0 if k_val > 0 else float("inf")
        eps_mertens = 0.0 if k_val > 0 else float("inf")
        eps_pintz_global = 0.0 if k_val > 0 else float("inf")

    return {
        "D_emp": D_emp,
        "D_upper": D_upper,
        "kT2": kT2,
        "eps_emp": eps_emp,
        "eps_emp_upper": eps_emp_upper,
        "eps_mertens": eps_mertens,
        "eps_pintz_global": eps_pintz_global,
    }


def solve_T_min_for_disproof(k_val, target_margin, y_val, D_upper=8.0e-6):
    if k_val <= 0 or target_margin <= 0:
        return {}

    def n_zeros_at_T(T):
        return (T / (2.0 * math.pi)) * math.log(T / (2.0 * math.pi * math.e))

    lo, hi = 100.0, 1e8
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        kT2 = k_val * mid * mid
        val = math.log(D_upper / (2.0 * k_val * mid)) - kT2
        if val <= math.log(target_margin):
            hi = mid
        else:
            lo = mid
    T_emp = hi

    lo, hi = 100.0, 1e8
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        kT2 = k_val * mid * mid
        S_b = 0.137 * math.log(mid) + 0.443 * math.log(math.log(mid)) + 4.350
        pref = math.log(mid / (2.0 * math.pi)) / (2.0 * math.pi * k_val * mid) + 4.0 * S_b
        val = math.log(pref) - kT2
        if val <= math.log(target_margin):
            hi = mid
        else:
            lo = mid
    T_mertens = hi

    lo, hi = 100.0, 1e8
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        kT2 = k_val * mid * mid
        pref = (math.e * y_val) / (2.0 * math.pi * k_val * mid)
        val = math.log(pref) - kT2
        if val <= math.log(target_margin):
            hi = mid
        else:
            lo = mid
    T_pintz_glob = hi

    return {
        "T_emp": T_emp,
        "N_emp": n_zeros_at_T(T_emp),
        "T_mertens": T_mertens,
        "N_mertens": n_zeros_at_T(T_mertens),
        "T_pintz_glob": T_pintz_glob,
        "N_pintz_glob": n_zeros_at_T(T_pintz_glob),
    }


def build_basis_G_tensor(gamma_arr, T_val, M, basis_type="jp"):
    n_zeros = len(gamma_arr)
    G = np.zeros((M + 1, M + 1, n_zeros), dtype=np.float64)
    L = 0.5 * (T_val - gamma_arr)
    half_g = 0.5 * gamma_arr

    alphas = np.zeros(M + 1, dtype=np.float64)
    omegas = np.zeros(M + 1, dtype=np.float64)
    for m in range(M + 1):
        if basis_type == "jp":
            alphas[m] = math.sqrt(2.0 / T_val)
            omegas[m] = (2 * m + 1) * math.pi / T_val
        else:
            alphas[m] = math.sqrt(1.0 / T_val) if m == 0 else math.sqrt(2.0 / T_val)
            omegas[m] = (2 * m) * math.pi / T_val

    def sinc_rad(x):
        return np.sinc(x / np.pi)

    for m in range(M + 1):
        for n in range(m, M + 1):
            w_plus = omegas[m] + omegas[n]
            w_minus = omegas[m] - omegas[n]
            term1 = np.cos(w_minus * half_g) * sinc_rad(w_plus * L)
            term2 = np.cos(w_plus * half_g) * sinc_rad(w_minus * L)
            val = alphas[m] * alphas[n] * L * (term1 + term2)
            G[m, n, :] = val
            if m != n:
                G[n, m, :] = val

    return G, omegas ** 2, alphas


def numerical_autoconv_kernel(gamma_arr, T_val, phi_func, n_grid=16384):
    u = np.linspace(-0.5 * T_val, 0.5 * T_val, n_grid, endpoint=True)
    du = T_val / (n_grid - 1)
    vals = phi_func(u)
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    norm2 = trapz_fn(vals * vals, u)
    vals = vals / math.sqrt(norm2)
    dvals = np.gradient(vals, du)
    sigma_x2 = float(trapz_fn(dvals * dvals, u))

    n_fft = 1 << (2 * n_grid).bit_length()
    F = np.fft.rfft(vals, n=n_fft)
    acf = np.fft.irfft(F * np.conj(F), n=n_fft)[:n_grid] * du
    acf[0] = 1.0
    g_grid = np.arange(n_grid) * du
    g_at_zeros = np.interp(gamma_arr, g_grid, acf, right=0.0)
    return g_at_zeros, sigma_x2


def main():
    t_start = time.time()
    out_lines = []

    def log(msg=""):
        print(msg, flush=True)
        out_lines.append(msg)

    log("=" * 100)
    log("PATH 1 REPORT: ANALYTICAL KERNEL & GAUSSIAN PARAMETER OPTIMIZATION FOR THE MERTENS BOUND")
    log("=" * 100)

    zdata = load_zeros_npz()
    gamma_all = zdata["gamma"]
    alpha_all = zdata["alpha"]
    n_total = len(gamma_all)
    n_71250 = 71250
    n_98625 = int(np.sum(gamma_all < 74000.0))
    n_100k = 100000
    idx_old_71250 = np.r_[0:70750, 89750:90250]

    log(f"\n[TASK 1 SUMMARY: ZETA ZEROS & BINARY TABLE CONSTRUCTION]")
    log(f"  Total refined zeros in chunks/ (400 chunks): {n_total}")
    log(f"  Old 285-chunk set (zeros.bin, 71,250 zeros): [0..70749] (gamma <= {gamma_all[70749]:.6f}) + [89750..90249] (gamma <= {gamma_all[90249]:.6f})")
    log(f"  Contiguous N = 71,250 cutoff : gamma[71249] = {gamma_all[n_71250-1]:.12f}")
    log(f"  Contiguous N = 98,625 cutoff : gamma[98624] = {gamma_all[n_98625-1]:.12f} (< 74000), gamma[98625] = {gamma_all[n_98625]:.12f}")
    log(f"  Contiguous N = 100,000 cutoff: gamma[99999] = {gamma_all[n_100k-1]:.12f}")
    log(f"  Constructed binary tables: zeros_98625.bin ({n_98625} zeros) and zeros_100k.bin ({n_100k} zeros)")

    log("\nLoading 100,000 zeros at 256-bit precision for exact verification on primary peaks...")
    gammas_mp, re_mp, im_mp = load_high_precision_chunks(n_100k)

    primary_peaks = [
        ("Kim-Nguyen Record", "19571878850562201959.215107"),
        ("Peak1 (1.3860e19)", "13859539710197847064.0623176514"),
        ("Peak2 (3.4850e18)", "3484971332733132396.5383310879"),
        ("Peak3 (8.0037e18)", "8003670347688035135.9983565242"),
        ("Peak4 (1.1703e19)", "11702632930821509802.7452619724"),
    ]

    prim_cache = os.path.join(ROOT, "path1_primary_AB.npz")
    primary_AB = {}
    if os.path.exists(prim_cache):
        pdata = np.load(prim_cache)
        for i, (name, y_str) in enumerate(primary_peaks):
            primary_AB[name] = (y_str, pdata[f"A_{i}"], pdata[f"B_{i}"])
    else:
        save_dict = {}
        for i, (name, y_str) in enumerate(primary_peaks):
            A, B = compute_exact_mpfr_AB(y_str, gammas_mp, re_mp, im_mp)
            primary_AB[name] = (y_str, A, B)
            save_dict[f"A_{i}"] = A
            save_dict[f"B_{i}"] = B
        np.savez(prim_cache, **save_dict)

    # =========================================================================
    # TASK 2: Evaluate 71,250 / 98,625 / 100,000-zero sums at k = 3e-9
    # =========================================================================
    log("\n" + "=" * 100)
    log("[TASK 2: FULL 71,250 / 98,625 (T=74000) / 100,000 (T=74920.83) ZERO EVALUATIONS AT k = 3e-9]")
    log("=" * 100)

    w_str = np.exp(-3e-9 * (gamma_all ** 2))
    # Both contiguous [71250..98624] and missing chunks from old_71250
    mask_missing_from_old = np.ones(n_98625, dtype=bool)
    mask_missing_from_old[idx_old_71250] = False
    tail_abs_missing_old = float(np.sum(2.0 * alpha_all[:n_98625][mask_missing_from_old] * w_str[:n_98625][mask_missing_from_old]))
    tail_abs_68363_74000 = float(np.sum(2.0 * alpha_all[90250:n_98625] * w_str[90250:n_98625]))
    tail_abs_98625_100k = float(np.sum(2.0 * alpha_all[n_98625:n_100k] * w_str[n_98625:n_100k]))
    log(f"  Maximum possible worst-case phase-aligned contribution at k = 3e-9:")
    log(f"    Missing 27,375 zeros (old 285-chunk set -> 98,625): sum 2*alpha*e^(-k*g^2) = {tail_abs_missing_old:.12e}")
    log(f"    Zeros 90,250..98,624 (68363 <= gamma < 74000)     : sum 2*alpha*e^(-k*g^2) = {tail_abs_68363_74000:.12e}")
    log(f"    Zeros 98,625..99,999 (74000 <= gamma <= 74920.83) : sum 2*alpha*e^(-k*g^2) = {tail_abs_98625_100k:.12e}")

    log("\n  --- 2A. Exact 256-bit MPFR Evaluation on Primary Peaks (k = 3e-9) ---")
    header = (
        f"  {'Peak Name':<20s} | {'h_71250(y0)':>14s} | {'h_98625(y0)':>14s} | "
        f"{'dh(71k->98.6k)':>14s} | {'dh(68.3k->74k)':>14s} | {'h_100k(y0)':>14s} | {'|h_98625(y*)|':>14s}"
    )
    log(header)
    log("  " + "-" * (len(header) - 2))
    for name, y_str in primary_peaks:
        _, A, B = primary_AB[name]
        h71 = float(np.dot(w_str[idx_old_71250], A[idx_old_71250]))
        h90250 = float(np.dot(w_str[:90250], A[:90250]))
        h98 = float(np.dot(w_str[:n_98625], A[:n_98625]))
        h100 = float(np.dot(w_str[:n_100k], A[:n_100k]))
        dh_71_98 = h98 - h71
        dh_90_98 = h98 - h90250
        _, h98_pol, ddy98 = polish_single_peak(A, B, gamma_all, w_str, n_98625, grid_hw=4e-5, grid_steps=20)
        log(
            f"  {name:<20s} | {h71:14.10f} | {h98:14.10f} | "
            f"{dh_71_98:+14.10e} | {dh_90_98:+14.10e} | {h100:14.10f} | {abs(h98_pol):14.10f}"
        )

    log_dir = os.path.join(ROOT, "logs") if os.path.isdir(os.path.join(ROOT, "logs")) else ROOT
    req_logs = [
        os.path.join(log_dir, "hr_wave1.log"),
        os.path.join(log_dir, "pc_famA.log"),
        os.path.join(log_dir, "hits_exact_y_multicenter.log"),
    ]
    all_logs = sorted(glob.glob(os.path.join(log_dir, "*.log")) + glob.glob(os.path.join(ROOT, "*.log")))
    clusters_req = parse_log_clusters(req_logs, min_abs_h=0.96)
    clusters_all = parse_log_clusters(all_logs, min_abs_h=0.96)

    log(f"\n  --- 2B. All Unique Sub-Record Clusters with |h| >= 0.96 in hr_wave1.log, pc_famA.log, hits_exact_y_multicenter.log ---")
    log(f"  Found {len(clusters_req)} unique clusters with |h| >= 0.96 in the 3 specified logs (and {len(clusters_all)} across all workspace logs).")

    n_cl = len(clusters_all)
    x_lo_arr = np.array([c["x"] & ((1 << 64) - 1) for c in clusters_all], dtype=np.uint64)
    x_hi_arr = np.array([(c["x"] >> 64) & ((1 << 64) - 1) for c in clusters_all], dtype=np.uint64)
    dy0_arr = np.array([c["dy"] for c in clusters_all], dtype=np.float64)

    A_cl = np.empty((n_cl, n_100k), dtype=np.float64)
    B_cl = np.empty((n_cl, n_100k), dtype=np.float64)

    t_ab = time.time()
    lib.compute_peak_AB(
        n_cl,
        x_lo_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        x_hi_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        dy0_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        n_100k,
        alpha_all.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        gamma_all.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        zdata["ghi_lo"].ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        zdata["ghi_hi"].ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        zdata["glo_lo"].ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        zdata["glo_hi"].ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        zdata["psi_lo"].ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        zdata["psi_hi"].ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        A_cl.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        B_cl.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    log(f"  Precomputed 100,000-zero phase matrices (A, B) for {n_cl} unique clusters in {time.time()-t_ab:.3f}s.")

    h71_unpol = A_cl[:, idx_old_71250] @ w_str[idx_old_71250]
    h90250_unpol = A_cl[:, :90250] @ w_str[:90250]
    h98_unpol = A_cl[:, :n_98625] @ w_str[:n_98625]
    h100_unpol = A_cl[:, :n_100k] @ w_str[:n_100k]

    _, h98_p_out, ddy98_out = fast_polish_batch_blas(A_cl[:, :n_98625], B_cl[:, :n_98625], gamma_all[:n_98625], w_str[:n_98625])
    _, h100_p_out, ddy100_out = fast_polish_batch_blas(A_cl, B_cl, gamma_all, w_str)

    dh_arr = h98_unpol - h71_unpol
    dh_68_74 = h98_unpol - h90250_unpol
    log(f"  Summary of dh at k = 3e-9 across all {n_cl} clusters:")
    log(f"    max |h_98625 - h_71250(old)| = {np.max(np.abs(dh_arr)):.10e} (rms = {np.sqrt(np.mean(dh_arr**2)):.10e})")
    log(f"    max |dh(68363 <= g < 74000)| = {np.max(np.abs(dh_68_74)):.10e} (rms = {np.sqrt(np.mean(dh_68_74**2)):.10e})")
    log(f"    max |h_100k - h_98625|       = {np.max(np.abs(h100_unpol - h98_unpol)):.10e}")

    log("\n  Top 25 Unique Sub-Record Clusters (sorted by polished |h_98625| at k = 3e-9):")
    order_98 = np.argsort(-np.abs(h98_p_out))
    log(
        f"  {'Rank':>4s} | {'Source Log':<28s} | {'y (approx)':>22s} | {'|h_71250(y0)|':>13s} | "
        f"{'|h_98625(y*)|':>13s} | {'|h_100k(y*)|':>13s} | {'dh(71k->98k)':>13s}"
    )
    log("  " + "-" * 120)
    for rank, idx in enumerate(order_98[:25], 1):
        c = clusters_all[idx]
        log(
            f"  {rank:4d} | {c['source']:<28s} | {c['y_approx']:22.6f} | {abs(h71_unpol[idx]):13.9f} | "
            f"{abs(h98_p_out[idx]):13.9f} | {abs(h100_p_out[idx]):13.9f} | {dh_arr[idx]:+13.5e}"
        )

    # =========================================================================
    # TASK 3: Analytical Gaussian Parameter Optimization (k < 3e-9)
    # =========================================================================
    log("\n" + "=" * 100)
    log("[TASK 3: ANALYTICAL GAUSSIAN PARAMETER OPTIMIZATION (k in [0, 3e-9]) & PINTZ/StR ERROR BOUNDS]")
    log("=" * 100)

    k_grid = [
        3.0e-9,
        2.5e-9,
        2.0e-9,
        1.5e-9,
        1.2e-9,
        1.0e-9,
        8.0e-10,
        6.0e-10,
        5.0e-10,
        4.0e-10,
        3.0e-10,
        2.0e-10,
        1.5e-10,
        1.0e-10,
        5.0e-11,
        2.0e-11,
        1.0e-11,
        0.0,
    ]

    for p_name in ["Peak2 (3.4850e18)", "Peak1 (1.3860e19)", "Peak3 (8.0037e18)", "Peak4 (1.1703e19)"]:
        y_str, A, B = primary_AB[p_name]
        y_val = float(y_str)
        log(f"\n  --- 3A. Gaussian Parameter Scan on {p_name}: y0 = {y_str} ---")
        log(
            f"  {'k':>10s} | {'|h_k(71250,y*)|':>15s} | {'|h_k(98625,y*)|':>15s} | "
            f"{'|h_k(100k,y*)|':>15s} | {'ddy*(100k)':>11s} | {'eps_emp(74921)':>14s} | {'eps_Mert(74921)':>15s}"
        )
        log("  " + "-" * 112)
        for kv in k_grid:
            wk = np.exp(-kv * (gamma_all ** 2))
            _, h71_p, _ = polish_single_peak(A[idx_old_71250], B[idx_old_71250], gamma_all[idx_old_71250], wk[idx_old_71250], n_71250, grid_hw=4e-5, grid_steps=40)
            _, h98_p, _ = polish_single_peak(A, B, gamma_all, wk, n_98625, grid_hw=4e-5, grid_steps=40)
            _, h100_p, ddy100 = polish_single_peak(A, B, gamma_all, wk, n_100k, grid_hw=4e-5, grid_steps=40)
            eb = compute_analytical_error_bounds(kv, gamma_all[-1], y_val, gamma_all, alpha_all)
            log(
                f"  {kv:10.3e} | {abs(h71_p):15.9f} | {abs(h98_p):15.9f} | "
                f"{abs(h100_p):15.9f} | {ddy100:+11.3e} | {eb['eps_emp_upper']:14.4e} | {eb['eps_mertens']:15.4e}"
            )

    log(f"\n  --- 3B. Full Scan of All {n_cl} Unique Sub-Record Clusters Across k in [0, 3e-9] ---")
    scan_ks = [3.0e-9, 2.0e-9, 1.5e-9, 1.0e-9, 7.5e-10, 5.0e-10, 3.0e-10, 2.0e-10, 1.5e-10, 1.0e-10, 5.0e-11, 0.0]

    best_per_cluster_100k = np.zeros(n_cl, dtype=np.float64)
    best_k_per_cluster_100k = np.zeros(n_cl, dtype=np.float64)
    best_per_cluster_98k = np.zeros(n_cl, dtype=np.float64)
    best_per_cluster_71k = np.zeros(n_cl, dtype=np.float64)
    best_k_per_cluster_71k = np.zeros(n_cl, dtype=np.float64)

    h_mat_100k = np.zeros((len(scan_ks), n_cl), dtype=np.float64)

    A_71_sub = np.ascontiguousarray(A_cl[:, idx_old_71250])
    B_71_sub = np.ascontiguousarray(B_cl[:, idx_old_71250])
    g_71_sub = np.ascontiguousarray(gamma_all[idx_old_71250])
    A_98_sub = np.ascontiguousarray(A_cl[:, :n_98625])
    B_98_sub = np.ascontiguousarray(B_cl[:, :n_98625])

    t_scan = time.time()
    for ki, kv in enumerate(scan_ks):
        wk = np.exp(-kv * (gamma_all ** 2))
        _, hp100, _ = fast_polish_batch_blas(A_cl, B_cl, gamma_all, wk)
        _, hp98, _ = fast_polish_batch_blas(A_98_sub, B_98_sub, gamma_all[:n_98625], wk[:n_98625])
        _, hp71, _ = fast_polish_batch_blas(A_71_sub, B_71_sub, g_71_sub, wk[idx_old_71250])

        h_mat_100k[ki, :] = hp100
        imp100 = np.abs(hp100) > best_per_cluster_100k
        best_per_cluster_100k[imp100] = np.abs(hp100[imp100])
        best_k_per_cluster_100k[imp100] = kv

        imp98 = np.abs(hp98) > best_per_cluster_98k
        best_per_cluster_98k[imp98] = np.abs(hp98[imp98])

        imp71 = np.abs(hp71) > best_per_cluster_71k
        best_per_cluster_71k[imp71] = np.abs(hp71[imp71])
        best_k_per_cluster_71k[imp71] = kv

    log(f"  Completed 12-point k-scan across {n_cl} clusters x 3 zero cutoffs in {time.time()-t_scan:.2f}s.")
    n_over_1_100k = int(np.sum(best_per_cluster_100k >= 1.0))
    n_over_1_71k = int(np.sum(best_per_cluster_71k >= 1.0))
    log(f"  Number of unique sub-record clusters achieving |h_k(y)| >= 1.000: {n_over_1_100k} on 100k zeros ({n_over_1_71k} on 71.25k zeros).")

    log("\n  Top 25 Sub-Record Clusters Ranked by max_{k in [0, 3e-9]} |h_k(y)| on 100,000 / 71,250 Zeros:")
    order_bestk = np.argsort(-np.maximum(best_per_cluster_100k, best_per_cluster_71k))
    log(
        f"  {'Rank':>4s} | {'Source Log':<26s} | {'y (approx)':>22s} | {'|h_3e-9(100k)|':>14s} | "
        f"{'max|h_k(71k)|':>13s} | {'max|h_k(98.6k)|':>15s} | {'max|h_k(100k)|':>14s} | {'opt k(100k)':>11s}"
    )
    log("  " + "-" * 132)
    for rank, idx in enumerate(order_bestk[:25], 1):
        c = clusters_all[idx]
        log(
            f"  {rank:4d} | {c['source']:<26s} | {c['y_approx']:22.6f} | {abs(h_mat_100k[0, idx]):14.9f} | "
            f"{best_per_cluster_71k[idx]:13.9f} | {best_per_cluster_98k[idx]:15.9f} | {best_per_cluster_100k[idx]:14.9f} | "
            f"{best_k_per_cluster_100k[idx]:11.3e}"
        )

    # Also specifically report on hr_wave1.log clusters as requested in Task 3
    hr_clusters = parse_log_clusters([os.path.join(log_dir, "hr_wave1.log")], min_abs_h=0.90)
    log(f"\n  --- 3C. Dedicated Scan of All {len(hr_clusters)} Unique Clusters in hr_wave1.log Across k in [0, 3e-9] ---")
    n_hr = len(hr_clusters)
    x_lo_hr = np.array([c["x"] & ((1 << 64) - 1) for c in hr_clusters], dtype=np.uint64)
    x_hi_hr = np.array([(c["x"] >> 64) & ((1 << 64) - 1) for c in hr_clusters], dtype=np.uint64)
    dy0_hr = np.array([c["dy"] for c in hr_clusters], dtype=np.float64)
    A_hr = np.empty((n_hr, n_100k), dtype=np.float64)
    B_hr = np.empty((n_hr, n_100k), dtype=np.float64)
    lib.compute_peak_AB(
        n_hr,
        x_lo_hr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        x_hi_hr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        dy0_hr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        n_100k,
        alpha_all.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        gamma_all.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        zdata["ghi_lo"].ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        zdata["ghi_hi"].ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        zdata["glo_lo"].ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        zdata["glo_hi"].ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        zdata["psi_lo"].ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        zdata["psi_hi"].ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        A_hr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        B_hr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    best_hr_100k = np.zeros(n_hr, dtype=np.float64)
    best_hr_k100 = np.zeros(n_hr, dtype=np.float64)
    best_hr_71k = np.zeros(n_hr, dtype=np.float64)
    hr_3e9_100k = np.zeros(n_hr, dtype=np.float64)
    for ki, kv in enumerate(scan_ks):
        wk = np.exp(-kv * (gamma_all ** 2))
        _, hp100, _ = fast_polish_batch_blas(A_hr, B_hr, gamma_all, wk)
        _, hp71, _ = fast_polish_batch_blas(
            np.ascontiguousarray(A_hr[:, idx_old_71250]),
            np.ascontiguousarray(B_hr[:, idx_old_71250]),
            g_71_sub,
            wk[idx_old_71250],
        )
        if ki == 0:
            hr_3e9_100k = np.abs(hp100)
        imp = np.abs(hp100) > best_hr_100k
        best_hr_100k[imp] = np.abs(hp100[imp])
        best_hr_k100[imp] = kv
        imp71 = np.abs(hp71) > best_hr_71k
        best_hr_71k[imp71] = np.abs(hp71[imp71])

    order_hr = np.argsort(-best_hr_100k)
    log(
        f"  {'Rank':>4s} | {'y (approx)':>22s} | {'|h_3e-9(100k)|':>14s} | "
        f"{'max|h_k(71k)|':>13s} | {'max|h_k(100k)|':>14s} | {'opt k(100k)':>11s}"
    )
    log("  " + "-" * 90)
    for rank, idx in enumerate(order_hr[:15], 1):
        c = hr_clusters[idx]
        log(
            f"  {rank:4d} | {c['y_approx']:22.6f} | {hr_3e9_100k[idx]:14.9f} | "
            f"{best_hr_71k[idx]:13.9f} | {best_hr_100k[idx]:14.9f} | {best_hr_k100[idx]:11.3e}"
        )

    # Detailed Pintz / StR threshold table for Peak2 (y = 3.4850e18)
    log("\n  --- 3D. Exact (k, T) Threshold Required for Rigorous Disproof at Peak2 (y = 3.4850e18) ---")
    _, A_p2, B_p2 = primary_AB["Peak2 (3.4850e18)"]
    y_p2 = 3484971332733132396.5383310879
    log(
        f"  {'k':>10s} | {'|h_k(71250)|':>13s} | {'|h_k(100k)|':>13s} | {'Margin(>1)':>12s} | "
        f"{'T_min(emp)':>11s} | {'N_zeros(emp)':>13s} | {'T_min(Mert)':>11s} | {'N_zeros(Mert)':>13s}"
    )
    log("  " + "-" * 116)
    for kv in [2e-11, 5e-11, 1e-10, 1.5e-10, 2e-10, 2.2e-10, 2.5e-10, 3e-10, 5e-10, 1e-9, 2e-9, 3e-9]:
        wk = np.exp(-kv * (gamma_all ** 2))
        _, h71p, _ = polish_single_peak(A_p2[idx_old_71250], B_p2[idx_old_71250], gamma_all[idx_old_71250], wk[idx_old_71250], n_71250, grid_hw=4e-5, grid_steps=40)
        _, h100p, _ = polish_single_peak(A_p2, B_p2, gamma_all, wk, n_100k, grid_hw=4e-5, grid_steps=40)
        margin_actual = abs(h100p) - 1.0
        margin_ref = margin_actual if margin_actual > 0 else 5.0e-4
        t_req = solve_T_min_for_disproof(kv, margin_ref, y_p2)
        m_str = f"{margin_actual:+.6e}" if margin_actual > 0 else f"({margin_actual:+.2e})"
        log(
            f"  {kv:10.2e} | {abs(h71p):13.9f} | {abs(h100p):13.9f} | {m_str:>12s} | "
            f"{t_req['T_emp']:11.1f} | {t_req['N_emp']:13.0f} | {t_req['T_mertens']:11.1f} | {t_req['N_mertens']:13.0f}"
        )

    # =========================================================================
    # TASK 4: Bandlimited Positive-Definite Kernel Optimization (g = phi * ~phi)
    # =========================================================================
    log("\n" + "=" * 100)
    log("[TASK 4: BANDLIMITED POSITIVE-DEFINITE KERNEL OPTIMIZATION (g = phi * ~phi on [-T, T])]")
    log("=" * 100)
    log("  Note: Every kernel g = phi * ~phi with supp(phi) subset [-T/2, T/2] and ||phi||_2 = 1 satisfies:")
    log("    (1) g(0) = 1")
    log("    (2) supp(g) subset [-T, T] => Frequency tail above T is IDENTICALLY ZERO (eps_tail(T) = 0)!")
    log("    (3) Spatial kernel K(x) = |hat(phi)(x)|^2 >= 0 is strictly non-negative on R with int K(x) dx = 1.")

    T_100k = float(gamma_all[-1])  # 74920.82749899
    g_100k = gamma_all[:n_100k]

    std_kernels = []
    g_fejer = 1.0 - g_100k / T_100k
    std_kernels.append(("Fejer / Bartlett (1 - |t|/T)", g_fejer, float("inf")))

    u_jp = g_100k / T_100k
    g_jp = (1.0 - u_jp) * np.cos(math.pi * u_jp) + (1.0 / math.pi) * np.sin(math.pi * u_jp)
    sigma2_jp = (math.pi / T_100k) ** 2
    std_kernels.append(("Jurkat-Peyerimhoff g_JP (p=1)", g_jp, sigma2_jp))

    for p in [2, 3, 4, 6, 8, 12, 16, 24, 32]:
        g_cp, sig2_cp = numerical_autoconv_kernel(
            g_100k, T_100k, lambda u, p=p: np.cos(math.pi * u / T_100k) ** p
        )
        k_eff = p * (math.pi ** 2) / (4.0 * (T_100k ** 2))
        std_kernels.append((f"Cosine-Power p={p:<2d} (k_eff={k_eff:.2e})", g_cp, sig2_cp))

    for k_eff in [1e-10, 3e-10, 1e-9, 2e-9, 3e-9]:
        alpha_g = k_eff * (T_100k ** 2)
        g_gw, sig2_gw = numerical_autoconv_kernel(
            g_100k,
            T_100k,
            lambda u, a=alpha_g: np.exp(-a * ((u / T_100k) ** 2)) * np.cos(math.pi * u / T_100k),
        )
        std_kernels.append((f"Bandlim-Gauss k_eff={k_eff:.1e}", g_gw, sig2_gw))

    log(f"\n  --- 4A. Standard Compactly-Supported Positive-Definite Kernels on [-T, T] (T = {T_100k:.2f}, N = 100,000) ---")
    log(
        f"  {'Kernel Name':<34s} | {'sigma_x':>10s} | {'|h_g(Peak2, y*)|':>16s} | "
        f"{'|h_g(Peak1, y*)|':>16s} | {'|h_g(Peak3, y*)|':>16s} | {'Max Over 2549 Clusters':>22s}"
    )
    log("  " + "-" * 126)

    _, A_p1, B_p1 = primary_AB["Peak1 (1.3860e19)"]
    _, A_p2, B_p2 = primary_AB["Peak2 (3.4850e18)"]
    _, A_p3, B_p3 = primary_AB["Peak3 (8.0037e18)"]

    for kname, w_kern, sig2 in std_kernels:
        _, hp2, _ = polish_single_peak(A_p2, B_p2, g_100k, w_kern, n_100k, grid_hw=4e-5, grid_steps=40)
        _, hp1, _ = polish_single_peak(A_p1, B_p1, g_100k, w_kern, n_100k, grid_hw=4e-5, grid_steps=40)
        _, hp3, _ = polish_single_peak(A_p3, B_p3, g_100k, w_kern, n_100k, grid_hw=4e-5, grid_steps=40)
        _, hp_all, _ = fast_polish_batch_blas(A_cl, B_cl, g_100k, w_kern)
        max_cl = float(np.max(np.abs(hp_all)))
        sig_str = f"{math.sqrt(sig2):.3e}" if math.isfinite(sig2) else "inf"
        log(
            f"  {kname:<34s} | {sig_str:>10s} | {abs(hp2):16.10f} | "
            f"{abs(hp1):16.10f} | {abs(hp3):16.10f} | {max_cl:22.10f}"
        )

    log(f"\n  --- 4B. Smooth Basis Expansion Optimization on [-T/2, T/2] (T = {T_100k:.2f}, N = 100,000) ---")
    M_max = 16
    G_even, D_even, alphas_even = build_basis_G_tensor(g_100k, T_100k, M_max, basis_type="even")
    G_jp, D_jp, alphas_jp = build_basis_G_tensor(g_100k, T_100k, M_max, basis_type="jp")

    for target_name in ["Peak2 (3.4850e18)", "Peak1 (1.3860e19)"]:
        y_str, A_t, B_t = primary_AB[target_name]
        _, _, ddy_opt = polish_single_peak(A_t, B_t, g_100k, w_str, n_100k, grid_hw=4e-5, grid_steps=40)
        th_shift = g_100k * ddy_opt
        val_t = A_t * np.cos(th_shift) - B_t * np.sin(th_shift)

        H_even_full = np.tensordot(G_even, val_t, axes=([2], [0]))
        H_jp_full = np.tensordot(G_jp, val_t, axes=([2], [0]))

        log(f"\n  Target Peak: {target_name} (y = {y_str}, polished ddy = {ddy_opt:+.3e})")
        log(
            f"  {'M':>3s} | {'Even Basis (unconstr)':>21s} | {'Even Basis (phi(T/2)=0)':>23s} | "
            f"{'JP Basis (unconstr)':>19s} | {'JP Basis (var<=2*JP)':>20s} | {'JP Basis (var<=1.2*JP)':>22s}"
        )
        log("  " + "-" * 120)

        for M in [1, 2, 3, 4, 6, 8, 12, 16]:
            He = H_even_full[: M + 1, : M + 1]
            Hj = H_jp_full[: M + 1, : M + 1]
            Dj = D_jp[: M + 1]

            evals_e = np.linalg.eigvalsh(He)
            max_e_un = float(np.max(np.abs(evals_e)))

            b_vec = np.array([((-1) ** m) * alphas_even[m] for m in range(M + 1)], dtype=np.float64)
            b_vec /= np.linalg.norm(b_vec)
            P = np.eye(M + 1) - np.outer(b_vec, b_vec)
            u_svd, _, _ = np.linalg.svd(P)
            Q = u_svd[:, :M]
            He_proj = Q.T @ He @ Q
            evals_e_bc = np.linalg.eigvalsh(He_proj)
            max_e_bc = float(np.max(np.abs(evals_e_bc)))

            evals_j = np.linalg.eigvalsh(Hj)
            max_j_un = float(np.max(np.abs(evals_j)))

            def solve_var_constrained(H_mat, D_vec, kappa):
                target_var = kappa * D_vec[0]
                best_val = 0.0
                for sign in [1.0, -1.0]:
                    S_mat = sign * H_mat
                    ev, evec = np.linalg.eigh(S_mat)
                    c_top = evec[:, -1]
                    if float(np.dot(c_top * c_top, D_vec)) <= target_var:
                        best_val = max(best_val, float(ev[-1]))
                        continue
                    lo_mu, hi_mu = 0.0, 1e12
                    for _ in range(70):
                        mid_mu = 0.5 * (lo_mu + hi_mu)
                        ev_m, evec_m = np.linalg.eigh(S_mat - mid_mu * np.diag(D_vec))
                        c_m = evec_m[:, -1]
                        var_m = float(np.dot(c_m * c_m, D_vec))
                        if var_m > target_var:
                            lo_mu = mid_mu
                        else:
                            hi_mu = mid_mu
                    ev_f, evec_f = np.linalg.eigh(S_mat - hi_mu * np.diag(D_vec))
                    c_f = evec_f[:, -1]
                    val_f = float(c_f.T @ S_mat @ c_f)
                    best_val = max(best_val, val_f)
                return best_val

            max_j_v2 = solve_var_constrained(Hj, Dj, 2.0)
            max_j_v12 = solve_var_constrained(Hj, Dj, 1.2)

            log(
                f"  {M:3d} | {max_e_un:21.10f} | {max_e_bc:23.10f} | "
                f"{max_j_un:19.10f} | {max_j_v2:20.10f} | {max_j_v12:22.10f}"
            )

    log("\n  --- 4C. Joint (y, c) Optimization & Spatial Localization Analysis at Peak2 and Peak1 ---")
    for target_name in ["Peak2 (3.4850e18)", "Peak1 (1.3860e19)"]:
        y_str, A_t, B_t = primary_AB[target_name]
        log(f"\n  Joint (ddy, c) optimization for {target_name}:")
        for M in [1, 2, 3, 4]:
            ddys = np.linspace(-3.0e-5, 3.0e-5, 31)
            best_un = (0.0, 0.0, None)
            best_bc = (0.0, 0.0, None)
            b_vec = np.array([((-1) ** m) * alphas_even[m] for m in range(M + 1)], dtype=np.float64)
            b_vec /= np.linalg.norm(b_vec)
            P = np.eye(M + 1) - np.outer(b_vec, b_vec)
            u_svd, _, _ = np.linalg.svd(P)
            Q = u_svd[:, :M]

            for ddy in ddys:
                th = g_100k * ddy
                val_t = A_t * np.cos(th) - B_t * np.sin(th)
                He = np.tensordot(G_even[: M + 1, : M + 1, :], val_t, axes=([2], [0]))
                ev, evec = np.linalg.eigh(He)
                if abs(ev[0]) > abs(ev[-1]):
                    v_un, c_un = abs(ev[0]), evec[:, 0]
                else:
                    v_un, c_un = abs(ev[-1]), evec[:, -1]
                if v_un > best_un[0]:
                    best_un = (float(v_un), float(ddy), c_un)

                He_bc = Q.T @ He @ Q
                ev_b, evec_b = np.linalg.eigh(He_bc)
                if abs(ev_b[0]) > abs(ev_b[-1]):
                    v_bc, c_bc = abs(ev_b[0]), Q @ evec_b[:, 0]
                else:
                    v_bc, c_bc = abs(ev_b[-1]), Q @ evec_b[:, -1]
                if v_bc > best_bc[0]:
                    best_bc = (float(v_bc), float(ddy), c_bc)

            var_bc = float(np.dot(best_bc[2] ** 2, D_even[: M + 1]))
            c_un_str = ", ".join(f"{x:+.5f}" for x in best_un[2])
            c_bc_str = ", ".join(f"{x:+.5f}" for x in best_bc[2])
            log(
                f"    M={M}: Even-Unconstrained |h_g| = {best_un[0]:.10f} (ddy={best_un[1]:+.2e}, c=[{c_un_str}])\n"
                f"         Even-BoundaryZero  |h_g| = {best_bc[0]:.10f} (ddy={best_bc[1]:+.2e}, sigma_x={math.sqrt(var_bc):.3e}, c=[{c_bc_str}])"
            )

    log(f"\nCompleted full Path 1 analysis in {time.time()-t_start:.2f}s.")
    with open(REPORT_PATH, "w") as rf:
        rf.write("\n".join(out_lines) + "\n")
    print(f"Saved report to {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
