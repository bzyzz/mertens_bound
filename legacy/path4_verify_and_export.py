#!/usr/bin/env python3
"""
Task 1 & 2 Verification and Export for Path 4 (Discrete Phase-Offset Target Coset Enumeration).
"""
import math
import struct
import time
import numpy as np
import gmpy2
from fpylll import IntegerMatrix, LLL, BKZ, FPLLL
from make_pure_compact_ensemble import W_ORIG, G_ORIG, T1_ORIG, T2_ORIG, TWO_ASTAR, GAMMA
from make_highrank_ensemble import compute_mults

FPLLL.set_precision(300)
gmpy2.get_context().precision = 300

def get_phases_exact(x_int, dy, n_z=68, force_sign=0):
    """
    Computes exact phase residuals theta_j in (-pi, pi] and fractional residuals e_j = theta_j/(2*pi) in (-0.5, 0.5]
    using 130-bit G_ORIG, W_ORIG, T1_ORIG and GAMMA.
    """
    raw_angs = np.zeros(n_z, dtype=np.float64)
    for j in range(n_z):
        w = W_ORIG[j]
        rem = (x_int * G_ORIG[j] - T1_ORIG[j]) % w
        if rem > w // 2:
            rem -= w
        frac = float(gmpy2.mpfr(rem) / gmpy2.mpfr(w)) + (GAMMA[j] * dy) / (2.0 * math.pi)
        frac = (frac + 0.5) % 1.0 - 0.5
        raw_angs[j] = frac * 2.0 * math.pi
    h68 = float(np.sum(TWO_ASTAR[:n_z] * np.cos(raw_angs)))
    tsign = force_sign if force_sign != 0 else (1 if h68 >= 0 else -1)
    thetas = np.zeros(n_z, dtype=np.float64)
    for j in range(n_z):
        ang = raw_angs[j] - (math.pi if tsign < 0 else 0.0)
        ang = (ang + math.pi) % (2.0 * math.pi) - math.pi
        thetas[j] = ang
    return thetas, thetas / (2.0 * math.pi), tsign, h68

def get_phases_from_y_str(y_str, n_z=68):
    y_mp = gmpy2.mpfr(y_str)
    x_int = int(gmpy2.rint(y_mp * (1 << 30)))
    dy = float(y_mp - gmpy2.mpfr(x_int) / (1 << 30))
    return get_phases_exact(x_int, dy, n_z, 0) + (x_int, dy)

peaks_x = [
    ("Peak1_KN01", 14881567448228267707440294418, -0.0009095, -1, 0.99589577),
    ("Peak2",      3741959475396584484692060629,  +0.0004640, +1, 0.99570314),
    ("Peak3",      8593875597821265029902433064,  +0.0004940, -1, 0.99539831),
    ("Peak4",      12565606428742753754033740394, -0.0001515, -1, 0.99365549),
    ("Peak5",      5954553542112766845732472455,  -0.0002495, -1, 0.98914322),
    ("Peak6",      15114789910867931519573849160, +0.0002100, +1, 0.98893936),
    ("Peak7",      15079151625700385180499797097, -0.0005890, -1, 0.98801725),
]

with open("Test-ST.txt", "r") as f:
    kn_lines = [line.strip() for line in f if line.strip()]

print("=== STEP 1A: Continuous Lattice Distance Verification on hr_00 (sel=range(60), beta=0.16, floor_r=1.00, sm=4) ===")
sel = list(range(60))
beta = 0.16
floor_r = 1.00
sm = 4
mults = compute_mults(sel, beta, floor_r)
w_orig = np.array([float(W_ORIG[j]) for j in range(60)])
g_centered = np.array([
    float((G_ORIG[j] % W_ORIG[j]) - (W_ORIG[j] if (G_ORIG[j] % W_ORIG[j]) > W_ORIG[j] // 2 else 0))
    for j in range(60)
])
y_unit_flt = float((1 << 30) * 1024 * int(sm))
w_eff = w_orig * mults.astype(np.float64)
r0_vec = g_centered * mults.astype(np.float64)
r0_norm2 = float(np.sum(r0_vec * r0_vec) + y_unit_flt * y_unit_flt)
log_det_B = math.log(y_unit_flt) + float(np.sum(np.log(w_eff)))
log_det_proj = log_det_B - 0.5 * math.log(r0_norm2)
gh_act = math.exp(math.lgamma(60 / 2.0 + 1.0) / 60.0) / math.sqrt(math.pi) * math.exp(log_det_proj / 60.0)
denom = 1.2935 * gh_act
scale_v = w_eff / denom
r0_unit = r0_vec / math.sqrt(r0_norm2)

def eval_norm_dist(e_vec, y_val_e18, yc_e18=9.0):
    V = e_vec[:60] * scale_v
    proj = float(np.dot(V, r0_unit))
    dy_norm = (y_val_e18 - yc_e18) * 1e18 * float(1 << 30) * y_unit_flt / denom
    return math.sqrt(float(np.sum(V * V)) - proj * proj + dy_norm * dy_norm)

for name, x_int, dy, tsign, h_val in peaks_x:
    y_e18 = (x_int / float(1 << 30)) / 1e18
    yc_best = min([3.0, 9.0, 15.0], key=lambda c: abs(y_e18 - c))
    thetas, e_vec, _, h68 = get_phases_exact(x_int, dy, 68, tsign)
    d0 = eval_norm_dist(e_vec, y_e18, yc_best)
    rogues = [(j, thetas[j], e_vec[j]) for j in range(60) if abs(thetas[j]) >= 0.85]
    rogues.sort(key=lambda x: abs(x[1]), reverse=True)
    best_1 = (d0, -1, 0)
    for a in range(8, 55):
        for sgn in [-1, 1]:
            e_sh = e_vec.copy()
            e_sh[a] = ((e_sh[a] - sgn * 0.18 + 0.5) % 1.0) - 0.5
            d1 = eval_norm_dist(e_sh, y_e18, yc_best)
            if d1 < best_1[0]:
                best_1 = (d1, a, sgn)
    best_2 = (best_1[0], -1, 0, -1, 0)
    for a1 in range(8, 55):
        s1 = 1 if e_vec[a1] > 0 else -1
        for a2 in range(a1 + 1, 55):
            s2 = 1 if e_vec[a2] > 0 else -1
            e_sh = e_vec.copy()
            e_sh[a1] = ((e_sh[a1] - s1 * 0.18 + 0.5) % 1.0) - 0.5
            e_sh[a2] = ((e_sh[a2] - s2 * 0.18 + 0.5) % 1.0) - 0.5
            d2 = eval_norm_dist(e_sh, y_e18, yc_best)
            if d2 < best_2[0]:
                best_2 = (d2, a1, s1, a2, s2)
    rogue_str = ", ".join([f"j={j}:{th:+.3f}rad({ev:+.3f})" for j, th, ev in rogues[:5]])
    print(f"{name:10s} (y={y_e18:6.3f}e18, |h|={h_val:.6f}, h68={h68:+.5f}): d0={d0:.4f} -> d_1shift={best_1[0]:.4f} (j={best_1[1]},{best_1[2]:+d}) -> d_2shift={best_2[0]:.4f} (j={best_2[1]}{best_2[2]:+d},{best_2[3]}{best_2[4]:+d}) | rogues: {rogue_str}")

print("\n=== KN_00..KN_13 Phase Residuals and Coset Distance Reductions ===")
for idx, y_str in enumerate(kn_lines):
    thetas, e_vec, tsign, h68, x_int, dy = get_phases_from_y_str(y_str, 68)
    y_flt = float(y_str) / 1e18
    yc_ref = min([3.0, 9.0, 15.0], key=lambda c: abs(y_flt - c)) if y_flt < 20.0 else y_flt
    d0 = eval_norm_dist(e_vec, y_flt, yc_ref)
    rogues = [(j, thetas[j], e_vec[j]) for j in range(60) if abs(thetas[j]) >= 0.85]
    rogues.sort(key=lambda x: abs(x[1]), reverse=True)
    best_1 = (d0, -1, 0)
    for a in range(8, 55):
        for sgn in [-1, 1]:
            e_sh = e_vec.copy()
            e_sh[a] = ((e_sh[a] - sgn * 0.18 + 0.5) % 1.0) - 0.5
            d1 = eval_norm_dist(e_sh, y_flt, yc_ref)
            if d1 < best_1[0]:
                best_1 = (d1, a, sgn)
    best_2 = (best_1[0], -1, 0, -1, 0)
    for a1 in range(8, 55):
        s1 = 1 if e_vec[a1] > 0 else -1
        for a2 in range(a1 + 1, 55):
            s2 = 1 if e_vec[a2] > 0 else -1
            e_sh = e_vec.copy()
            e_sh[a1] = ((e_sh[a1] - s1 * 0.18 + 0.5) % 1.0) - 0.5
            e_sh[a2] = ((e_sh[a2] - s2 * 0.18 + 0.5) % 1.0) - 0.5
            d2 = eval_norm_dist(e_sh, y_flt, yc_ref)
            if d2 < best_2[0]:
                best_2 = (d2, a1, s1, a2, s2)
    rogue_str = ", ".join([f"j={j}:{th:+.3f}rad({ev:+.3f})" for j, th, ev in rogues[:5]])
    print(f"KN_{idx:02d} (y={y_flt:8.3f}e18, h68={h68:+.5f}): d0={d0:.4f} -> d_1shift={best_1[0]:.4f} (j={best_1[1]},{best_1[2]:+d}) -> d_2shift={best_2[0]:.4f} (j={best_2[1]}{best_2[2]:+d},{best_2[3]}{best_2[4]:+d}) | rogues: {rogue_str}")
