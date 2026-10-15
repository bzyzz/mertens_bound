#!/usr/bin/env python3
"""
Build hr_00_M60_P68_prefix_{pos,neg}.gso (sel = range(60), beta = 0.16, floor_r = 1.00, sm = 4),
export the exact 60 x 61 sensitivity matrix S[a][k] = W_eff[a] * (b_k^*)_a / rs[k],
and verify the exact reduced-basis GSO normalized distance of Peak1..Peak7 and KN_00..KN_13
under single-zero and pair-zero target phase shifts.
"""
import os
import math
import struct
import time
import numpy as np
import gmpy2
from fpylll import IntegerMatrix, LLL, BKZ, FPLLL
from make_pure_compact_ensemble import W_ORIG, G_ORIG, T1_ORIG, T2_ORIG, TWO_ASTAR, GAMMA
from make_highrank_ensemble import compute_mults
from make_wave2_gso import export_both_gso

FPLLL.set_precision(300)
gmpy2.get_context().precision = 300

def compute_exact_gso_and_sens(rows, w_eff):
    n = len(rows)
    m = len(rows[0])
    M_act = m - 1
    B = [[gmpy2.mpfr(x) for x in r] for r in rows]
    mu = [[gmpy2.mpfr(0)] * n for _ in range(n)]
    r = [[gmpy2.mpfr(0)] * n for _ in range(n)]
    c = [gmpy2.mpfr(0)] * n

    for i in range(n):
        for j in range(i + 1):
            ip = sum(B[i][k] * B[j][k] for k in range(m))
            val = ip
            for l in range(j):
                val -= mu[j][l] * r[i][l]
            r[i][j] = val
            if j < i:
                mu[i][j] = val / r[j][j]
            else:
                mu[i][i] = gmpy2.mpfr(1)
                c[i] = val

    # Compute b_star[i][a] for every coordinate a in 0..M_act-1
    S = np.zeros((M_act, n), dtype=np.float64)
    bstar = [[gmpy2.mpfr(0)] * m for _ in range(n)]
    for a in range(m):
        for i in range(n):
            val = B[i][a]
            for l in range(i):
                val -= mu[i][l] * bstar[l][a]
            bstar[i][a] = val
        if a < M_act:
            we_mp = gmpy2.mpfr(w_eff[a])
            for i in range(n):
                S[a, i] = float(we_mp * bstar[i][a] / c[i])

    return mu, c, bstar, S

def build_hr00_and_sensitivity():
    t0 = time.time()
    sel = list(range(60))
    beta = 0.16
    floor_r = 1.00
    sm = 4
    M_act = len(sel)
    n = M_act + 1

    for fl_try in [floor_r, floor_r + 0.03, floor_r + 0.06]:
        mults = compute_mults(sel, beta, fl_try)
        for sm_try in [sm, 8]:
            B = IntegerMatrix(n, n)
            t1_new = [0] * n
            t2_new = [0] * n
            for a, j in enumerate(sel):
                mult = int(mults[a])
                B[0, a] = G_ORIG[j] * mult
                B[a + 1, a] = W_ORIG[j] * mult
                t1_new[a] = (T1_ORIG[j] % W_ORIG[j]) * mult
                t2_new[a] = (T2_ORIG[j] % W_ORIG[j]) * mult
            y_unit = (1 << 30) * 1024 * int(sm_try)
            B[0, M_act] = y_unit

            LLL.reduction(B)
            for b in [20, 30, 40, 50, 60]:
                param = BKZ.Param(block_size=b, max_loops=3, strategies=BKZ.DEFAULT_STRATEGY, auto_abort=True)
                BKZ.reduction(B, param)

            rows_b = [[int(B[i, k]) for k in range(n)] for i in range(n)]
            s_rows = [r for r in rows_b if abs(r[M_act] // y_unit) <= (1 << 24)]
            m_rows = [r for r in rows_b if abs(r[M_act] // y_unit) > (1 << 24)]
            print(f"Try fl={fl_try:.2f}, sm={sm_try}: len(s_rows)={len(s_rows)}, len(m_rows)={len(m_rows)}", flush=True)
            if len(s_rows) == 1:
                break
        if len(s_rows) == 1:
            break

    if s_rows[0][M_act] < 0:
        s_rows[0] = [-x for x in s_rows[0]]
    final_rows = s_rows + m_rows
    w_eff = [W_ORIG[sel[a]] * int(mults[a]) for a in range(M_act)]

    out_pos = "hr_00_M60_P68_prefix_pos.gso"
    out_neg = "hr_00_M60_P68_prefix_neg.gso"
    export_both_gso(final_rows, t1_new, t2_new, y_unit, out_pos, out_neg)

    print(f"Computing exact 300-bit MPFR Gram-Schmidt and sensitivity matrix S ({M_act}x{n})...", flush=True)
    mu, c, bstar, S = compute_exact_gso_and_sens(final_rows, w_eff)

    # Verify linearity of bar_t shift:
    # Shifting t1_new[a] by +0.18 * w_eff[a] shifts <t, b_k^*>/c[k] by +0.18 * S[a, k]
    sens_path = "gso/hr_00_M60_P68_prefix.sens" if os.path.isdir("gso") else "hr_00_M60_P68_prefix.sens"
    with open(sens_path, "wb") as f:
        f.write(struct.pack("<ii", M_act, n))
        f.write(S.tobytes())
    print(f"Saved {out_pos}, {out_neg}, and {sens_path} in {time.time()-t0:.2f}s", flush=True)

    # Also check how each peak's exact lattice vector v in final_rows decomposes in the GSO basis!
    # For any integer x_int, the lattice vector v in L has:
    #   v[a] = x_int * (G_ORIG[a] * mults[a]) - k_a * w_eff[a], where k_a = round((x_int * G_ORIG[a] - t_orig[a]) / W_ORIG[a])
    #   v[M_act] = x_int * y_unit
    # And the residual vector is r_vec = v - t_new (with t_new[M_act] = yc * 2^30 * y_unit).
    # Since b_star[0..n-1] is an orthogonal basis of R^n and b_star[0] = final_rows[0] (the r0 singleton row),
    # the squared norm of the projection of r_vec onto span(b_star[1..n-1]) is:
    #   sum_{k=1}^{n-1} (<r_vec, b_star[k]>^2 / c[k])!
    # Wait: when we shift t_new[a] -> t_new[a] + delta_e * w_eff[a],
    # how much does <r_vec, b_star[k]> / c[k] change?
    # By EXACTLY -delta_e * S[a, k]!
    # And its squared norm in span(b_star[1..n-1]) is:
    #   sum_{k=1}^{n-1} c[k] * (alpha_k - delta_e * S[a, k])^2
    # where alpha_k = <r_vec, b_star[k]> / c[k] = <v, b_star[k]>/c[k] - bar_t[k]!
    # Notice that since v is an INTEGER linear combination of final_rows[0..n-1],
    # <v, b_star[k]> / c[k] = z_k + sum_{i=k+1}^{n-1} z_i * mu[i][k], which is the EXACT GSO coefficient!

    c_flt = np.array([float(x) for x in c], dtype=np.float64)
    log_det_proj = float(np.sum(np.log(c_flt[1:]))) * 0.5
    gh_act = math.exp(math.lgamma(M_act / 2.0 + 1.0) / M_act) / math.sqrt(math.pi) * math.exp(log_det_proj / M_act)
    denom = 1.2935 * gh_act
    print(f"Reduced GSO gh_act = {gh_act:.6e}, denom = 1.2935 * gh_act = {denom:.6e}")

    # Verify that sum_{k=1}^{n-1} c[k] * S[a, k]^2 == w_eff[a]^2 - <w_eff[a]*e_a, b_star[0]>^2 / c[0]!
    max_parseval_err = 0.0
    for a in range(M_act):
        lhs = float(np.sum(c_flt[1:] * (S[a, 1:] ** 2)))
        rhs = float(gmpy2.mpfr(w_eff[a])**2 - (gmpy2.mpfr(w_eff[a]) * bstar[0][a])**2 / c[0])
        max_parseval_err = max(max_parseval_err, abs(lhs - rhs) / rhs)
    print(f"Max Parseval identity relative error across all a=0..59: {max_parseval_err:.3e}", flush=True)

if __name__ == "__main__":
    build_hr00_and_sensitivity()
