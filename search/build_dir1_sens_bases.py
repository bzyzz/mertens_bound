#!/usr/bin/env python3
"""
build_dir1_sens_bases.py
Builds 6 distinct sub-3.485e18 high-rank BKZ-60 bases (with sm in {20, 24}, beta in {0.00, 0.04})
and exports BOTH gso/dir1_baseXX_{pos,neg}.gso and gso/dir1_baseXX.sens from the exact same
BKZ-reduced basis rows `final_rows`.
"""
import os
import math
import struct
import time
import multiprocessing as mp
import numpy as np
import gmpy2
from fpylll import IntegerMatrix, LLL, BKZ, FPLLL
from make_pure_compact_ensemble import W_ORIG, G_ORIG, T1_ORIG, T2_ORIG
from make_wave2_gso import export_both_gso

FPLLL.set_precision(300)
gmpy2.get_context().precision = 300


def compute_mults(sel, beta=0.04, floor_r=1.00, ref_idx=None):
    if ref_idx is None:
        w_ref = float(W_ORIG[sel[-1]])
    else:
        w_ref = float(W_ORIG[ref_idx])
    mults = []
    for j in sel:
        r = max((float(W_ORIG[j]) / w_ref) ** (2.0 * beta), floor_r)
        m = int(round(r * w_ref * 1024.0 / float(W_ORIG[j])))
        mults.append(max(1, m))
    return np.array(mults, dtype=np.int64)


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


def get_base_specs():
    return [
        (0, "pref_M60_b00_sm24", list(range(60)), 0.00, 24),
        (1, "pref_M59_b04_sm24", list(range(59)), 0.04, 24),
        (2, "drop_14_21_33_45_M60_sm24", [j for j in range(64) if j not in {14, 21, 33, 45}], 0.04, 24),
        (3, "drop_11_17_29_41_M60_sm24", [j for j in range(64) if j not in {11, 17, 29, 41}], 0.04, 24),
        (4, "pref_M60_b04_sm20", list(range(60)), 0.04, 20),
        (5, "drop_12_19_35_44_M60_sm20", [j for j in range(64) if j not in {12, 19, 35, 44}], 0.00, 20),
    ]


def build_one_base(args):
    idx, name, sel, beta, sm = args
    FPLLL.set_precision(300)
    gmpy2.get_context().precision = 300
    t0 = time.time()
    M_act = len(sel)
    n = M_act + 1

    found_single = False
    for beta_try in [beta, min(beta, 0.02), 0.00]:
        for fl_try in [1.00, 1.04]:
            mults = compute_mults(sel, beta_try, fl_try, ref_idx=min(sel[-1], 58))
            B = IntegerMatrix(n, n)
            t1_new = [0] * n
            t2_new = [0] * n
            for a, j in enumerate(sel):
                mult = int(mults[a])
                B[0, a] = G_ORIG[j] * mult
                B[a + 1, a] = W_ORIG[j] * mult
                t1_new[a] = (T1_ORIG[j] % W_ORIG[j]) * mult
                t2_new[a] = (T2_ORIG[j] % W_ORIG[j]) * mult
            y_unit = (1 << 30) * 1024 * int(sm)
            B[0, M_act] = y_unit

            LLL.reduction(B)
            for b in [20, 30, 40, 50, min(60, n)]:
                param = BKZ.Param(block_size=b, max_loops=3, strategies=BKZ.DEFAULT_STRATEGY, auto_abort=True)
                BKZ.reduction(B, param)

            rows_b = [[int(B[i, k]) for k in range(n)] for i in range(n)]
            s_rows = [r for r in rows_b if abs(r[M_act] // y_unit) <= (1 << 24)]
            m_rows = [r for r in rows_b if abs(r[M_act] // y_unit) > (1 << 24)]
            if len(s_rows) == 1:
                found_single = True
                break
        if found_single:
            break

    w_eff_full = [W_ORIG[sel[a]] * int(mults[a]) for a in range(M_act)]

    if len(s_rows) > 1:
        s_rows.sort(key=lambda r: abs(r[M_act]))
        r0 = s_rows[0]
        if r0[M_act] < 0:
            r0 = [-x for x in r0]
        ej_cols = set()
        for r in s_rows[1:]:
            xk = r[M_act] // r0[M_act]
            for col in range(M_act):
                if r[col] - xk * r0[col] != 0:
                    ej_cols.add(col)
        keep_cols = [col for col in range(n) if col not in ej_cols]
        final_rows = [[r[col] for col in keep_cols] for r in ([r0] + m_rows)]
        t1_new = [t1_new[col] for col in keep_cols]
        t2_new = [t2_new[col] for col in keep_cols]
        w_eff = [w_eff_full[col] for col in keep_cols[:-1]]
    else:
        s_row = s_rows[0]
        if s_row[M_act] < 0:
            s_row = [-x for x in s_row]
        final_rows = [s_row] + m_rows
        w_eff = w_eff_full

    os.makedirs("gso", exist_ok=True)
    out_pos = f"gso/dir1_base{idx:02d}_pos.gso"
    out_neg = f"gso/dir1_base{idx:02d}_neg.gso"
    sens_path = f"gso/dir1_base{idx:02d}.sens"

    export_both_gso(final_rows, t1_new, t2_new, y_unit, out_pos, out_neg)
    mu, c, bstar, S = compute_exact_gso_and_sens(final_rows, w_eff)

    M_final = len(w_eff)
    n_final = len(final_rows)
    with open(sens_path, "wb") as f:
        f.write(struct.pack("<ii", M_final, n_final))
        f.write(S.tobytes())

    c_flt = np.array([float(x) for x in c], dtype=np.float64)
    max_parseval_err = 0.0
    for a in range(M_final):
        lhs = float(np.sum(c_flt[1:] * (S[a, 1:] ** 2)))
        rhs = float(gmpy2.mpfr(w_eff[a]) ** 2 - (gmpy2.mpfr(w_eff[a]) * bstar[0][a]) ** 2 / c[0])
        max_parseval_err = max(max_parseval_err, abs(lhs - rhs) / rhs)

    y_steps = [abs(r[-1] // y_unit) / (1 << 30) for r in final_rows[1:]]
    print(
        f"[dir1_base{idx:02d}] {name:26s}: M={M_final:2d}, n={n_final:2d}, "
        f"med_Y={np.median(y_steps):.2e}, parseval_err={max_parseval_err:.2e} ({time.time()-t0:.2f}s)",
        flush=True,
    )
    return sens_path, out_pos, out_neg


if __name__ == "__main__":
    t0 = time.time()
    specs = get_base_specs()
    with mp.Pool(len(specs)) as pool:
        pool.map(build_one_base, specs)
    print(f"All {len(specs)} dir1 bases (.gso + .sens) built in {time.time()-t0:.2f}s", flush=True)
