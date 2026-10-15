#!/usr/bin/env python3
"""
make_sub35_highrank.py
Builds 32 pairs (64 GSO files: gso/s35hr_00..31_{pos,neg}.gso) of Pure-Compact High-Rank
BKZ-60 lattices (M_act in {57, 58, 59, 60}, k_y_cut = 1) specifically localized to
the sub-Peak2 region y in (0, 3.4849e18) via sm in {20, 24, 28} and flat weight
exponents beta in [0.00, 0.05] (which prevent singleton ejection under high sm).
"""
import os
import math
import time
import multiprocessing as mp
import numpy as np
from fpylll import IntegerMatrix, LLL, BKZ, FPLLL
from make_pure_compact_ensemble import W_ORIG, G_ORIG, T1_ORIG, T2_ORIG, TWO_ASTAR, export_both_gso

FPLLL.set_precision(300)


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


def design_sub35_highrank_specs():
    rng = np.random.default_rng(20261016)
    specs = []

    # 1. Group A (00..07): 8 Pure Prefix High-Rank Lattices M in {57, 58, 59, 60}
    prefix_cfgs = [
        (0, "pref_M57_b04_sm24", list(range(57)), 0.04, 24),
        (1, "pref_M58_b00_sm24", list(range(58)), 0.00, 24),
        (2, "pref_M58_b04_sm24", list(range(58)), 0.04, 24),
        (3, "pref_M59_b00_sm24", list(range(59)), 0.00, 24),
        (4, "pref_M59_b04_sm24", list(range(59)), 0.04, 24),
        (5, "pref_M59_b04_sm32", list(range(59)), 0.04, 32),
        (6, "pref_M60_b00_sm24", list(range(60)), 0.00, 24),
        (7, "pref_M60_b03_sm20", list(range(60)), 0.03, 20),
    ]
    for idx, name, sel, beta, sm in prefix_cfgs:
        specs.append((idx, name, sel, beta, sm))

    # 2. Group B (08..23): 16 Greedy Multi-Band Rogue-Zero Covering Lattices over 0..67
    M_POOL = 68
    two_a = TWO_ASTAR[:M_POOL]
    w_orig = np.array([float(W_ORIG[j]) for j in range(M_POOL)])
    g_centered = np.array([
        float((G_ORIG[j] % W_ORIG[j]) - (W_ORIG[j] if (G_ORIG[j] % W_ORIG[j]) > W_ORIG[j] // 2 else 0))
        for j in range(M_POOL)
    ])

    def make_fast_eval(sel, beta=0.04, sm=24):
        sel_arr = np.array(sel, dtype=int)
        M_act = len(sel_arr)
        mults = compute_mults(sel_arr, beta, 1.00).astype(np.float64)
        y_unit = float((1 << 30) * 1024 * int(sm))
        r0 = g_centered[sel_arr] * mults
        r0_norm2 = float(np.sum(r0 * r0) + y_unit * y_unit)
        log_det_B = math.log(y_unit) + float(np.sum(np.log(w_orig[sel_arr] * mults)))
        log_det_proj = log_det_B - 0.5 * math.log(r0_norm2)
        gh_act = (
            math.exp(math.lgamma(M_act / 2.0 + 1.0) / M_act)
            / math.sqrt(math.pi)
            * math.exp(log_det_proj / M_act)
        )
        denom = 1.2935 * gh_act
        scale_v = (w_orig[sel_arr] * mults) / denom
        r0_unit = r0 / math.sqrt(r0_norm2)
        return sel_arr, scale_v, r0_unit

    def eval_lat_batch(lat, E):
        sel_arr, scale_v, r0_unit = lat
        V = E[:, sel_arr] * scale_v[None, :]
        proj = V @ r0_unit
        return np.sqrt(np.sum(V * V, axis=1) - proj * proj + 0.020)

    N = 40000
    sigma_j = 0.023 * ((two_a[0] / two_a) ** 0.34)
    E = rng.normal(0.0, sigma_j[None, :], size=(N, M_POOL))
    for row in range(N):
        h_idx = rng.choice(np.arange(9, 26), size=rng.choice([1, 2], p=[0.50, 0.50]), replace=False)
        E[row, h_idx] = rng.uniform(0.26, 0.47, size=len(h_idx)) * rng.choice([-1, 1], size=len(h_idx))
        l_idx = rng.choice(np.arange(26, M_POOL), size=rng.integers(5, 9), replace=False)
        E[row, l_idx] = rng.uniform(0.28, 0.48, size=len(l_idx)) * rng.choice([-1, 1], size=len(l_idx))
    E = (E + 0.5) % 1.0 - 0.5
    h68 = np.sum(two_a[None, :] * np.cos(2.0 * np.pi * E), axis=1)
    E_target = E[h68 >= 0.875][:3500]

    cur_min = np.minimum(
        eval_lat_batch(make_fast_eval(list(range(59)), 0.04, 24), E_target),
        eval_lat_batch(make_fast_eval(list(range(60)), 0.00, 24), E_target),
    )

    candidates = []
    for _ in range(600):
        M_pool_loc = int(rng.choice([66, 67, 68]))
        M_act_loc = int(rng.choice([58, 59, 60], p=[0.30, 0.50, 0.20]))
        n_hi = int(rng.choice([1, 2], p=[0.45, 0.55]))
        drop_hi = set(int(x) for x in rng.choice(np.arange(9, 26), size=n_hi, replace=False))
        n_lo = M_pool_loc - M_act_loc - n_hi
        drop_lo = set(int(x) for x in rng.choice(np.arange(26, M_pool_loc), size=n_lo, replace=False))
        sel = [j for j in range(M_pool_loc) if j not in (drop_hi | drop_lo)]
        beta = float(rng.choice([0.02, 0.04]))
        sm = int(rng.choice([20, 24, 28]))
        candidates.append((sorted(drop_hi), M_pool_loc, M_act_loc, beta, sm, sel, make_fast_eval(sel, beta, sm)))

    dist_mat = np.stack([eval_lat_batch(c[6], E_target) for c in candidates], axis=0)
    chosen = set()
    for step in range(8, 24):
        best_i = -1
        best_score = (-1, -1, -999.0)
        for i in range(len(candidates)):
            if i in chosen:
                continue
            nm = np.minimum(cur_min, dist_mat[i])
            cov1 = int(np.sum(nm <= 0.990))
            cov2 = int(np.sum(nm <= 1.010))
            mean_d = -float(np.mean(nm))
            if (cov1, cov2, mean_d) > best_score:
                best_score = (cov1, cov2, mean_d)
                best_i = i
        chosen.add(best_i)
        cur_min = np.minimum(cur_min, dist_mat[best_i])
        drop_hi, M_pool_loc, M_act_loc, beta, sm, sel, _ = candidates[best_i]
        hi_str = "_".join(str(x) for x in drop_hi)
        name = f"M{M_act_loc}_P{M_pool_loc}_hi{hi_str}_s{sm}"
        specs.append((step, name, sel, beta, sm))

    # 3. Group C (24..31): 8 Mid-Band Boosted 59-Zero Lattices (0..43 + 15 zeros in 44..88)
    prefix44 = list(range(44))
    mid_cfgs = [
        (24, "mb59_even44_72_s24", prefix44 + list(range(44, 74, 2)), 0.03, 24),
        (25, "mb59_odd45_73_s24", prefix44 + list(range(45, 75, 2)), 0.03, 24),
        (26, "mb59_str3A_s24", prefix44 + list(range(44, 89, 3)), 0.03, 24),
        (27, "mb59_str3B_s24", prefix44 + list(range(45, 90, 3)), 0.03, 24),
        (28, "mb59_str3C_s24", prefix44 + list(range(46, 91, 3)), 0.03, 24),
        (29, "mb59_top44_54_hi_s24", prefix44 + list(range(44, 54)) + [56, 59, 62, 65, 68], 0.03, 24),
        (30, "mb59_top44_52_hi_s24", prefix44 + list(range(44, 52)) + [54, 57, 60, 63, 66, 70, 74], 0.03, 24),
        (31, "mb59_top44_50_hi_s28", prefix44 + list(range(44, 50)) + [52, 55, 58, 61, 64, 67, 71, 75, 79], 0.02, 28),
    ]
    for idx, name, sel, beta, sm in mid_cfgs:
        specs.append((idx, name, sel, beta, sm))

    return specs


def build_one(args):
    idx, name, sel, beta, sm = args
    FPLLL.set_precision(300)
    t0 = time.time()
    M_act = len(sel)
    n = M_act + 1

    # Try beta down to 0.00 so that zero singletons are ejected (len(s_rows) == 1)
    for beta_try in [beta, min(beta, 0.02), 0.00]:
        mults = compute_mults(sel, beta_try, 1.00, ref_idx=min(sel[-1], 58))
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
            break

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
    else:
        s_row = s_rows[0]
        if s_row[M_act] < 0:
            s_row = [-x for x in s_row]
        final_rows = [s_row] + m_rows

    out_pos = f"s35hr_{idx:02d}_{name}_pos.gso"
    out_neg = f"s35hr_{idx:02d}_{name}_neg.gso"
    export_both_gso(final_rows, t1_new, t2_new, y_unit, out_pos, out_neg)
    y_steps = [abs(r[-1] // y_unit) / (1 << 30) for r in final_rows[1:]]
    print(
        f"[s35hr_{idx:02d}] {name:22s}: m_act={len(final_rows)-1:2d}, "
        f"min_Y={min(y_steps):.2e}, med_Y={np.median(y_steps):.2e} ({time.time()-t0:.2f}s)",
        flush=True,
    )
    return out_pos, out_neg


if __name__ == "__main__":
    t0 = time.time()
    specs = design_sub35_highrank_specs()
    with mp.Pool(min(32, len(specs))) as pool:
        pool.map(build_one, specs)
    print(f"All 64 Sub-3.485e18 High-Rank GSO files built in {time.time()-t0:.2f}s", flush=True)
