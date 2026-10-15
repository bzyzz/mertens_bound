#!/usr/bin/env python3
"""
Generate 12 Pairs (24 GSO files) of 60-Zero Mid-Band Boosted BKZ-60 Lattices
(mb60_00..mb60_11, M_act = 60, k_y_cut = 1):
- Keeps 0..44 (45 zeros) fixed in every basis.
- Samples 15 zeros from the wider mid-band pool 45..95 (51 zeros, weighted by alpha_j^*)
  using greedy covering over simulated mid-band-boosted peaks.
- Uses w_ref = float(W_ORIG[44]) and floor_r = 1.00 so that k_y_cut = 1 (all 60 zeros
  active in the macro lattice).
"""
import math
import multiprocessing as mp
import os
import time
from decimal import Decimal, getcontext
import numpy as np
from fpylll import IntegerMatrix, LLL, BKZ, FPLLL
from make_pure_compact_ensemble import W_ORIG, G_ORIG, T1_ORIG, T2_ORIG, TWO_ASTAR, export_both_gso

FPLLL.set_precision(300)
getcontext().prec = 60


def compute_mults(sel, beta=0.12, floor_r=1.00, ref_idx=44):
    w_ref = float(W_ORIG[ref_idx])
    mults = []
    for j in sel:
        r = max((float(W_ORIG[j]) / w_ref) ** (2.0 * beta), floor_r)
        m = int(round(r * w_ref * 1024.0 / float(W_ORIG[j])))
        mults.append(max(1, m))
    return np.array(mults, dtype=np.int64)


def design_midband60_specs():
    rng = np.random.default_rng(20261015)
    M_POOL = 96
    two_a = TWO_ASTAR[:M_POOL]
    w_orig = np.array([float(W_ORIG[j]) for j in range(M_POOL)])
    g_centered = np.array([
        float((G_ORIG[j] % W_ORIG[j]) - (W_ORIG[j] if (G_ORIG[j] % W_ORIG[j]) > W_ORIG[j] // 2 else 0))
        for j in range(M_POOL)
    ])

    def make_fast_eval(sel, beta=0.12, floor_r=1.00, sm=4):
        sel_arr = np.array(sel, dtype=int)
        M_act = len(sel_arr)
        mults = compute_mults(sel_arr, beta, floor_r, ref_idx=44).astype(np.float64)
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
        return np.sqrt(np.sum(V * V, axis=1) - proj * proj + 0.025)

    # Simulate N = 50,000 multi-band peaks with strong mid-band 45..95 alignment
    N = 50000
    sigma_j = 0.024 * ((two_a[0] / two_a) ** 0.30)
    E = rng.normal(0.0, sigma_j[None, :], size=(N, M_POOL))
    mid_pool = np.arange(45, M_POOL)
    w_mid = two_a[45:M_POOL]
    p_mid_inv = (1.0 / w_mid) / np.sum(1.0 / w_mid)
    for row in range(N):
        # 2 to 4 anti-aligned zeros in 15..44 (causing h_0..44 to be 0.72..0.81)
        h_idx = rng.choice(np.arange(15, 45), size=rng.integers(2, 5), replace=False)
        E[row, h_idx] = rng.uniform(0.24, 0.45, size=len(h_idx)) * rng.choice([-1, 1], size=len(h_idx))
        # 16 to 22 unaligned zeros in 45..95 (so 29 to 35 of 51 zeros in 45..95 are aligned, matching Test-ST.txt)
        l_idx = rng.choice(mid_pool, size=rng.integers(16, 23), replace=False, p=p_mid_inv)
        E[row, l_idx] = rng.uniform(0.26, 0.48, size=len(l_idx)) * rng.choice([-1, 1], size=len(l_idx))
    E = (E + 0.5) % 1.0 - 0.5
    h96 = np.sum(two_a[None, :] * np.cos(2.0 * np.pi * E), axis=1)
    h45 = np.sum(two_a[None, :45] * np.cos(2.0 * np.pi * E[:, :45]), axis=1)
    h_mid = h96 - h45
    mask = (h96 >= 0.885) & (h_mid >= 0.115)
    E_target = E[mask][:4000]
    print(
        f"Mid-band target set: {len(E_target)} peaks with h_0..95 >= 0.885 and h_45..95 >= 0.115 "
        f"(mean h_0..44={np.mean(h45[mask]):.4f}, mean h_45..95={np.mean(h_mid[mask]):.4f})",
        flush=True,
    )

    prefix45 = list(range(45))
    candidates = []

    # Structured mid-band 15-zero subsets in 45..95
    structured_mid15 = [
        ("top45_59", list(range(45, 60)), 0.12),
        ("even45_73", list(range(45, 75, 2)), 0.12),
        ("odd46_74", list(range(46, 76, 2)), 0.12),
        ("str3_45_87", list(range(45, 90, 3)), 0.12),
        ("str3_46_88", list(range(46, 91, 3)), 0.12),
        ("str3_47_89", list(range(47, 92, 3)), 0.12),
    ]
    for tag, mid15, beta in structured_mid15:
        sel = prefix45 + mid15
        candidates.append((tag, mid15, beta, sel, make_fast_eval(sel, beta, 1.00, 4)))

    # 800 alpha_j^*-weighted random subsets of 15 zeros from 45..95
    for idx in range(800):
        gamma = float(rng.choice([0.75, 1.0, 1.25, 1.5]))
        p_mid = (w_mid ** gamma) / np.sum(w_mid ** gamma)
        mid15 = sorted(int(x) for x in rng.choice(mid_pool, size=15, replace=False, p=p_mid))
        beta = float(rng.choice([0.11, 0.12, 0.13]))
        sel = prefix45 + mid15
        tag = f"m{mid15[0]}_{mid15[7]}_{mid15[-1]}_b{int(round(beta*100))}"
        candidates.append((tag, mid15, beta, sel, make_fast_eval(sel, beta, 1.00, 4)))

    dist_mat = np.stack([eval_lat_batch(c[4], E_target) for c in candidates], axis=0)
    cur_min = np.full(len(E_target), 999.0)
    chosen = set()
    specs = []

    for step in range(12):
        best_i = -1
        best_score = (-1, -1, -999.0)
        for i in range(len(candidates)):
            if i in chosen:
                continue
            nm = np.minimum(cur_min, dist_mat[i])
            cov1 = int(np.sum(nm <= 1.005))
            cov2 = int(np.sum(nm <= 1.025))
            mean_d = -float(np.mean(nm))
            if (cov1, cov2, mean_d) > best_score:
                best_score = (cov1, cov2, mean_d)
                best_i = i
        chosen.add(best_i)
        cur_min = np.minimum(cur_min, dist_mat[best_i])
        tag, mid15, beta, sel, _ = candidates[best_i]
        specs.append((step, tag, sel, beta, 1.00, 4, mid15))
        print(
            f"Step {step:2d} [mb60_{step:02d}_{tag:18s}]: cov<=1.005={100*np.mean(cur_min<=1.005):5.1f}% "
            f"| cov<=1.025={100*np.mean(cur_min<=1.025):5.1f}% | mid15={mid15}",
            flush=True,
        )

    return specs


def build_one(args):
    idx, name, sel, beta, floor_r, sm, mid15 = args
    prefix_dir = "gso/" if os.path.isdir("gso") else ""
    out_pos = f"{prefix_dir}mb60_{idx:02d}_{name}_pos.gso"
    out_neg = f"{prefix_dir}mb60_{idx:02d}_{name}_neg.gso"
    if os.path.exists(out_pos) and os.path.exists(out_neg):
        import struct as _st
        with open(out_pos, "rb") as _f:
            _n = _st.unpack("<i", _f.read(4))[0]
        if _n == 61:
            print(f"[mb60_{idx:02d}] {name:22s}: already built with m_act=60 (k_y_cut=1)", flush=True)
            return out_pos, out_neg
    FPLLL.set_precision(300)
    t0 = time.time()
    M_act = len(sel)
    n = M_act + 1

    # Try (floor_r, beta, sm) with ref_idx=44 so all 60 zeros stay active (len(s_rows) == 1, k_y_cut = 1)
    for fl_try, b_try, sm_try in [
        (floor_r, beta, sm),
        (1.00, 0.12, 4),
        (1.00, 0.14, 4),
        (1.03, 0.14, 4),
        (1.06, 0.15, 4),
        (1.10, 0.16, 4),
        (1.03, 0.12, 8),
        (1.06, 0.14, 8),
    ]:
        mults = compute_mults(sel, b_try, fl_try, ref_idx=44)
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
        for b in [20, 30, 40, 50, min(60, n)]:
            param = BKZ.Param(block_size=b, max_loops=3, strategies=BKZ.DEFAULT_STRATEGY, auto_abort=True)
            BKZ.reduction(B, param)

        rows_b = [[int(B[i, k]) for k in range(n)] for i in range(n)]
        s_rows = [r for r in rows_b if abs(r[M_act] // y_unit) <= (1 << 24)]
        m_rows = [r for r in rows_b if abs(r[M_act] // y_unit) > (1 << 24)]
        if len(s_rows) == 1:
            break

    if len(s_rows) > 1:
        next_j = next(j for j in range(45, 96) if j not in sel)
        new_sel = sorted(sel[:-1] + [next_j])
        new_mid15 = new_sel[45:]
        return build_one((idx, name, new_sel, 0.13, 1.00, 4, new_mid15))

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

    out_pos = f"mb60_{idx:02d}_{name}_pos.gso"
    out_neg = f"mb60_{idx:02d}_{name}_neg.gso"
    export_both_gso(final_rows, t1_new, t2_new, y_unit, out_pos, out_neg)
    print(
        f"[mb60_{idx:02d}] {name:22s}: m_act={len(final_rows)-1:2d} (s_rows={len(s_rows)}) "
        f"fl={fl_try:.2f} b={b_try:.2f} sm={sm_try} mid15={mid15[0]}..{mid15[-1]} ({time.time()-t0:.2f}s)",
        flush=True,
    )
    return out_pos, out_neg


if __name__ == "__main__":
    t0 = time.time()
    specs = design_midband60_specs()
    with mp.Pool(min(12, len(specs))) as pool:
        pool.map(build_one, specs)
    print(f"All 24 Mid-Band 60-Zero GSO files (mb60_00..mb60_11) built in {time.time()-t0:.2f}s", flush=True)
