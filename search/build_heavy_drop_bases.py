#!/usr/bin/env python3
"""
build_heavy_drop_bases.py
Technique 2: Targeted Heavy-Zero Drop (j in 4..20) BKZ-60 Lattice Ensemble Search
Builds 48 BKZ-60 bases (gso/hdrop_00..47_pos.gso and gso/hdrop_00..47_neg.gso, total 96 .gso files)
with M in {58, 59, 60} out of P = 68 and K_scale = 3.485e18:
  - Keep j in {0, 1, 2, 3} always active.
  - Group 1 (hdrop_00..17, 18 bases, M=60): Drop 1 heavy zero j1 in {4..21},
    1 medium-heavy zero j2 in {12..28} \ {j1}, and 6 mid-band zeros from {25..67}.
  - Group 2 (hdrop_18..35, 18 bases, M=59): Drop 2 heavy zeros (j1, j2) from {4..18}
    (covering the 18 key pairs) plus 7 mid-band zeros from {22..67}.
  - Group 3 (hdrop_36..47, 12 bases, M=58): Drop 2 zeros from {5..20} and 2 zeros from {21..35}
    plus 6 zeros from {36..67}, using varied exponent beta in {0.50, 0.54, 0.58} in W_j ~ (alpha_j*)^beta.
"""
import os
import math
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import gmpy2
from fpylll import IntegerMatrix, LLL, BKZ, FPLLL
from make_pure_compact_ensemble import W_ORIG, G_ORIG, T1_ORIG, T2_ORIG, TWO_ASTAR
from make_wave2_gso import export_both_gso

FPLLL.set_precision(300)
gmpy2.get_context().precision = 300


def compute_mults_hdrop(sel, beta_exp=0.50, ref_idx=18, tail_beta=0.04, floor_r=1.00):
    """
    Computes integer column multipliers for active zero indices `sel`.
    Since W_ORIG[j] ~ (alpha_j*)^0.50:
      - For heavy zeros j <= ref_idx, W_eff[j] = W_ORIG[j] * mult[j] ~ (alpha_j*)^beta_exp
      - For mid/tail zeros j > ref_idx, we use a gentle slope `tail_beta` and floor `floor_r`
        relative to W_ORIG[min(sel[-1], 58)] so that BKZ-60 never ejects tail singletons
        (preserving k_y_cut == 1 and full macro rank M_act).
    """
    w_tail_ref = float(W_ORIG[min(sel[-1], 58)])
    w_heavy_ref = float(W_ORIG[ref_idx])
    r_heavy_at_ref = max((w_heavy_ref / w_tail_ref) ** (2.0 * tail_beta), floor_r)

    mults = []
    for j in sel:
        wj = float(W_ORIG[j])
        if j <= ref_idx:
            # W_eff[j] proportional to (alpha_j*)^beta_exp = (W_ORIG[j])^(2 * beta_exp)
            r = r_heavy_at_ref * ((wj / w_heavy_ref) ** (2.0 * beta_exp))
        else:
            r = max((wj / w_tail_ref) ** (2.0 * tail_beta), floor_r)
        m = int(round(r * w_tail_ref * 1024.0 / wj))
        mults.append(max(1, m))
    return np.array(mults, dtype=np.int64)


def design_hdrop_specs():
    rng = np.random.default_rng(2026101502)
    P = 68
    specs = []

    # -------------------------------------------------------------------------
    # Group 1 (hdrop_00..17, 18 bases, M=60):
    # Drop 1 heavy zero j1 in {4..21}, 1 medium-heavy zero j2 in {12..28}\{j1},
    # and 6 mid-band zeros from {25..67}\{j1, j2}.
    # -------------------------------------------------------------------------
    j2_Cycle = [19, 22, 24, 18, 21, 25, 20, 23, 26, 27, 28, 12, 13, 14, 15, 16, 17, 24]
    for i in range(18):
        j1 = 4 + i  # 4..21
        j2 = j2_Cycle[i]
        if j2 == j1:
            j2 = 28 if j1 != 28 else 27
        # Pick 6 well-spread mid-band zeros from {25..67} \ {j1, j2}
        mid_pool = [j for j in range(25, P) if j not in (j1, j2)]
        # Deterministic stratified selection + slight rotation across i
        idxs = [(i * 5 + k * 7) % len(mid_pool) for k in range(6)]
        drop_mid = set()
        for ix in idxs:
            while mid_pool[ix] in drop_mid:
                ix = (ix + 1) % len(mid_pool)
            drop_mid.add(mid_pool[ix])
        drops = sorted({j1, j2} | drop_mid)
        assert len(drops) == 8
        sel = [j for j in range(P) if j not in drops]
        assert len(sel) == 60
        assert all(j in sel for j in (0, 1, 2, 3))
        beta = [0.50, 0.52, 0.54][i % 3]
        sm = [20, 24, 22][i % 3]
        desc = f"G1_M60_j{j1}_{j2}_b{int(round(beta*100))}"
        specs.append((i, desc, sel, drops, beta, sm))

    # -------------------------------------------------------------------------
    # Group 2 (hdrop_18..35, 18 bases, M=59):
    # Systematically drop 2 heavy zeros (j1, j2) from {4..18} plus 7 mid-band zeros from {22..67}.
    # -------------------------------------------------------------------------
    g2_pairs = [
        (4, 9), (5, 10), (6, 11), (7, 12), (8, 13), (5, 8),
        (6, 9), (7, 10), (8, 14), (9, 15), (10, 16), (11, 17),
        (12, 18), (4, 7), (6, 12), (7, 15), (9, 14), (13, 17),
    ]
    assert len(g2_pairs) == 18
    for k_idx, (j1, j2) in enumerate(g2_pairs):
        idx = 18 + k_idx
        mid_pool = [j for j in range(22, P) if j not in (j1, j2)]
        idxs = [(k_idx * 5 + m * 6 + (m // 2)) % len(mid_pool) for m in range(7)]
        drop_mid = set()
        for ix in idxs:
            while mid_pool[ix] in drop_mid:
                ix = (ix + 1) % len(mid_pool)
            drop_mid.add(mid_pool[ix])
        drops = sorted({j1, j2} | drop_mid)
        assert len(drops) == 9
        sel = [j for j in range(P) if j not in drops]
        assert len(sel) == 59
        assert all(j in sel for j in (0, 1, 2, 3))
        beta = [0.50, 0.54, 0.52][k_idx % 3]
        sm = [20, 24, 24][k_idx % 3]
        desc = f"G2_M59_j{j1}_{j2}_b{int(round(beta*100))}"
        specs.append((idx, desc, sel, drops, beta, sm))

    # -------------------------------------------------------------------------
    # Group 3 (hdrop_36..47, 12 bases, M=58):
    # Drop 2 zeros from {5..20} and 2 zeros from {21..35} plus 6 zeros from {36..67},
    # using varied exponent beta in {0.50, 0.54, 0.58} in W_j ~ (alpha_j*)^beta.
    # -------------------------------------------------------------------------
    g3_heavy_pairs = [
        (5, 11), (6, 13), (7, 14), (8, 15), (9, 16), (10, 17),
        (11, 18), (12, 19), (13, 20), (5, 16), (7, 18), (9, 20),
    ]
    g3_med_pairs = [
        (21, 28), (22, 29), (23, 30), (24, 31), (25, 32), (26, 33),
        (27, 34), (21, 35), (22, 33), (23, 34), (24, 35), (26, 35),
    ]
    betas_g3 = [0.50, 0.54, 0.58]
    for k_idx in range(12):
        idx = 36 + k_idx
        h1, h2 = g3_heavy_pairs[k_idx]
        m1, m2 = g3_med_pairs[k_idx]
        beta = betas_g3[k_idx % 3]
        mid_pool = list(range(36, P))
        idxs = [(k_idx * 3 + m * 5) % len(mid_pool) for m in range(6)]
        drop_mid = set()
        for ix in idxs:
            while mid_pool[ix] in drop_mid:
                ix = (ix + 1) % len(mid_pool)
            drop_mid.add(mid_pool[ix])
        drops = sorted({h1, h2, m1, m2} | drop_mid)
        assert len(drops) == 10
        sel = [j for j in range(P) if j not in drops]
        assert len(sel) == 58
        assert all(j in sel for j in (0, 1, 2, 3))
        sm = [20, 22, 24][k_idx % 3]
        desc = f"G3_M58_j{h1}_{h2}_{m1}_{m2}_b{int(round(beta*100))}"
        specs.append((idx, desc, sel, drops, beta, sm))

    return specs


def build_one(args):
    idx, desc, sel, drops, beta, sm = args
    FPLLL.set_precision(300)
    gmpy2.get_context().precision = 300
    t0 = time.time()
    M_act = len(sel)
    n = M_act + 1

    # Try progressive (ref_idx, tail_beta) settings so that zero singletons are ejected
    # and the reduced basis has k_y_cut == 1 with all M_act zeros active.
    try_configs = [
        (16, 0.04, sm),
        (12, 0.03, sm),
        (8, 0.02, sm),
        (4, 0.00, sm),
        (0, 0.00, 20),
    ]
    for ref_idx, tail_beta, sm_try in try_configs:
        mults = compute_mults_hdrop(sel, beta_exp=beta, ref_idx=ref_idx, tail_beta=tail_beta, floor_r=1.00)
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

    os.makedirs("gso", exist_ok=True)
    out_pos = f"gso/hdrop_{idx:02d}_pos.gso"
    out_neg = f"gso/hdrop_{idx:02d}_neg.gso"
    export_both_gso(final_rows, t1_new, t2_new, y_unit, out_pos, out_neg)

    y_steps = [abs(r[-1] // y_unit) / (1 << 30) for r in final_rows[1:]]
    med_y = float(np.median(y_steps))
    min_y = float(min(y_steps))
    m_final = len(final_rows) - 1
    dt = time.time() - t0
    print(
        f"[hdrop_{idx:02d}] {desc:28s} | M={m_final:2d}/{M_act:2d} | beta={beta:.2f} | "
        f"drops={drops[:4]}.. | min_Y={min_y:.2e} med_Y={med_y:.2e} ({dt:.2f}s)",
        flush=True,
    )
    return idx, desc, m_final, M_act, drops, beta, med_y, out_pos, out_neg


if __name__ == "__main__":
    t0 = time.time()
    specs = design_hdrop_specs()
    print(f"Building {len(specs)} Heavy-Zero Drop BKZ-60 bases (96 .gso files) with max_workers=32...", flush=True)
    with ProcessPoolExecutor(max_workers=32) as ex:
        results = list(ex.map(build_one, specs))
    dt = time.time() - t0
    full_rank = sum(1 for r in results if r[2] == r[3])
    print(
        f"\nCompleted all {len(results)} pairs ({2*len(results)} .gso files) in {dt:.2f}s | "
        f"Full-rank (no ejected singletons): {full_rank}/{len(results)}",
        flush=True,
    )
