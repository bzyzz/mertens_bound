#!/usr/bin/env python3
"""
build_gpu_m36_bases.py
Builds 16 BKZ-reduced medium-rank GSO bases (M = 36 and M = 38) localized to
K_scale = 3.485e18 (gso/gpu_m36_00..15_*_{pos,neg}.gso):
  - Base 0: pure prefix j in 0..35 (M=36, P=36)
  - Bases 1..11: drop 1 or 2 zeros from j in 10..37 (M=36, P=37..38) covering rogue zeros in 10..37
  - Bases 12..15: rank M=38 bases (pure prefix 0..37 and 2-zero dropouts from 12..39)
"""
import os
import math
import time
import multiprocessing as mp
import numpy as np
from fpylll import IntegerMatrix, LLL, BKZ, FPLLL
from make_pure_compact_ensemble import W_ORIG, G_ORIG, T1_ORIG, T2_ORIG, TWO_ASTAR, export_both_gso

FPLLL.set_precision(300)


def compute_mults(sel, beta=0.35, floor_r=1.00, ref_idx=None):
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


def get_gpu_m36_specs():
    # K_scale = 3.485e18 -> sm = 24 (y_unit = 2^30 * 1024 * 24)
    specs = [
        # Base 0: pure prefix j in 0..35 (M=36, P=36)
        (0, "pref_M36_P36", list(range(36)), 0.36, 24),
        # Bases 1..11: M=36 bases dropping 1 or 2 zeros in 10..37 (P=37 or P=38)
        (1, "drop_19_34_M36", [j for j in range(38) if j not in {19, 34}], 0.36, 24),
        (2, "drop_25_28_M36", [j for j in range(38) if j not in {25, 28}], 0.36, 24),
        (3, "drop_20_35_M36", [j for j in range(38) if j not in {20, 35}], 0.36, 24),
        (4, "drop_16_23_M36", [j for j in range(38) if j not in {16, 23}], 0.36, 24),
        (5, "drop_11_22_M36", [j for j in range(38) if j not in {11, 22}], 0.36, 24),
        (6, "drop_27_33_M36", [j for j in range(38) if j not in {27, 33}], 0.36, 24),
        (7, "drop_14_24_M36", [j for j in range(38) if j not in {14, 24}], 0.36, 24),
        (8, "drop_12_30_M36", [j for j in range(38) if j not in {12, 30}], 0.36, 24),
        (9, "drop_31_36_M36", [j for j in range(38) if j not in {31, 36}], 0.36, 24),
        (10, "drop_17_29_M36", [j for j in range(38) if j not in {17, 29}], 0.36, 24),
        (11, "drop_26_32_M36", [j for j in range(38) if j not in {26, 32}], 0.36, 24),
        # Bases 12..15: rank M=38 bases
        (12, "pref_M38_P38", list(range(38)), 0.34, 24),
        (13, "drop_19_36_M38", [j for j in range(40) if j not in {19, 36}], 0.34, 24),
        (14, "drop_22_38_M38", [j for j in range(40) if j not in {22, 38}], 0.34, 24),
        (15, "drop_25_34_M38", [j for j in range(40) if j not in {25, 34}], 0.34, 24),
    ]
    return specs


def build_one(args):
    idx, name, sel, beta, sm = args
    FPLLL.set_precision(300)
    t0 = time.time()
    M_act = len(sel)
    n = M_act + 1

    for beta_try in [beta, min(beta, 0.20), 0.08, 0.00]:
        mults = compute_mults(sel, beta_try, 1.00, ref_idx=sel[-1])
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
        for b in [20, 30, n]:
            param = BKZ.Param(block_size=b, max_loops=5, strategies=BKZ.DEFAULT_STRATEGY, auto_abort=True)
            BKZ.reduction(B, param)

        rows_b = [[int(B[i, k]) for k in range(n)] for i in range(n)]
        s_rows = [r for r in rows_b if abs(r[M_act] // y_unit) <= (1 << 24)]
        m_rows = [r for r in rows_b if abs(r[M_act] // y_unit) > (1 << 24)]
        if len(s_rows) == 1:
            break

    s_row = s_rows[0]
    if s_row[M_act] < 0:
        s_row = [-x for x in s_row]
    final_rows = [s_row] + m_rows

    os.makedirs("gso", exist_ok=True)
    out_pos = f"gso/gpu_m36_{idx:02d}_{name}_pos.gso"
    out_neg = f"gso/gpu_m36_{idx:02d}_{name}_neg.gso"
    export_both_gso(final_rows, t1_new, t2_new, y_unit, out_pos, out_neg)
    y_steps = [abs(r[-1] // y_unit) / (1 << 30) for r in final_rows[1:]]
    print(
        f"[gpu_m36_{idx:02d}] {name:18s}: m_act={len(final_rows)-1:2d}, "
        f"y1={y_steps[0]:.2e}, med_Y={np.median(y_steps):.2e} ({time.time()-t0:.2f}s)",
        flush=True,
    )
    return out_pos, out_neg


if __name__ == "__main__":
    t0 = time.time()
    specs = get_gpu_m36_specs()
    with mp.Pool(min(16, len(specs))) as pool:
        pairs = pool.map(build_one, specs)
    print(f"All {len(specs)} pairs ({2*len(specs)} GSO files) built in {time.time()-t0:.2f}s", flush=True)
