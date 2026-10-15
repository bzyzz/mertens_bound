#!/usr/bin/env python3
"""
Generate 16 Pairs (32 GSO files) of Pure Compact Zero-Dropout Covering Lattices
(zd_00..zd_15, M_act in {51, 52}, n = M_act + 1, k_y_cut = 1).

Key Mathematical Principle:
Every extreme peak of |h_StR(y)| has all first 24 zeros j in [0..23] tightly aligned,
while 2 to 4 zeros in the mid-band j in [24..56] are anti-aligned (fractional phase error 0.40..0.50).
In a prefix lattice (like Kim & Nguyen's SR.ENUM), those 2-4 anti-aligned mid-band zeros add a massive
Euclidean norm penalty that pushes the lattice point outside the enumeration radius.
By constructing a Combinatorial Covering Code of Zero-Dropout Lattices over j in [0..56] that always keeps
[0..23] and drops structured 5-to-6-zero subsets of [24..56], any peak with 2-4 anti-aligned zeros in [24..56]
falls deep inside the enumeration sphere of at least one covering lattice.
"""
import math
import multiprocessing as mp
import struct
import time
import numpy as np
from fpylll import IntegerMatrix, LLL, BKZ, FPLLL
from make_pure_compact_ensemble import W_ORIG, G_ORIG, T1_ORIG, T2_ORIG, export_both_gso

FPLLL.set_precision(300)


def make_zd_specs():
    specs = [
        # Group 1 (zd_00..zd_07): Structured Combinatorial Covering of [24..56] (M_act = 51 or 52)
        ("drop_34_36_42_48_50_55", [j for j in range(57) if j not in {34, 36, 42, 48, 50, 55}], 0.38, 4),
        ("drop_30_31_35_47_52_53", [j for j in range(57) if j not in {30, 31, 35, 47, 52, 53}], 0.38, 4),
        ("drop_29_43_51_52_54_55", [j for j in range(57) if j not in {29, 43, 51, 52, 54, 55}], 0.38, 4),
        ("drop_30_44_45_50_51_52", [j for j in range(57) if j not in {30, 44, 45, 50, 51, 52}], 0.38, 4),
        ("drop_24_27_30_37_38_41", [j for j in range(57) if j not in {24, 27, 30, 37, 38, 41}], 0.38, 4),
        ("drop_25_32_38_44_46_52", [j for j in range(57) if j not in {25, 32, 38, 44, 46, 52}], 0.38, 4),
        ("drop_29_33_40_45_49_55", [j for j in range(57) if j not in {29, 33, 40, 45, 49, 55}], 0.38, 4),
        ("drop_27_37_40_41_42_55", [j for j in range(57) if j not in {27, 37, 40, 41, 42, 55}], 0.38, 4),
        # Group 2 (zd_08..zd_15): Residue & Sliding Window Covering of [25..56]
        ("drop_28_31_34_38_40_50", [j for j in range(57) if j not in {28, 31, 34, 38, 40, 50}], 0.38, 4),
        ("drop_25_37_41_42_46_49", [j for j in range(57) if j not in {25, 37, 41, 42, 46, 49}], 0.38, 4),
        ("drop_39_44_49_50_51_56", [j for j in range(57) if j not in {39, 44, 49, 50, 51, 56}], 0.38, 4),
        ("drop_27_28_34_35_39_55", [j for j in range(57) if j not in {27, 28, 34, 35, 39, 55}], 0.38, 4),
        ("drop_win26_31",          [j for j in range(57) if j not in {26, 27, 28, 29, 30, 31}], 0.38, 4),
        ("drop_win32_37",          [j for j in range(57) if j not in {32, 33, 34, 35, 36, 37}], 0.38, 4),
        ("drop_mod5_0_in_25_54",   [j for j in range(57) if not (25 <= j <= 54 and j % 5 == 0)], 0.38, 4),
        ("drop_mod5_2_in_25_54",   [j for j in range(57) if not (25 <= j <= 54 and j % 5 == 2)], 0.38, 4),
    ]
    out = []
    for idx, (name, sel, beta, sm) in enumerate(specs):
        w_ref = W_ORIG[sel[-1]]
        mults = [int(round(((W_ORIG[j] / w_ref) ** (2.0 * beta - 1.0)) * 1024)) for j in sel]
        out.append((idx, name, sel, mults, sm))
    return out


def build_one(args):
    idx, name, sel, mults, sm = args
    t0 = time.time()
    M_act = len(sel)
    n = M_act + 1
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
    for b in [20, 30, 40, 48, 54, min(60, n)]:
        param = BKZ.Param(block_size=b, max_loops=3, strategies=BKZ.DEFAULT_STRATEGY, auto_abort=True)
        BKZ.reduction(B, param)

    rows_b = [[int(B[i, k]) for k in range(n)] for i in range(n)]
    s_rows = [r for r in rows_b if abs(r[M_act] // y_unit) <= (1 << 24)]
    m_rows = [r for r in rows_b if abs(r[M_act] // y_unit) > (1 << 24)]
    s_rows.sort(key=lambda r: abs(r[M_act]))
    if s_rows[0][M_act] < 0:
        s_rows[0] = [-x for x in s_rows[0]]

    if len(s_rows) > 1:
        r0 = s_rows[0]
        ej_cols = set()
        for r in s_rows[1:]:
            xk = r[M_act] // r0[M_act]
            for col in range(M_act):
                if r[col] - xk * r0[col] != 0:
                    ej_cols.add(col)
        keep_cols = [col for col in range(n) if col not in ej_cols]
        final_rows = [[r[col] for col in keep_cols] for r in ([s_rows[0]] + m_rows)]
        t1_final = [t1_new[col] for col in keep_cols]
        t2_final = [t2_new[col] for col in keep_cols]
    else:
        final_rows = s_rows + m_rows
        t1_final = t1_new
        t2_final = t2_new

    out_pos = f"zd_{idx:02d}_{name}_pos.gso"
    out_neg = f"zd_{idx:02d}_{name}_neg.gso"
    export_both_gso(final_rows, t1_final, t2_final, y_unit, out_pos, out_neg)
    print(f"[zd_{idx:02d}] {name:26s}: m_act={len(final_rows)-1:2d} ({time.time()-t0:.2f}s)", flush=True)
    return out_pos, out_neg


if __name__ == "__main__":
    specs = make_zd_specs()
    t0 = time.time()
    with mp.Pool(min(16, len(specs))) as pool:
        pool.map(build_one, specs)
    print(f"All 32 Zero-Dropout GSO files built in {time.time()-t0:.2f}s")
