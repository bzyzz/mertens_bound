import os
import time
import struct
import numpy as np
import gmpy2
from multiprocessing import Pool
from make_rescaled_gso import read_ntl
from make_wave2_gso import export_both_gso, T1_ORIG, T2_ORIG, ROWS_ORIG
from fpylll import IntegerMatrix, LLL, BKZ, FPLLL

FPLLL.set_precision(300)
gmpy2.get_context().precision = 300


def build_pair(args):
    idx, name, col_mults, s_mult = args
    FPLLL.set_precision(300)
    gmpy2.get_context().precision = 300
    t0 = time.time()
    n, m = len(ROWS_ORIG), len(ROWS_ORIG[0])
    rows = [r[:] for r in ROWS_ORIG]
    t1 = T1_ORIG[:]
    t2 = T2_ORIG[:]
    for j in range(m - 1):
        mult = col_mults[j]
        for i in range(n):
            rows[i][j] *= mult
        t1[j] *= mult
        t2[j] *= mult
    for i in range(n):
        rows[i][m - 1] *= s_mult
    t1[m - 1] *= s_mult
    t2[m - 1] *= s_mult
    y_mult = s_mult * (1 << 30)

    M = IntegerMatrix.from_matrix(rows)
    LLL.reduction(M)
    for beta in [20, 30, 40, 48, 54, 60]:
        param = BKZ.Param(
            block_size=beta,
            max_loops=3,
            strategies=BKZ.DEFAULT_STRATEGY,
            auto_abort=True,
        )
        BKZ.reduction(M, param)

    rows = [[int(M[i, j]) for j in range(m)] for i in range(n)]
    singletons = [r for r in rows if abs(r[m - 1] // y_mult) <= (1 << 24)]
    macro = [r for r in rows if abs(r[m - 1] // y_mult) > (1 << 24)]
    out_pos = f"s35_{idx:02d}_{name}_pos.gso"
    out_neg = f"s35_{idx:02d}_{name}_neg.gso"
    export_both_gso(singletons + macro, t1, t2, y_mult, out_pos, out_neg)
    y_steps = [abs(r[m - 1] // y_mult) / (1 << 30) for r in macro]
    print(
        f"[{idx:02d}] {name:16s}: k_cut={len(singletons)} (macro={len(macro)}), "
        f"med_Y={np.median(y_steps):.2e} ({time.time()-t0:.1f}s)",
        flush=True,
    )
    return out_pos, out_neg


def make_specs():
    specs = []
    # Group A: Expanded macro dimension (top52..top58) with 24x..48x relative y-scaling
    # Base zero weight ~ 4..6, s_mult = 128 or 192 -> med_Y ~ 1.0e18 .. 1.5e18
    c_top52 = [5 if j < 35 else (6 if j < 52 else 4) for j in range(140)]
    c_top54 = [5 if j < 35 else (6 if j < 54 else 4) for j in range(140)]
    c_top56 = [5 if j < 35 else (6 if j < 56 else 4) for j in range(140)]
    c_top58 = [5 if j < 33 else (6 if j < 58 else 4) for j in range(140)]
    c_t55s1 = [5 if j < 34 else (4 if 40 <= j < 48 else (6 if j < 64 else 4)) for j in range(140)]
    c_t55s2 = [5 if j < 34 else (4 if 36 <= j < 44 else (6 if j < 64 else 4)) for j in range(140)]
    c_t55h  = [6 if j < 15 else (5 if j < 35 else (6 if j < 56 else 4)) for j in range(140)]
    c_top52sF = [5 if j < 35 else (4 if 44 <= j <= 48 else (6 if j <= 56 else 4)) for j in range(140)]

    for name, cm, sm in [
        ("top52_y128", c_top52, 128),
        ("top54_y128", c_top54, 128),
        ("top54_y192", c_top54, 192),
        ("top56_y128", c_top56, 128),
        ("top56_y192", c_top56, 192),
        ("top58_y128", c_top58, 128),
        ("top58_y192", c_top58, 192),
        ("t55s1_y128", c_t55s1, 128),
        ("t55s2_y128", c_t55s2, 128),
        ("t55h_y128",  c_t55h,  128),
        ("top52sF_y128", c_top52sF, 128),
    ]:
        specs.append((name, cm, sm))

    # Group B: Column-swapped top-49/50 bases with 24x..40x relative y-scaling
    # Base zero weight = 8, s_mult = 192 (24x) or 256 (32x) -> med_Y ~ 9.5e17 .. 1.3e18
    swap_defs = [
        ("swapA_y192", range(41, 49), range(49, 57), 192),
        ("swapA_y256", range(41, 49), range(49, 57), 256),
        ("swapB_y192", range(37, 45), range(49, 57), 192),
        ("swapB_y256", range(37, 45), range(49, 57), 256),
        ("swapC_y192", range(39, 49), range(53, 63), 192),
        ("swapD_y192", range(33, 41), range(49, 57), 192),
        ("swapE_y192", [j for j in range(35, 48, 2)], range(49, 56), 192),
        ("swapG_y192", range(43, 49), range(49, 55), 192),
        ("swapH_y192", range(35, 43), range(51, 59), 192),
        ("swapI_y192", [j for j in range(34, 49, 2)], range(49, 57), 192),
    ]
    for name, j_dn, j_up, sm in swap_defs:
        s_dn, s_up = set(j_dn), set(j_up)
        cm = [7 if j in s_dn else (10 if j in s_up else 8) for j in range(140)]
        specs.append((name, cm, sm))

    # Group C: Heavy low-zero boosted + localized y-scaling
    specs.append(("heavy12_y192", [10 if j < 12 else (8 if j < 50 else 7) for j in range(140)], 192))
    specs.append(("heavy20_y192", [9 if j < 20 else (8 if j < 51 else 7) for j in range(140)], 192))
    specs.append(("grad52_y192",  [10 if j < 15 else (9 if j < 35 else (8 if j < 52 else 7)) for j in range(140)], 192))

    return [(i, s[0], s[1], s[2]) for i, s in enumerate(specs)]


if __name__ == "__main__":
    t0 = time.time()
    tasks = make_specs()
    print(f"Building {len(tasks)} BKZ-60 reductions -> {2*len(tasks)} sub-3.485e18 GSO bases in parallel...")
    with Pool(24) as pool:
        pairs = pool.map(build_pair, tasks)
    all_files = []
    for p, n in pairs:
        all_files.extend([p, n])
    print(f"Done in {time.time() - t0:.2f}s!")
    print(",".join(all_files))
