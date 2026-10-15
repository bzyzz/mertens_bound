import os
import time
import numpy as np
import gmpy2
from multiprocessing import Pool
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
    out_pos = f"s35w3_{idx:02d}_{name}_pos.gso"
    out_neg = f"s35w3_{idx:02d}_{name}_neg.gso"
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
    # Core j < 32 has weight 5; boundary active j in [32, K) has weight 7 (ratio 7/5 = 1.40!)
    # Passive j >= K has weight 4; s_mult = 96 or 144 -> forces macro = 53..58 AND med_Y ~ 1e18!
    for K, sm in [
        (54, 96), (55, 96), (56, 96), (57, 96), (58, 96), (60, 96),
        (54, 144), (56, 144), (58, 144), (60, 144),
    ]:
        cm = [5 if j < 32 else (7 if j < K else 4) for j in range(140)]
        specs.append((f"m{K}_y{sm}", cm, sm))

    # Swapped active-54..57 bases with core=5, promote=7, demote=4
    swaps = [
        ("msw1_y96",  range(40, 48), range(48, 64), 96),
        ("msw2_y96",  range(36, 44), range(48, 64), 96),
        ("msw3_y96",  range(38, 46), range(48, 65), 96),
        ("msw4_y96",  range(33, 41), range(48, 64), 96),
        ("msw5_y96",  range(42, 48), range(48, 62), 96),
        ("msw6_y96",  [j for j in range(34, 48, 2)], range(48, 63), 96),
        ("msw7_y96",  [j for j in range(35, 49, 2)], range(48, 63), 96),
        ("msw8_y96",  [j for j in range(33, 48) if j % 3 == 0], range(48, 62), 96),
        ("msw1_y144", range(40, 48), range(48, 64), 144),
        ("msw2_y144", range(36, 44), range(48, 64), 144),
        ("msw3_y144", range(38, 46), range(48, 65), 144),
        ("msw5_y144", range(42, 48), range(48, 62), 144),
        ("msw9_y96",  range(44, 50), range(50, 64), 96),
        ("msw10_y96", range(30, 38), range(48, 64), 96),
    ]
    for name, j_dn, j_up, sm in swaps:
        s_dn, s_up = set(j_dn), set(j_up)
        cm = [5 if j < 30 else (4 if j in s_dn else (7 if (j < 48 or j in s_up) else 4)) for j in range(140)]
        specs.append((name, cm, sm))

    return [(i, s[0], s[1], s[2]) for i, s in enumerate(specs)]


if __name__ == "__main__":
    t0 = time.time()
    tasks = make_specs()
    print(f"Building {len(tasks)} true-macro-54..58 sub-3.485e18 BKZ-60 reductions -> {2*len(tasks)} GSO bases...")
    with Pool(24) as pool:
        pairs = pool.map(build_pair, tasks)
    all_files = []
    for p, n in pairs:
        all_files.extend([p, n])
    print(f"Done in {time.time() - t0:.2f}s!")
    print(",".join(all_files))
