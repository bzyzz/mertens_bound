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
    out_pos = f"s35w2_{idx:02d}_{name}_pos.gso"
    out_neg = f"s35w2_{idx:02d}_{name}_neg.gso"
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
    # Wide active/passive gap (active=7..8, passive=4 -> ratio 1.75..2.00)
    # Locks macro = 53..58 even under strong y-localization (s_mult = 96..192, med_Y ~ 1e18)
    for K, sm in [(53, 128), (54, 128), (55, 128), (56, 128), (57, 128), (58, 128), (55, 192), (57, 192)]:
        cm = [7 if j < 35 else (8 if j < K else 4) for j in range(140)]
        specs.append((f"lock{K}_y{sm}", cm, sm))

    # Swapped active-53..55 bases with wide gap (demote=4, base=7, promote=8)
    swaps = [
        ("lswA_y128", range(41, 49), range(49, 61), 128),
        ("lswB_y128", range(37, 45), range(49, 61), 128),
        ("lswC_y128", range(39, 47), range(49, 62), 128),
        ("lswD_y128", range(34, 42), range(49, 61), 128),
        ("lswE_y128", range(43, 49), range(49, 60), 128),
        ("lswF_y128", [j for j in range(35, 49, 2)], range(49, 61), 128),
        ("lswG_y128", [j for j in range(34, 49, 2)], range(49, 61), 128),
        ("lswH_y128", [j for j in range(33, 49) if j % 3 == 0], range(49, 60), 128),
        ("lswA_y192", range(41, 49), range(49, 61), 192),
        ("lswB_y192", range(37, 45), range(49, 61), 192),
        ("lswC_y192", range(39, 47), range(49, 62), 192),
        ("lswE_y192", range(43, 49), range(49, 60), 192),
    ]
    for name, j_dn, j_up, sm in swaps:
        s_dn, s_up = set(j_dn), set(j_up)
        cm = [4 if j in s_dn else (8 if j in s_up else (7 if j < 49 else 4)) for j in range(140)]
        specs.append((name, cm, sm))

    # Heavy low-zero + locked macro=54..55
    specs.append(("lhvy15_y128", [8 if j < 15 else (7 if j < 35 else (8 if j < 55 else 4)) for j in range(140)], 128))
    specs.append(("lhvy15_y192", [8 if j < 15 else (7 if j < 35 else (8 if j < 55 else 4)) for j in range(140)], 192))
    specs.append(("lhvy22_y128", [8 if j < 22 else (7 if j < 38 else (8 if j < 56 else 4)) for j in range(140)], 128))
    specs.append(("lhvy22_y192", [8 if j < 22 else (7 if j < 38 else (8 if j < 56 else 4)) for j in range(140)], 192))

    return [(i, s[0], s[1], s[2]) for i, s in enumerate(specs)]


if __name__ == "__main__":
    t0 = time.time()
    tasks = make_specs()
    print(f"Building {len(tasks)} locked-macro sub-3.485e18 BKZ-60 reductions -> {2*len(tasks)} GSO bases...")
    with Pool(24) as pool:
        pairs = pool.map(build_pair, tasks)
    all_files = []
    for p, n in pairs:
        all_files.extend([p, n])
    print(f"Done in {time.time() - t0:.2f}s!")
    print(",".join(all_files))
