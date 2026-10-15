import os
import time
from multiprocessing import Pool
from make_rescaled_gso import read_ntl, export_gso
from fpylll import IntegerMatrix, LLL, BKZ, FPLLL

def build_one(args):
    idx, src, name, col_mults, s_mult = args
    FPLLL.set_precision(300)
    t0 = time.time()
    rows, tgt = read_ntl(src)
    n, m = len(rows), len(rows[0])
    for j in range(m - 1):
        mult = col_mults[j]
        for i in range(n):
            rows[i][j] *= mult
        tgt[j] *= mult
    for i in range(n):
        rows[i][m - 1] *= s_mult
    tgt[m - 1] *= s_mult
    y_mult = s_mult * (1 << 30)

    M = IntegerMatrix.from_matrix(rows)
    LLL.reduction(M)
    for beta in [20, 30, 40, 48, 54, 60]:
        param = BKZ.Param(block_size=beta, max_loops=3, strategies=BKZ.DEFAULT_STRATEGY, auto_abort=True)
        BKZ.reduction(M, param)

    rows = [[int(M[i, j]) for j in range(m)] for i in range(n)]
    singletons = [r for r in rows if abs(r[m - 1] // y_mult) <= (1 << 24)]
    macro = [r for r in rows if abs(r[m - 1] // y_mult) > (1 << 24)]
    out_path = f"ens_{idx:02d}_{name}.gso"
    export_gso(singletons + macro, tgt, y_mult, out_path)
    dt = time.time() - t0
    print(f"[{idx:02d}] {out_path}: singletons={len(singletons)}, macro={len(macro)} in {dt:.1f}s", flush=True)
    return out_path

def make_specs():
    specs = []
    # 1. top54 (y_mult=4)
    c_top54 = [5 if j < 35 else (6 if j < 54 else 4) for j in range(140)]
    specs.append(("SR.ENUM", "SR1_top54", c_top54, 4))
    specs.append(("SR.ENUM2", "SR2_top54", c_top54, 4))

    # 2. top56 (y_mult=4)
    c_top56 = [5 if j < 35 else (6 if j < 56 else 4) for j in range(140)]
    specs.append(("SR.ENUM", "SR1_top56", c_top56, 4))
    specs.append(("SR.ENUM2", "SR2_top56", c_top56, 4))

    # 3. top58 (y_mult=4)
    c_top58 = [5 if j < 33 else (6 if j < 58 else 4) for j in range(140)]
    specs.append(("SR.ENUM", "SR1_top58", c_top58, 4))
    specs.append(("SR.ENUM2", "SR2_top58", c_top58, 4))

    # 4. top55 swap {40..47} <-> {56..63}
    c_t55_s1 = [5 if j < 34 else (4 if 40 <= j < 48 else (6 if j < 64 else 4)) for j in range(140)]
    specs.append(("SR.ENUM", "SR1_t55s1", c_t55_s1, 4))
    specs.append(("SR.ENUM2", "SR2_t55s1", c_t55_s1, 4))

    # 5. top55 swap {36..43} <-> {56..63}
    c_t55_s2 = [5 if j < 34 else (4 if 36 <= j < 44 else (6 if j < 64 else 4)) for j in range(140)]
    specs.append(("SR.ENUM", "SR1_t55s2", c_t55_s2, 4))
    specs.append(("SR.ENUM2", "SR2_t55s2", c_t55_s2, 4))

    # 6. top55 heavy0..14 boosted
    c_t55_h = [6 if j < 15 else (5 if j < 35 else (6 if j < 56 else 4)) for j in range(140)]
    specs.append(("SR.ENUM", "SR1_t55h", c_t55_h, 4))
    specs.append(("SR.ENUM2", "SR2_t55h", c_t55_h, 4))

    # 7. top49 swap A: {41..48} <-> {49..56}
    c_sA = [7 if 41 <= j <= 48 else (10 if 49 <= j <= 56 else 8) for j in range(140)]
    specs.append(("SR.ENUM", "SR1_swapA", c_sA, 8))
    specs.append(("SR.ENUM2", "SR2_swapA", c_sA, 8))

    # 8. top49 swap B: {37..44} <-> {49..56}
    c_sB = [7 if 37 <= j <= 44 else (10 if 49 <= j <= 56 else 8) for j in range(140)]
    specs.append(("SR.ENUM", "SR1_swapB", c_sB, 8))
    specs.append(("SR.ENUM2", "SR2_swapB", c_sB, 8))

    # 9. top49 swap C: {39..48} <-> {53..62}
    c_sC = [7 if 39 <= j <= 48 else (10 if 53 <= j <= 62 else 8) for j in range(140)]
    specs.append(("SR.ENUM", "SR1_swapC", c_sC, 8))
    specs.append(("SR.ENUM2", "SR2_swapC", c_sC, 8))

    # 10. top49 swap D: {33..40} <-> {49..56}
    c_sD = [7 if 33 <= j <= 40 else (10 if 49 <= j <= 56 else 8) for j in range(140)]
    specs.append(("SR.ENUM", "SR1_swapD", c_sD, 8))
    specs.append(("SR.ENUM2", "SR2_swapD", c_sD, 8))

    # 11. top49 swap E: odd j in 35..47 <-> 49..55
    c_sE = [7 if (35 <= j <= 47 and j % 2 == 1) else (10 if 49 <= j <= 55 else 8) for j in range(140)]
    specs.append(("SR.ENUM", "SR1_swapE", c_sE, 8))
    specs.append(("SR.ENUM2", "SR2_swapE", c_sE, 8))

    # 12. top52 swap F: K=52 with {44..48} <-> {52..56}
    c_sF = [5 if j < 35 else (4 if 44 <= j <= 48 else (6 if j <= 56 else 4)) for j in range(140)]
    specs.append(("SR.ENUM", "SR1_top52sF", c_sF, 4))
    specs.append(("SR.ENUM2", "SR2_top52sF", c_sF, 4))

    return [(i, s[0], s[1], s[2], s[3]) for i, s in enumerate(specs)]

if __name__ == "__main__":
    t0 = time.time()
    tasks = make_specs()
    print(f"Building {len(tasks)} BKZ-60 ensemble bases in parallel...")
    with Pool(24) as pool:
        paths = pool.map(build_one, tasks)
    print(f"All {len(paths)} bases built in {time.time() - t0:.2f}s:")
    print(",".join(paths))
