import time
import math
import struct
import numpy as np
import gmpy2
from multiprocessing import Pool
from make_rescaled_gso import read_ntl
from make_wave2_gso import export_both_gso
from fpylll import IntegerMatrix, LLL, BKZ, FPLLL

FPLLL.set_precision(300)
gmpy2.get_context().precision = 300

ROWS_ORIG, T1_ORIG = read_ntl("SR.ENUM")
_, T2_ORIG = read_ntl("SR.ENUM2")
M_COLS = len(ROWS_ORIG[0])
SINGS_ORIG = [r for r in ROWS_ORIG if abs(r[M_COLS - 1]) <= (1 << 54)]
R0_ORIG = SINGS_ORIG[0]

W_ORIG = []
G_ORIG = []
for j in range(140):
    g = 0
    for r in ROWS_ORIG:
        xk = r[-1] // (1 << 30)
        g = math.gcd(g, abs(r[j] - xk * R0_ORIG[j]))
    W_ORIG.append(g)
    G_ORIG.append(R0_ORIG[j] % g)

with open("zeros.bin", "rb") as f:
    nz = struct.unpack("<i", f.read(4))[0]
    TWO_ASTAR = np.frombuffer(f.read(8 * nz), dtype=np.float64)[:140].copy()
    GAMMA = np.frombuffer(f.read(8 * nz), dtype=np.float64)[:140].copy()


def build_compact_pair(args):
    idx, name, sel_indices, mult_nums, s_mult = args
    FPLLL.set_precision(300)
    gmpy2.get_context().precision = 300
    t0 = time.time()

    M_act = len(sel_indices)
    n = M_act + 1
    B = IntegerMatrix(n, n)
    t1_new = [0] * n
    t2_new = [0] * n

    for a, j in enumerate(sel_indices):
        mult = int(mult_nums[a])
        B[0, a] = G_ORIG[j] * mult
        B[a + 1, a] = W_ORIG[j] * mult
        t1_new[a] = (T1_ORIG[j] % W_ORIG[j]) * mult
        t2_new[a] = (T2_ORIG[j] % W_ORIG[j]) * mult

    y_unit = (1 << 30) * 1024 * int(s_mult)
    B[0, M_act] = y_unit

    LLL.reduction(B)
    for b in [20, 30, 40, 48, 54, min(60, n)]:
        param = BKZ.Param(block_size=b, max_loops=3, strategies=BKZ.DEFAULT_STRATEGY, auto_abort=True)
        BKZ.reduction(B, param)

    rows_b = [[int(B[i, k]) for k in range(n)] for i in range(n)]
    s_rows = [r for r in rows_b if abs(r[M_act] // y_unit) <= (1 << 24)]
    m_rows = [r for r in rows_b if abs(r[M_act] // y_unit) > (1 << 24)]
    s_rows.sort(key=lambda r: abs(r[M_act]))

    # Ensure s_rows[0] has positive y_unit (= +1 in xcol)
    if s_rows[0][M_act] < 0:
        s_rows[0] = [-x for x in s_rows[0]]

    # If any extra zero-singletons were ejected (len(s_rows) > 1), strip their columns and rows!
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

    out_pos = f"pc_{idx:02d}_{name}_pos.gso"
    out_neg = f"pc_{idx:02d}_{name}_neg.gso"
    export_both_gso(final_rows, t1_final, t2_final, y_unit, out_pos, out_neg)

    y_steps = [abs(r[-1] // y_unit) / (1 << 30) for r in final_rows[1:]]
    print(
        f"[{idx:02d}] {name:16s}: m_act={len(final_rows)-1:2d} (from {M_act:2d}), "
        f"med_Y={np.median(y_steps):.2e} ({time.time()-t0:.2f}s)",
        flush=True,
    )
    return out_pos, out_neg


def make_specs():
    specs = []

    # Family A (00..11): Expanded Prefix Lattices M in {50..58} with critical weight exponent beta
    prefix_configs = [
        ("M50_b42_y2", 50, 0.42, 2),
        ("M50_b42_y4", 50, 0.42, 4),
        ("M51_b38_y2", 51, 0.38, 2),
        ("M51_b38_y4", 51, 0.38, 4),
        ("M52_b35_y2", 52, 0.35, 2),
        ("M52_b35_y4", 52, 0.35, 4),
        ("M53_b31_y2", 53, 0.31, 2),
        ("M53_b31_y4", 53, 0.31, 4),
        ("M54_b28_y2", 54, 0.28, 2),
        ("M54_b28_y4", 54, 0.28, 4),
        ("M55_b25_y3", 55, 0.25, 3),
        ("M56_b22_y3", 56, 0.22, 3),
    ]
    for name, M_act, beta, sm in prefix_configs:
        sel = list(range(M_act))
        w_ref = W_ORIG[M_act - 1]
        mults = [int(round(((W_ORIG[j] / w_ref) ** (2.0 * beta - 1.0)) * 1024)) for j in sel]
        specs.append((name, sel, mults, sm))

    # Family B (12..23): Rigidity-Weighted Low-gamma Priority Lattices
    # Priorities score_j = two_astar[j] * gamma[j]^(-eta) bring slow-phase rigid zeros into the active lattice!
    rigidity_configs = [
        ("rig_M51_e12_y2", 51, 0.36, 0.12, 2),
        ("rig_M51_e20_y3", 51, 0.36, 0.20, 3),
        ("rig_M52_e12_y2", 52, 0.33, 0.12, 2),
        ("rig_M52_e20_y3", 52, 0.33, 0.20, 3),
        ("rig_M52_e28_y3", 52, 0.33, 0.28, 3),
        ("rig_M53_e15_y2", 53, 0.30, 0.15, 2),
        ("rig_M53_e24_y3", 53, 0.30, 0.24, 3),
        ("rig_M54_e15_y3", 54, 0.27, 0.15, 3),
        ("rig_M54_e24_y3", 54, 0.27, 0.24, 3),
        ("rig_M54_e32_y4", 54, 0.27, 0.32, 4),
        ("rig_M55_e18_y3", 55, 0.24, 0.18, 3),
        ("rig_M55_e28_y4", 55, 0.24, 0.28, 4),
    ]
    for name, M_act, beta, eta, sm in rigidity_configs:
        scores = TWO_ASTAR * (GAMMA ** (-eta))
        sel = sorted(int(x) for x in np.argsort(-scores)[:M_act])
        w_ref = W_ORIG[sel[-1]]
        g_ref = GAMMA[sel[-1]]
        mults = []
        for j in sel:
            # Desired weight proportional to (two_astar[j])^beta * (gamma[j])^(-0.35*eta)
            scale = ((W_ORIG[j] / w_ref) ** (2.0 * beta - 1.0)) * ((GAMMA[j] / g_ref) ** (-0.35 * eta))
            mults.append(int(round(scale * 1024)))
        specs.append((name, sel, mults, sm))

    # Family C (24..35): Combinatorial Zero-Subset & Tail-Boosted Lattices (M = 52..54)
    subset_configs = [
        ("sub_top38_ev", list(range(38)) + [j for j in range(38, 68, 2)], 0.31, 3),
        ("sub_top38_od", list(range(38)) + [j for j in range(39, 69, 2)], 0.31, 3),
        ("sub_skip45_48", list(range(45)) + list(range(49, 57)), 0.33, 3),
        ("sub_skip41_44", list(range(41)) + list(range(45, 56)), 0.34, 3),
        ("sub_skip37_40", list(range(37)) + list(range(41, 56)), 0.34, 3),
        ("sub_hi54_58", list(range(47)) + [49, 50, 51, 54, 55, 56], 0.31, 3),
        ("sub_hi50_60", list(range(45)) + [47, 48, 50, 52, 54, 56, 58, 60], 0.30, 3),
        ("sub_mod3_0", list(range(40)) + [j for j in range(40, 64) if j % 3 != 0][:13], 0.31, 3),
        ("sub_mod3_1", list(range(40)) + [j for j in range(40, 64) if j % 3 != 1][:13], 0.31, 3),
        ("sub_mod3_2", list(range(40)) + [j for j in range(40, 64) if j % 3 != 2][:13], 0.31, 3),
        ("sub_hvy12_M52", list(range(52)), 0.38, 3),
        ("sub_flat_M54", list(range(54)), 0.24, 3),
    ]
    for name, sel, beta, sm in subset_configs:
        w_ref = W_ORIG[sel[-1]]
        mults = [int(round(((W_ORIG[j] / w_ref) ** (2.0 * beta - 1.0)) * 1024)) for j in sel]
        specs.append((name, sel, mults, sm))

    return [(i, s[0], s[1], s[2], s[3]) for i, s in enumerate(specs)]


if __name__ == "__main__":
    t0 = time.time()
    tasks = make_specs()
    print(f"Building {len(tasks)} pure compact BKZ-60 lattices -> {2*len(tasks)} GSO files in parallel...")
    with Pool(24) as pool:
        pool.map(build_compact_pair, tasks)
    print(f"All {2*len(tasks)} pure compact GSO bases built in {time.time() - t0:.2f}s!")
