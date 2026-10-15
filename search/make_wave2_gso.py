import os
import time
import struct
import numpy as np
import gmpy2
from multiprocessing import Pool
from make_rescaled_gso import read_ntl
from fpylll import IntegerMatrix, LLL, BKZ, FPLLL

FPLLL.set_precision(300)
gmpy2.get_context().precision = 300

_, T1_ORIG = read_ntl("SR.ENUM")
_, T2_ORIG = read_ntl("SR.ENUM2")
ROWS_ORIG, _ = read_ntl("SR.ENUM2")

def export_both_gso(rows, t1, t2, y_mult, out_pos, out_neg):
    n = len(rows)
    m = len(rows[0])
    B = [[gmpy2.mpfr(x) for x in r] for r in rows]
    B1 = [gmpy2.mpfr(x) for x in t1]
    B2 = [gmpy2.mpfr(x) for x in t2]
    mu = [[gmpy2.mpfr(0)] * n for _ in range(n)]
    r = [[gmpy2.mpfr(0)] * n for _ in range(n)]
    c = [gmpy2.mpfr(0)] * n

    for i in range(n):
        for j in range(i + 1):
            ip = sum(B[i][k] * B[j][k] for k in range(m))
            val = ip
            for l in range(j):
                val -= mu[j][l] * r[i][l]
            r[i][j] = val
            if j < i:
                mu[i][j] = val / r[j][j]
            else:
                mu[i][i] = gmpy2.mpfr(1)
                c[i] = val

    # Project B1 and B2 onto b_j^*
    r1 = [gmpy2.mpfr(0)] * n
    r2 = [gmpy2.mpfr(0)] * n
    sig1 = np.zeros(n, dtype=np.float64)
    sig2 = np.zeros(n, dtype=np.float64)
    for j in range(n):
        v1 = sum(B1[k] * B[j][k] for k in range(m))
        v2 = sum(B2[k] * B[j][k] for k in range(m))
        for l in range(j):
            v1 -= mu[j][l] * r1[l]
            v2 -= mu[j][l] * r2[l]
        r1[j] = v1
        r2[j] = v2
        sig1[j] = float(v1 / c[j])
        sig2[j] = float(v2 / c[j])

    bstar_last = [gmpy2.mpfr(0)] * n
    for i in range(n):
        val = B[i][m - 1]
        for l in range(i):
            val -= mu[i][l] * bstar_last[l]
        bstar_last[i] = val

    scale_y1 = gmpy2.mpfr(y_mult) * gmpy2.exp2(30)
    rs_arr = np.array([float(c[i]) for i in range(n)], dtype=np.float64)
    dsig_arr = np.array([float(scale_y1 * bstar_last[i] / c[i]) for i in range(n)], dtype=np.float64)
    mu_arr = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            mu_arr[i, j] = float(mu[i][j])

    lo_arr = np.zeros(n, dtype=np.uint64)
    hi_arr = np.zeros(n, dtype=np.uint64)
    MASK64 = (1 << 64) - 1
    MASK128 = (1 << 128) - 1
    for i in range(n):
        xval = rows[i][m - 1] // y_mult
        u = xval & MASK128
        lo_arr[i] = u & MASK64
        hi_arr[i] = (u >> 64) & MASK64

    for path, sig_arr in [(out_pos, sig1), (out_neg, sig2)]:
        if os.path.dirname(path) == "" and os.path.isdir("gso"):
            path = os.path.join("gso", path)
        with open(path, "wb") as f:
            f.write(struct.pack("<i", n))
            f.write(rs_arr.tobytes())
            f.write(sig_arr.tobytes())
            f.write(dsig_arr.tobytes())
            f.write(mu_arr.tobytes())
            f.write(lo_arr.tobytes())
            f.write(hi_arr.tobytes())

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
        param = BKZ.Param(block_size=beta, max_loops=3, strategies=BKZ.DEFAULT_STRATEGY, auto_abort=True)
        BKZ.reduction(M, param)

    rows = [[int(M[i, j]) for j in range(m)] for i in range(n)]
    singletons = [r for r in rows if abs(r[m - 1] // y_mult) <= (1 << 24)]
    macro = [r for r in rows if abs(r[m - 1] // y_mult) > (1 << 24)]
    out_pos = f"w2_{idx:02d}_{name}_pos.gso"
    out_neg = f"w2_{idx:02d}_{name}_neg.gso"
    export_both_gso(singletons + macro, t1, t2, y_mult, out_pos, out_neg)
    y_steps = [abs(r[m - 1] // y_mult) / (1 << 30) for r in macro]
    print(f"[{idx:02d}] {name:12s}: k_cut={len(singletons)} (macro={len(macro)}), med_Y={np.median(y_steps):.2e} ({time.time()-t0:.1f}s)", flush=True)
    return out_pos, out_neg

def make_specs():
    specs = []
    # Localized y-scaling (s_mult = 2x..6x base weight) + zero swaps around j=30..65
    # 0..3: Pure exact zero weights with s_mult = 5, 10, 14, 20
    for sm in [5, 10, 14, 20]:
        specs.append((f"exact_y{sm}", [1]*140, sm))

    # 4..15: Swapped top-49/50 bases with localized y-scaling (base=8, s_mult=32 or 64 -> 4x or 8x y-localization!)
    swap_ranges = [
        ("sw41_48", range(41, 49), range(49, 57), 32),
        ("sw41_48y64", range(41, 49), range(49, 57), 64),
        ("sw37_44", range(37, 45), range(49, 57), 32),
        ("sw33_40", range(33, 41), range(49, 57), 32),
        ("sw39_48_53", range(39, 49), range(53, 63), 32),
        ("sw35_44_51", range(35, 45), range(51, 61), 32),
        ("sw43_48_55", range(43, 49), range(55, 61), 32),
        ("sw30_37_49", range(30, 38), range(49, 57), 32),
        ("sw_even", [j for j in range(34, 49, 2)], range(49, 57), 32),
        ("sw_odd", [j for j in range(35, 49, 2)], range(50, 57), 32),
        ("sw_mod3a", [j for j in range(30, 49) if j % 3 == 0], range(49, 56), 32),
        ("sw_mod3b", [j for j in range(30, 49) if j % 3 == 1], range(52, 58), 32),
    ]
    for name, j_dn, j_up, sm in swap_ranges:
        s_dn, s_up = set(j_dn), set(j_up)
        cm = [7 if j in s_dn else (10 if j in s_up else 8) for j in range(140)]
        specs.append((name, cm, sm))

    # 16..23: Heavy-boosted + expanded top-51..54 bases with monotonically non-increasing zero weights on 0..34!
    # Notice: keep 0..34 at 9 or 10 (NEVER smaller than 35..54!) while boosting 49..54 to 9 and keeping 55..139 at 7!
    specs.append(("mono_top53_y24", [9 if j < 53 else 7 for j in range(140)], 24))
    specs.append(("mono_top53_y40", [9 if j < 53 else 7 for j in range(140)], 40))
    specs.append(("mono_top55_y24", [9 if j < 55 else 7 for j in range(140)], 24))
    specs.append(("mono_top55_y40", [9 if j < 55 else 7 for j in range(140)], 40))
    specs.append(("heavy10_y24", [10 if j < 10 else (8 if j < 50 else 7) for j in range(140)], 24))
    specs.append(("heavy10_y48", [10 if j < 10 else (8 if j < 50 else 7) for j in range(140)], 48))
    specs.append(("heavy20_y24", [9 if j < 20 else (8 if j < 51 else 7) for j in range(140)], 24))
    specs.append(("heavy20_y48", [9 if j < 20 else (8 if j < 51 else 7) for j in range(140)], 48))

    return [(i, s[0], s[1], s[2]) for i, s in enumerate(specs)]

if __name__ == "__main__":
    t0 = time.time()
    tasks = make_specs()
    print(f"Building {len(tasks)} BKZ-60 reductions -> {2*len(tasks)} GSO bases in parallel...")
    with Pool(24) as pool:
        pairs = pool.map(build_pair, tasks)
    print(f"Done in {time.time() - t0:.2f}s!")
