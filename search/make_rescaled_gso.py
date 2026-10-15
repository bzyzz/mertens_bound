import os
import sys
import struct
import time
import numpy as np
import gmpy2
from fpylll import IntegerMatrix, LLL, BKZ, GSO, FPLLL

FPLLL.set_precision(300)
gmpy2.get_context().precision = 300

def read_ntl(path):
    with open(path, "r") as f:
        text = f.read()
    # Parse matrix [[...], [...]] and target [...]
    idx = text.rfind("\n[")
    mat_str = text[:idx]
    tgt_str = text[idx:]
    rows = []
    for line in mat_str.splitlines():
        line = line.strip()
        if line.startswith("["):
            content = line.strip("[] ")
            if content:
                rows.append([int(x.split(".")[0]) for x in content.split()])
    tgt = [int(x.split(".")[0]) for x in tgt_str.replace("[", " ").replace("]", " ").split()]
    return rows, tgt

def export_gso(rows, tgt, y_mult, out_path):
    n = len(rows)
    m = len(rows[0])
    # Exact Gram-Schmidt using gmpy2 mpfr (300 bits)
    B = [[gmpy2.mpfr(x) for x in r] for r in rows] + [[gmpy2.mpfr(x) for x in tgt]]
    mu = [[gmpy2.mpfr(0)] * (n + 1) for _ in range(n + 1)]
    r = [[gmpy2.mpfr(0)] * (n + 1) for _ in range(n + 1)]
    c = [gmpy2.mpfr(0)] * (n + 1)

    for i in range(n + 1):
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

    bstar_last = [gmpy2.mpfr(0)] * n
    for i in range(n):
        val = B[i][m - 1]
        for l in range(i):
            val -= mu[i][l] * bstar_last[l]
        bstar_last[i] = val

    scale_y1 = gmpy2.mpfr(y_mult) * gmpy2.exp2(30)
    rs_arr = np.array([float(c[i]) for i in range(n)], dtype=np.float64)
    sig_arr = np.array([float(mu[n][i]) for i in range(n)], dtype=np.float64)
    dsig_arr = np.array([float(scale_y1 * bstar_last[i] / c[i]) for i in range(n)], dtype=np.float64)
    mu_arr = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            mu_arr[i, j] = float(mu[i][j])

    lo_arr = np.zeros(n, dtype=np.uint64)
    hi_arr = np.zeros(n, dtype=np.uint64)
    MASK64 = (1 << 64) - 1
    MASK128 = (1 << 128) - 1
    k_y_cut = 0
    for i in range(n):
        xval = rows[i][m - 1] // y_mult
        if i == k_y_cut and abs(xval) <= (1 << 24):
            k_y_cut += 1
        u = xval & MASK128
        lo_arr[i] = u & MASK64
        hi_arr[i] = (u >> 64) & MASK64

    if os.path.dirname(out_path) == "" and os.path.isdir("gso"):
        out_path = os.path.join("gso", out_path)
    with open(out_path, "wb") as f:
        f.write(struct.pack("<i", n))
        f.write(rs_arr.tobytes())
        f.write(sig_arr.tobytes())
        f.write(dsig_arr.tobytes())
        f.write(mu_arr.tobytes())
        f.write(lo_arr.tobytes())
        f.write(hi_arr.tobytes())
    print(f"Exported {out_path} (k_y_cut={k_y_cut})", flush=True)

if __name__ == "__main__":
    in_file = sys.argv[1]
    out_file = sys.argv[2]
    arg3 = sys.argv[3]
    max_beta = int(sys.argv[4])

    t0 = time.time()
    rows, tgt = read_ntl(in_file)
    n = len(rows)
    m = len(rows[0])
    if arg3.startswith("s"):
        s_mult = int(arg3[1:])
        for i in range(n):
            rows[i][m - 1] *= s_mult
        tgt[m - 1] *= s_mult
        y_mult = s_mult * (1 << 30)
    else:
        b_shift = int(arg3)
        for i in range(n):
            if b_shift >= 0:
                rows[i][m - 1] <<= b_shift
            else:
                rows[i][m - 1] >>= (-b_shift)
        if b_shift >= 0:
            tgt[m - 1] <<= b_shift
        else:
            tgt[m - 1] >>= (-b_shift)
        y_mult = (1 << (30 + b_shift))

    if max_beta > 0:
        M = IntegerMatrix.from_matrix(rows)
        print(f"Running LLL on {in_file} (arg3={arg3})...", flush=True)
        LLL.reduction(M)
        betas = [b for b in [20, 30, 40, 48, 54, 60] if b <= max_beta]
        for beta in betas:
            print(f"Running BKZ-{beta}...", flush=True)
            param = BKZ.Param(block_size=beta, max_loops=3, strategies=BKZ.DEFAULT_STRATEGY, auto_abort=True)
            BKZ.reduction(M, param)
        rows = [[int(M[i, j]) for j in range(m)] for i in range(n)]

    export_gso(rows, tgt, y_mult, out_file)
    print(f"Done in {time.time() - t0:.2f}s", flush=True)
