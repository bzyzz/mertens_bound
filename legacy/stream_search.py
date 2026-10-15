"""Long pruned enumeration at the Gaussian radius, scored as points arrive."""
import math
import os
import struct
import subprocess
import time
from multiprocessing import Pool

import gmpy2
from fpylll import BKZ, GSO, LLL

from diverse_enum import ITEMS, PI, fast_h
from lattice_search import build_lattice, gs_target, partial_h

ROOT = os.path.dirname(os.path.abspath(__file__))
ENUM = os.path.join(ROOT, "enum_cvp")


def kn_prune(n, lin=3):
    prune = []
    for k in range(n):
        i = n - k
        rr1 = i / n
        rr2 = (lin - 1) * math.sqrt(i * (n - i) / (n / 2 + 1)) / n
        prune.append(min(1.0, rr1 + rr2))
    return prune


def gaussian_radius(rs, gh):
    n = len(rs)
    logdet = sum(0.5 * math.log(r) for r in rs)
    heur = math.lgamma(1 + n / 2) / n - 0.5 * math.log(math.pi)
    return math.exp(logdet / n) * math.exp(heur) * gh


def one(job):
    N, nu, nu_y, nu_t, negative, gh, node_cap = job
    t0 = time.time()
    B, target = build_lattice(ITEMS, N, nu, nu_y, nu_t, PI, negative)
    LLL.reduction(B)
    BKZ.reduction(B, BKZ.Param(40, max_loops=1))
    M = GSO.Mat(B, float_type="dpe")
    M.update_gso()
    n = N + 1
    rs = [M.get_r(i, i) for i in range(n)]
    sigma = gs_target(B, target, M)
    mu = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i):
            mu[i][j] = M.get_mu(i, j)
    radius2 = gaussian_radius(rs, gh) ** 2
    step = 1 << nu_t
    xcol = []
    for i in range(n):
        last = int(B[i, n - 1])
        if last % step != 0:
            return f"bad-col {job}"
        xcol.append(last // step)
        if abs(xcol[-1]).bit_length() > 120:
            return f"wide-col {xcol[-1].bit_length()} {job}"
    bucket = max(1, int(0.02 * 2 ** (nu - nu_y)))
    prune = kn_prune(n)
    blob = struct.pack("<iii", n, 0, 200000)
    blob += struct.pack("<d", float(radius2))
    blob += struct.pack("<" + "d" * n, *[float(z) for z in rs])
    blob += struct.pack("<" + "d" * n, *[float(z) for z in sigma])
    flat = []
    for j in range(n):
        flat.extend(float(mu[j][i]) for i in range(n))
    blob += struct.pack("<" + "d" * (n * n), *flat)
    blob += struct.pack("<" + "d" * n, *prune)
    blob += struct.pack("<q", int(bucket))
    lo, hi = [], []
    for z in xcol:
        zmod = int(z) & ((1 << 128) - 1)
        lo.append(zmod & ((1 << 64) - 1))
        hi.append((zmod >> 64) & ((1 << 64) - 1))
    blob += struct.pack("<" + "Q" * n, *lo)
    blob += struct.pack("<" + "Q" * n, *hi)
    env = os.environ.copy()
    env["ENUM_NODE_CAP"] = str(node_cap)
    proc = subprocess.Popen(
        [ENUM], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env
    )
    proc.stdin.write(blob)
    proc.stdin.close()
    best = None
    small = None
    nsol = 0
    scale = gmpy2.exp2(nu_y - nu)
    for raw in proc.stderr:
        line = raw.decode().strip()
        if line.startswith("progress"):
            msg = f"{job} {line} best={None if best is None else round(best[0], 4)} t={time.time()-t0:.0f}"
            print(msg, flush=True)
            continue
        if not line.startswith("X "):
            if line.startswith("nodes"):
                print(f"{job} {line}", flush=True)
            continue
        x = int(line[2:])
        if x <= 0:
            continue
        y = gmpy2.mpfr(x) * scale
        if not (gmpy2.mpfr("1e12") < y < gmpy2.mpfr("1e24")):
            continue
        nsol += 1
        h = fast_h(y)
        if best is None or abs(h) > abs(best[0]):
            best = (h, str(y))
            print(f"HIT {h:.5f} y={y} job={job} n={nsol}", flush=True)
            if abs(h) > 0.985:
                full = partial_h(y, ITEMS, len(ITEMS))
                print(f"FULL {full} y={y} job={job}", flush=True)
                with open(os.path.join(ROOT, "stream_hits.txt"), "a") as f:
                    f.write(f"FULL {full} y={y} job={job}\n")
        if y < gmpy2.mpfr("1.95e19") and (small is None or abs(h) > abs(small[0])):
            small = (h, str(y))
            print(f"SMALL {h:.5f} y={y} job={job}", flush=True)
    proc.wait()
    return f"done {job} nsol={nsol} best={best} small={small} t={time.time()-t0:.0f}"


def main():
    cap = int(os.environ.get("STREAM_CAP", "20000000000"))
    jobs = []
    for nu_y in (100, 98, 96, 94):
        jobs.append((140, 130, nu_y, 30, True, 1.35, cap))
        jobs.append((140, 130, nu_y, 30, False, 1.35, cap))
    print(f"jobs {len(jobs)} cap {cap}", flush=True)
    with Pool(len(jobs)) as pool:
        for msg in pool.imap_unordered(one, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
