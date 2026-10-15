#!/usr/bin/env python3
import gzip, os, sys, time, math
from multiprocessing import Pool
import mpmath as mp
from mpmath.functions.rszeta import Rzeta_set

DPS = 40
GAMMA_MAX = mp.mpf("75000")
ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = ROOT if os.path.isdir(os.path.join(ROOT, "chunks")) else os.path.dirname(ROOT)
OUT_DIR = os.path.join(BASE_DIR, "chunks")
os.makedirs(OUT_DIR, exist_ok=True)


def zeta_and_deriv_half(ctx, t):
    wpinitial = ctx.prec
    s = ctx.mpf("0.5") + ctx.j * t
    ctx.prec = 53
    X = ctx.sqrt(abs(s))
    M1 = 2 * ctx.sqrt(t / (2 * ctx.pi))
    abst = abs(0.5 - s)
    T = 2 * abst * math.log(abst)
    wpbasic = max(
        6,
        3 + ctx.mag(t),
        2 + ctx.mag(2.12 * M1 + 21.2 * M1 * X + 1.3 * M1 * X * T) + wpinitial + 1,
    )
    wptheta = max(4, 3 + ctx.mag(2.7 * M1 * X) + wpinitial + 1)
    wpR = 3 + ctx.mag(1.1 + 2 * X) + wpinitial + 1
    ctx.prec = wptheta
    theta = ctx.siegeltheta(t)
    ps1 = (ctx._re(ctx.psi(0, s / 2))) / 2 - ctx.ln(ctx.pi) / 2
    ctx.prec = wpR
    xrz = Rzeta_set(ctx, s, [0, 1])
    yrz0 = ctx.conj(xrz[0])
    yrz1 = ctx.conj(xrz[1])
    ctx.prec = wpbasic
    exptheta = ctx.expj(-2 * theta)
    z0 = xrz[0] + exptheta * yrz0
    z1 = xrz[1] + exptheta * (-yrz1 - 2 * yrz0 * ps1)
    ctx.prec = wpinitial
    return +z0, +z1


def refine_chunk(args):
    chunk_id, gammas = args
    out_path = os.path.join(OUT_DIR, f"chunk_{chunk_id:06d}.txt")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        with open(out_path, "r") as f:
            if sum(1 for _ in f) == len(gammas):
                return chunk_id, len(gammas), True
    lines = []
    for g0 in gammas:
        t_float = float(g0)
        if t_float > 2000.0:
            with mp.workdps(18):
                t = mp.mpf(g0)
                z, zp = zeta_and_deriv_half(mp.mp, t)
                t = t - (z / (1j * zp)).real
            with mp.workdps(32):
                t = mp.mpf(t)
                z, zp = zeta_and_deriv_half(mp.mp, t)
                t = t - (z / (1j * zp)).real
            mp.mp.dps = DPS
            t = mp.mpf(t)
            z, zp = zeta_and_deriv_half(mp.mp, t)
            corr = z / (1j * zp)
            t = t - corr.real
            if abs(corr) >= mp.mpf("1e-28"):
                z, zp = zeta_and_deriv_half(mp.mp, t)
                corr = z / (1j * zp)
                t = t - corr.real
            s = mp.mpf("0.5") + 1j * t
            coeff = 1 / (s * zp)
        else:
            mp.mp.dps = DPS
            half = mp.mpf("0.5")
            t = mp.mpf(g0)
            for _ in range(3):
                s = half + 1j * t
                z = mp.zeta(s)
                zp = mp.zeta(s, derivative=1)
                corr = z / (1j * zp)
                t = t - corr.real
                if abs(corr) < mp.mpf("1e-28"):
                    break
            s = half + 1j * t
            zp = mp.zeta(s, derivative=1)
            coeff = 1 / (s * zp)
        re_c = coeff.real
        im_c = coeff.imag
        lines.append(
            f"{mp.nstr(t, 42, strip_zeros=False)} "
            f"{mp.nstr(re_c, 30, strip_zeros=False)} "
            f"{mp.nstr(im_c, 30, strip_zeros=False)}\n"
        )
    tmp = out_path + f".tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        f.writelines(lines)
    os.replace(tmp, out_path)
    return chunk_id, len(gammas), False


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    chunk = int(sys.argv[2]) if len(sys.argv) > 2 else 250
    seeds = []
    with gzip.open(os.path.join(BASE_DIR, "zeros1.gz"), "rt") as f:
        for line in f:
            s = line.strip()
            if not s or not (s[0].isdigit() or s[0] == "."):
                continue
            if float(s) > float(GAMMA_MAX):
                break
            seeds.append(s)
    tasks = []
    for i in range(0, len(seeds), chunk):
        out_path = os.path.join(OUT_DIR, f"chunk_{i:06d}.txt")
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            with open(out_path, "r") as f:
                if sum(1 for _ in f) == len(seeds[i : i + chunk]):
                    continue
        tasks.append((i, seeds[i : i + chunk]))
    print(f"seeds={len(seeds)} pending_chunks={len(tasks)} workers={workers}", flush=True)
    t0 = time.time()
    done = 0
    with Pool(workers) as p:
        for cid, n, skipped in p.imap_unordered(refine_chunk, tasks):
            done += 1
            if done % 5 == 0 or done == len(tasks):
                el = time.time() - t0
                print(
                    f"[{done}/{len(tasks)}] chunk={cid} el={el:.1f}s rate={done/el:.2f} chunks/s",
                    flush=True,
                )


if __name__ == "__main__":
    main()
