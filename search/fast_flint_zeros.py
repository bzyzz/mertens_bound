#!/usr/bin/env python3
import gzip, os, sys, time
from multiprocessing import Pool
import flint

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "chunks")
os.makedirs(OUT_DIR, exist_ok=True)


def refine_chunk(args):
    chunk_id, gammas = args
    out_path = os.path.join(OUT_DIR, f"chunk_{chunk_id:06d}.txt")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        with open(out_path, "r") as f:
            if sum(1 for _ in f) == len(gammas):
                return chunk_id, len(gammas), True

    half_22 = None
    lines = []
    for g0 in gammas:
        flint.ctx.dps = 22
        t = flint.arb(g0)
        s = flint.acb_series([flint.acb(flint.arb("0.5"), t), 1], prec=2).zeta()
        corr = (s[0] / (flint.acb(0, 1) * s[1])).real
        t1_str = (t - corr).mid().str(22, radius=False)

        flint.ctx.dps = 40
        t = flint.arb(t1_str)
        s = flint.acb_series([flint.acb(flint.arb("0.5"), t), 1], prec=2).zeta()
        dt1 = -(s[0] / (flint.acb(0, 1) * s[1])).real
        t_final = (t + dt1).mid()
        rho = flint.acb(flint.arb("0.5"), t_final)
        coeff = (1 / (rho * s[1])).mid()
        lines.append(
            f"{t_final.str(38, radius=False)} "
            f"{coeff.real.str(26, radius=False)} "
            f"{coeff.imag.str(26, radius=False)}\n"
        )

    tmp = out_path + f".tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        f.writelines(lines)
    os.replace(tmp, out_path)
    return chunk_id, len(gammas), False


def main():
    max_zeros = int(sys.argv[1]) if len(sys.argv) > 1 else 250000
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 112
    chunk = 250

    seeds = []
    with gzip.open(os.path.join(ROOT, "zeros6.gz"), "rt") as f:
        for line in f:
            s = line.strip()
            if not s or not (s[0].isdigit() or s[0] == "."):
                continue
            seeds.append(s)
            if len(seeds) >= max_zeros:
                break

    tasks = []
    for i in range(0, len(seeds), chunk):
        out_path = os.path.join(OUT_DIR, f"chunk_{i:06d}.txt")
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            with open(out_path, "r") as f:
                if sum(1 for _ in f) == len(seeds[i : i + chunk]):
                    continue
        tasks.append((i, seeds[i : i + chunk]))

    print(f"target_zeros={len(seeds)} pending_chunks={len(tasks)} workers={workers}", flush=True)
    if not tasks:
        print("All chunks already computed!", flush=True)
        return

    t0 = time.time()
    done = 0
    with Pool(workers) as p:
        for cid, n, skipped in p.imap_unordered(refine_chunk, tasks):
            done += 1
            if done % 25 == 0 or done == len(tasks):
                el = time.time() - t0
                rate = (done * chunk) / max(el, 1e-6)
                print(
                    f"[{done}/{len(tasks)}] chunk={cid:06d} el={el:.1f}s rate={rate:.0f} zeros/s",
                    flush=True,
                )


if __name__ == "__main__":
    main()
