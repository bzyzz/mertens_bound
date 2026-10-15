#!/usr/bin/env python3
"""
eval_tech2_hdrop_600k.py
Parses logs/tech2_hdrop_gh1055.log and logs/tech2_hdrop_gh1080.log,
reports total GSO nodes visited and per-basis statistics, and refines/evaluates
all sub-3.4849e18 candidates with |h_10k| >= 0.975 (and top 100 overall)
on zeros_600k.bin (N = 100,000, 300,000, and 600,000 zeros).
"""
import os
import re
import glob
import time
import struct
import ctypes
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))


def load_zeros_bin(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        gamma = np.frombuffer(f.read(n * 8), dtype=np.float64).copy()
        alpha = np.frombuffer(f.read(n * 8), dtype=np.float64).copy()
        raw = np.frombuffer(f.read(n * 48), dtype=np.uint64).reshape(n, 6)
        ghi_lo = np.ascontiguousarray(raw[:, 0])
        ghi_hi = np.ascontiguousarray(raw[:, 1])
        glo_lo = np.ascontiguousarray(raw[:, 2])
        glo_hi = np.ascontiguousarray(raw[:, 3])
        psi_lo = np.ascontiguousarray(raw[:, 4])
        psi_hi = np.ascontiguousarray(raw[:, 5])
    return n, gamma, alpha, ghi_lo, ghi_hi, glo_lo, glo_hi, psi_lo, psi_hi


def parse_tech2_logs():
    pat_hit = re.compile(
        r"\|h\|=([0-9.]+)\s+h=([+-]?[0-9.]+)\s+y=([0-9.]+)\s+\(x=([0-9]+),\s*dy=([+-]?[0-9.eE+-]+)\)"
    )
    pat_done = re.compile(
        r"ALL DONE in ([0-9.]+) s \| nodes=([0-9]+) \(([0-9.]+) Gnodes/s\) \| pruned_at_cut=([0-9]+) \| unique_clusters=([0-9]+) \| best_abs=([0-9.]+)"
    )
    pat_file = re.compile(
        r"\[(gso/hdrop_[0-9]+_(?:pos|neg)\.gso) \([0-9]+/[0-9]+\) yc=[0-9.]+e18 \([0-9]+/[0-9]+\)\] ([0-9.]+) s \| new_clusters=([0-9]+) \| total_unique=([0-9]+) \| best=([0-9.]+)"
    )

    clusters = {}
    log_summaries = {}
    log_files = [
        os.path.join(ROOT, "logs", "tech2_hdrop_gh1055.log"),
        os.path.join(ROOT, "logs", "tech2_hdrop_gh1080.log"),
    ]
    for lf in log_files:
        if not os.path.exists(lf):
            continue
        bname = os.path.basename(lf)
        cur_hits = []
        file_stats = []
        done_info = None
        with open(lf, "r", errors="ignore") as f:
            for line in f:
                m = pat_hit.search(line)
                if m:
                    abs_h = float(m.group(1))
                    h_val = float(m.group(2))
                    y_val = float(m.group(3))
                    x_int = int(m.group(4))
                    dy0 = float(m.group(5))
                    cur_hits.append((abs_h, h_val, y_val, x_int, dy0))
                    if 0 < y_val < 3.4849e18:
                        key = x_int >> 32
                        if key not in clusters or abs_h > clusters[key][0]:
                            clusters[key] = (abs_h, h_val, y_val, x_int, dy0, bname)
                    continue
                mf = pat_file.search(line)
                if mf:
                    file_stats.append((mf.group(1), float(mf.group(2)), int(mf.group(3)), int(mf.group(4)), float(mf.group(5))))
                    continue
                md = pat_done.search(line)
                if md:
                    done_info = (
                        float(md.group(1)),
                        int(md.group(2)),
                        float(md.group(3)),
                        int(md.group(4)),
                        int(md.group(5)),
                        float(md.group(6)),
                    )
        log_summaries[bname] = (done_info, file_stats, len(cur_hits))

    return sorted(clusters.values(), key=lambda t: -t[0]), log_summaries


def main():
    t0 = time.time()
    cands, log_summaries = parse_tech2_logs()

    print("=== TECHNIQUE 2 SWEEP SUMMARY ===")
    tot_nodes_all = 0
    tot_sec_all = 0.0
    for bname, (done_info, file_stats, n_hits) in log_summaries.items():
        if done_info:
            sec, nodes, gnps, pruned, uniq, best_abs = done_info
            tot_nodes_all += nodes
            tot_sec_all += sec
            print(
                f"  {bname}: {len(file_stats)} files | {sec:.2f}s | nodes={nodes:,} ({gnps:.2f} Gnodes/s) | "
                f"pruned_at_cut={pruned:,} | unique_clusters={uniq:,} | best_10k={best_abs:.8f}"
            )
        else:
            print(f"  {bname}: {len(file_stats)} files completed so far, {n_hits} hits logged")
    print(f"  TOTAL GSO NODES VISITED ACROSS TECH2 SWEEPS: {tot_nodes_all:,} ({tot_nodes_all/1e9:.2f} Gnodes in {tot_sec_all:.2f}s)\n")

    cands_975 = [c for c in cands if c[0] >= 0.975]
    print(f"Unique sub-3.4849e18 clusters discovered in Tech2 logs: {len(cands)}")
    print(f"Candidates with |h_10k| >= 0.975 and y < 3.4849e18: {len(cands_975)}")
    for i, (ah, hv, yv, xi, dy, src) in enumerate(cands_975):
        print(f"  #{i+1:2d}: |h_10k|={ah:.8f} (h={hv:+.8f})  y={yv:.9e}  x={xi}  dy={dy:+.7f}  [{src}]")

    if not cands:
        return

    n_zeros, gamma, alpha, ghi_lo, ghi_hi, glo_lo, glo_hi, psi_lo, psi_hi = load_zeros_bin(
        os.path.join(ROOT, "zeros_600k.bin")
    )
    print(f"\nLoaded zeros_600k.bin: N={n_zeros}, gamma_max={gamma[-1]:.6f} ({time.time()-t0:.2f}s)")

    lib = ctypes.CDLL(os.path.join(ROOT, "libpath1_eval.so"))
    u64_p = ctypes.POINTER(ctypes.c_uint64)
    f64_p = ctypes.POINTER(ctypes.c_double)

    lib.compute_peak_AB.argtypes = [
        ctypes.c_int, u64_p, u64_p, f64_p,
        ctypes.c_int, f64_p, f64_p, u64_p, u64_p, u64_p, u64_p, u64_p, u64_p,
        f64_p, f64_p,
    ]
    lib.polish_peaks_with_weights.argtypes = [
        ctypes.c_int, ctypes.c_int, f64_p, f64_p, f64_p, f64_p,
        ctypes.c_double, ctypes.c_int, f64_p, f64_p, f64_p,
    ]

    top_cands = [c for c in cands if c[0] >= 0.960]
    if len(top_cands) < min(100, len(cands)):
        top_cands = cands[: min(100, len(cands))]
    n_p = len(top_cands)
    print(f"Refining and evaluating {n_p} candidates (including all {len(cands_975)} with |h_10k| >= 0.975) on N=600,000 zeros...")

    x_lo_arr = np.array([c[3] & 0xFFFFFFFFFFFFFFFF for c in top_cands], dtype=np.uint64)
    x_hi_arr = np.array([c[3] >> 64 for c in top_cands], dtype=np.uint64)
    dy0_arr = np.array([c[4] for c in top_cands], dtype=np.float64)

    A = np.empty((n_p, n_zeros), dtype=np.float64)
    B = np.empty((n_p, n_zeros), dtype=np.float64)

    t1 = time.time()
    lib.compute_peak_AB(
        n_p,
        x_lo_arr.ctypes.data_as(u64_p),
        x_hi_arr.ctypes.data_as(u64_p),
        dy0_arr.ctypes.data_as(f64_p),
        n_zeros,
        alpha.ctypes.data_as(f64_p),
        gamma.ctypes.data_as(f64_p),
        ghi_lo.ctypes.data_as(u64_p),
        ghi_hi.ctypes.data_as(u64_p),
        glo_lo.ctypes.data_as(u64_p),
        glo_hi.ctypes.data_as(u64_p),
        psi_lo.ctypes.data_as(u64_p),
        psi_hi.ctypes.data_as(u64_p),
        A.ctypes.data_as(f64_p),
        B.ctypes.data_as(f64_p),
    )
    print(f"Computed A, B matrix ({n_p} x {n_zeros}) in {time.time()-t1:.2f}s")

    T_max = gamma[-1]
    u = gamma / T_max
    w_jp_600k = np.ascontiguousarray((1.0 - u) * np.cos(np.pi * u) + np.sin(np.pi * u) / np.pi, dtype=np.float64)
    w_g15_600k = np.ascontiguousarray(np.exp(-1.5e-10 * gamma * gamma), dtype=np.float64)
    w_g10_600k = np.ascontiguousarray(np.exp(-1.0e-10 * gamma * gamma), dtype=np.float64)
    w_raw_600k = np.ones(n_zeros, dtype=np.float64)

    def eval_weights(w_vec):
        h_unpol = np.empty(n_p, dtype=np.float64)
        h_pol = np.empty(n_p, dtype=np.float64)
        ddy_pol = np.empty(n_p, dtype=np.float64)
        lib.polish_peaks_with_weights(
            n_p, n_zeros,
            A.ctypes.data_as(f64_p),
            B.ctypes.data_as(f64_p),
            gamma.ctypes.data_as(f64_p),
            w_vec.ctypes.data_as(f64_p),
            2.5e-4, 40,
            h_unpol.ctypes.data_as(f64_p),
            h_pol.ctypes.data_as(f64_p),
            ddy_pol.ctypes.data_as(f64_p),
        )
        return h_unpol, h_pol, ddy_pol

    _, h_jp600, ddy_jp600 = eval_weights(w_jp_600k)
    _, h_g15, ddy_g15 = eval_weights(w_g15_600k)
    _, h_g10, ddy_g10 = eval_weights(w_g10_600k)
    _, h_raw600, ddy_raw600 = eval_weights(w_raw_600k)

    print("\n=== ALL CANDIDATES WITH |h_10k| >= 0.975 EVALUATED AT N = 600,000 ZEROS ===")
    print(f"{'Rank':>4s}  {'|h_10k|':>10s}  {'|h_best600k|':>12s}  {'h_JP(600k)':>12s}  {'h_G15(600k)':>12s}  {'h_G10(600k)':>12s}  {'h_raw(600k)':>12s}  {'y':>16s}  {'Source'}")
    for i, c in enumerate(top_cands):
        ah10, hv10, yv, xi, dy, src = c
        if ah10 >= 0.975:
            best600 = max(abs(h_jp600[i]), abs(h_g15[i]), abs(h_g10[i]))
            print(
                f"{i+1:4d}  {ah10:10.8f}  {best600:12.8f}  {h_jp600[i]:+12.8f}  "
                f"{h_g15[i]:+12.8f}  {h_g10[i]:+12.8f}  {h_raw600[i]:+12.8f}  {yv:16.9e}  {src}"
            )

    results = []
    for i, c in enumerate(top_cands):
        ah10, hv10, yv, xi, dy, src = c
        best600 = max(abs(h_jp600[i]), abs(h_g15[i]), abs(h_g10[i]))
        results.append((
            best600, abs(h_jp600[i]), h_jp600[i], ddy_jp600[i],
            abs(h_g15[i]), h_g15[i], ddy_g15[i],
            abs(h_g10[i]), h_g10[i], ddy_g10[i],
            h_raw600[i], ah10, hv10, yv, xi, dy, src,
        ))

    results.sort(key=lambda t: -t[0])
    print("\n=== TOP 25 TECH2 SUB-3.4849e18 PEAKS BY |h_best(600k)| ===")
    print(f"{'Rank':>4s}  {'|h_best600k|':>12s}  {'h_JP(600k)':>12s}  {'h_G15(600k)':>12s}  {'h_G10(600k)':>12s}  {'h_raw(600k)':>12s}  {'|h_10k|':>10s}  {'y':>16s}  {'x_int'}")
    for idx, r in enumerate(results[:25]):
        best600, ajp, hjp, djp, ag15, hg15, dg15, ag10, hg10, dg10, hraw, ah10, hv10, yv, xi, dy, src = r
        print(
            f"{idx+1:4d}  {best600:12.8f}  {hjp:+12.8f}  {hg15:+12.8f}  "
            f"{hg10:+12.8f}  {hraw:+12.8f}  {ah10:10.8f}  {yv:16.9e}  {xi}"
        )


if __name__ == "__main__":
    main()
