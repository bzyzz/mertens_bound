#!/usr/bin/env python3
"""
eval_tech1_600k.py
Parse Technique 1 sweep logs (logs/tech1_heavy_shifts.log, logs/tech1_3flips.log, logs/tech1_4flips.log),
summarize cosets swept and total GSO nodes visited, and refine/evaluate the top sub-Peak2 candidates
against 600,000 zeta zeros (zeros_600k.bin) via libpath1_eval.so.
"""
import ctypes
import glob
import os
import re
import struct
import sys
import time
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))

LIB_PATH = os.path.join(ROOT, "libpath1_eval.so")
ZEROS_600K = os.path.join(ROOT, "zeros_600k.bin")
LOG_FILES = [
    ("Sweep 1 (heavy-shifts)", os.path.join(ROOT, "logs/tech1_heavy_shifts.log")),
    ("Sweep 2 (3flips)",       os.path.join(ROOT, "logs/tech1_3flips.log")),
    ("Sweep 3 (4flips)",       os.path.join(ROOT, "logs/tech1_4flips.log")),
]

CLUSTER_RE = re.compile(
    r"CLUSTER #(\d+)\s+\[coset=([^\]]+)\]\s+\|h\|=([0-9.]+)\s+h=([+\-0-9.]+)\s+m64=([0-9.]+)\s+y=([0-9.]+)\s+\(x=(\d+),\s*dy=([+\-0-9.]+)\)"
)
DONE_FILE_RE = re.compile(
    r"\[([^\s]+)\s+yc=([0-9.]+e18)\s+\|\s+(\d+)\s+cosets\s+\|\s+(\d+)\s+prefixes\]\s+([0-9.]+)\s+s\s+\|\s+nodes=([0-9.]+)B\s+\(([0-9.]+)\s+Gnodes/s\)\s+\|\s+new_clusters=(\d+)"
)
ALL_DONE_RE = re.compile(
    r"ALL DONE in ([0-9.]+) s \| nodes=(\d+) \(([0-9.]+)B, ([0-9.]+) Gnodes/s\) \| pruned_at_cut=(\d+) \| unique_clusters=(\d+) \| best_abs=([0-9.]+)"
)


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


def parse_logs():
    sweep_stats = []
    all_candidates = []

    for label, path in LOG_FILES:
        if not os.path.exists(path):
            print(f"[WARN] Log not found: {path}")
            continue
        file_runs = []
        summary = None
        cands_in_log = 0
        cands_960 = 0
        cands_975 = 0

        with open(path, "r") as f:
            for line in f:
                m_c = CLUSTER_RE.search(line)
                if m_c:
                    cid = int(m_c.group(1))
                    coset = m_c.group(2)
                    ah = float(m_c.group(3))
                    h_val = float(m_c.group(4))
                    m64 = float(m_c.group(5))
                    y_str = m_c.group(6)
                    x_int = int(m_c.group(7))
                    dy = float(m_c.group(8))
                    cands_in_log += 1
                    if ah >= 0.960:
                        cands_960 += 1
                    if ah >= 0.975:
                        cands_975 += 1
                    all_candidates.append({
                        "sweep": label,
                        "cid": cid,
                        "coset": coset,
                        "ah_71k": ah,
                        "h_71k": h_val,
                        "m64": m64,
                        "y_str": y_str,
                        "x": x_int,
                        "dy": dy,
                    })
                    continue

                m_f = DONE_FILE_RE.search(line)
                if m_f:
                    file_runs.append({
                        "gso": m_f.group(1),
                        "yc": m_f.group(2),
                        "cosets": int(m_f.group(3)),
                        "prefixes": int(m_f.group(4)),
                        "sec": float(m_f.group(5)),
                        "nodes_B": float(m_f.group(6)),
                        "gnodes_s": float(m_f.group(7)),
                        "new_clusters": int(m_f.group(8)),
                    })
                    continue

                m_a = ALL_DONE_RE.search(line)
                if m_a:
                    summary = {
                        "sec": float(m_a.group(1)),
                        "nodes": int(m_a.group(2)),
                        "nodes_B": float(m_a.group(3)),
                        "gnodes_s": float(m_a.group(4)),
                        "pruned": int(m_a.group(5)),
                        "unique_clusters": int(m_a.group(6)),
                        "best_abs": float(m_a.group(7)),
                    }

        # Deduplicate file_runs if printed to both stdout and stderr
        unique_runs = []
        seen_keys = set()
        for r in file_runs:
            k = (r["gso"], r["yc"], r["prefixes"])
            if k not in seen_keys:
                seen_keys.add(k)
                unique_runs.append(r)

        sweep_stats.append({
            "label": label,
            "path": path,
            "runs": unique_runs,
            "summary": summary,
            "cands_960": cands_960,
            "cands_975": cands_975,
        })

    return sweep_stats, all_candidates


def deduplicate_candidates(cands):
    # Sort descending by ah_71k
    cands_sorted = sorted(cands, key=lambda c: c["ah_71k"], reverse=True)
    unique = []
    # 1 unit of y is 2^30 in x; deduplicate within dy = 0.05 -> dx = 0.05 * 2^30 ~ 5.37e7
    THRESH_X = int(0.05 * (1 << 30))
    for c in cands_sorted:
        peak_x = c["x"] + int(round(c["dy"] * (1 << 30)))
        c["peak_x"] = peak_x
        if any(abs(peak_x - u["peak_x"]) < THRESH_X for u in unique):
            continue
        unique.append(c)
    return unique


def main():
    sweep_stats, all_candidates = parse_logs()

    print("=" * 105)
    print("TECHNIQUE 1 SWEEP SUMMARY")
    print("=" * 105)
    grand_nodes = 0
    grand_sec = 0.0
    grand_clusters = 0
    for st in sweep_stats:
        print(f"\n--- {st['label']} ({st['path']}) ---")
        for r in st["runs"]:
            print(f"  [{r['gso']} yc={r['yc']}] cosets={r['cosets']} prefixes={r['prefixes']} "
                  f"time={r['sec']:.2f}s nodes={r['nodes_B']:.3f}B ({r['gnodes_s']:.2f} Gnodes/s) new_clusters={r['new_clusters']}")
        if st["summary"]:
            s = st["summary"]
            grand_nodes += s["nodes"]
            grand_sec += s["sec"]
            grand_clusters += s["unique_clusters"]
            print(f"  TOTAL: time={s['sec']:.2f}s | nodes={s['nodes']:,} ({s['nodes_B']:.3f}B, {s['gnodes_s']:.2f} Gnodes/s) "
                  f"| unique_clusters={s['unique_clusters']:,} | hits>=0.960: {st['cands_960']} | hits>=0.975: {st['cands_975']} | best_71k={s['best_abs']:.8f}")

    print("\n" + "=" * 105)
    print(f"GRAND TOTAL ACROSS SWEEPS: time={grand_sec:.2f}s | nodes={grand_nodes:,} ({grand_nodes/1e9:.3f}B) | unique_clusters={grand_clusters:,}")
    print("=" * 105)

    unique_cands = deduplicate_candidates(all_candidates)
    cands_960 = [c for c in unique_cands if c["ah_71k"] >= 0.960]
    hits_975 = [c for c in unique_cands if c["ah_71k"] >= 0.975]
    print(f"\nTotal logged cluster lines: {len(all_candidates)} -> Unique peaks: {len(unique_cands)}")
    print(f"Unique peaks with |h| >= 0.960: {len(cands_960)}")
    print(f"Unique peaks with |h| >= 0.975 and y < 3.4849e18: {len(hits_975)}")

    to_eval = unique_cands
    if not to_eval:
        print("No candidates found.")
        return

    t0 = time.time()
    n_zeros, gamma, alpha, ghi_lo, ghi_hi, glo_lo, glo_hi, psi_lo, psi_hi = load_zeros_bin(ZEROS_600K)
    print(f"\nLoaded {ZEROS_600K}: N={n_zeros}, gamma_max={gamma[-1]:.6f} ({time.time()-t0:.2f}s)")

    lib = ctypes.CDLL(LIB_PATH)
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

    n_p = len(to_eval)
    x_lo_arr = np.array([c["x"] & 0xFFFFFFFFFFFFFFFF for c in to_eval], dtype=np.uint64)
    x_hi_arr = np.array([(c["x"] >> 64) & 0xFFFFFFFFFFFFFFFF for c in to_eval], dtype=np.uint64)
    dy0_arr = np.array([c["dy"] for c in to_eval], dtype=np.float64)

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
    print(f"Computed exact A, B matrix ({n_p} x {n_zeros}) in {time.time()-t1:.2f}s")

    T_max = gamma[-1]
    u = gamma / T_max
    w_jp_600k = np.ascontiguousarray((1.0 - u) * np.cos(np.pi * u) + np.sin(np.pi * u) / np.pi, dtype=np.float64)
    w_g15_600k = np.ascontiguousarray(np.exp(-1.5e-10 * gamma * gamma), dtype=np.float64)
    w_g10_600k = np.ascontiguousarray(np.exp(-1.0e-10 * gamma * gamma), dtype=np.float64)
    w_raw_600k = np.ones(n_zeros, dtype=np.float64)
    w_10k = np.zeros(n_zeros, dtype=np.float64)
    w_10k[:10000] = 1.0

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
    _, h_10k_pol, ddy_10k = eval_weights(w_10k)

    for i, c in enumerate(to_eval):
        c["h_jp600"] = float(h_jp600[i])
        c["h_g15"] = float(h_g15[i])
        c["h_g10"] = float(h_g10[i])
        c["h_raw600"] = float(h_raw600[i])
        c["h_10k"] = float(h_10k_pol[i])
        c["ah_10k"] = abs(float(h_10k_pol[i]))
        c["best600"] = max(abs(c["h_jp600"]), abs(c["h_g15"]), abs(c["h_g10"]), abs(c["h_raw600"]))
        best_dy = c["dy"] + float(ddy_raw600[i])
        x_int = c["x"]
        y_int = x_int >> 30
        y_frac = (x_int & ((1 << 30) - 1)) / float(1 << 30) + best_dy
        while y_frac < 0.0:
            y_int -= 1
            y_frac += 1.0
        while y_frac >= 1.0:
            y_int += 1
            y_frac -= 1.0
        c["y_exact"] = f"{y_int}.{int(round(y_frac * 1e9)):09d}"

    hits_10k_975 = [c for c in to_eval if c["ah_10k"] >= 0.975 or c["ah_71k"] >= 0.975]
    print(f"\nCandidates with |h_10k| >= 0.975 or |h_71k| >= 0.975: {len(hits_10k_975)}")

    to_eval_by_71k = sorted(to_eval, key=lambda c: c["ah_71k"], reverse=True)
    print("\n" + "=" * 145)
    print("TOP SUB-3.4849e18 PEAKS BY |h_71k| EVALUATED AT N = 600,000 ZEROS")
    print("=" * 145)
    print(f"{'Rank':<5} {'|h_71k|':<11} {'|h_10k|':<11} {'|h_best600k|':<13} {'h_raw(600k)':<13} {'h_JP(600k)':<13} {'h_G10(600k)':<13} {'y (exact < 3.4849e18)':<30} {'coset':<26} {'sweep'}")
    print("-" * 145)
    for i, c in enumerate(to_eval_by_71k[:25], 1):
        print(f"#{i:<4d} {c['ah_71k']:<11.8f} {c['ah_10k']:<11.8f} {c['best600']:<13.8f} {c['h_raw600']:<+13.8f} "
              f"{c['h_jp600']:<+13.8f} {c['h_g10']:<+13.8f} {c['y_exact']:<30} {c['coset']:<26} {c['sweep']}")
    print("=" * 145)

    to_eval_by_600k = sorted(to_eval, key=lambda c: c["best600"], reverse=True)
    print("\n" + "=" * 145)
    print("TOP SUB-3.4849e18 PEAKS BY |h_best(600k)| EVALUATED AT N = 600,000 ZEROS")
    print("=" * 145)
    print(f"{'Rank':<5} {'|h_best600k|':<13} {'h_raw(600k)':<13} {'h_JP(600k)':<13} {'h_G10(600k)':<13} {'|h_71k|':<11} {'|h_10k|':<11} {'y (exact < 3.4849e18)':<30} {'coset':<26} {'sweep'}")
    print("-" * 145)
    for i, c in enumerate(to_eval_by_600k[:25], 1):
        print(f"#{i:<4d} {c['best600']:<13.8f} {c['h_raw600']:<+13.8f} {c['h_jp600']:<+13.8f} {c['h_g10']:<+13.8f} "
              f"{c['ah_71k']:<11.8f} {c['ah_10k']:<11.8f} {c['y_exact']:<30} {c['coset']:<26} {c['sweep']}")
    print("=" * 145)


if __name__ == "__main__":
    main()
