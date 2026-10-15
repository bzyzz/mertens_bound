#!/usr/bin/env python3
"""
eval_sub35_candidates.py
Evaluates all discovered sub-3.4849e18 candidate peaks across logs/*.log
on N = 100,000, 300,000, and 600,000 zeros (zeros_600k.bin) using libpath1_eval.so.
"""
import os, re, glob, time, struct, ctypes
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

def parse_logs():
    pat = re.compile(
        r"\|h\|=([0-9.]+)\s+h=([+-]?[0-9.]+)\s+y=([0-9.]+)\s+\(x=([0-9]+),\s*dy=([+-]?[0-9.eE+-]+)\)"
    )
    clusters = {}
    log_files = sorted(glob.glob(os.path.join(ROOT, "logs", "*.log")))
    for lf in log_files:
        with open(lf, "r", errors="ignore") as f:
            for line in f:
                m = pat.search(line)
                if m:
                    abs_h = float(m.group(1))
                    h_val = float(m.group(2))
                    y_val = float(m.group(3))
                    x_int = int(m.group(4))
                    dy0 = float(m.group(5))
                    if 0 < y_val < 3.4849e18:
                        key = x_int >> 32
                        if key not in clusters or abs_h > clusters[key][0]:
                            clusters[key] = (abs_h, h_val, y_val, x_int, dy0, os.path.basename(lf))
    return sorted(clusters.values(), key=lambda t: -t[0])

def main():
    t0 = time.time()
    cands = parse_logs()
    print(f"Parsed {len(cands)} unique clusters in y < 3.4849e18 across logs/*.log")
    if not cands:
        return

    print("Top 25 by 10k-zero |h|:")
    for i, (ah, hv, yv, xi, dy, src) in enumerate(cands[:25]):
        print(f"  #{i+1:2d}: |h_10k|={ah:.8f} (h={hv:+.8f})  y={yv:.6e}  x={xi}  dy={dy:+.7f}  [{src}]")

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
        f64_p, f64_p
    ]
    lib.polish_peaks_with_weights.argtypes = [
        ctypes.c_int, ctypes.c_int, f64_p, f64_p, f64_p, f64_p,
        ctypes.c_double, ctypes.c_int, f64_p, f64_p, f64_p
    ]

    # Evaluate all candidates with |h_10k| >= 0.960 (or top 250)
    top_cands = [c for c in cands if c[0] >= 0.960]
    if len(top_cands) < min(200, len(cands)):
        top_cands = cands[:min(200, len(cands))]
    n_p = len(top_cands)
    print(f"Evaluating {n_p} candidates on N=600,000 zeros...")

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

    # Kernel weights on N = 600,000
    T_max = gamma[-1]
    u = gamma / T_max
    w_jp_600k = np.ascontiguousarray((1.0 - u) * np.cos(np.pi * u) + np.sin(np.pi * u) / np.pi, dtype=np.float64)
    w_g15_600k = np.ascontiguousarray(np.exp(-1.5e-10 * gamma * gamma), dtype=np.float64)
    w_g10_600k = np.ascontiguousarray(np.exp(-1.0e-10 * gamma * gamma), dtype=np.float64)

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

    results = []
    for i, c in enumerate(top_cands):
        ah10, hv10, yv, xi, dy, src = c
        best600 = max(abs(h_jp600[i]), abs(h_g15[i]), abs(h_g10[i]))
        results.append((
            best600, abs(h_jp600[i]), h_jp600[i], ddy_jp600[i],
            abs(h_g15[i]), h_g15[i], ddy_g15[i],
            abs(h_g10[i]), h_g10[i], ddy_g10[i],
            ah10, yv, xi, dy, src
        ))

    results.sort(key=lambda t: -t[0])
    print("\n=== TOP 30 SUB-3.4849e18 PEAKS EVALUATED AT N = 600,000 ZEROS ===")
    print(f"{'Rank':>4s}  {'|h_best600k|':>12s}  {'h_JP(600k)':>12s}  {'h_G15(600k)':>12s}  {'h_G10(600k)':>12s}  {'|h_10k|':>10s}  {'y':>14s}  {'x_int'}")
    for idx, r in enumerate(results[:30]):
        best600, ajp, hjp, djp, ag15, hg15, dg15, ag10, hg10, dg10, ah10, yv, xi, dy, src = r
        print(f"{idx+1:4d}  {best600:12.8f}  {hjp:+12.8f}  {hg15:+12.8f}  {hg10:+12.8f}  {ah10:10.8f}  {yv:14.6e}  {xi} (dy={dy+dg15:+.8f})")

if __name__ == "__main__":
    main()
