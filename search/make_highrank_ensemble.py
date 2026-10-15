#!/usr/bin/env python3
"""
Generate 24 Pairs (48 GSO files) of High-Rank 59/60-Zero Multi-Band Covering Lattices
(hr_00..hr_23, M_act in {59, 60}, k_y_cut = 1) over pool j in [0..67].

Why High-Rank M_act = 59..60?
1. Exact band decomposition of h_StR(y) shows that sum_{j=0}^{50} 2*alpha_j^* = 0.9830 < 1.000,
   whereas sum_{j=0}^{67} 2*alpha_j^* = 1.0700. Every known counterexample |h_StR(y)| > 1.0
   derives +0.040 to +0.060 from zeros j in [51..67].
2. By the Gaussian Heuristic on the M-dimensional torus with N_indep ~ 5e20,
   sqrt(M / (2*pi*e)) * N_indep^(-1/M) < 1 holds up to M = 68. Using a flat weight exponent
   beta in [0.14, 0.18] and a minimum effective period floor (floor_r >= 1.00) prevents
   BKZ-60 from ejecting singleton period vectors, keeping all 59-60 zeros active in the
   macro Diophantine lattice (k_y_cut = 1).
3. With the dim_scale = 1.2935 bug fixed, increasing M_act from 51 to 59-60 reduces the
   normalized distance tot / (1.2935 * gh_act) of true multi-band peaks from 1.11-1.19
   down to 0.88-1.01, while rejecting 51-zero false positives that lack mid-band alignment.
"""
import math
import multiprocessing as mp
import time
from decimal import Decimal, getcontext
import numpy as np
from fpylll import IntegerMatrix, LLL, BKZ, FPLLL
from make_pure_compact_ensemble import W_ORIG, G_ORIG, T1_ORIG, T2_ORIG, TWO_ASTAR, export_both_gso

FPLLL.set_precision(300)
getcontext().prec = 60


def compute_mults(sel, beta=0.16, floor_r=1.00):
    w_ref = float(W_ORIG[sel[-1]])
    mults = []
    for j in sel:
        r = max((float(W_ORIG[j]) / w_ref) ** (2.0 * beta), floor_r)
        m = int(round(r * w_ref * 1024.0 / float(W_ORIG[j])))
        mults.append(max(1, m))
    return np.array(mults, dtype=np.int64)


def design_highrank_specs():
    rng = np.random.default_rng(20261015)
    M_POOL = 68
    two_a = TWO_ASTAR[:M_POOL]
    w_orig = np.array([float(W_ORIG[j]) for j in range(M_POOL)])
    g_centered = np.array([
        float((G_ORIG[j] % W_ORIG[j]) - (W_ORIG[j] if (G_ORIG[j] % W_ORIG[j]) > W_ORIG[j] // 2 else 0))
        for j in range(M_POOL)
    ])

    def make_fast_eval(sel, beta=0.16, floor_r=1.00, sm=4):
        sel_arr = np.array(sel, dtype=int)
        M_act = len(sel_arr)
        mults = compute_mults(sel_arr, beta, floor_r).astype(np.float64)
        y_unit = float((1 << 30) * 1024 * int(sm))
        r0 = g_centered[sel_arr] * mults
        r0_norm2 = float(np.sum(r0 * r0) + y_unit * y_unit)
        log_det_B = math.log(y_unit) + float(np.sum(np.log(w_orig[sel_arr] * mults)))
        log_det_proj = log_det_B - 0.5 * math.log(r0_norm2)
        gh_act = (
            math.exp(math.lgamma(M_act / 2.0 + 1.0) / M_act)
            / math.sqrt(math.pi)
            * math.exp(log_det_proj / M_act)
        )
        denom = 1.2935 * gh_act
        scale_v = (w_orig[sel_arr] * mults) / denom
        r0_unit = r0 / math.sqrt(r0_norm2)
        return sel_arr, scale_v, r0_unit

    def eval_lat_batch(lat, E):
        sel_arr, scale_v, r0_unit = lat
        V = E[:, sel_arr] * scale_v[None, :]
        proj = V @ r0_unit
        # Include typical y contribution (~0.025 in squared normalized units)
        return np.sqrt(np.sum(V * V, axis=1) - proj * proj + 0.025)

    # Simulate N = 50,000 multi-band peaks with |h_68| >= 0.862:
    N = 50000
    sigma_j = 0.023 * ((two_a[0] / two_a) ** 0.34)
    E = rng.normal(0.0, sigma_j[None, :], size=(N, M_POOL))
    for row in range(N):
        h_idx = rng.choice(np.arange(10, 26), size=rng.choice([1, 2], p=[0.55, 0.45]), replace=False)
        E[row, h_idx] = rng.uniform(0.26, 0.47, size=len(h_idx)) * rng.choice([-1, 1], size=len(h_idx))
        l_idx = rng.choice(np.arange(26, M_POOL), size=rng.integers(5, 9), replace=False)
        E[row, l_idx] = rng.uniform(0.28, 0.48, size=len(l_idx)) * rng.choice([-1, 1], size=len(l_idx))
    E = (E + 0.5) % 1.0 - 0.5
    h68 = np.sum(two_a[None, :] * np.cos(2.0 * np.pi * E), axis=1)
    E_target = E[h68 >= 0.875][:4000]
    print(f"High-rank target set: {len(E_target)} multi-band peaks with h68 >= 0.875", flush=True)

    # Seed with 2 pure high-rank prefix lattices (M=59 and M=60):
    specs = []
    sel59 = list(range(59))
    specs.append((0, "pref_M59_b16", sel59, 0.16, 1.00, 4, [], []))
    sel60 = list(range(60))
    specs.append((1, "pref_M60_b15", sel60, 0.15, 1.00, 4, [], []))

    cur_min = np.minimum(
        eval_lat_batch(make_fast_eval(sel59, 0.16, 1.00, 4), E_target),
        eval_lat_batch(make_fast_eval(sel60, 0.15, 1.00, 4), E_target),
    )

    candidates = []
    for _ in range(800):
        M_pool_loc = int(rng.choice([66, 67, 68]))
        M_act_loc = int(rng.choice([58, 59, 60], p=[0.25, 0.55, 0.20]))
        n_hi = int(rng.choice([1, 2], p=[0.45, 0.55]))
        drop_hi = set(int(x) for x in rng.choice(np.arange(10, 26), size=n_hi, replace=False))
        n_lo = M_pool_loc - M_act_loc - n_hi
        drop_lo = set(int(x) for x in rng.choice(np.arange(26, M_pool_loc), size=n_lo, replace=False))
        sel = [j for j in range(M_pool_loc) if j not in (drop_hi | drop_lo)]
        beta = float(rng.choice([0.14, 0.16, 0.18]))
        candidates.append((
            sorted(drop_hi),
            sorted(drop_lo),
            M_pool_loc,
            M_act_loc,
            beta,
            sel,
            make_fast_eval(sel, beta, 1.00, 4),
        ))

    dist_mat = np.stack([eval_lat_batch(c[6], E_target) for c in candidates], axis=0)
    chosen = set()
    for step in range(2, 24):
        best_i = -1
        best_score = (-1, -1, -999.0)
        for i in range(len(candidates)):
            if i in chosen:
                continue
            nm = np.minimum(cur_min, dist_mat[i])
            cov1 = int(np.sum(nm <= 0.995))
            cov2 = int(np.sum(nm <= 1.015))
            mean_d = -float(np.mean(nm))
            if (cov1, cov2, mean_d) > best_score:
                best_score = (cov1, cov2, mean_d)
                best_i = i
        chosen.add(best_i)
        cur_min = np.minimum(cur_min, dist_mat[best_i])
        drop_hi, drop_lo, M_pool_loc, M_act_loc, beta, sel, _ = candidates[best_i]
        hi_str = "_".join(str(x) for x in drop_hi)
        name = f"M{M_act_loc}_P{M_pool_loc}_hi{hi_str}_b{int(round(beta*100))}"
        specs.append((step, name, sel, beta, 1.00, 4, drop_hi, drop_lo))

    print(
        f"Greedy 24-lattice High-Rank Code covers {100*np.mean(cur_min<=1.005):.1f}% at gh_top<=1.005 "
        f"and {100*np.mean(cur_min<=1.015):.1f}% at gh_top<=1.015 (median={np.median(cur_min):.4f})",
        flush=True,
    )
    return specs


def build_one(args):
    idx, name, sel, beta, floor_r, sm, drop_hi, drop_lo = args
    FPLLL.set_precision(300)
    t0 = time.time()
    M_act = len(sel)
    n = M_act + 1

    # Try increasing floor_r and sm until zero singletons are ejected (len(s_rows) == 1)
    for fl_try in [floor_r, floor_r + 0.03, floor_r + 0.06, floor_r + 0.10]:
        mults = compute_mults(sel, beta, fl_try)
        for sm_try in [sm, 8, 16]:
            B = IntegerMatrix(n, n)
            t1_new = [0] * n
            t2_new = [0] * n
            for a, j in enumerate(sel):
                mult = int(mults[a])
                B[0, a] = G_ORIG[j] * mult
                B[a + 1, a] = W_ORIG[j] * mult
                t1_new[a] = (T1_ORIG[j] % W_ORIG[j]) * mult
                t2_new[a] = (T2_ORIG[j] % W_ORIG[j]) * mult
            y_unit = (1 << 30) * 1024 * int(sm_try)
            B[0, M_act] = y_unit

            LLL.reduction(B)
            for b in [20, 30, 40, 50, min(60, n)]:
                param = BKZ.Param(block_size=b, max_loops=3, strategies=BKZ.DEFAULT_STRATEGY, auto_abort=True)
                BKZ.reduction(B, param)

            rows_b = [[int(B[i, k]) for k in range(n)] for i in range(n)]
            s_rows = [r for r in rows_b if abs(r[M_act] // y_unit) <= (1 << 24)]
            m_rows = [r for r in rows_b if abs(r[M_act] // y_unit) > (1 << 24)]
            if len(s_rows) == 1:
                break
        if len(s_rows) == 1:
            break

    # If still > 1 singleton row, strip ejected columns cleanly
    if len(s_rows) > 1:
        s_rows.sort(key=lambda r: abs(r[M_act]))
        r0 = s_rows[0]
        if r0[M_act] < 0:
            r0 = [-x for x in r0]
        ej_cols = set()
        for r in s_rows[1:]:
            xk = r[M_act] // r0[M_act]
            for col in range(M_act):
                if r[col] - xk * r0[col] != 0:
                    ej_cols.add(col)
        keep_cols = [col for col in range(n) if col not in ej_cols]
        final_rows = [[r[col] for col in keep_cols] for r in ([r0] + m_rows)]
        t1_new = [t1_new[col] for col in keep_cols]
        t2_new = [t2_new[col] for col in keep_cols]
    else:
        s_row = s_rows[0]
        if s_row[M_act] < 0:
            s_row = [-x for x in s_row]
        final_rows = [s_row] + m_rows

    out_pos = f"hr_{idx:02d}_{name}_pos.gso"
    out_neg = f"hr_{idx:02d}_{name}_neg.gso"
    export_both_gso(final_rows, t1_new, t2_new, y_unit, out_pos, out_neg)
    print(
        f"[hr_{idx:02d}] {name:24s}: m_act={len(final_rows)-1:2d} drop_hi={drop_hi} ({time.time()-t0:.2f}s)",
        flush=True,
    )
    return out_pos, out_neg


if __name__ == "__main__":
    t0 = time.time()
    specs = design_highrank_specs()
    with mp.Pool(min(24, len(specs))) as pool:
        pool.map(build_one, specs)
    print(f"All 48 High-Rank GSO files built in {time.time()-t0:.2f}s", flush=True)
