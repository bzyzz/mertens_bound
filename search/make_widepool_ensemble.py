#!/usr/bin/env python3
"""
Generate 24 Pairs (48 GSO files) of Deep Wide-Pool Covering Lattices
(wp_00..wp_23, M_act = 51, n = 52, k_y_cut = 1) over pool j in [0..65].

Why Wide-Pool [0..65]?
1. Zeros j in [61..65] still contribute ~0.045 to h_StR(y). Any peak y in (0, 1.957e19)
   that achieves |h_StR(y)| > 1.0 by aligning 51 zeros out of the first 66 zeros [0..65]
   while having 1-2 anti-aligned zeros in [10..23] and 12-14 anti-aligned zeros in [24..65]
   is missed by all previous ensembles (which were restricted to pools <= 61).
2. Extending the pool to M_pool in [62..66] while keeping M_act = 51 simultaneously:
   - Lowers w_ref = W_ORIG[sel[-1]], slightly increasing mults and det(L_proj)^(1/51),
     which increases GH(L_proj) and shrinks the normalized distance gh_top!
   - Gives 12-15 dropout slots across [10..65] to absorb phase outliers.
"""
import math
import multiprocessing as mp
import time
import numpy as np
from fpylll import IntegerMatrix, LLL, BKZ, FPLLL
from make_pure_compact_ensemble import W_ORIG, G_ORIG, T1_ORIG, T2_ORIG, TWO_ASTAR, export_both_gso
from make_zero_dropout_ensemble import make_zd_specs
from make_multiband_ensemble import design_multiband_specs

FPLLL.set_precision(300)


def design_widepool_specs():
    rng = np.random.default_rng(20261014)
    M_POOL = 66
    two_a = TWO_ASTAR[:M_POOL]
    w_orig = np.array([float(W_ORIG[j]) for j in range(M_POOL)])
    g_centered = np.array([
        float((G_ORIG[j] % W_ORIG[j]) - (W_ORIG[j] if (G_ORIG[j] % W_ORIG[j]) > W_ORIG[j] // 2 else 0))
        for j in range(M_POOL)
    ])

    def make_fast_eval(sel, beta=0.38, sm=4):
        sel = np.array(sel, dtype=int)
        M_act = len(sel)
        w_ref = w_orig[sel[-1]]
        mults = np.round(((w_orig[sel] / w_ref) ** (2.0 * beta - 1.0)) * 1024.0)
        y_unit = float((1 << 30) * 1024 * int(sm))
        r0 = g_centered[sel] * mults
        r0_norm2 = np.sum(r0 * r0) + y_unit * y_unit
        log_det_B = math.log(y_unit) + np.sum(np.log(w_orig[sel] * mults))
        log_det_proj = log_det_B - 0.5 * math.log(r0_norm2)
        gh_act = math.exp(math.lgamma(M_act / 2.0 + 1.0) / M_act) / math.sqrt(math.pi) * math.exp(log_det_proj / M_act)
        dim_scale = 1.2935 * (2.15 ** (49.0 / M_act - 1.0))
        denom = dim_scale * gh_act
        scale_v = (w_orig[sel] * mults) / denom
        r0_unit = r0 / math.sqrt(r0_norm2)
        return sel, scale_v, r0_unit

    def eval_lat_batch(lat, E):
        sel, scale_v, r0_unit = lat
        V = E[:, sel] * scale_v[None, :]
        proj = V @ r0_unit
        return np.sqrt(np.sum(V * V, axis=1) - proj * proj)

    pref49_eval = make_fast_eval(list(range(49)), 0.50, 1)
    zd_evals = [make_fast_eval(sel, 0.38, 4) for _, _, sel, _, _ in make_zd_specs()]
    mb_evals = [make_fast_eval(sel, 0.38, 4) for _, _, sel, _, _, _, _ in design_multiband_specs()]

    N = 60000
    sigma_j = 0.024 * ((two_a[0] / two_a) ** 0.37)
    E = rng.normal(0.0, sigma_j[None, :], size=(N, M_POOL))
    for row in range(N):
        h_idx = rng.choice(np.arange(9, 24), size=rng.choice([1, 2, 3], p=[0.35, 0.45, 0.20]), replace=False)
        E[row, h_idx] = rng.uniform(0.27, 0.48, size=len(h_idx)) * rng.choice([-1, 1], size=len(h_idx))
        l_idx = rng.choice(np.arange(24, 66), size=rng.integers(8, 14), replace=False)
        E[row, l_idx] = rng.uniform(0.30, 0.49, size=len(l_idx)) * rng.choice([-1, 1], size=len(l_idx))
    E = (E + 0.5) % 1.0 - 0.5
    h66 = np.sum(two_a[None, :] * np.cos(2.0 * np.pi * E), axis=1)

    d_pref = eval_lat_batch(pref49_eval, E)
    d_zd = np.min(np.stack([eval_lat_batch(z, E) for z in zd_evals], axis=0), axis=0)
    d_mb = np.min(np.stack([eval_lat_batch(m, E) for m in mb_evals], axis=0), axis=0)

    mask = (h66 >= 0.840) & (d_pref > 1.12) & (d_zd > 1.09) & (d_mb > 1.10)
    E_miss = E[mask]
    print(f"Wide-pool target: {len(E_miss)} high-h66 peaks missed by SR.ENUM, zd_00..15, and mb_00..23", flush=True)

    candidates = []
    for _ in range(3600):
        M_pool_loc = int(rng.choice([63, 64, 65, 66]))
        n_hi = int(rng.choice([1, 2, 3], p=[0.25, 0.55, 0.20]))
        drop_hi = set(int(x) for x in rng.choice(np.arange(9, 24), size=n_hi, replace=False))
        n_lo = M_pool_loc - 51 - n_hi
        drop_lo = set(int(x) for x in rng.choice(np.arange(24, M_pool_loc), size=n_lo, replace=False))
        sel = [j for j in range(M_pool_loc) if j not in (drop_hi | drop_lo)]
        beta = float(rng.choice([0.35, 0.37, 0.39]))
        candidates.append((sorted(drop_hi), sorted(drop_lo), M_pool_loc, beta, sel, make_fast_eval(sel, beta, 4)))

    dist_mat = np.stack([eval_lat_batch(c[5], E_miss) for c in candidates], axis=0)
    cur_min = np.full(len(E_miss), 999.0)
    chosen = []
    out = []
    for step in range(24):
        best_i = -1
        best_score = (-1, -1, -999.0)
        for i in range(len(candidates)):
            if i in chosen:
                continue
            nm = np.minimum(cur_min, dist_mat[i])
            cov1 = int(np.sum(nm <= 1.07))
            cov2 = int(np.sum(nm <= 1.11))
            mean_d = -float(np.mean(nm))
            if (cov1, cov2, mean_d) > best_score:
                best_score = (cov1, cov2, mean_d)
                best_i = i
        chosen.append(best_i)
        cur_min = np.minimum(cur_min, dist_mat[best_i])
        drop_hi, drop_lo, M_pool_loc, beta, sel, _ = candidates[best_i]
        hi_str = "_".join(str(x) for x in drop_hi)
        name = f"P{M_pool_loc}_hi{hi_str}_b{int(round(beta*100))}"
        w_ref = W_ORIG[sel[-1]]
        mults = [int(round(((W_ORIG[j] / w_ref) ** (2.0 * beta - 1.0)) * 1024)) for j in sel]
        out.append((step, name, sel, mults, 4, drop_hi, drop_lo))
    print(
        f"Greedy 24-lattice Wide-Pool Code covers {100*np.mean(cur_min<=1.12):.1f}% of residual missed high-h66 peaks "
        f"at gh_top<=1.12 (median={np.median(cur_min):.3f})",
        flush=True,
    )
    return out


def build_one(args):
    idx, name, sel, mults, sm, drop_hi, drop_lo = args
    t0 = time.time()
    M_act = len(sel)
    n = M_act + 1
    for sm_try in [sm, 8, 16, 32, 64]:
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
        for b in [20, 30, 40, 48, 54, min(60, n)]:
            param = BKZ.Param(block_size=b, max_loops=3, strategies=BKZ.DEFAULT_STRATEGY, auto_abort=True)
            BKZ.reduction(B, param)

        rows_b = [[int(B[i, k]) for k in range(n)] for i in range(n)]
        s_rows = [r for r in rows_b if abs(r[M_act] // y_unit) <= (1 << 24)]
        m_rows = [r for r in rows_b if abs(r[M_act] // y_unit) > (1 << 24)]
        if len(s_rows) == 1:
            break
    if len(s_rows) != 1:
        next_j = next(j for j in range(70) if j not in sel)
        sel = sorted(list(sel[:-1]) + [next_j])
        w_ref = W_ORIG[sel[-1]]
        mults = [int(round(((W_ORIG[j] / w_ref) ** (2.0 * 0.38 - 1.0)) * 1024)) for j in sel]
        return build_one((idx, name, sel, mults, 4, drop_hi, drop_lo))
    s_row = s_rows[0]
    if s_row[M_act] < 0:
        s_row = [-x for x in s_row]
    final_rows = [s_row] + m_rows

    out_pos = f"wp_{idx:02d}_{name}_pos.gso"
    out_neg = f"wp_{idx:02d}_{name}_neg.gso"
    export_both_gso(final_rows, t1_new, t2_new, y_unit, out_pos, out_neg)
    print(
        f"[wp_{idx:02d}] {name:24s}: m_act={len(final_rows)-1:2d} drop_hi={drop_hi} ({time.time()-t0:.2f}s)",
        flush=True,
    )
    return out_pos, out_neg


if __name__ == "__main__":
    t0 = time.time()
    specs = design_widepool_specs()
    with mp.Pool(min(24, len(specs))) as pool:
        pool.map(build_one, specs)
    print(f"All 48 Wide-Pool GSO files built in {time.time()-t0:.2f}s", flush=True)
