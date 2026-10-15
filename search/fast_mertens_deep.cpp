#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>
#include <omp.h>

static constexpr int MAXN = 160;
static constexpr int STRIDE = 160;
static constexpr int K_PRE = 256;
static constexpr int M_PRE = 38;
static constexpr int N_PRE = 2 * M_PRE + 1; // 77 pts, step 1.5e-4 over [-0.0057, +0.0057]
static constexpr int K0 = 512;
static constexpr int M0 = 228;
static constexpr int N0 = 2 * M0 + 1;       // 457 pts, step 2.5e-5 over [-0.0057, +0.0057]
static constexpr int K1 = 2048;
static constexpr int M1 = 60;
static constexpr int N1 = 2 * M1 + 1;       // 121 pts, step 1.0e-5 over [-0.0006, +0.0006]

struct ZeroData {
    int n_zeros;
    std::vector<double> two_astar;
    std::vector<double> gamma;
    std::vector<unsigned __int128> ghi;
    std::vector<unsigned __int128> glo;
    std::vector<unsigned __int128> psi;
    std::vector<double> C_pre, S_pre;
    std::vector<double> C0, S0;
    std::vector<double> C1, S1;
    alignas(64) float cos_lut[16384];
    uint32_t step_idx[128];

    bool load(const char *path) {
        FILE *f = fopen(path, "rb");
        if (!f) return false;
        if (fread(&n_zeros, 4, 1, f) != 1) { fclose(f); return false; }
        two_astar.resize(n_zeros);
        gamma.resize(n_zeros);
        std::vector<uint64_t> ghi_lo(n_zeros), ghi_hi(n_zeros);
        std::vector<uint64_t> glo_lo(n_zeros), glo_hi(n_zeros);
        std::vector<uint64_t> psi_lo(n_zeros), psi_hi(n_zeros);
        if (fread(two_astar.data(), 8, n_zeros, f) != (size_t)n_zeros) return false;
        if (fread(gamma.data(), 8, n_zeros, f) != (size_t)n_zeros) return false;
        if (fread(ghi_lo.data(), 8, n_zeros, f) != (size_t)n_zeros) return false;
        if (fread(ghi_hi.data(), 8, n_zeros, f) != (size_t)n_zeros) return false;
        if (fread(glo_lo.data(), 8, n_zeros, f) != (size_t)n_zeros) return false;
        if (fread(glo_hi.data(), 8, n_zeros, f) != (size_t)n_zeros) return false;
        if (fread(psi_lo.data(), 8, n_zeros, f) != (size_t)n_zeros) return false;
        if (fread(psi_hi.data(), 8, n_zeros, f) != (size_t)n_zeros) return false;
        fclose(f);
        ghi.resize(n_zeros);
        glo.resize(n_zeros);
        psi.resize(n_zeros);
        for (int i = 0; i < n_zeros; ++i) {
            ghi[i] = (unsigned __int128)ghi_lo[i] | ((unsigned __int128)ghi_hi[i] << 64);
            glo[i] = (unsigned __int128)glo_lo[i] | ((unsigned __int128)glo_hi[i] << 64);
            psi[i] = (unsigned __int128)psi_lo[i] | ((unsigned __int128)psi_hi[i] << 64);
        }
        C_pre.resize(N_PRE * K_PRE);
        S_pre.resize(N_PRE * K_PRE);
        for (int m = -M_PRE; m <= M_PRE; ++m) {
            double dy = m * 1.5e-4;
            int idx = m + M_PRE;
            for (int i = 0; i < K_PRE; ++i) {
                double th = gamma[i] * dy;
                C_pre[idx * K_PRE + i] = std::cos(th);
                S_pre[idx * K_PRE + i] = std::sin(th);
            }
        }
        C0.resize(N0 * K0);
        S0.resize(N0 * K0);
        for (int m = -M0; m <= M0; ++m) {
            double dy = m * 2.5e-5;
            int idx = m + M0;
            for (int i = 0; i < K0; ++i) {
                double th = gamma[i] * dy;
                C0[idx * K0 + i] = std::cos(th);
                S0[idx * K0 + i] = std::sin(th);
            }
        }
        C1.resize(N1 * K1);
        S1.resize(N1 * K1);
        for (int m = -M1; m <= M1; ++m) {
            double dy = m * 1.0e-5;
            int idx = m + M1;
            for (int i = 0; i < K1; ++i) {
                double th = gamma[i] * dy;
                C1[idx * K1 + i] = std::cos(th);
                S1[idx * K1 + i] = std::sin(th);
            }
        }
        for (int k = 0; k < 16384; ++k) {
            cos_lut[k] = (float)std::cos(2.0 * M_PI * (double)k / 16384.0);
        }
        for (int i = 0; i < 128; ++i) {
            step_idx[i] = ((uint32_t)std::llround(gamma[i] * 0.0014 * (16384.0 / (2.0 * M_PI)))) & 16383u;
        }
        return true;
    }

    inline bool fast_screen(unsigned __int128 x) const {
        uint64_t x_hi = (uint64_t)(x >> 48);
        uint64_t x_lo = (uint64_t)(x & ((1ULL << 48) - 1));
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f, s4 = 0.0f, s5 = 0.0f, s6 = 0.0f;
        for (int i = 0; i < 64; ++i) {
            unsigned __int128 ph = (unsigned __int128)x_hi * ghi[i] + (unsigned __int128)x_lo * glo[i] - psi[i];
            uint32_t idx0 = (uint32_t)((uint64_t)(ph >> 64) >> 50);
            uint32_t st = step_idx[i];
            float w = (float)two_astar[i];
            s0 += w * cos_lut[(idx0 - 3 * st) & 16383u];
            s1 += w * cos_lut[(idx0 - 2 * st) & 16383u];
            s2 += w * cos_lut[(idx0 - st) & 16383u];
            s3 += w * cos_lut[idx0];
            s4 += w * cos_lut[(idx0 + st) & 16383u];
            s5 += w * cos_lut[(idx0 + 2 * st) & 16383u];
            s6 += w * cos_lut[(idx0 + 3 * st) & 16383u];
        }
        float m64 = std::max({std::abs(s0), std::abs(s1), std::abs(s2), std::abs(s3), std::abs(s4), std::abs(s5), std::abs(s6)});
        if (m64 < 0.790f) return false;
        for (int i = 64; i < 128; ++i) {
            unsigned __int128 ph = (unsigned __int128)x_hi * ghi[i] + (unsigned __int128)x_lo * glo[i] - psi[i];
            uint32_t idx0 = (uint32_t)((uint64_t)(ph >> 64) >> 50);
            uint32_t st = step_idx[i];
            float w = (float)two_astar[i];
            s0 += w * cos_lut[(idx0 - 3 * st) & 16383u];
            s1 += w * cos_lut[(idx0 - 2 * st) & 16383u];
            s2 += w * cos_lut[(idx0 - st) & 16383u];
            s3 += w * cos_lut[idx0];
            s4 += w * cos_lut[(idx0 + st) & 16383u];
            s5 += w * cos_lut[(idx0 + 2 * st) & 16383u];
            s6 += w * cos_lut[(idx0 + 3 * st) & 16383u];
        }
        float m128 = std::max({std::abs(s0), std::abs(s1), std::abs(s2), std::abs(s3), std::abs(s4), std::abs(s5), std::abs(s6)});
        return m128 >= 0.860f;
    }

    std::pair<double, double> polish(unsigned __int128 x) const {
        uint64_t x_hi = (uint64_t)(x >> 48);
        uint64_t x_lo = (uint64_t)(x & ((1ULL << 48) - 1));
        constexpr double SCALE = 3.14159265358979323846 / 9223372036854775808.0;

        alignas(64) double ang0[K0];
        alignas(64) double A0[K0];
        alignas(64) double B0[K0];
        for (int i = 0; i < K_PRE; ++i) {
            unsigned __int128 ph = (unsigned __int128)x_hi * ghi[i] + (unsigned __int128)x_lo * glo[i] - psi[i];
            int64_t top = (int64_t)(ph >> 64);
            double a = (double)top * SCALE;
            ang0[i] = a;
            A0[i] = two_astar[i] * std::cos(a);
            B0[i] = two_astar[i] * std::sin(a);
        }
        double pre_best = 0.0, best_h = 0.0, best_dy = 0.0;
        for (int idx = 0; idx < N_PRE; ++idx) {
            const double *cp = &C_pre[idx * K_PRE];
            const double *sp = &S_pre[idx * K_PRE];
            double s = 0.0;
            #pragma omp simd reduction(+:s)
            for (int i = 0; i < K_PRE; ++i) {
                s += A0[i] * cp[i] - B0[i] * sp[i];
            }
            if (std::abs(s) > pre_best) {
                pre_best = std::abs(s);
                best_h = s;
                best_dy = (idx - M_PRE) * 1.5e-4;
            }
        }
        if (pre_best < 0.895) return {best_h, best_dy};

        for (int i = K_PRE; i < K0; ++i) {
            unsigned __int128 ph = (unsigned __int128)x_hi * ghi[i] + (unsigned __int128)x_lo * glo[i] - psi[i];
            int64_t top = (int64_t)(ph >> 64);
            double a = (double)top * SCALE;
            ang0[i] = a;
            A0[i] = two_astar[i] * std::cos(a);
            B0[i] = two_astar[i] * std::sin(a);
        }
        double best_abs = 0.0;
        for (int idx = 0; idx < N0; ++idx) {
            const double *cp = &C0[idx * K0];
            const double *sp = &S0[idx * K0];
            double s = 0.0;
            #pragma omp simd reduction(+:s)
            for (int i = 0; i < K0; ++i) {
                s += A0[i] * cp[i] - B0[i] * sp[i];
            }
            if (std::abs(s) > best_abs) {
                best_abs = std::abs(s);
                best_h = s;
                best_dy = (idx - M0) * 2.5e-5;
            }
        }
        if (best_abs < 0.915) return {best_h, best_dy};

        double center = best_dy;
        alignas(64) double ang1[K1];
        alignas(64) double A1[K1];
        alignas(64) double B1[K1];
        for (int i = 0; i < K0; ++i) ang1[i] = ang0[i];
        for (int i = K0; i < K1; ++i) {
            unsigned __int128 ph = (unsigned __int128)x_hi * ghi[i] + (unsigned __int128)x_lo * glo[i] - psi[i];
            int64_t top = (int64_t)(ph >> 64);
            ang1[i] = (double)top * SCALE;
        }
        for (int i = 0; i < K1; ++i) {
            double a = ang1[i] + gamma[i] * center;
            A1[i] = two_astar[i] * std::cos(a);
            B1[i] = two_astar[i] * std::sin(a);
        }
        best_abs = 0.0;
        for (int idx = 0; idx < N1; ++idx) {
            const double *cp = &C1[idx * K1];
            const double *sp = &S1[idx * K1];
            double s = 0.0;
            #pragma omp simd reduction(+:s)
            for (int i = 0; i < K1; ++i) {
                s += A1[i] * cp[i] - B1[i] * sp[i];
            }
            if (std::abs(s) > best_abs) {
                best_abs = std::abs(s);
                best_h = s;
                best_dy = center + (idx - M1) * 1.0e-5;
            }
        }
        if (best_abs < 0.945) return {best_h, best_dy};

        std::vector<double> full_ang(n_zeros);
        for (int i = 0; i < K1; ++i) full_ang[i] = ang1[i];
        for (int i = K1; i < n_zeros; ++i) {
            unsigned __int128 ph = (unsigned __int128)x_hi * ghi[i] + (unsigned __int128)x_lo * glo[i] - psi[i];
            int64_t top = (int64_t)(ph >> 64);
            full_ang[i] = (double)top * SCALE;
        }
        center = best_dy;
        best_abs = 0.0;
        const int K2 = 15000;
        for (int m = -320; m <= 320; ++m) {
            double dy = center + m * 2.5e-6;
            double s = 0.0;
            for (int i = 0; i < K2; ++i) {
                s += two_astar[i] * std::cos(full_ang[i] + gamma[i] * dy);
            }
            if (std::abs(s) > best_abs) {
                best_abs = std::abs(s);
                best_dy = dy;
            }
        }

        center = best_dy;
        best_abs = 0.0;
        for (int m = -120; m <= 120; ++m) {
            double dy = center + m * 5e-7;
            double s = 0.0;
            for (int i = 0; i < n_zeros; ++i) {
                s += two_astar[i] * std::cos(full_ang[i] + gamma[i] * dy);
            }
            if (std::abs(s) > best_abs) {
                best_abs = std::abs(s);
                best_h = s;
                best_dy = dy;
            }
        }
        return {best_h, best_dy};
    }
};

struct Prefix {
    int k_start;
    int jumps;
    double dist;
    __int128 cur_x;
    int64_t x[MAXN];
};

struct ConcurrentBitset {
    static constexpr uint64_t MASK = (1ULL << 32) - 1;
    std::vector<uint64_t> words;
    ConcurrentBitset() : words(1ULL << 26, 0ULL) {}
    void clear() {
        std::fill(words.begin(), words.end(), 0ULL);
    }
    bool test_and_set(uint64_t v) {
        uint64_t h = (v ^ (v >> 30)) * 0xbf58476d1ce4e5b9ULL;
        h = (h ^ (h >> 27)) * 0x94d049bb133111ebULL;
        h = h ^ (h >> 31);
        uint64_t idx = (h & MASK) >> 6;
        uint64_t bit = 1ULL << (h & 63);
        uint64_t old = __atomic_fetch_or(&words[idx], bit, __ATOMIC_RELAXED);
        return (old & bit) == 0;
    }
};

static void print_int128(__int128 v, char *out) {
    if (v == 0) { out[0] = '0'; out[1] = '\0'; return; }
    bool neg = (v < 0);
    unsigned __int128 u = neg ? (unsigned __int128)(-v) : (unsigned __int128)v;
    char buf[64];
    int n = 0;
    while (u > 0) {
        buf[n++] = '0' + (int)(u % 10);
        u /= 10;
    }
    int p = 0;
    if (neg) out[p++] = '-';
    while (n > 0) out[p++] = buf[--n];
    out[p] = '\0';
}

int main(int argc, char **argv) {
    const char *fname_arg = (argc > 1) ? argv[1] : "hr_00_pref_M59_b16_neg.gso";
    double gh_top = (argc > 2) ? atof(argv[2]) : 1.045;
    double gh_leaf = (argc > 3) ? atof(argv[3]) : gh_top;
    double lin = (argc > 4) ? atof(argv[4]) : 3.0;
    const char *yc_arg = (argc > 5) ? argv[5] : "2.0,5.0,8.0,11.0,14.0,17.0";
    int k_split = (argc > 6) ? atoi(argv[6]) : 0;

    std::vector<std::string> fnames;
    {
        std::string s(fname_arg);
        size_t pos = 0;
        while (pos < s.size()) {
            size_t comma = s.find(',', pos);
            if (comma == std::string::npos) {
                fnames.push_back(s.substr(pos));
                break;
            } else {
                fnames.push_back(s.substr(pos, comma - pos));
                pos = comma + 1;
            }
        }
    }

    std::vector<double> yc_list;
    {
        const char *p = yc_arg;
        while (*p) {
            char *endp = nullptr;
            double v = strtod(p, &endp);
            if (endp == p) break;
            if (*endp == ':') {
                double v_start = v;
                p = endp + 1;
                double v_step = strtod(p, &endp);
                if (*endp == ':') {
                    p = endp + 1;
                    double v_end = strtod(p, &endp);
                    if (v_step > 0) {
                        for (double cur = v_start; cur <= v_end + 1e-9; cur += v_step) yc_list.push_back(cur);
                    } else if (v_step < 0) {
                        for (double cur = v_start; cur >= v_end - 1e-9; cur += v_step) yc_list.push_back(cur);
                    }
                } else {
                    yc_list.push_back(v_start);
                }
            } else {
                yc_list.push_back(v);
            }
            if (*endp == ',') p = endp + 1;
            else break;
        }
        if (yc_list.empty()) yc_list.push_back(0.0);
    }

    ZeroData zd;
    if (!zd.load("zeros.bin")) {
        fprintf(stderr, "Failed to load zeros.bin\n");
        return 1;
    }

    double max_y_e18 = 3.484900;
    if (const char *env_max_y = getenv("MAX_Y_E18")) {
        double my = atof(env_max_y);
        if (my > 0.0) max_y_e18 = my;
    }
    __int128 max_x = (__int128)(max_y_e18 * 1e18 * (double)(1ULL << 30));
    __int128 margin_x = (__int128)10 << 30;

    auto t_start = std::chrono::steady_clock::now();
    std::atomic<uint64_t> total_nodes{0};
    std::atomic<uint64_t> total_clusters{0};
    std::atomic<uint64_t> pruned_at_cut{0};
    std::mutex out_mtx;
    ConcurrentBitset seen_y_buckets;
    double global_best_abs = 0.0;

    for (size_t f_idx = 0; f_idx < fnames.size(); ++f_idx) {
        const char *fname = fnames[f_idx].c_str();
        FILE *f = fopen(fname, "rb");
        if (!f) {
            std::string alt = std::string("gso/") + fname;
            f = fopen(alt.c_str(), "rb");
        }
        if (!f) {
            fprintf(stderr, "Failed to load %s\n", fname);
            continue;
        }
        int n;
        if (fread(&n, 4, 1, f) != 1) { fclose(f); continue; }
        std::vector<double> rs(n), base_sigma(n), dsigma(n), mu(n * n);
        std::vector<uint64_t> lo(n), hi(n);
        if (fread(rs.data(), 8, n, f) != (size_t)n) { fclose(f); continue; }
        if (fread(base_sigma.data(), 8, n, f) != (size_t)n) { fclose(f); continue; }
        if (fread(dsigma.data(), 8, n, f) != (size_t)n) { fclose(f); continue; }
        if (fread(mu.data(), 8, n * n, f) != (size_t)(n * n)) { fclose(f); continue; }
        if (fread(lo.data(), 8, n, f) != (size_t)n) { fclose(f); continue; }
        if (fread(hi.data(), 8, n, f) != (size_t)n) { fclose(f); continue; }
        fclose(f);

        std::vector<__int128> xcol(n);
        for (int i = 0; i < n; ++i) {
            unsigned __int128 u = (unsigned __int128)lo[i] | ((unsigned __int128)hi[i] << 64);
            xcol[i] = (__int128)u;
        }

        int k_y_cut = 0;
        while (k_y_cut < n) {
            __int128 v = xcol[k_y_cut] >= 0 ? xcol[k_y_cut] : -xcol[k_y_cut];
            if (v > ((__int128)1 << 24)) break;
            k_y_cut++;
        }

        std::vector<double> mu_T(n * STRIDE, 0.0);
        for (int k = 0; k < n; ++k) {
            for (int i = k + 1; i < n; ++i) {
                mu_T[k * STRIDE + i] = mu[i * n + k];
            }
        }

        bool bad_rs = false;
        for (int i = 0; i < n; ++i) {
            if (!(rs[i] > 1e30) || rs[i] > 1e85) { bad_rs = true; break; }
            if (i >= k_y_cut && rs[i] < 1e41) { bad_rs = true; break; }
        }
        if (bad_rs || k_y_cut != 1) {
            fprintf(stderr, "Skipping degenerate GSO %s (rank-deficient or ill-conditioned rs, k_y_cut=%d)\n", fname, k_y_cut);
            continue;
        }
        double logdet = 0.0;
        for (int i = 0; i < n; ++i) logdet += 0.5 * std::log(rs[i]);
        double heur = std::lgamma(1.0 + n / 2.0) / n - 0.5 * std::log(M_PI);
        double base_gh = std::exp(logdet / n) * std::exp(heur);

        int m_act = n - k_y_cut;
        double logdet_act = 0.0;
        for (int i = k_y_cut; i < n; ++i) logdet_act += 0.5 * std::log(rs[i]);
        double heur_act = std::lgamma(1.0 + m_act / 2.0) / m_act - 0.5 * std::log(M_PI);
        double gh_act = std::exp(logdet_act / m_act) * std::exp(heur_act);
        double dim_scale = 1.2935;
        double r2_cut = (dim_scale * gh_act * gh_top) * (dim_scale * gh_act * gh_top);

        // Variance-envelope upper-tree & per-step pruning arrays
        std::vector<double> bounds(n);
        std::vector<double> max_inc(n);
        std::vector<int> max_jumps(n);
        std::vector<int> floor_jumps(n, 0);
        double r2_leaf_global = (base_gh * gh_leaf) * (base_gh * gh_leaf);
        for (int k = 0; k < n; ++k) {
            if (k >= k_y_cut) {
                int i_lvl = n - k;
                double u = (double)i_lvl / (double)m_act;
                // Piecewise convex variance-envelope upper-tree pruning curve p_deep(u):
                // Tight in the top 30% of the tree (i <= 18) where prefixes are generated,
                // while opening smoothly in the middle tree (i = 19..45) for zero pinching.
                double p_deep = (u <= 0.30)
                    ? (0.040 + 0.68 * u + 0.30 * u * u)
                    : std::min(1.0, 0.062 + 0.88 * u + 0.12 * u * u);
                bounds[k] = r2_cut * p_deep;

                // Per-level integer step residual bound |a_k| <= max_a(i_lvl)
                // Because rs[k] grows 18x-20x from top (i=1) to bottom (i=m_act),
                // |a_k| = sqrt(d_inc / rs[k]) shrinks monotonically with depth i_lvl.
                double max_a = (i_lvl <= 4)  ? 2.60 :
                               (i_lvl <= 16) ? 1.85 :
                               (i_lvl <= 26) ? 1.15 :
                               (i_lvl <= 45) ? 0.90 : 0.65;
                max_inc[k] = max_a * max_a * rs[k];

                // Cumulative non-Babai step count (|a_k| > 0.5, i.e., step >= 1) envelope
                max_jumps[k] = (i_lvl <= 3)  ? 3 :
                               (i_lvl <= 6)  ? 5 :
                               (i_lvl <= 12) ? 8 :
                               (i_lvl <= 18) ? 10 :
                               (i_lvl <= 26) ? 12 : 14;

                // Remaining-jump floor: applied only where max_jumps[k] == 14 (i_lvl >= 27)
                floor_jumps[k] = (i_lvl >= 42) ? 14 :
                                 (i_lvl >= 35) ? 13 :
                                 (i_lvl >= 27) ? 11 : 0;
            } else {
                double i = (double)(n - k);
                double rr1 = i / n;
                double rr2 = (lin - 1.0) * std::sqrt(i * (n - i) / (n / 2.0 + 1.0)) / n;
                double p = std::min(1.0, rr1 + rr2);
                bounds[k] = (k_y_cut == 1) ? (r2_cut * 10.0) : (r2_leaf_global * std::min(1.0, p + 0.12));
                max_inc[k] = bounds[k];
                max_jumps[k] = 60;
                floor_jumps[k] = 0;
            }
        }

        int cur_k_split = (k_split >= n - 14 && k_split < n - 4) ? k_split : (n - 10);

        for (size_t yc_idx = 0; yc_idx < yc_list.size(); ++yc_idx) {
            double y_center_e18 = yc_list[yc_idx];
            std::vector<double> sigma = base_sigma;
            if (y_center_e18 != 0.0) {
                double yc = y_center_e18 * 1e18;
                for (int i = 0; i < n; ++i) sigma[i] += yc * dsigma[i];
            }

            __int128 win_lo = -margin_x;
            __int128 win_hi = max_x + margin_x;

            auto t0 = std::chrono::steady_clock::now();
            std::vector<Prefix> prefixes;
            Prefix init_p;
            init_p.jumps = 0;
            init_p.dist = 0.0;
            init_p.cur_x = 0;
            memset(init_p.x, 0, sizeof(init_p.x));

            auto gen_prefixes = [&](auto &self, int k, Prefix &p) -> void {
                if (prefixes.size() >= 2000000 || (k <= cur_k_split && (k <= cur_k_split - 6 || p.dist >= 0.65 * bounds[k + 1]))) {
                    p.k_start = k;
                    prefixes.push_back(p);
                    return;
                }
                double c = sigma[k];
                const double *mt = &mu_T[k * STRIDE];
                for (int i = k + 1; i < n; ++i) c -= (double)p.x[i] * mt[i];
                int64_t x0 = (int64_t)std::llround(c);
                int dir = (c >= (double)x0) ? 1 : -1;
                double eff_bnd = std::min(bounds[k], p.dist + max_inc[k]);
                int mj = max_jumps[k];
                int fj = floor_jumps[k];
                for (int64_t step = 0;; ++step) {
                    int nj = p.jumps + (step > 0 ? 1 : 0);
                    if (nj > mj) break;
                    int64_t xx;
                    if (step == 0) xx = x0;
                    else {
                        int64_t mag = (step + 1) >> 1;
                        xx = (step & 1) ? (x0 + dir * mag) : (x0 - dir * mag);
                    }
                    double a = (double)xx - c;
                    double nd = p.dist + a * a * rs[k];
                    if (nd > eff_bnd) break;
                    p.x[k] = xx;
                    __int128 nx = p.cur_x + (__int128)xx * xcol[k];
                    double old_d = p.dist;
                    __int128 old_x = p.cur_x;
                    int old_j = p.jumps;
                    p.dist = nd;
                    p.cur_x = nx;
                    p.jumps = (nj < fj) ? fj : nj;
                    self(self, k - 1, p);
                    p.dist = old_d;
                    p.cur_x = old_x;
                    p.jumps = old_j;
                }
            };

            gen_prefixes(gen_prefixes, n - 1, init_p);
            uint64_t prev_clusters = total_clusters.load();

            #pragma omp parallel
            {
                uint64_t local_nodes = 0;
                uint64_t local_pruned = 0;
                double x_dbl[MAXN];
                int64_t x_int[MAXN];

                #pragma omp for schedule(dynamic, 1)
                for (size_t idx = 0; idx < prefixes.size(); ++idx) {
                    const Prefix &pf = prefixes[idx];
                    for (int i = pf.k_start + 1; i < n; ++i) {
                        x_int[i] = pf.x[i];
                        x_dbl[i] = (double)pf.x[i];
                    }

                    double sigma_pf[MAXN];
                    double sigma_mid[MAXN];
                    double sigma_low[MAXN];
                    const int k_mid = k_y_cut + (n - k_y_cut) * 4 / 10;
                    const int k_low = k_y_cut + (n - k_y_cut) * 2 / 10;
                    for (int j = 0; j <= pf.k_start; ++j) {
                        double s = sigma[j];
                        const double *mt = &mu_T[j * STRIDE];
                        for (int i = pf.k_start + 1; i < n; ++i) s -= x_dbl[i] * mt[i];
                        sigma_pf[j] = s;
                    }

                    auto rec_top = [&](auto &self, int k, double dist, __int128 cur_x, int jumps) -> void {
                        local_nodes++;
                        if (k < k_y_cut) {
                            if (cur_x <= win_lo || cur_x >= win_hi) {
                                local_pruned++;
                                return;
                            }
                            __int128 full_x = cur_x;
                            if (k_y_cut == 1) {
                                double c = sigma_low[0];
                                const double *mt = &mu_T[0];
                                for (int i = 1; i < k_low; ++i) c -= x_dbl[i] * mt[i];
                                int64_t x0 = (int64_t)std::llround(c);
                                full_x += (__int128)x0 * xcol[0];
                            } else {
                                double d = dist;
                                if (d > bounds[k_y_cut - 1]) return;
                                for (int j = k_y_cut - 1; j >= 0; --j) {
                                    double c = sigma_mid[j];
                                    const double *mt = &mu_T[j * STRIDE];
                                    for (int i = j + 1; i < k_mid; ++i) c -= x_dbl[i] * mt[i];
                                    int64_t x0 = (int64_t)std::llround(c);
                                    x_dbl[j] = (double)x0;
                                    double a = (double)x0 - c;
                                    d += a * a * rs[j];
                                    if (d > bounds[j]) return;
                                    full_x += (__int128)x0 * xcol[j];
                                }
                            }
                            if (full_x > 0 && full_x < max_x) {
                                if (!zd.fast_screen((unsigned __int128)full_x)) return;
                                uint64_t y_bkt = (uint64_t)(full_x >> 32);
                                if (!seen_y_buckets.test_and_set(y_bkt)) return;
                                uint64_t cid = ++total_clusters;
                                auto [h_val, dy] = zd.polish((unsigned __int128)full_x);
                                double ah = std::abs(h_val);
                                if (ah >= 0.955 || ah > global_best_abs) {
                                    std::lock_guard<std::mutex> lk(out_mtx);
                                    if (ah > global_best_abs) global_best_abs = ah;
                                    char xbuf[64];
                                    print_int128(full_x, xbuf);
                                    double y_approx = (double)full_x / (double)(1ULL << 30);
                                    printf("CLUSTER #%llu |h|=%.8f h=%+.8f y=%.6f (x=%s, dy=%+.7f)%s\n",
                                           (unsigned long long)cid, ah, h_val, y_approx, xbuf, dy,
                                           (ah > 1.00000006) ? " *** RECORD BREAKING HIT ***" : "");
                                    fflush(stdout);
                                }
                            }
                            return;
                        }
                        if (k == k_y_cut) {
                            double eff_bnd = bounds[k];
                            if (dist > eff_bnd) return;
                            double c = sigma_low[k];
                            const double *mt = &mu_T[k * STRIDE];
                            for (int i = k + 1; i < k_low; ++i) c -= x_dbl[i] * mt[i];
                            double rad = std::sqrt((eff_bnd - dist) / rs[k]);
                            int64_t gso_lo = (int64_t)std::ceil(c - rad);
                            int64_t gso_hi = (int64_t)std::floor(c + rad);
                            if (gso_lo > gso_hi) return;

                            double inv_W = 1.0 / (double)(xcol[k] >= 0 ? xcol[k] : -xcol[k]);
                            int64_t y_lo, y_hi;
                            if (xcol[k] > 0) {
                                y_lo = (int64_t)std::floor((double)(win_lo - cur_x) * inv_W);
                                y_hi = (int64_t)std::ceil((double)(win_hi - cur_x) * inv_W);
                            } else {
                                y_lo = (int64_t)std::floor((double)(cur_x - win_hi) * inv_W);
                                y_hi = (int64_t)std::ceil((double)(cur_x - win_lo) * inv_W);
                            }
                            int64_t lo_xx = std::max(gso_lo, y_lo);
                            int64_t hi_xx = std::min(gso_hi, y_hi);
                            if (lo_xx > hi_xx) {
                                local_pruned++;
                                return;
                            }
                            for (int64_t xx = hi_xx; xx >= lo_xx; --xx) {
                                __int128 nx = cur_x + (__int128)xx * xcol[k];
                                if (nx <= win_lo || nx >= win_hi) continue;
                                double a = (double)xx - c;
                                double nd = dist + a * a * rs[k];
                                if (nd <= eff_bnd) {
                                    x_int[k] = xx;
                                    x_dbl[k] = (double)xx;
                                    self(self, k - 1, nd, nx, jumps);
                                }
                            }
                            return;
                        }
                        double c;
                        const double *mt = &mu_T[k * STRIDE];
                        if (k < k_low) {
                            c = sigma_low[k];
                            for (int i = k + 1; i < k_low; ++i) c -= x_dbl[i] * mt[i];
                        } else if (k < k_mid) {
                            c = sigma_mid[k];
                            for (int i = k + 1; i < k_mid; ++i) c -= x_dbl[i] * mt[i];
                        } else {
                            c = sigma_pf[k];
                            for (int i = k + 1; i <= pf.k_start; ++i) c -= x_dbl[i] * mt[i];
                        }
                        int64_t x0 = (int64_t)std::llround(c);
                        int dir = (c >= (double)x0) ? 1 : -1;
                        double eff_bnd = std::min(bounds[k], dist + max_inc[k]);
                        int mj = max_jumps[k];
                        int fj = floor_jumps[k];
                        for (int64_t step = 0;; ++step) {
                            int nj = jumps + (step > 0 ? 1 : 0);
                            if (nj > mj) break;
                            int64_t xx;
                            if (step == 0) xx = x0;
                            else {
                                int64_t mag = (step + 1) >> 1;
                                xx = (step & 1) ? (x0 + dir * mag) : (x0 - dir * mag);
                            }
                            double a = (double)xx - c;
                            double nd = dist + a * a * rs[k];
                            if (nd > eff_bnd) break;
                            x_int[k] = xx;
                            x_dbl[k] = (double)xx;
                            if (k == k_mid) {
                                for (int j = 0; j < k_mid; ++j) {
                                    double s = sigma_pf[j];
                                    const double *mtj = &mu_T[j * STRIDE];
                                    for (int i = k_mid; i <= pf.k_start; ++i) s -= x_dbl[i] * mtj[i];
                                    sigma_mid[j] = s;
                                }
                            } else if (k == k_low) {
                                int j_start = (k_y_cut == 1) ? 0 : k_y_cut;
                                for (int j = j_start; j < k_low; ++j) {
                                    double s = sigma_mid[j];
                                    const double *mtj = &mu_T[j * STRIDE];
                                    for (int i = k_low; i < k_mid; ++i) s -= x_dbl[i] * mtj[i];
                                    sigma_low[j] = s;
                                }
                            }
                            self(self, k - 1, nd, cur_x + (__int128)xx * xcol[k], (nj < fj) ? fj : nj);
                        }
                    };

                    rec_top(rec_top, pf.k_start, pf.dist, pf.cur_x, pf.jumps);
                }
                total_nodes += local_nodes;
                pruned_at_cut += local_pruned;
            }
            auto t1 = std::chrono::steady_clock::now();
            double sec = std::chrono::duration<double>(t1 - t0).count();
            fprintf(stderr, "[%s (%zu/%zu) yc=%.2fe18 (%zu/%zu)] %.2f s | new_clusters=%llu | total_unique=%llu | best=%.8f\n",
                    fname, f_idx + 1, fnames.size(), y_center_e18, yc_idx + 1, yc_list.size(), sec,
                    (unsigned long long)(total_clusters.load() - prev_clusters),
                    (unsigned long long)total_clusters.load(), global_best_abs);
        }
    }

    auto t_end = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(t_end - t_start).count();
    fprintf(stderr, "ALL DONE in %.2f s | nodes=%llu (%.2f Gnodes/s) | pruned_at_cut=%llu | unique_clusters=%llu | best_abs=%.8f\n",
            sec,
            (unsigned long long)total_nodes.load(),
            (double)total_nodes.load() / (sec * 1e9),
            (unsigned long long)pruned_at_cut.load(),
            (unsigned long long)total_clusters.load(),
            global_best_abs);
    return 0;
}
