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
static constexpr int K_MID = 140;
static constexpr int K_PRE = 256;
static constexpr int M_PRE = 38;
static constexpr int N_PRE = 2 * M_PRE + 1; // 77 pts, step 1.5e-4 over [-0.0057, +0.0057]
static constexpr int K0 = 512;
static constexpr int M0 = 228;
static constexpr int N0 = 2 * M0 + 1;       // 457 pts, step 2.5e-5 over [-0.0057, +0.0057]
static constexpr int K1 = 2048;
static constexpr int M1 = 60;
static constexpr int N1 = 2 * M1 + 1;       // 121 pts, step 1.0e-5 over [-0.0006, +0.0006]
static constexpr int K2 = 15000;
static constexpr int M2 = 320;
static constexpr int N2 = 2 * M2 + 1;       // 641 pts, step 2.5e-6 over [-0.0008, +0.0008]

struct PolishResult {
    double h_val;
    double dy;
    double h140_val;
    double dy140;
    double h256_val;
    bool full_refined;
};

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
    std::vector<double> C2, S2;
    alignas(64) float cos_lut[16384];
    uint32_t step_idx[K_MID];

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
        C2.resize((size_t)N2 * K2);
        S2.resize((size_t)N2 * K2);
        for (int m = -M2; m <= M2; ++m) {
            double dy = m * 2.5e-6;
            size_t idx = (size_t)(m + M2);
            for (int i = 0; i < K2; ++i) {
                double th = gamma[i] * dy;
                C2[idx * K2 + i] = std::cos(th);
                S2[idx * K2 + i] = std::sin(th);
            }
        }
        for (int k = 0; k < 16384; ++k) {
            cos_lut[k] = (float)std::cos(2.0 * M_PI * (double)k / 16384.0);
        }
        for (int i = 0; i < K_MID; ++i) {
            step_idx[i] = ((uint32_t)std::llround(gamma[i] * 0.0014 * (16384.0 / (2.0 * M_PI)))) & 16383u;
        }
        return true;
    }

    // Two-stage mid-band-aware vectorized leaf filter at k == k_y_cut:
    // Stage A: j = 0..55 (|h_{0..55}| >= 0.740)
    // Stage B: j = 56..139 (|h_{0..139}| >= 0.880)
    inline bool fast_screen(unsigned __int128 x, float &out_m140) const {
        uint64_t x_hi = (uint64_t)(x >> 48);
        uint64_t x_lo = (uint64_t)(x & ((1ULL << 48) - 1));
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f, s4 = 0.0f, s5 = 0.0f, s6 = 0.0f;
        for (int i = 0; i < 56; ++i) {
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
        float m56 = std::max({std::abs(s0), std::abs(s1), std::abs(s2), std::abs(s3), std::abs(s4), std::abs(s5), std::abs(s6)});
        if (m56 < 0.740f) return false;
        for (int i = 56; i < K_MID; ++i) {
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
        float m140 = std::max({std::abs(s0), std::abs(s1), std::abs(s2), std::abs(s3), std::abs(s4), std::abs(s5), std::abs(s6)});
        out_m140 = m140;
        return m140 >= 0.880f;
    }

    PolishResult polish(unsigned __int128 x) const {
        uint64_t x_hi = (uint64_t)(x >> 48);
        uint64_t x_lo = (uint64_t)(x & ((1ULL << 48) - 1));
        constexpr double SCALE = 3.14159265358979323846 / 9223372036854775808.0;

        // Stage -1: K_PRE = 256 zeros, also tracking exact h_{0..139}
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
        double best_abs_140 = 0.0, best_h140 = 0.0, best_dy140 = 0.0;
        double pre_best = 0.0, best_h256 = 0.0, best_dy = 0.0;
        for (int idx = 0; idx < N_PRE; ++idx) {
            const double *cp = &C_pre[idx * K_PRE];
            const double *sp = &S_pre[idx * K_PRE];
            double s140 = 0.0;
            #pragma omp simd reduction(+:s140)
            for (int i = 0; i < K_MID; ++i) {
                s140 += A0[i] * cp[i] - B0[i] * sp[i];
            }
            double s_tail = 0.0;
            #pragma omp simd reduction(+:s_tail)
            for (int i = K_MID; i < K_PRE; ++i) {
                s_tail += A0[i] * cp[i] - B0[i] * sp[i];
            }
            double s256 = s140 + s_tail;
            if (std::abs(s140) > best_abs_140) {
                best_abs_140 = std::abs(s140);
                best_h140 = s140;
                best_dy140 = (idx - M_PRE) * 1.5e-4;
            }
            if (std::abs(s256) > pre_best) {
                pre_best = std::abs(s256);
                best_h256 = s256;
                best_dy = (idx - M_PRE) * 1.5e-4;
            }
        }

        // Refine h_{0..139} with 2 Newton steps if |h_{0..139}| >= 0.890
        if (best_abs_140 >= 0.890) {
            double dy_cur = best_dy140;
            for (int iter = 0; iter < 2; ++iter) {
                double s = 0.0, hp = 0.0, hpp = 0.0;
                for (int i = 0; i < K_MID; ++i) {
                    double th = ang0[i] + gamma[i] * dy_cur;
                    double c = two_astar[i] * std::cos(th);
                    double sn = two_astar[i] * std::sin(th);
                    s += c;
                    hp -= gamma[i] * sn;
                    hpp -= (gamma[i] * gamma[i]) * c;
                }
                if (std::abs(s) > best_abs_140) {
                    best_abs_140 = std::abs(s);
                    best_h140 = s;
                    best_dy140 = dy_cur;
                }
                if (std::abs(hpp) > 1e-9) {
                    double step = hp / hpp;
                    if (std::abs(step) > 2e-4) step = (step > 0) ? 2e-4 : -2e-4;
                    dy_cur -= step;
                }
            }
            double s = 0.0;
            for (int i = 0; i < K_MID; ++i) {
                s += two_astar[i] * std::cos(ang0[i] + gamma[i] * dy_cur);
            }
            if (std::abs(s) > best_abs_140) {
                best_abs_140 = std::abs(s);
                best_h140 = s;
                best_dy140 = dy_cur;
            }
        }

        bool force_full = (best_abs_140 >= 0.915) || (pre_best >= 0.915);
        double precheck_thresh = (best_abs_140 >= 0.905) ? 0.905 : 0.910;
        if (!force_full && pre_best < precheck_thresh && best_abs_140 < 0.905) {
            return {best_h256, best_dy, best_h140, best_dy140, best_h256, false};
        }

        // Stage 0: K0 = 512 zeros, 457 pts (step 2.5e-5 over [-0.0057, +0.0057])
        for (int i = K_PRE; i < K0; ++i) {
            unsigned __int128 ph = (unsigned __int128)x_hi * ghi[i] + (unsigned __int128)x_lo * glo[i] - psi[i];
            int64_t top = (int64_t)(ph >> 64);
            double a = (double)top * SCALE;
            ang0[i] = a;
            A0[i] = two_astar[i] * std::cos(a);
            B0[i] = two_astar[i] * std::sin(a);
        }
        double best_abs = 0.0, best_h = best_h256;
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
        if (!force_full && best_abs < 0.915) {
            return {best_h, best_dy, best_h140, best_dy140, best_h256, false};
        }

        // Stage 1: K1 = 2048 zeros, 121 pts (step 1e-5 over [-0.0006, +0.0006] around best_dy)
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
        if (!force_full && best_abs < 0.940) {
            return {best_h, best_dy, best_h140, best_dy140, best_h256, false};
        }

        // Stage 2: K2 = 15000 zeros, SIMD scan step 2.5e-6 over [-0.0004, +0.0004] around best_dy
        std::vector<double> full_ang(n_zeros);
        for (int i = 0; i < K1; ++i) full_ang[i] = ang1[i];
        for (int i = K1; i < n_zeros; ++i) {
            unsigned __int128 ph = (unsigned __int128)x_hi * ghi[i] + (unsigned __int128)x_lo * glo[i] - psi[i];
            int64_t top = (int64_t)(ph >> 64);
            full_ang[i] = (double)top * SCALE;
        }
        center = best_dy;
        std::vector<double> A2(K2), B2(K2);
        for (int i = 0; i < K2; ++i) {
            double a = full_ang[i] + gamma[i] * center;
            A2[i] = two_astar[i] * std::cos(a);
            B2[i] = two_astar[i] * std::sin(a);
        }
        best_abs = 0.0;
        for (int idx = 0; idx < N2; ++idx) {
            const double *cp = &C2[(size_t)idx * K2];
            const double *sp = &S2[(size_t)idx * K2];
            double s = 0.0;
            #pragma omp simd reduction(+:s)
            for (int i = 0; i < K2; ++i) {
                s += A2[i] * cp[i] - B2[i] * sp[i];
            }
            if (std::abs(s) > best_abs) {
                best_abs = std::abs(s);
                best_h = s;
                best_dy = center + (idx - M2) * 2.5e-6;
            }
        }

        // Stage 3: Full 71,250-zero 9-point local stencil (step 1.25e-6 over +-5e-6) + 3-step Newton-Raphson refiner
        center = best_dy;
        best_abs = 0.0;
        for (int m = -4; m <= 4; ++m) {
            double dy_test = center + m * 1.25e-6;
            double s = 0.0;
            #pragma omp simd reduction(+:s)
            for (int i = 0; i < n_zeros; ++i) {
                s += two_astar[i] * std::cos(full_ang[i] + gamma[i] * dy_test);
            }
            if (std::abs(s) > best_abs) {
                best_abs = std::abs(s);
                best_h = s;
                best_dy = dy_test;
            }
        }

        // Exact 71,250-zero Newton-Raphson polish
        double dy_nr = best_dy;
        for (int iter = 0; iter < 3; ++iter) {
            double s = 0.0, hp = 0.0, hpp = 0.0;
            for (int i = 0; i < n_zeros; ++i) {
                double th = full_ang[i] + gamma[i] * dy_nr;
                double c = two_astar[i] * std::cos(th);
                double sn = two_astar[i] * std::sin(th);
                s += c;
                hp -= gamma[i] * sn;
                hpp -= (gamma[i] * gamma[i]) * c;
            }
            if (std::abs(s) > best_abs) {
                best_abs = std::abs(s);
                best_h = s;
                best_dy = dy_nr;
            }
            if (std::abs(hpp) > 1e-9) {
                double step = hp / hpp;
                if (std::abs(step) > 2e-6) step = (step > 0) ? 2e-6 : -2e-6;
                dy_nr -= step;
                if (std::abs(step) < 1e-11) break;
            } else {
                break;
            }
        }
        {
            double s = 0.0;
            for (int i = 0; i < n_zeros; ++i) {
                s += two_astar[i] * std::cos(full_ang[i] + gamma[i] * dy_nr);
            }
            if (std::abs(s) > best_abs) {
                best_abs = std::abs(s);
                best_h = s;
                best_dy = dy_nr;
            }
        }
        return {best_h, best_dy, best_h140, best_dy140, best_h256, true};
    }
};

struct Prefix {
    int k_start;
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
    const char *fname_arg = (argc > 1) ? argv[1] : "SR.ENUM2.gso";
    double gh_top = (argc > 2) ? atof(argv[2]) : 1.005;
    double gh_leaf = (argc > 3) ? atof(argv[3]) : gh_top;
    double lin = (argc > 4) ? atof(argv[4]) : 3.0;
    const char *yc_arg = (argc > 5) ? argv[5] : "0.0";
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

    __int128 max_x = ((__int128)19571878850562ULL * 1000000ULL + 201959ULL) * ((__int128)1 << 30);
    __int128 margin_x = (__int128)10 << 30;

    auto t_start = std::chrono::steady_clock::now();
    std::atomic<uint64_t> total_nodes{0};
    std::atomic<uint64_t> total_clusters{0};
    std::atomic<uint64_t> total_newton{0};
    std::atomic<uint64_t> pruned_at_cut{0};
    std::mutex out_mtx;
    ConcurrentBitset seen_y_buckets;
    ConcurrentBitset seen_print_buckets;
    double global_best_abs = 0.0;
    double global_best_h140 = 0.0;

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

        std::vector<double> bounds(n);
        double r2_leaf_global = (base_gh * gh_leaf) * (base_gh * gh_leaf);
        for (int k = 0; k < n; ++k) {
            if (k >= k_y_cut) {
                double i = (double)(n - k);
                double u = i / (double)m_act;
                double p_act = std::min(1.0, 0.035 + 0.965 * u + 0.055 * std::sin(M_PI * u));
                bounds[k] = r2_cut * p_act;
            } else {
                double i = (double)(n - k);
                double rr1 = i / n;
                double rr2 = (lin - 1.0) * std::sqrt(i * (n - i) / (n / 2.0 + 1.0)) / n;
                double p = std::min(1.0, rr1 + rr2);
                bounds[k] = (k_y_cut == 1) ? (r2_cut * 10.0) : (r2_leaf_global * std::min(1.0, p + 0.12));
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

            auto t0 = std::chrono::steady_clock::now();
            std::vector<Prefix> prefixes;
            Prefix init_p;
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
                double bnd = bounds[k];
                for (int64_t step = 0;; ++step) {
                    int64_t xx;
                    if (step == 0) xx = x0;
                    else {
                        int64_t mag = (step + 1) >> 1;
                        xx = (step & 1) ? (x0 + dir * mag) : (x0 - dir * mag);
                    }
                    double a = (double)xx - c;
                    double nd = p.dist + a * a * rs[k];
                    if (nd > bnd) break;
                    p.x[k] = xx;
                    __int128 nx = p.cur_x + (__int128)xx * xcol[k];
                    double old_d = p.dist;
                    __int128 old_x = p.cur_x;
                    p.dist = nd;
                    p.cur_x = nx;
                    self(self, k - 1, p);
                    p.dist = old_d;
                    p.cur_x = old_x;
                }
            };

            gen_prefixes(gen_prefixes, n - 1, init_p);
            uint64_t prev_clusters = total_clusters.load();
            uint64_t prev_newton = total_newton.load();

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

                    auto rec_top = [&](auto &self, int k, double dist, __int128 cur_x) -> void {
                        local_nodes++;
                        if (k < k_y_cut) {
                            if (cur_x <= -margin_x || cur_x >= max_x + margin_x) {
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
                                float screen_m140 = 0.0f;
                                if (!zd.fast_screen((unsigned __int128)full_x, screen_m140)) return;
                                uint64_t y_bkt = (uint64_t)(full_x >> 22);
                                if (screen_m140 >= 0.900f) y_bkt ^= 0x5a5a5a5a5a5a5a5aULL;
                                if (!seen_y_buckets.test_and_set(y_bkt)) return;
                                uint64_t cid = ++total_clusters;
                                PolishResult pr = zd.polish((unsigned __int128)full_x);
                                if (pr.full_refined) ++total_newton;
                                double ah = std::abs(pr.h_val);
                                double ah140 = std::abs(pr.h140_val);
                                double ah256 = std::abs(pr.h256_val);
                                if (ah >= 0.960 || ah > global_best_abs || ah140 >= 0.928 || ah140 > global_best_h140) {
                                    std::lock_guard<std::mutex> lk(out_mtx);
                                    bool new_best_h = (ah > global_best_abs);
                                    bool new_best_140 = (ah140 > global_best_h140);
                                    if (new_best_h) global_best_abs = ah;
                                    if (new_best_140) global_best_h140 = ah140;
                                    __int128 peak_x = full_x + (__int128)std::llround(pr.dy * (double)(1ULL << 30));
                                    uint64_t p_bkt = (uint64_t)(peak_x >> 25);
                                    bool first_print = seen_print_buckets.test_and_set(p_bkt);
                                    if (new_best_h || new_best_140 || (first_print && (ah >= 0.960 || ah140 >= 0.928))) {
                                        char xbuf[64];
                                        print_int128(full_x, xbuf);
                                        double y_approx = (double)full_x / (double)(1ULL << 30);
                                        printf("CLUSTER #%llu |h|=%.8f h=%+.8f |h140|=%.6f h140=%+.6f |h256|=%.6f y=%.6f (x=%s, dy=%+.7f)%s\n",
                                               (unsigned long long)cid, ah, pr.h_val, ah140, pr.h140_val, ah256,
                                               y_approx, xbuf, pr.dy,
                                               (ah > 1.00000000) ? " *** COUNTEREXAMPLE |h|>1 ***" : "");
                                        fflush(stdout);
                                    }
                                }
                            }
                            return;
                        }
                        if (k == k_y_cut) {
                            double inv_W = 1.0 / (double)(xcol[k] >= 0 ? xcol[k] : -xcol[k]);
                            double f_hi = (xcol[k] < 0)
                                ? ((double)(cur_x + margin_x) * inv_W)
                                : ((double)(max_x + margin_x - cur_x) * inv_W);
                            int64_t xx_hi = (int64_t)std::floor(f_hi);
                            __int128 nx0 = cur_x + (__int128)xx_hi * xcol[k];
                            if (nx0 <= -margin_x || nx0 >= max_x + margin_x) {
                                local_pruned++;
                                return;
                            }
                            double c = sigma_low[k];
                            const double *mt = &mu_T[k * STRIDE];
                            for (int i = k + 1; i < k_low; ++i) c -= x_dbl[i] * mt[i];
                            double bnd = bounds[k];
                            for (int64_t xx = xx_hi;; --xx) {
                                __int128 nx = cur_x + (__int128)xx * xcol[k];
                                if (nx <= -margin_x || nx >= max_x + margin_x) break;
                                double a = (double)xx - c;
                                double nd = dist + a * a * rs[k];
                                if (nd <= bnd) {
                                    x_int[k] = xx;
                                    x_dbl[k] = (double)xx;
                                    self(self, k - 1, nd, nx);
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
                        double bnd = bounds[k];
                        for (int64_t step = 0;; ++step) {
                            int64_t xx;
                            if (step == 0) xx = x0;
                            else {
                                int64_t mag = (step + 1) >> 1;
                                xx = (step & 1) ? (x0 + dir * mag) : (x0 - dir * mag);
                            }
                            double a = (double)xx - c;
                            double nd = dist + a * a * rs[k];
                            if (nd > bnd) break;
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
                            self(self, k - 1, nd, cur_x + (__int128)xx * xcol[k]);
                        }
                    };

                    rec_top(rec_top, pf.k_start, pf.dist, pf.cur_x);
                }
                total_nodes += local_nodes;
                pruned_at_cut += local_pruned;
            }
            auto t1 = std::chrono::steady_clock::now();
            double sec = std::chrono::duration<double>(t1 - t0).count();
            fprintf(stderr, "[%s (%zu/%zu) yc=%.2fe18 (%zu/%zu)] %.2f s | new_clus=%llu (newton=%llu) | tot_uniq=%llu | best_h=%.8f | best_h140=%.6f\n",
                    fname, f_idx + 1, fnames.size(), y_center_e18, yc_idx + 1, yc_list.size(), sec,
                    (unsigned long long)(total_clusters.load() - prev_clusters),
                    (unsigned long long)(total_newton.load() - prev_newton),
                    (unsigned long long)total_clusters.load(),
                    global_best_abs, global_best_h140);
        }
    }

    auto t_end = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(t_end - t_start).count();
    fprintf(stderr, "ALL DONE in %.2f s | nodes=%llu (%.2f Gnodes/s) | pruned_at_cut=%llu | unique_clusters=%llu (newton=%llu) | best_abs=%.8f | best_h140=%.6f\n",
            sec,
            (unsigned long long)total_nodes.load(),
            (double)total_nodes.load() / (sec * 1e9),
            (unsigned long long)pruned_at_cut.load(),
            (unsigned long long)total_clusters.load(),
            (unsigned long long)total_newton.load(),
            global_best_abs,
            global_best_h140);
    return 0;
}
