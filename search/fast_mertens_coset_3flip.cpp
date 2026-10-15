// fast_mertens_coset_3flip.cpp
// Technique 1: Multi-Rogue 3-Flip/4-Flip Cosets & Heavy-Zero (j in 3..8) Phase Shifts
// Combines exact sensitivity-matrix coset shifts (from fast_mertens_coset_dir1.cpp)
// with jump/step envelope pruning (max_jumps=14, max_a=2.60 from fast_mertens_deep.cpp).

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <omp.h>

static constexpr int MAXN = 80;
static constexpr int STRIDE = 80;
static constexpr int K_PRE = 256;
static constexpr int M_PRE = 38;
static constexpr int N_PRE = 2 * M_PRE + 1;
static constexpr int K0 = 512;
static constexpr int M0 = 228;
static constexpr int N0 = 2 * M0 + 1;
static constexpr int K1 = 2048;
static constexpr int M1 = 60;
static constexpr int N1 = 2 * M1 + 1;

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

    inline bool fast_screen(unsigned __int128 x, float &out_m64) const {
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
        out_m64 = m64;
        if (m64 < 0.785f) return false;
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
        return m128 >= 0.840f;
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
        if (pre_best < 0.885) return {best_h, best_dy};

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

struct ConcurrentBitset {
    static constexpr uint64_t MASK = (1ULL << 29) - 1;
    std::vector<uint64_t> words;
    ConcurrentBitset() : words(1ULL << 23, 0ULL) {}
    inline bool test_and_set(uint64_t key) {
        uint64_t h = key * 0x9E3779B97F4A7C15ULL;
        h ^= (h >> 32);
        uint64_t bit = h & MASK;
        uint64_t idx = bit >> 6;
        uint64_t mask = 1ULL << (bit & 63);
        uint64_t old = __atomic_fetch_or(&words[idx], mask, __ATOMIC_RELAXED);
        return (old & mask) == 0;
    }
};

static void print_int128(__int128 v, char *buf) {
    if (v == 0) { buf[0] = '0'; buf[1] = '\0'; return; }
    bool neg = v < 0;
    unsigned __int128 u = neg ? (unsigned __int128)(-v) : (unsigned __int128)v;
    char tmp[64];
    int p = 0;
    while (u > 0) {
        tmp[p++] = (char)('0' + (u % 10));
        u /= 10;
    }
    int out = 0;
    if (neg) buf[out++] = '-';
    while (p > 0) buf[out++] = tmp[--p];
    buf[out] = '\0';
}

static void format_exact_y(__int128 full_x, double dy, char *buf, size_t buf_sz) {
    int64_t y_int = (int64_t)(full_x >> 30);
    double y_frac = (double)(full_x & ((1ULL << 30) - 1)) / (double)(1ULL << 30) + dy;
    while (y_frac < 0.0) { y_int -= 1; y_frac += 1.0; }
    while (y_frac >= 1.0) { y_int += 1; y_frac -= 1.0; }
    long long frac_9 = (long long)std::llround(y_frac * 1e9);
    if (frac_9 >= 1000000000LL) { y_int += 1; frac_9 -= 1000000000LL; }
    snprintf(buf, buf_sz, "%lld.%09lld", (long long)y_int, frac_9);
}

struct CosetSpec {
    std::string label;
    int n_shifts;
    int a[4];
    double delta_e[4];
};

struct WorkItem {
    int coset_idx;
    int k_start;
    int jumps;
    double dist;
    __int128 cur_x;
    int64_t x[MAXN];
};

struct HitRecord {
    uint64_t cid;
    std::string coset;
    std::string gso_file;
    double ah;
    double h_val;
    double m64;
    __int128 full_x;
    double dy;
};

enum class SweepMode {
    HEAVY_SHIFTS,
    FLIPS_3,
    FLIPS_4
};

static std::vector<CosetSpec> build_cosets(int M_sens, SweepMode mode) {
    std::vector<CosetSpec> cosets;

    if (mode == SweepMode::HEAVY_SHIFTS) {
        // 1. Single shifts on a in 3..8 by Delta e_a in {-0.18, +0.18, -0.28, +0.28, 0.50} (30 cosets)
        const double single_de[5] = {-0.18, +0.18, -0.28, +0.28, 0.50};
        for (int a = 3; a <= 8 && a < M_sens; ++a) {
            for (int d = 0; d < 5; ++d) {
                char buf[64];
                snprintf(buf, sizeof(buf), "heavy1_j%d_%+.2f", a, single_de[d]);
                cosets.push_back({buf, 1, {a, 0, 0, 0}, {single_de[d], 0.0, 0.0, 0.0}});
            }
        }
        // 2. Pair shifts where a1 in 3..8 (Delta e_a1 in {-0.22, +0.22, 0.50})
        //    and a2 in 8..49 (Delta e_a2 in {-0.25, +0.25, 0.50}) (6 * 3 * 42 * 3 = 2,268 cosets)
        const double de1_list[3] = {-0.22, +0.22, 0.50};
        const double de2_list[3] = {-0.25, +0.25, 0.50};
        for (int a1 = 3; a1 <= 8 && a1 < M_sens; ++a1) {
            for (int d1 = 0; d1 < 3; ++d1) {
                for (int a2_raw = 8; a2_raw <= 49 && a2_raw < M_sens; ++a2_raw) {
                    int a2 = (a1 == a2_raw && 50 < M_sens) ? 50 : a2_raw;
                    for (int d2 = 0; d2 < 3; ++d2) {
                        char buf[64];
                        snprintf(buf, sizeof(buf), "heavy2_j%d_%+.2f_j%d_%+.2f",
                                 a1, de1_list[d1], a2, de2_list[d2]);
                        cosets.push_back({buf, 2, {a1, a2, 0, 0}, {de1_list[d1], de2_list[d2], 0.0, 0.0}});
                    }
                }
            }
        }
    } else if (mode == SweepMode::FLIPS_3) {
        // 3-Zero Simultaneous Flips: C(24, 3) = 2,024 triples from A_24 with Delta e = 0.50
        const int A24_raw[24] = {
            10, 12, 14, 15, 16, 17, 18, 20, 22, 24, 26, 28,
            30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 54
        };
        std::vector<int> A24;
        for (int i = 0; i < 24; ++i) {
            if (A24_raw[i] < M_sens) A24.push_back(A24_raw[i]);
        }
        const int sz = (int)A24.size();
        for (int i = 0; i < sz; ++i) {
            for (int j = i + 1; j < sz; ++j) {
                for (int k = j + 1; k < sz; ++k) {
                    char buf[64];
                    snprintf(buf, sizeof(buf), "3flip_j%d_j%d_j%d", A24[i], A24[j], A24[k]);
                    cosets.push_back({buf, 3, {A24[i], A24[j], A24[k], 0}, {0.50, 0.50, 0.50, 0.0}});
                }
            }
        }
    } else if (mode == SweepMode::FLIPS_4) {
        // Guided 4-Zero Simultaneous Flips: C(14, 4) = 1,001 quadruples from A_14 with Delta e = 0.50
        const int A14_raw[14] = {
            14, 17, 20, 23, 26, 28, 31, 34, 38, 42, 46, 48, 50, 54
        };
        std::vector<int> A14;
        for (int i = 0; i < 14; ++i) {
            if (A14_raw[i] < M_sens) A14.push_back(A14_raw[i]);
        }
        const int sz = (int)A14.size();
        for (int i = 0; i < sz; ++i) {
            for (int j = i + 1; j < sz; ++j) {
                for (int k = j + 1; k < sz; ++k) {
                    for (int l = k + 1; l < sz; ++l) {
                        char buf[64];
                        snprintf(buf, sizeof(buf), "4flip_j%d_j%d_j%d_j%d", A14[i], A14[j], A14[k], A14[l]);
                        cosets.push_back({buf, 4, {A14[i], A14[j], A14[k], A14[l]}, {0.50, 0.50, 0.50, 0.50}});
                    }
                }
            }
        }
    }

    return cosets;
}

int main(int argc, char **argv) {
    if (argc == 4 && std::string(argv[1]) == "--repolish") {
        ZeroData zd;
        if (!zd.load(argv[2])) return 1;
        FILE *fl = fopen(argv[3], "r");
        if (!fl) return 1;
        char line[1024];
        while (fgets(line, sizeof(line), fl)) {
            if (strncmp(line, "CLUSTER #", 9) != 0) continue;
            const char *px = strstr(line, "(x=");
            if (!px) continue;
            px += 3;
            unsigned __int128 x = 0;
            while (*px >= '0' && *px <= '9') {
                x = x * 10 + (unsigned)(*px - '0');
                ++px;
            }
            auto [h, dy] = zd.polish(x);
            double ah = std::abs(h);
            char xbuf[64], ybuf[64];
            print_int128((__int128)x, xbuf);
            format_exact_y((__int128)x, dy, ybuf, sizeof(ybuf));
            char cbuf[128] = {0};
            const char *pc = strchr(line, '[');
            const char *pc2 = pc ? strchr(pc, ']') : nullptr;
            if (pc && pc2 && pc2 - pc < 120) memcpy(cbuf, pc, pc2 - pc + 1);
            printf("REPOLISHED %s |h|=%.8f h=%+.8f y=%s (x=%s, dy=%+.7f)\n",
                   cbuf, ah, h, ybuf, xbuf, dy);
        }
        fclose(fl);
        return 0;
    }

    SweepMode mode = SweepMode::FLIPS_3;
    const char *mode_name = "3flips";
    std::vector<char*> pos_args;
    pos_args.push_back(argv[0]);

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--heavy-shifts") == 0) {
            mode = SweepMode::HEAVY_SHIFTS;
            mode_name = "heavy-shifts";
        } else if (std::strcmp(argv[i], "--3flips") == 0) {
            mode = SweepMode::FLIPS_3;
            mode_name = "3flips";
        } else if (std::strcmp(argv[i], "--4flips") == 0) {
            mode = SweepMode::FLIPS_4;
            mode_name = "4flips";
        } else if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            ++i;
            if (std::strcmp(argv[i], "heavy-shifts") == 0 || std::strcmp(argv[i], "heavy") == 0) {
                mode = SweepMode::HEAVY_SHIFTS;
                mode_name = "heavy-shifts";
            } else if (std::strcmp(argv[i], "3flips") == 0 || std::strcmp(argv[i], "3flip") == 0) {
                mode = SweepMode::FLIPS_3;
                mode_name = "3flips";
            } else if (std::strcmp(argv[i], "4flips") == 0 || std::strcmp(argv[i], "4flip") == 0) {
                mode = SweepMode::FLIPS_4;
                mode_name = "4flips";
            } else {
                fprintf(stderr, "Unknown --mode value: %s\n", argv[i]);
                return 1;
            }
        } else {
            pos_args.push_back(argv[i]);
        }
    }

    if (pos_args.size() < 5) {
        fprintf(stderr, "Usage: %s [--heavy-shifts|--3flips|--4flips|--mode <m>] <zeros.bin> <sens_file> <gh_top> <yc_csv> [gso1 gso2 ...]\n", argv[0]);
        return 1;
    }
    const char *zeros_path = pos_args[1];
    const char *sens_path = pos_args[2];
    double gh_top = std::atof(pos_args[3]);
    std::string yc_str = pos_args[4];

    // Parse yc_csv (supports comma-separated values as well as start:step:end ranges)
    std::vector<double> yc_list;
    {
        const char *p = yc_str.c_str();
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
    }

    // Parse comma-separated sens_path list
    std::vector<std::string> sens_files;
    {
        std::string sp_all(sens_path);
        size_t pos = 0;
        while (pos < sp_all.size()) {
            size_t comma = sp_all.find(',', pos);
            if (comma == std::string::npos) {
                sens_files.push_back(sp_all.substr(pos));
                break;
            } else {
                sens_files.push_back(sp_all.substr(pos, comma - pos));
                pos = comma + 1;
            }
        }
    }

    std::vector<std::string> explicit_fnames;
    for (size_t i = 5; i < pos_args.size(); ++i) explicit_fnames.push_back(pos_args[i]);

    ZeroData zd;
    if (!zd.load(zeros_path)) {
        fprintf(stderr, "Failed to load %s\n", zeros_path);
        return 1;
    }

    // y_min = 0.0, y_max = 3.4849e18
    __int128 max_x = ((__int128)3484900000000ULL * 1000000ULL) * ((__int128)1 << 30);
    __int128 margin_x = (__int128)10 << 30;

    auto t_start = std::chrono::steady_clock::now();
    std::atomic<uint64_t> total_nodes{0};
    std::atomic<uint64_t> total_clusters{0};
    std::atomic<uint64_t> pruned_at_cut{0};
    std::mutex out_mtx;
    ConcurrentBitset seen_y_buckets;
    double global_best_abs = 0.0;
    std::vector<HitRecord> recorded_hits;
    size_t total_gso_files = 0;
    size_t last_cosets_count = 0;

    for (size_t s_idx = 0; s_idx < sens_files.size(); ++s_idx) {
        const std::string &cur_sens_path = sens_files[s_idx];
        FILE *fs = fopen(cur_sens_path.c_str(), "rb");
        if (!fs) {
            std::string alt_s = std::string("gso/") + cur_sens_path;
            fs = fopen(alt_s.c_str(), "rb");
        }
        if (!fs) {
            fprintf(stderr, "Failed to load %s\n", cur_sens_path.c_str());
            return 1;
        }
        int M_sens = 0, n_sens = 0;
        if (fread(&M_sens, 4, 1, fs) != 1 || fread(&n_sens, 4, 1, fs) != 1) {
            fclose(fs);
            return 1;
        }
        std::vector<double> S(M_sens * n_sens);
        if (fread(S.data(), 8, M_sens * n_sens, fs) != (size_t)(M_sens * n_sens)) {
            fclose(fs);
            return 1;
        }
        fclose(fs);

        std::vector<CosetSpec> cosets = build_cosets(M_sens, mode);
        last_cosets_count = cosets.size();
        fprintf(stderr, "Loaded %d zeros, sensitivity matrix %s (%d x %d), mode=%s (%zu cosets), gh_top=%.4f, max_jumps=14, max_a=2.60\n",
                zd.n_zeros, cur_sens_path.c_str(), M_sens, n_sens, mode_name, cosets.size(), gh_top);
        printf("Loaded %d zeros, sensitivity matrix %s (%d x %d), mode=%s (%zu cosets), gh_top=%.4f, max_jumps=14, max_a=2.60\n",
               zd.n_zeros, cur_sens_path.c_str(), M_sens, n_sens, mode_name, cosets.size(), gh_top);
        fflush(stdout);

        std::vector<std::string> fnames = explicit_fnames;
        if (fnames.empty()) {
            if (cur_sens_path.size() > 5 && cur_sens_path.substr(cur_sens_path.size() - 5) == ".sens") {
                std::string stem = cur_sens_path.substr(0, cur_sens_path.size() - 5);
                fnames.push_back(stem + "_neg.gso");
                fnames.push_back(stem + "_pos.gso");
            }
        }
        total_gso_files += fnames.size();

    for (size_t f_idx = 0; f_idx < fnames.size(); ++f_idx) {
        const char *fname = fnames[f_idx].c_str();
        FILE *f = fopen(fname, "rb");
        if (!f) {
            std::string alt = std::string("gso/") + fname;
            f = fopen(alt.c_str(), "rb");
        }
        if (!f) { fprintf(stderr, "Failed to load %s\n", fname); continue; }
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

        if (n != n_sens) {
            fprintf(stderr, "Dimension mismatch: GSO n=%d vs sens n=%d\n", n, n_sens);
            continue;
        }

        std::vector<__int128> xcol(n);
        for (int i = 0; i < n; ++i) {
            unsigned __int128 u = (unsigned __int128)lo[i] | ((unsigned __int128)hi[i] << 64);
            xcol[i] = (__int128)u;
        }

        int k_y_cut = 1;
        std::vector<double> mu_T(n * STRIDE, 0.0);
        for (int k = 0; k < n; ++k) {
            for (int i = k + 1; i < n; ++i) {
                mu_T[k * STRIDE + i] = mu[i * n + k];
            }
        }

        int m_act = n - k_y_cut;
        double logdet_act = 0.0;
        for (int i = k_y_cut; i < n; ++i) logdet_act += 0.5 * std::log(rs[i]);
        double heur_act = std::lgamma(1.0 + m_act / 2.0) / m_act - 0.5 * std::log(M_PI);
        double gh_act = std::exp(logdet_act / m_act) * std::exp(heur_act);
        double dim_scale = 1.2935;
        double r2_cut = (dim_scale * gh_act * gh_top) * (dim_scale * gh_act * gh_top);

        // Variance-envelope upper-tree & per-step pruning arrays (max_jumps = 14, max_a = 2.60)
        std::vector<double> bounds(n);
        std::vector<double> max_inc(n);
        std::vector<int> max_jumps(n);
        std::vector<int> floor_jumps(n, 0);
        for (int k = 0; k < n; ++k) {
            if (k >= k_y_cut) {
                int i_lvl = n - k;
                double u = (double)i_lvl / (double)m_act;
                double p_deep = (u <= 0.30)
                    ? (0.040 + 0.68 * u + 0.30 * u * u)
                    : std::min(1.0, 0.062 + 0.88 * u + 0.12 * u * u);
                bounds[k] = r2_cut * p_deep;

                double max_a = (i_lvl <= 4)  ? 2.60 :
                               (i_lvl <= 16) ? 1.85 :
                               (i_lvl <= 26) ? 1.15 :
                               (i_lvl <= 45) ? 0.90 : 0.65;
                max_inc[k] = max_a * max_a * rs[k];

                max_jumps[k] = (i_lvl <= 3)  ? 3 :
                               (i_lvl <= 6)  ? 5 :
                               (i_lvl <= 12) ? 8 :
                               (i_lvl <= 18) ? 10 :
                               (i_lvl <= 26) ? 12 : 14;

                floor_jumps[k] = (i_lvl >= 42) ? 14 :
                                 (i_lvl >= 35) ? 13 :
                                 (i_lvl >= 27) ? 11 : 0;
            } else {
                bounds[k] = r2_cut * 10.0;
                max_inc[k] = bounds[k];
                max_jumps[k] = 60;
                floor_jumps[k] = 0;
            }
        }

        const int cur_k_split = n - 9;
        const size_t n_cosets = cosets.size();

        for (size_t yc_idx = 0; yc_idx < yc_list.size(); ++yc_idx) {
            double y_center_e18 = yc_list[yc_idx];
            if (y_center_e18 > 1e12) y_center_e18 *= 1e-18;
            double yc = y_center_e18 * 1e18;
            auto t0 = std::chrono::steady_clock::now();

            // Precompute shifted sigma table [n_cosets][MAXN]
            std::vector<double> coset_sigma(n_cosets * MAXN, 0.0);
            for (size_t c_idx = 0; c_idx < n_cosets; ++c_idx) {
                const CosetSpec &cs = cosets[c_idx];
                double *sig_c = &coset_sigma[c_idx * MAXN];
                for (int k = 0; k < n; ++k) {
                    double v = base_sigma[k] + yc * dsigma[k];
                    if (k >= 1) {
                        for (int r = 0; r < cs.n_shifts; ++r) {
                            v += cs.delta_e[r] * S[cs.a[r] * n + k];
                        }
                    }
                    sig_c[k] = v;
                }
            }

            // Generate prefixes in parallel across all cosets
            std::vector<std::vector<WorkItem>> per_coset_items(n_cosets);
            #pragma omp parallel for schedule(dynamic, 1)
            for (size_t c_idx = 0; c_idx < n_cosets; ++c_idx) {
                const double *sig_c = &coset_sigma[c_idx * MAXN];
                auto &vec = per_coset_items[c_idx];
                WorkItem init_w;
                init_w.coset_idx = (int)c_idx;
                init_w.jumps = 0;
                init_w.dist = 0.0;
                init_w.cur_x = 0;
                memset(init_w.x, 0, sizeof(init_w.x));

                auto gen_pf = [&](auto &self, int k, WorkItem &w) -> void {
                    if (vec.size() >= 25000 || (k <= cur_k_split && (k <= cur_k_split - 4 || w.dist >= 0.65 * bounds[k + 1]))) {
                        w.k_start = k;
                        vec.push_back(w);
                        return;
                    }
                    double c = sig_c[k];
                    const double *mt = &mu_T[k * STRIDE];
                    for (int i = k + 1; i < n; ++i) c -= (double)w.x[i] * mt[i];
                    int64_t x0 = (int64_t)std::llround(c);
                    int dir = (c >= (double)x0) ? 1 : -1;
                    double eff_bnd = std::min(bounds[k], w.dist + max_inc[k]);
                    int mj = max_jumps[k];
                    int fj = floor_jumps[k];
                    for (int64_t step = 0;; ++step) {
                        int nj = w.jumps + (step > 0 ? 1 : 0);
                        if (nj > mj) break;
                        int64_t xx;
                        if (step == 0) xx = x0;
                        else {
                            int64_t mag = (step + 1) >> 1;
                            xx = (step & 1) ? (x0 + dir * mag) : (x0 - dir * mag);
                        }
                        double a = (double)xx - c;
                        double nd = w.dist + a * a * rs[k];
                        if (nd > eff_bnd) break;
                        w.x[k] = xx;
                        __int128 nx = w.cur_x + (__int128)xx * xcol[k];
                        double old_d = w.dist;
                        __int128 old_x = w.cur_x;
                        int old_j = w.jumps;
                        w.dist = nd;
                        w.cur_x = nx;
                        w.jumps = (nj < fj) ? fj : nj;
                        self(self, k - 1, w);
                        w.dist = old_d;
                        w.cur_x = old_x;
                        w.jumps = old_j;
                    }
                };
                gen_pf(gen_pf, n - 1, init_w);
            }

            std::vector<WorkItem> all_items;
            size_t tot_pf = 0;
            for (size_t c_idx = 0; c_idx < n_cosets; ++c_idx) tot_pf += per_coset_items[c_idx].size();
            all_items.reserve(tot_pf);
            for (size_t c_idx = 0; c_idx < n_cosets; ++c_idx) {
                all_items.insert(all_items.end(), per_coset_items[c_idx].begin(), per_coset_items[c_idx].end());
            }

            uint64_t prev_clusters = total_clusters.load();
            uint64_t prev_nodes = total_nodes.load();

            #pragma omp parallel
            {
                uint64_t local_nodes = 0;
                uint64_t local_pruned = 0;
                double x_dbl[MAXN];
                int64_t x_int[MAXN];
                double sigma_pf[MAXN];
                double sigma_hi[MAXN];
                double sigma_mid[MAXN];
                const int k_hi  = k_y_cut + (n - k_y_cut) * 65 / 100; // ~40
                const int k_mid = k_y_cut + (n - k_y_cut) * 35 / 100; // ~22

                #pragma omp for schedule(dynamic, 8)
                for (size_t idx = 0; idx < all_items.size(); ++idx) {
                    const WorkItem &pf = all_items[idx];
                    const double *sig_c = &coset_sigma[pf.coset_idx * MAXN];
                    for (int i = pf.k_start + 1; i < n; ++i) {
                        x_int[i] = pf.x[i];
                        x_dbl[i] = (double)pf.x[i];
                    }
                    for (int j = 0; j <= pf.k_start; ++j) {
                        double s = sig_c[j];
                        const double *mt = &mu_T[j * STRIDE];
                        for (int i = pf.k_start + 1; i < n; ++i) s -= x_dbl[i] * mt[i];
                        sigma_pf[j] = s;
                    }

                    auto eval_cut_and_leaf = [&](double dist_at_cut, __int128 cur_x_at_cut) -> void {
                        local_nodes++;
                        double inv_W = 1.0 / (double)(xcol[k_y_cut] >= 0 ? xcol[k_y_cut] : -xcol[k_y_cut]);
                        double f_hi = (xcol[k_y_cut] < 0)
                            ? ((double)(cur_x_at_cut + margin_x) * inv_W)
                            : ((double)(max_x + margin_x - cur_x_at_cut) * inv_W);
                        double f_lo = (xcol[k_y_cut] < 0)
                            ? ((double)(cur_x_at_cut - max_x - margin_x) * inv_W)
                            : ((double)(-margin_x - cur_x_at_cut) * inv_W);
                        int64_t xx_hi = (int64_t)std::floor(f_hi);
                        int64_t xx_lo = (int64_t)std::ceil(f_lo);
                        if (xx_lo > xx_hi) {
                            local_pruned++;
                            return;
                        }
                        double bnd = bounds[k_y_cut];
                        double rem = bnd - dist_at_cut;
                        if (rem < 0.0) {
                            local_pruned++;
                            return;
                        }
                        double c = sigma_mid[k_y_cut];
                        const double *mt = &mu_T[k_y_cut * STRIDE];
                        for (int i = k_y_cut + 1; i < k_mid; ++i) c -= x_dbl[i] * mt[i];
                        double rad = std::sqrt(rem / rs[k_y_cut]);
                        int64_t gso_lo = (int64_t)std::ceil(c - rad);
                        int64_t gso_hi = (int64_t)std::floor(c + rad);
                        if (gso_lo > xx_lo) xx_lo = gso_lo;
                        if (gso_hi < xx_hi) xx_hi = gso_hi;
                        if (xx_lo > xx_hi) {
                            local_pruned++;
                            return;
                        }
                        for (int64_t xx = xx_hi; xx >= xx_lo; --xx) {
                            __int128 nx = cur_x_at_cut + (__int128)xx * xcol[k_y_cut];
                            if (nx <= -margin_x || nx >= max_x + margin_x) continue;
                            double a = (double)xx - c;
                            double nd = dist_at_cut + a * a * rs[k_y_cut];
                            if (nd <= bnd) {
                                x_int[k_y_cut] = xx;
                                x_dbl[k_y_cut] = (double)xx;
                                local_nodes++;
                                double c0 = sigma_mid[0];
                                const double *mt0 = &mu_T[0];
                                for (int i = 1; i < k_mid; ++i) c0 -= x_dbl[i] * mt0[i];
                                int64_t x0 = (int64_t)std::llround(c0);
                                __int128 full_x = nx + (__int128)x0 * xcol[0];
                                if (full_x > 0 && full_x < max_x) {
                                    float m64 = 0.0f;
                                    if (!zd.fast_screen((unsigned __int128)full_x, m64)) continue;
                                    uint64_t y_bkt = (uint64_t)(full_x >> 21);
                                    if (!seen_y_buckets.test_and_set(y_bkt)) continue;
                                    uint64_t cid = ++total_clusters;
                                    auto [h_val, dy] = zd.polish((unsigned __int128)full_x);
                                    double ah = std::abs(h_val);
                                    if (ah >= 0.960 || ah > global_best_abs) {
                                        std::lock_guard<std::mutex> lk(out_mtx);
                                        if (ah > global_best_abs) global_best_abs = ah;
                                        char xbuf[64], ybuf[64];
                                        print_int128(full_x, xbuf);
                                        format_exact_y(full_x, dy, ybuf, sizeof(ybuf));
                                        if (ah >= 0.960) {
                                            recorded_hits.push_back({
                                                cid, cosets[pf.coset_idx].label, fname,
                                                ah, h_val, (double)m64, full_x, dy
                                            });
                                        }
                                        printf("CLUSTER #%llu [coset=%s] |h|=%.8f h=%+.8f m64=%.4f y=%s (x=%s, dy=%+.7f)%s\n",
                                               (unsigned long long)cid, cosets[pf.coset_idx].label.c_str(),
                                               ah, h_val, m64, ybuf, xbuf, dy,
                                               (ah > 1.00000006) ? " *** RECORD BREAKING HIT ***" : "");
                                        fflush(stdout);
                                    }
                                }
                            }
                        }
                    };

                    auto rec_top = [&](auto &self, int k, double dist, __int128 cur_x, int jumps) -> void {
                        if (k == k_y_cut) {
                            eval_cut_and_leaf(dist, cur_x);
                            return;
                        }
                        // Fast straight-line Babai path when jumps == 14 (no more step > 0 allowed)
                        if (jumps >= 14) {
                            double d = dist;
                            __int128 cx = cur_x;
                            for (int kk = k; kk > k_y_cut; --kk) {
                                local_nodes++;
                                double c;
                                const double *mt = &mu_T[kk * STRIDE];
                                if (kk < k_mid) {
                                    c = sigma_mid[kk];
                                    for (int i = kk + 1; i < k_mid; ++i) c -= x_dbl[i] * mt[i];
                                } else if (kk < k_hi) {
                                    c = sigma_hi[kk];
                                    for (int i = kk + 1; i < k_hi; ++i) c -= x_dbl[i] * mt[i];
                                } else {
                                    c = sigma_pf[kk];
                                    for (int i = kk + 1; i <= pf.k_start; ++i) c -= x_dbl[i] * mt[i];
                                }
                                int64_t xx = (int64_t)std::llround(c);
                                double a = (double)xx - c;
                                double eff_bnd = std::min(bounds[kk], d + max_inc[kk]);
                                d += a * a * rs[kk];
                                if (d > eff_bnd) return;
                                x_int[kk] = xx;
                                x_dbl[kk] = (double)xx;
                                cx += (__int128)xx * xcol[kk];
                                if (kk == k_hi) {
                                    for (int j = 0; j < k_hi; ++j) {
                                        double s = sigma_pf[j];
                                        const double *mtj = &mu_T[j * STRIDE];
                                        for (int i = k_hi; i <= pf.k_start; ++i) s -= x_dbl[i] * mtj[i];
                                        sigma_hi[j] = s;
                                    }
                                } else if (kk == k_mid) {
                                    for (int j = 0; j < k_mid; ++j) {
                                        double s = sigma_hi[j];
                                        const double *mtj = &mu_T[j * STRIDE];
                                        for (int i = k_mid; i < k_hi; ++i) s -= x_dbl[i] * mtj[i];
                                        sigma_mid[j] = s;
                                    }
                                }
                            }
                            eval_cut_and_leaf(d, cx);
                            return;
                        }

                        local_nodes++;
                        double c;
                        const double *mt = &mu_T[k * STRIDE];
                        if (k < k_mid) {
                            c = sigma_mid[k];
                            for (int i = k + 1; i < k_mid; ++i) c -= x_dbl[i] * mt[i];
                        } else if (k < k_hi) {
                            c = sigma_hi[k];
                            for (int i = k + 1; i < k_hi; ++i) c -= x_dbl[i] * mt[i];
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
                            if (k == k_hi) {
                                for (int j = 0; j < k_hi; ++j) {
                                    double s = sigma_pf[j];
                                    const double *mtj = &mu_T[j * STRIDE];
                                    for (int i = k_hi; i <= pf.k_start; ++i) s -= x_dbl[i] * mtj[i];
                                    sigma_hi[j] = s;
                                }
                            } else if (k == k_mid) {
                                for (int j = 0; j < k_mid; ++j) {
                                    double s = sigma_hi[j];
                                    const double *mtj = &mu_T[j * STRIDE];
                                    for (int i = k_mid; i < k_hi; ++i) s -= x_dbl[i] * mtj[i];
                                    sigma_mid[j] = s;
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
            uint64_t file_nodes = total_nodes.load() - prev_nodes;
            fprintf(stderr, "[%s yc=%.2fe18 | %zu cosets | %zu prefixes] %.2f s | nodes=%.3fB (%.2f Gnodes/s) | new_clusters=%llu | total_unique=%llu | best=%.8f\n",
                    fname, y_center_e18, n_cosets, all_items.size(), sec,
                    (double)file_nodes / 1e9, (double)file_nodes / (std::max(sec, 1e-6) * 1e9),
                    (unsigned long long)(total_clusters.load() - prev_clusters),
                    (unsigned long long)total_clusters.load(), global_best_abs);
            printf("[%s yc=%.2fe18 | %zu cosets | %zu prefixes] %.2f s | nodes=%.3fB (%.2f Gnodes/s) | new_clusters=%llu | total_unique=%llu | best=%.8f\n",
                   fname, y_center_e18, n_cosets, all_items.size(), sec,
                   (double)file_nodes / 1e9, (double)file_nodes / (std::max(sec, 1e-6) * 1e9),
                   (unsigned long long)(total_clusters.load() - prev_clusters),
                   (unsigned long long)total_clusters.load(), global_best_abs);
            fflush(stdout);
        }
    }
    } // end sens_files loop

    auto t_end = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(t_end - t_start).count();
    fprintf(stderr, "ALL DONE in %.2f s | nodes=%llu (%.2f Gnodes/s) | pruned_at_cut=%llu | unique_clusters=%llu | best_abs=%.8f\n",
            sec,
            (unsigned long long)total_nodes.load(),
            (double)total_nodes.load() / (std::max(sec, 1e-6) * 1e9),
            (unsigned long long)pruned_at_cut.load(),
            (unsigned long long)total_clusters.load(),
            global_best_abs);
    printf("\n=== SUMMARY (mode=%s, cosets=%zu, gso_files=%zu, gh_top=%.4f) ===\n",
           mode_name, last_cosets_count, total_gso_files, gh_top);
    printf("ALL DONE in %.2f s | nodes=%llu (%.3fB, %.2f Gnodes/s) | pruned_at_cut=%llu | unique_clusters=%llu | best_abs=%.8f\n",
           sec,
           (unsigned long long)total_nodes.load(),
           (double)total_nodes.load() / 1e9,
           (double)total_nodes.load() / (std::max(sec, 1e-6) * 1e9),
           (unsigned long long)pruned_at_cut.load(),
           (unsigned long long)total_clusters.load(),
           global_best_abs);

    if (!recorded_hits.empty()) {
        std::sort(recorded_hits.begin(), recorded_hits.end(), [](const HitRecord &a, const HitRecord &b) {
            return a.ah > b.ah;
        });
        printf("\n=== TOP HITS (|h_10k| >= 0.960, total=%zu) ===\n", recorded_hits.size());
        printf("%-4s  %-10s  %-11s  %-7s  %-29s  %-28s  %-24s\n",
               "Rank", "|h_10k|", "h_10k", "m64", "y", "coset", "gso_file");
        size_t show = std::min<size_t>(recorded_hits.size(), 50);
        for (size_t i = 0; i < show; ++i) {
            const auto &r = recorded_hits[i];
            char xbuf[64], ybuf[64];
            print_int128(r.full_x, xbuf);
            format_exact_y(r.full_x, r.dy, ybuf, sizeof(ybuf));
            printf("#%-3zu  %10.8f  %+11.8f  %7.4f  %-29s  %-28s  %-24s (x=%s, dy=%+.7f)\n",
                   i + 1, r.ah, r.h_val, r.m64, ybuf, r.coset.c_str(), r.gso_file.c_str(), xbuf, r.dy);
        }
    }
    fflush(stdout);
    return 0;
}
