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

struct CosetSpec {
    std::string label;
    int n_shifts;
    int a[3];
    double delta_e[3];
};

struct WorkItem {
    int coset_idx;
    int k_start;
    double dist;
    __int128 cur_x;
    int64_t x[MAXN];
};

static std::vector<CosetSpec> build_cosets(int M_sens, bool pairs_and_flips_only) {
    std::vector<CosetSpec> cosets;
    if (!pairs_and_flips_only) {
        // 0: Unshifted baseline
        cosets.push_back({"unshifted", 0, {0, 0, 0}, {0.0, 0.0, 0.0}});

        // Single-coordinate phase-offset cosets a in [8..M_sens-1], delta_e in {-0.20, +0.20}
        for (int a = 8; a < M_sens; ++a) {
            for (int sgn : {-1, 1}) {
                char buf[64];
                snprintf(buf, sizeof(buf), "1sh_j%d_%+d", a, sgn);
                cosets.push_back({buf, 1, {a, 0, 0}, {sgn * 0.20, 0.0, 0.0}});
            }
        }
    }

    // Single-coordinate half-period sign-flip cosets (delta_e = 0.50) for a in [12..M_sens-1]
    for (int a = 12; a < M_sens; ++a) {
        char buf[64];
        snprintf(buf, sizeof(buf), "1flip_j%d", a);
        cosets.push_back({buf, 1, {a, 0, 0}, {0.50, 0.0, 0.0}});
    }

    // Comprehensive 2-zero half-period sign-flip cosets (2flip_j<a1>_j<a2>, delta_e = {0.50, 0.50})
    // covering high-weight mid-band pairs a1 in [12..30], a2 in [25..48] with a1 < a2 (76 well-spaced pairs)
    std::set<std::pair<int, int>> seen_2flip;
    for (int a1 = 12; a1 <= 30; ++a1) {
        if (a1 >= M_sens) continue;
        for (int k = 0; k < 4; ++k) {
            int a2;
            if (a1 < 25) {
                a2 = 25 + ((a1 - 12) * 5 + k * 6) % 24;
            } else {
                int span = 48 - a1;
                a2 = (a1 + 1) + ((a1 - 25) * 3 + k * 5) % span;
            }
            if (a1 < a2 && a2 <= 48 && a2 < M_sens && seen_2flip.insert({a1, a2}).second) {
                char buf[64];
                snprintf(buf, sizeof(buf), "2flip_j%d_j%d", a1, a2);
                cosets.push_back({buf, 2, {a1, a2, 0}, {0.50, 0.50, 0.0}});
            }
        }
    }

    // Known rogue pair cosets from Peak1..Peak7 and KN_00..KN_13 + high-sensitivity mid-band pairs
    struct PairDef { int a1; double e1; int a2; double e2; const char *tag; };
    std::vector<PairDef> pairs = {
        {38, +0.18, 44, +0.18, "P1_18"}, {38, -0.18, 44, -0.18, "P1_m18"}, {38, 0.50, 44, 0.50, "P1_50"},
        {34, +0.18, 48, +0.18, "P2_18"}, {34, 0.50, 48, 0.50, "P2_50"}, {9, -0.18, 34, 0.50, "P2_mix1"}, {9, -0.18, 48, 0.50, "P2_mix2"},
        {31, +0.18, 47, -0.18, "P3_18"}, {31, 0.50, 47, 0.50, "P3_50"}, {31, 0.50, 52, 0.50, "P3_50b"}, {30, -0.18, 31, 0.50, "P3_mix"},
        {23, +0.18, 40, -0.18, "P4_18"}, {40, 0.50, 42, 0.50, "P4_50"}, {23, 0.50, 40, 0.50, "P4_50b"},
        {29, +0.18, 52, +0.18, "P5_18"}, {29, 0.50, 52, 0.50, "P5_50"}, {52, 0.50, 54, 0.50, "P5_50b"}, {13, 0.50, 52, 0.50, "P5_50c"},
        {16, +0.18, 17, +0.18, "P5_hm"}, {16, -0.18, 17, -0.18, "P5_hmm"},
        {45, +0.18, 52, -0.18, "P6_18"}, {45, 0.50, 52, 0.50, "P6_50"}, {11, +0.18, 45, 0.50, "P6_mix1"}, {11, 0.50, 52, 0.50, "P6_mix2"},
        {29, -0.18, 45, +0.18, "P7_18"}, {29, 0.50, 45, 0.50, "P7_50"}, {45, 0.50, 49, 0.50, "P7_50b"}, {29, 0.50, 49, 0.50, "P7_50c"},
        {36, -0.18, 37, +0.18, "KN00"}, {35, -0.18, 39, -0.18, "KN02"}, {36, +0.18, 44, -0.18, "KN03"},
        {25, -0.18, 44, -0.18, "KN04"}, {38, -0.18, 54, -0.18, "KN05"}, {39, +0.18, 40, -0.18, "KN06"},
        {9,  -0.18, 38, -0.18, "KN07"}, {17, +0.18, 42, +0.18, "KN08"}, {29, -0.18, 47, -0.18, "KN09"},
        {23, -0.18, 35, +0.18, "KN10"}, {37, -0.18, 54, +0.18, "KN11"}, {39, -0.18, 47, -0.18, "KN12"},
        {27, -0.18, 32, +0.18, "KN13"}
    };
    for (const auto &pr : pairs) {
        if (pr.a1 < M_sens && pr.a2 < M_sens) {
            char buf[64];
            snprintf(buf, sizeof(buf), "2sh_%s_j%d_j%d", pr.tag, pr.a1, pr.a2);
            cosets.push_back({buf, 2, {pr.a1, pr.a2, 0}, {pr.e1, pr.e2, 0.0}});
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
            double y_approx = (double)x / 1073741824.0 + dy;
            char xbuf[64];
            print_int128((__int128)x, xbuf);
            char cbuf[128] = {0};
            const char *pc = strchr(line, '[');
            const char *pc2 = pc ? strchr(pc, ']') : nullptr;
            if (pc && pc2 && pc2 - pc < 120) memcpy(cbuf, pc, pc2 - pc + 1);
            printf("REPOLISHED %s |h|=%.8f h=%+.8f y=%.6f (x=%s, dy=%+.7f)\n",
                   cbuf, ah, h, y_approx, xbuf, dy);
        }
        fclose(fl);
        return 0;
    }

    bool flag_pairs_and_flips = false;
    std::vector<char*> pos_args;
    pos_args.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--pairs-and-flips-only") {
            flag_pairs_and_flips = true;
        } else {
            pos_args.push_back(argv[i]);
        }
    }

    if (pos_args.size() < 5) {
        fprintf(stderr, "Usage: %s [--pairs-and-flips-only] <zeros.bin> <sens_file> <gh_top> <yc_csv> [gso1 gso2 ...]\n", argv[0]);
        return 1;
    }
    const char *zeros_path = pos_args[1];
    const char *sens_path = pos_args[2];
    double gh_top = std::atof(pos_args[3]);
    std::string yc_str = pos_args[4];

    std::vector<double> yc_list;
    size_t pos = 0;
    while (pos < yc_str.size()) {
        size_t comma = yc_str.find(',', pos);
        if (comma == std::string::npos) comma = yc_str.size();
        yc_list.push_back(std::atof(yc_str.substr(pos, comma - pos).c_str()));
        pos = comma + 1;
    }

    std::vector<std::string> fnames;
    for (size_t i = 5; i < pos_args.size(); ++i) fnames.push_back(pos_args[i]);

    ZeroData zd;
    if (!zd.load(zeros_path)) {
        fprintf(stderr, "Failed to load %s\n", zeros_path);
        return 1;
    }

    FILE *fs = fopen(sens_path, "rb");
    if (!fs) {
        std::string alt_s = std::string("gso/") + sens_path;
        fs = fopen(alt_s.c_str(), "rb");
    }
    if (!fs) {
        fprintf(stderr, "Failed to load %s\n", sens_path);
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

    bool pairs_and_flips_only = flag_pairs_and_flips || (gh_top > 0.990);
    std::vector<CosetSpec> cosets = build_cosets(M_sens, pairs_and_flips_only);
    fprintf(stderr, "Loaded %d zeros, sensitivity matrix (%d x %d), %zu cosets (pairs_and_flips_only=%d), gh_top=%.4f\n",
            zd.n_zeros, M_sens, n_sens, cosets.size(), (int)pairs_and_flips_only, gh_top);

    __int128 max_x = ((__int128)3484900000000ULL * 1000000ULL) * ((__int128)1 << 30);
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

        std::vector<double> bounds(n);
        for (int k = 0; k < n; ++k) {
            if (k >= k_y_cut) {
                double i = (double)(n - k);
                double u = i / (double)m_act;
                double p_act = std::min(1.0, std::max(0.11 + 0.72 * u, 0.025 + 0.975 * std::pow(u, 1.05)));
                bounds[k] = r2_cut * p_act;
            } else {
                bounds[k] = r2_cut * 10.0;
            }
        }

        const int cur_k_split = n - 7;
        const size_t n_cosets = cosets.size();

        for (size_t yc_idx = 0; yc_idx < yc_list.size(); ++yc_idx) {
            double y_center_e18 = yc_list[yc_idx];
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
                    double bnd = bounds[k];
                    for (int64_t step = 0;; ++step) {
                        int64_t xx;
                        if (step == 0) xx = x0;
                        else {
                            int64_t mag = (step + 1) >> 1;
                            xx = (step & 1) ? (x0 + dir * mag) : (x0 - dir * mag);
                        }
                        double a = (double)xx - c;
                        double nd = w.dist + a * a * rs[k];
                        if (nd > bnd) break;
                        w.x[k] = xx;
                        __int128 nx = w.cur_x + (__int128)xx * xcol[k];
                        double old_d = w.dist;
                        __int128 old_x = w.cur_x;
                        w.dist = nd;
                        w.cur_x = nx;
                        self(self, k - 1, w);
                        w.dist = old_d;
                        w.cur_x = old_x;
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
                double sigma_mid[MAXN];
                double sigma_low[MAXN];
                const int k_mid = k_y_cut + (n - k_y_cut) * 4 / 10;
                const int k_low = k_y_cut + (n - k_y_cut) * 2 / 10;

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

                    auto rec_top = [&](auto &self, int k, double dist, __int128 cur_x) -> void {
                        local_nodes++;
                        if (k < k_y_cut) {
                            if (cur_x <= -margin_x || cur_x >= max_x + margin_x) {
                                local_pruned++;
                                return;
                            }
                            double c = sigma_low[0];
                            const double *mt = &mu_T[0];
                            for (int i = 1; i < k_low; ++i) c -= x_dbl[i] * mt[i];
                            int64_t x0_base = (int64_t)std::llround(c);
                            for (int64_t dx0 = -1; dx0 <= 1; ++dx0) {
                                int64_t x0 = x0_base + dx0;
                                __int128 full_x = cur_x + (__int128)x0 * xcol[0];
                                if (full_x > 0 && full_x < max_x) {
                                    if (!zd.fast_screen((unsigned __int128)full_x)) continue;
                                    uint64_t y_bkt = (uint64_t)(full_x >> 21);
                                    if (!seen_y_buckets.test_and_set(y_bkt)) continue;
                                    uint64_t cid = ++total_clusters;
                                    auto [h_val, dy] = zd.polish((unsigned __int128)full_x);
                                    double ah = std::abs(h_val);
                                    if (ah >= 0.96 || ah > global_best_abs) {
                                        std::lock_guard<std::mutex> lk(out_mtx);
                                        if (ah > global_best_abs) global_best_abs = ah;
                                        char xbuf[64];
                                        print_int128(full_x, xbuf);
                                        double y_approx = (double)full_x / (double)(1ULL << 30);
                                        printf("CLUSTER #%llu [coset=%s] |h|=%.8f h=%+.8f y=%.6f (x=%s, dy=%+.7f)%s\n",
                                               (unsigned long long)cid, cosets[pf.coset_idx].label.c_str(),
                                               ah, h_val, y_approx, xbuf, dy,
                                               (ah > 1.00000006) ? " *** RECORD BREAKING HIT ***" : "");
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
                            double f_lo = (xcol[k] < 0)
                                ? ((double)(cur_x - max_x - margin_x) * inv_W)
                                : ((double)(-margin_x - cur_x) * inv_W);
                            int64_t xx_hi = (int64_t)std::floor(f_hi);
                            int64_t xx_lo = (int64_t)std::ceil(f_lo);
                            double c = sigma_low[k];
                            const double *mt = &mu_T[k * STRIDE];
                            for (int i = k + 1; i < k_low; ++i) c -= x_dbl[i] * mt[i];
                            double bnd = bounds[k];
                            double rem = bnd - dist;
                            if (rem < 0.0) {
                                local_pruned++;
                                return;
                            }
                            double rad = std::sqrt(rem / rs[k]);
                            int64_t gso_lo = (int64_t)std::ceil(c - rad);
                            int64_t gso_hi = (int64_t)std::floor(c + rad);
                            if (gso_lo > xx_lo) xx_lo = gso_lo;
                            if (gso_hi < xx_hi) xx_hi = gso_hi;
                            if (xx_lo > xx_hi) {
                                local_pruned++;
                                return;
                            }
                            for (int64_t xx = xx_hi; xx >= xx_lo; --xx) {
                                __int128 nx = cur_x + (__int128)xx * xcol[k];
                                if (nx <= -margin_x || nx >= max_x + margin_x) continue;
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
                                for (int j = 0; j < k_low; ++j) {
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
            fprintf(stderr, "[%s yc=%.2fe18 | %zu cosets | %zu prefixes] %.2f s | nodes=%.2fM | new_clusters=%llu | total_unique=%llu | best=%.8f\n",
                    fname, y_center_e18, n_cosets, all_items.size(), sec,
                    (double)(total_nodes.load() - prev_nodes) / 1e6,
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
