#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <omp.h>

int main(int argc, char **argv) {
    const char *fname = (argc > 1) ? argv[1] : "SR.ENUM.gso";
    double gh = (argc > 2) ? atof(argv[2]) : 1.19;
    double lin = (argc > 3) ? atof(argv[3]) : 3.0;

    FILE *f = fopen(fname, "rb");
    if (!f) return 1;
    int n;
    if (fread(&n, 4, 1, f) != 1) return 1;
    std::vector<double> rs(n), sigma(n), mu(n * n);
    std::vector<uint64_t> lo(n), hi(n);
    if (fread(rs.data(), 8, n, f) != (size_t)n) return 1;
    if (fread(sigma.data(), 8, n, f) != (size_t)n) return 1;
    if (fread(mu.data(), 8, n * n, f) != (size_t)(n * n)) return 1;
    if (fread(lo.data(), 8, n, f) != (size_t)n) return 1;
    if (fread(hi.data(), 8, n, f) != (size_t)n) return 1;
    fclose(f);

    std::vector<__int128> xcol(n);
    for (int i = 0; i < n; ++i) {
        unsigned __int128 u = (unsigned __int128)lo[i] + ((unsigned __int128)hi[i] << 64);
        xcol[i] = (__int128)u;
    }

    double logdet = 0.0;
    for (int i = 0; i < n; ++i) logdet += 0.5 * std::log(rs[i]);
    double heur = std::lgamma(1.0 + n / 2.0) / n - 0.5 * std::log(M_PI);
    double myradius = std::exp(logdet / n) * std::exp(heur) * gh;
    double radius2 = myradius * myradius;

    std::vector<double> bounds(n);
    for (int k = 0; k < n; ++k) {
        double i = (double)(n - k);
        double rr1 = i / n;
        double rr2 = (lin - 1.0) * std::sqrt(i * (n - i) / (n / 2.0 + 1.0)) / n;
        double p = std::min(1.0, rr1 + rr2);
        bounds[k] = radius2 * p;
    }

    // Max x corresponding to y = 1.9571878850562201959e19 * 2^30
    __int128 max_x = ((__int128)19571878850562ULL * 1000000ULL) * ((__int128)1 << 30);
    __int128 margin_x = (__int128)10 << 30; // +/- 10 in y

    // Let's test on the first 50M top nodes first to see speed and hit rate!
    uint64_t top_nodes = 0;
    uint64_t reached_k90 = 0;
    uint64_t in_range_k90 = 0;
    uint64_t babai_pass = 0;
    uint64_t babai_near = 0;
    std::vector<int64_t> x(n, 0);

    auto rec = [&](auto &self, int k, double dist, __int128 cur_x) -> void {
        if (top_nodes > 50000000ULL) return;
        top_nodes++;
        if (k == 90) {
            reached_k90++;
            if (cur_x <= -margin_x || cur_x >= max_x + margin_x) return;
            in_range_k90++;
            // Compute Babai down from k = 90 to 0
            double d = dist;
            __int128 full_x = cur_x;
            for (int j = 90; j >= 0; --j) {
                double c = sigma[j];
                const double *col = &mu[0 * n + j];
                for (int i = j + 1; i < n; ++i) c -= (double)x[i] * mu[(size_t)i * n + j];
                int64_t x0 = (int64_t)std::llround(c);
                x[j] = x0;
                double a = (double)x0 - c;
                d += a * a * rs[j];
                full_x += (__int128)x0 * xcol[j];
                // Quick early abort if d gets huge
                if (d > radius2 * 1.15) return;
            }
            if (full_x > 0 && full_x < max_x) {
                if (d <= radius2) {
                    babai_pass++;
                    double y = (double)full_x / (double)(1ULL << 30);
                    printf("PASS y=%.6f d/R2=%.5f\n", y, d / radius2);
                } else if (d <= radius2 * 1.08) {
                    babai_near++;
                    double y = (double)full_x / (double)(1ULL << 30);
                    printf("NEAR y=%.6f d/R2=%.5f\n", y, d / radius2);
                }
            }
            return;
        }
        double c = sigma[k];
        for (int i = k + 1; i < n; ++i) c -= (double)x[i] * mu[(size_t)i * n + k];
        int64_t x0 = (int64_t)std::llround(c);
        int dir = (c >= (double)x0) ? 1 : -1;
        double bnd = bounds[k];
        for (int64_t step = 0;; ++step) {
            if (top_nodes > 50000000ULL) return;
            int64_t xx;
            if (step == 0) xx = x0;
            else {
                int64_t mag = (step + 1) >> 1;
                xx = (step & 1) ? (x0 + dir * mag) : (x0 - dir * mag);
            }
            double a = (double)xx - c;
            double nd = dist + a * a * rs[k];
            if (nd > bnd) break;
            x[k] = xx;
            self(self, k - 1, nd, cur_x + (__int128)xx * xcol[k]);
        }
    };

    rec(rec, n - 1, 0.0, 0);
    fprintf(stderr, "top_nodes=%llu reached_k90=%llu in_range=%llu pass=%llu near=%llu\n",
            (unsigned long long)top_nodes,
            (unsigned long long)reached_k90,
            (unsigned long long)in_range_k90,
            (unsigned long long)babai_pass,
            (unsigned long long)babai_near);
    return 0;
}
