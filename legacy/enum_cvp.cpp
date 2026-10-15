// Pruned CVP enumeration. The shortest Gram-Schmidt direction is not zigzagged:
// only the nearest coefficient along that line is kept.
//
// Input (binary, little endian):
//   int32 n
//   int32 short_level   (-1 = auto = argmin r)
//   int32 max_sols
//   double radius2
//   double r[n]
//   double sigma[n]
//   double mu[n*n] row-major, mu[j*n + i] = mu_{j,i}
//   double prune[n]     relative squared-radius bounds at each level
//
// Output:
//   int32 nsols
//   then nsols records of: double dist, int64 x[n]
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <unordered_set>
#include <vector>

static int n;
static int short_level;
static std::vector<char> collapse;
static int max_sols;
static double radius2;
static std::vector<double> rdiag, sigma, mu, prune;
static std::vector<int64_t> x;
static std::vector<std::pair<double, std::vector<int64_t>>> sols;
struct Key {
    uint64_t lo, hi;
    bool operator==(const Key &o) const { return lo == o.lo && hi == o.hi; }
};
struct KeyHash {
    size_t operator()(const Key &k) const {
        return (size_t)(k.lo ^ (k.hi * 0x9E3779B185EBCA87ull));
    }
};
static std::vector<__int128> xcol;
static int64_t bucket = 0;
static uint64_t nodes = 0;
static uint64_t node_cap = 30000000000ULL;
static int max_depth = 0;
static std::unordered_set<Key, KeyHash> seen_key;
static bool full = false;

static void consider(double dist) {
    if ((int)sols.size() >= max_sols) {
        full = true;
        return;
    }
    if (bucket > 0 && !xcol.empty()) {
        __int128 acc = 0;
        for (int i = 0; i < n; ++i) acc += (__int128)x[i] * xcol[i];
        if (acc <= 0) return;
        __int128 key = acc / bucket;
        Key kk{(uint64_t)key, (uint64_t)(key >> 64)};
        if (!seen_key.insert(kk).second) return;
        // Stream the exact Mertens integer x, y = x * 2^{nu_y-nu}.
        std::fprintf(stderr, "X ");
        char buf[80];
        int nch = 0;
        __int128 v = acc;
        if (v < 0) {
            std::fputc('-', stderr);
            v = -v;
        }
        if (v == 0) buf[nch++] = '0';
        while (v > 0) {
            buf[nch++] = char('0' + (int)(v % 10));
            v /= 10;
        }
        while (nch--) std::fputc(buf[nch], stderr);
        std::fputc('\n', stderr);
        std::fflush(stderr);
    }
    sols.emplace_back(dist, x);
    if ((int)sols.size() >= max_sols) full = true;
}

static void rec(int k, double dist) {
    if (full) return;
    if (n - 1 - k > max_depth) max_depth = n - 1 - k;
    if ((++nodes & 0x1ffffff) == 0) {
        std::fprintf(stderr, "progress nodes %llu sols %zu\n",
                     (unsigned long long)nodes, sols.size());
    }
    if (nodes > node_cap) return;
    if (k < 0) {
        consider(dist);
        return;
    }
    double c = sigma[k];
    const double *mu_k = mu.data();  // mu[j*n + k]
    for (int j = k + 1; j < n; ++j) c -= (double)x[j] * mu[(size_t)j * n + k];
    double budget = radius2 * prune[k] - dist;
    if (budget < 0) return;
    double rem = budget / rdiag[k];
    if (rem < 0) return;
    double s = std::sqrt(rem);
    int64_t lo = (int64_t)std::ceil(c - s - 1e-9);
    int64_t hi = (int64_t)std::floor(c + s + 1e-9);
    int64_t x0 = (int64_t)std::llround(c);
    if (!collapse.empty() && collapse[k]) {
        if (x0 < lo || x0 > hi) return;
        x[k] = x0;
        double a = (double)x0 - c;
        rec(k - 1, dist + a * a * rdiag[k]);
        return;
    }
    // Zigzag outward from the rounded center, biased toward c.
    int dir = (c >= (double)x0) ? 1 : -1;
    for (int64_t step = 0;; ++step) {
        if (full || nodes > node_cap) return;
        int64_t xx;
        if (step == 0) {
            xx = x0;
        } else {
            int64_t mag = (step + 1) / 2;
            int sgn = (step % 2 == 1) ? dir : -dir;
            xx = x0 + sgn * mag;
        }
        if (xx < lo || xx > hi) {
            int64_t mag = (step + 1) / 2;
            if (x0 - mag < lo && x0 + mag > hi) break;
            continue;
        }
        x[k] = xx;
        double a = (double)xx - c;
        double nd = dist + a * a * rdiag[k];
        if (nd <= radius2 * prune[k] + 1e-8) rec(k - 1, nd);
    }
}

int main() {
    if (const char *env = std::getenv("ENUM_NODE_CAP"))
        node_cap = std::strtoull(env, nullptr, 10);
    if (std::fread(&n, 4, 1, stdin) != 1) return 1;
    if (std::fread(&short_level, 4, 1, stdin) != 1) return 1;
    if (std::fread(&max_sols, 4, 1, stdin) != 1) return 1;
    if (std::fread(&radius2, 8, 1, stdin) != 1) return 1;
    rdiag.resize(n);
    sigma.resize(n);
    mu.resize((size_t)n * n);
    prune.resize(n);
    x.assign(n, 0);
    if (std::fread(rdiag.data(), 8, n, stdin) != (size_t)n) return 1;
    if (std::fread(sigma.data(), 8, n, stdin) != (size_t)n) return 1;
    if (std::fread(mu.data(), 8, (size_t)n * n, stdin) != (size_t)n * n) return 1;
    if (std::fread(prune.data(), 8, n, stdin) != (size_t)n) return 1;
    if (std::fread(&bucket, 8, 1, stdin) != 1) bucket = 0;
    if (bucket > 0) {
        std::vector<uint64_t> lo(n), hi(n);
        if (std::fread(lo.data(), 8, n, stdin) != (size_t)n) return 1;
        if (std::fread(hi.data(), 8, n, stdin) != (size_t)n) return 1;
        xcol.resize(n);
        for (int i = 0; i < n; ++i) {
            unsigned __int128 u =
                (unsigned __int128)lo[i] + ((unsigned __int128)hi[i] << 64);
            xcol[i] = (__int128)u;
        }
        seen_key.reserve(1 << 20);
    }
    collapse.assign(n, 0);
    if (short_level < 0) {
        std::vector<double> sorted = rdiag;
        std::sort(sorted.begin(), sorted.end());
        // Collapse only the unusually short Gram-Schmidt directions.
        double cutoff = sorted[0] * 8.0;
        for (int i = 0; i < n; ++i)
            if (rdiag[i] <= cutoff) collapse[i] = 1;
    } else if (short_level < n) {
        collapse[short_level] = 1;
    }
    int ncollapse = 0;
    for (int i = 0; i < n; ++i) ncollapse += collapse[i];
    short_level = ncollapse;
    rec(n - 1, 0.0);
    std::sort(sols.begin(), sols.end(),
              [](auto &a, auto &b) { return a.first < b.first; });
    int32_t ns = (int32_t)sols.size();
    std::fwrite(&ns, 4, 1, stdout);
    for (auto &s : sols) {
        std::fwrite(&s.first, 8, 1, stdout);
        std::fwrite(s.second.data(), 8, n, stdout);
    }
    std::fprintf(stderr, "nodes %llu sols %d short %d depth %d\n",
                 (unsigned long long)nodes, ns, short_level, max_depth);
    return 0;
}
