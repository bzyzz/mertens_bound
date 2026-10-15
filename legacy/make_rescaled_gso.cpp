#include "NTL/LLL.h"
#include <NTL/mat_RR.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

using namespace NTL;
using namespace std;

void ModifiedGS(const mat_ZZ &B, mat_RR &mu, vec_RR &c) {
    long k = B.NumRows();
    mat_RR r;
    ZZ z1;
    mu.SetDims(k, k);
    r.SetDims(k, k);
    c.SetLength(k);
    for (long i = 1; i <= k; i++) {
        for (long j = 1; j <= i; j++) {
            InnerProduct(z1, B(i), B(j));
            conv(r(i, j), z1);
            for (long l = 1; l < j; l++)
                r(i, j) = r(i, j) - (mu(j, l) * r(i, l));
            mu(i, j) = r(i, j) / r(j, j);
            if (j == i) c(i) = r(i, i);
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 5) {
        cerr << "Usage: " << argv[0] << " <in.ENUM> <out.gso> <b_shift> <bkz_beta> [prune]" << endl;
        return 1;
    }
    const char *in_file = argv[1];
    const char *out_file = argv[2];
    long b_shift = atol(argv[3]);
    long bkz_beta = atol(argv[4]);
    long bkz_prune = (argc > 5) ? atol(argv[5]) : 10;

    RR::SetPrecision(300);

    ifstream fin(in_file);
    if (!fin) {
        cerr << "Cannot open " << in_file << endl;
        return 1;
    }
    mat_ZZ M;
    vec_ZZ target;
    fin >> M >> target;
    fin.close();

    long dim = M.NumRows();
    long m = M.NumCols();
    cerr << "Loaded " << in_file << " dim=" << dim << " cols=" << m << endl;

    if (b_shift > 0) {
        for (long i = 1; i <= dim; i++) M(i, m) <<= b_shift;
        target(m) <<= b_shift;
    } else if (b_shift < 0) {
        for (long i = 1; i <= dim; i++) M(i, m) >>= (-b_shift);
        target(m) >>= (-b_shift);
    }

    if (bkz_beta > 0) {
        cerr << "Running LLL_RR..." << endl;
        LLL_RR(M, 0.99);
        if (bkz_beta >= 10) {
            cerr << "Running BKZ_RR beta=" << bkz_beta << " prune=" << bkz_prune << "..." << endl;
            BKZ_RR(M, 0.99, bkz_beta, bkz_prune);
        }
    }

    // Count how many bottom rows have tiny y-step
    long k_y_cut = 0;
    long total_shift = 30 + b_shift;
    for (long i = 1; i <= dim; i++) {
        ZZ xval = M(i, m) >> total_shift;
        if (abs(xval) > (ZZ(1) << 24)) break;
        k_y_cut++;
    }
    cerr << "After reduction: k_y_cut=" << k_y_cut << endl;

    mat_ZZ M2;
    M2.SetDims(dim + 1, m);
    for (long i = 1; i <= dim; i++) M2(i) = M(i);
    M2(dim + 1) = target;

    mat_RR mu2;
    vec_RR c2;
    ModifiedGS(M2, mu2, c2);

    // Also compute dsigma: change in mu2(dim+1, i) when target(m) increases by 2^(60 + b_shift) (i.e. y_center += 1)
    // Since target_shifted = target + delta_t * e_m, <target_shifted, b*_j> = <target, b*_j> + delta_t * b*_j[m].
    // More directly, let's compute mu2 with target(m) += 2^(60 + b_shift) * 1e18 and divide difference by 1e18!
    mat_ZZ M3 = M2;
    ZZ delta_1e18 = (ZZ(1000000000000000000LL)) << (60 + b_shift);
    M3(dim + 1, m) += delta_1e18;
    mat_RR mu3;
    vec_RR c3;
    ModifiedGS(M3, mu3, c3);

    int n = (int)dim;
    vector<double> rs(n), sigma(n), dsigma(n), mu(n * n, 0.0);
    vector<uint64_t> lo(n), hi(n);

    for (int i = 0; i < n; i++) {
        rs[i] = conv<double>(c2(i + 1));
        sigma[i] = conv<double>(mu2(dim + 1, i + 1));
        dsigma[i] = conv<double>((mu3(dim + 1, i + 1) - mu2(dim + 1, i + 1)) / to_RR(1e18));
        for (int j = 0; j < n; j++) {
            mu[i * n + j] = conv<double>(mu2(i + 1, j + 1));
        }
        ZZ xval = M(i + 1, m) >> total_shift;
        bool neg = (xval < 0);
        ZZ ux = abs(xval);
        unsigned __int128 u = 0;
        for (long b = 127; b >= 0; b--) {
            u = (u << 1) | (unsigned __int128)bit(ux, b);
        }
        __int128 s = neg ? -(__int128)u : (__int128)u;
        unsigned __int128 su = (unsigned __int128)s;
        lo[i] = (uint64_t)(su & 0xFFFFFFFFFFFFFFFFULL);
        hi[i] = (uint64_t)(su >> 64);
    }

    FILE *f = fopen(out_file, "wb");
    fwrite(&n, 4, 1, f);
    fwrite(rs.data(), 8, n, f);
    fwrite(sigma.data(), 8, n, f);
    fwrite(dsigma.data(), 8, n, f);
    fwrite(mu.data(), 8, n * n, f);
    fwrite(lo.data(), 8, n, f);
    fwrite(hi.data(), 8, n, f);
    fclose(f);
    cerr << "Saved " << out_file << endl;
    return 0;
}
