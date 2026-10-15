#include <cmath>
#include <cstdint>
#include <omp.h>

extern "C" {

// Compute base phase cos(theta_j) and sin(theta_j) where
// theta_j = ((gamma_j * (x / 2^30 + dy0)) mod 2pi) - psi_j
// so that 2 * alpha_j * cos(theta_j + gamma_j * ddy) = 2 * Re(e^{i gamma_j (y0 + ddy)} / (rho_j zeta'(rho_j)))
// Specifically:
// Let A_j = 2 * alpha_j * cos(theta_j)
// Let B_j = 2 * alpha_j * sin(theta_j)
// Then for any additional shift ddy, the j-th term (unweighted by g(gamma_j)) is:
//   A_j * cos(gamma_j * ddy) - B_j * sin(gamma_j * ddy)
void compute_peak_AB(
    int n_peaks,
    const uint64_t* x_lo_arr,
    const uint64_t* x_hi_arr,
    const double* dy0_arr,
    int n_zeros,
    const double* alpha,
    const double* gamma,
    const uint64_t* ghi_lo,
    const uint64_t* ghi_hi,
    const uint64_t* glo_lo,
    const uint64_t* glo_hi,
    const uint64_t* psi_lo,
    const uint64_t* psi_hi,
    double* out_A, // shape (n_peaks, n_zeros)
    double* out_B  // shape (n_peaks, n_zeros)
) {
    const double scale = 2.0 * M_PI * std::ldexp(1.0, -64);
    #pragma omp parallel for schedule(static) num_threads(16)
    for (int p = 0; p < n_peaks; ++p) {
        unsigned __int128 x = ((unsigned __int128)x_hi_arr[p] << 64) | (unsigned __int128)x_lo_arr[p];
        uint64_t xh = (uint64_t)(x >> 48);
        uint64_t xl = (uint64_t)(x & ((1ULL << 48) - 1));
        double dy0 = dy0_arr[p];
        double* A_p = out_A + (size_t)p * n_zeros;
        double* B_p = out_B + (size_t)p * n_zeros;
        for (int i = 0; i < n_zeros; ++i) {
            unsigned __int128 ghi = ((unsigned __int128)ghi_hi[i] << 64) | (unsigned __int128)ghi_lo[i];
            unsigned __int128 glo = ((unsigned __int128)glo_hi[i] << 64) | (unsigned __int128)glo_lo[i];
            unsigned __int128 psi = ((unsigned __int128)psi_hi[i] << 64) | (unsigned __int128)psi_lo[i];
            unsigned __int128 ph = (unsigned __int128)xh * ghi + (unsigned __int128)xl * glo - psi;
            uint64_t top64 = (uint64_t)(ph >> 64);
            double th = (double)top64 * scale + gamma[i] * dy0;
            double two_a = 2.0 * alpha[i];
            A_p[i] = two_a * std::cos(th);
            B_p[i] = two_a * std::sin(th);
        }
    }
}

// Given precomputed A_p, B_p for n_peaks and n_zeros, and a weight vector w[n_zeros],
// polish ddy via Newton's method starting from the best of a small grid around 0,
// and return polished (h, ddy) for each peak.
void polish_peaks_with_weights(
    int n_peaks,
    int n_zeros,
    const double* A,       // (n_peaks, n_zeros)
    const double* B,       // (n_peaks, n_zeros)
    const double* gamma,   // (n_zeros,)
    const double* w,       // (n_zeros,)
    double grid_half_width,
    int grid_steps,
    double* out_h_unpol,   // (n_peaks,)
    double* out_h_pol,     // (n_peaks,)
    double* out_ddy_pol    // (n_peaks,)
) {
    #pragma omp parallel for schedule(dynamic, 4) num_threads(16)
    for (int p = 0; p < n_peaks; ++p) {
        const double* A_p = A + (size_t)p * n_zeros;
        const double* B_p = B + (size_t)p * n_zeros;

        // 1. Unpolished at ddy = 0
        double h0 = 0.0;
        for (int i = 0; i < n_zeros; ++i) {
            h0 += w[i] * A_p[i];
        }
        out_h_unpol[p] = h0;

        double best_ddy = 0.0;
        double best_abs = std::fabs(h0);

        // Coarse grid scan using first 4096 zeros if grid_steps > 0, then refine
        int k_scan = (n_zeros < 4096) ? n_zeros : 4096;
        if (grid_steps > 0) {
            for (int m = -grid_steps; m <= grid_steps; ++m) {
                double ddy = (grid_half_width * m) / grid_steps;
                double hs = 0.0;
                for (int i = 0; i < k_scan; ++i) {
                    double th = gamma[i] * ddy;
                    hs += w[i] * (A_p[i] * std::cos(th) - B_p[i] * std::sin(th));
                }
                if (std::fabs(hs) > best_abs) {
                    best_abs = std::fabs(hs);
                    best_ddy = ddy;
                }
            }
        }

        // Newton-Raphson polishing on all n_zeros
        double ddy = best_ddy;
        for (int it = 0; it < 6; ++it) {
            double f = 0.0, fp = 0.0, fpp = 0.0;
            for (int i = 0; i < n_zeros; ++i) {
                double g = gamma[i];
                double th = g * ddy;
                double c = std::cos(th);
                double s = std::sin(th);
                double val = A_p[i] * c - B_p[i] * s;
                double dval = g * (-A_p[i] * s - B_p[i] * c);
                double ddval = -g * g * val;
                double wi = w[i];
                f += wi * val;
                fp += wi * dval;
                fpp += wi * ddval;
            }
            if (std::fabs(fpp) > 1e-12) {
                double step = fp / fpp;
                if (step > 1e-4) step = 1e-4;
                if (step < -1e-4) step = -1e-4;
                ddy -= step;
                if (std::fabs(step) < 1e-13) break;
            } else {
                break;
            }
        }
        double h_final = 0.0;
        for (int i = 0; i < n_zeros; ++i) {
            double th = gamma[i] * ddy;
            h_final += w[i] * (A_p[i] * std::cos(th) - B_p[i] * std::sin(th));
        }
        if (std::fabs(h0) > std::fabs(h_final)) {
            out_h_pol[p] = h0;
            out_ddy_pol[p] = 0.0;
        } else {
            out_h_pol[p] = h_final;
            out_ddy_pol[p] = ddy;
        }
    }
}

} // extern "C"
