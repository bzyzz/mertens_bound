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
#include <vector>
#include <omp.h>
#include <cuda_runtime.h>

static constexpr int MAXN = 64;
static constexpr int STRIDE = 64;
static constexpr int K_GPU = 9;             // Levels k = 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 run on GPU
static constexpr int GPU_BATCH = 262144;    // 2^18 level-10 nodes per GPU launch
static constexpr int MAX_HITS_BATCH = 65536;

static constexpr int K_PRE = 256;
static constexpr int M_PRE = 38;
static constexpr int N_PRE = 2 * M_PRE + 1;
static constexpr int K0 = 512;
static constexpr int M0 = 228;
static constexpr int N0 = 2 * M0 + 1;
static constexpr int K1 = 2048;
static constexpr int M1 = 60;
static constexpr int N1 = 2 * M1 + 1;

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

struct alignas(16) GpuNode {
    float s[10];
    float dist;
    int32_t jumps;
    uint64_t cx_lo;
    int64_t cx_hi;
};

struct GpuHit {
    uint64_t x_lo;
    uint64_t x_hi;
    float m128;
    float m64;
};

// Constant memory for level 0..9 lattice parameters (all FP32 normalized by r2_cut)
__constant__ float c_rs[10];
__constant__ float c_inv_rs1;
__constant__ float c_bounds[10];
__constant__ float c_max_inc[10];
__constant__ float c_mu[10][10]; // c_mu[i][k] = mu[i * n + k] for 0 <= k < i <= 9
__constant__ uint64_t c_xcol_lo[10];
__constant__ int64_t c_xcol_hi[10];
__constant__ int c_max_jumps[10];
__constant__ int c_floor_jumps[10];
__constant__ uint64_t c_max_x_lo;
__constant__ int64_t c_max_x_hi;
__constant__ uint64_t c_win_lo_lo;
__constant__ int64_t c_win_lo_hi;
__constant__ uint64_t c_win_hi_lo;
__constant__ int64_t c_win_hi_hi;

// Constant memory for 256-zero exact phase arithmetic (9 KB total in constant memory)
__constant__ uint64_t c_G0[256];
__constant__ uint64_t c_G1[256];
__constant__ uint64_t c_G2[256];
__constant__ uint64_t c_G3[256];
__constant__ uint64_t c_PSI64[256];
__constant__ float c_two_astar[256];
__constant__ uint32_t c_step_idx[256];

#define BEGIN_LEVEL(L, D_IN, J_IN) \
    int32_t x0_##L = __float2int_rn(c##L); \
    int32_t dir_##L = (c##L >= (float)x0_##L) ? 1 : -1; \
    float eff_bnd_##L = fminf(c_bounds[L], (D_IN) + c_max_inc[L]); \
    int32_t mj_##L = c_max_jumps[L]; \
    int32_t fj_##L = c_floor_jumps[L]; \
    _Pragma("unroll 1") \
    for (int32_t st_##L = 0; ; ++st_##L) { \
        int32_t nj_##L = (J_IN) + (st_##L > 0 ? 1 : 0); \
        if (nj_##L > mj_##L) break; \
        int32_t z_##L = (st_##L == 0) ? x0_##L : ( \
            (st_##L & 1) ? (x0_##L + dir_##L * ((st_##L + 1) >> 1)) \
                         : (x0_##L - dir_##L * ((st_##L + 1) >> 1)) ); \
        float a_##L = (float)z_##L - c##L; \
        float d_##L = (D_IN) + a_##L * a_##L * c_rs[L]; \
        if (d_##L > eff_bnd_##L) break; \
        local_nodes++; \
        float zf_##L = (float)z_##L; \
        int32_t j_##L = (nj_##L < fj_##L) ? fj_##L : nj_##L; \
        (void)j_##L;

#define END_LEVEL(L) }

__global__ __launch_bounds__(256, 2)
void gpu_leaf_sweep_kernel(
    const GpuNode * __restrict__ d_nodes,
    int n_nodes,
    GpuHit * __restrict__ d_hits,
    uint32_t * __restrict__ d_hit_count,
    unsigned long long * __restrict__ d_leaf_cands,
    unsigned long long * __restrict__ d_tree_nodes,
    float m64_thresh,
    float m128_thresh
) {
    __shared__ float s_cos_lut[4096];
    for (int k = threadIdx.x; k < 4096; k += blockDim.x) {
        s_cos_lut[k] = __cosf(2.0f * 3.14159265358979323846f * (float)k / 4096.0f);
    }
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t local_cands = 0;
    uint64_t local_nodes = 0;

    if (tid < n_nodes) {
        GpuNode node = d_nodes[tid];
        float s0_in = node.s[0], s1_in = node.s[1], s2_in = node.s[2], s3_in = node.s[3], s4_in = node.s[4];
        float s5_in = node.s[5], s6_in = node.s[6], s7_in = node.s[7], s8_in = node.s[8], s9_in = node.s[9];
        __int128 cx10 = ((__int128)node.cx_hi << 64) | (unsigned __int128)node.cx_lo;
        __int128 win_lo = ((__int128)c_win_lo_hi << 64) | (unsigned __int128)c_win_lo_lo;
        __int128 win_hi = ((__int128)c_win_hi_hi << 64) | (unsigned __int128)c_win_hi_lo;
        __int128 max_x  = ((__int128)c_max_x_hi << 64) | (unsigned __int128)c_max_x_lo;

        float c9 = s9_in;
        BEGIN_LEVEL(9, node.dist, node.jumps)
            float c8 = s8_in - zf_9 * c_mu[9][8];
            BEGIN_LEVEL(8, d_9, j_9)
                float c7 = s7_in - zf_9 * c_mu[9][7] - zf_8 * c_mu[8][7];
                BEGIN_LEVEL(7, d_8, j_8)
                    float c6 = s6_in - zf_9 * c_mu[9][6] - zf_8 * c_mu[8][6] - zf_7 * c_mu[7][6];
                    BEGIN_LEVEL(6, d_7, j_7)
                        float c5 = s5_in - zf_9 * c_mu[9][5] - zf_8 * c_mu[8][5] - zf_7 * c_mu[7][5] - zf_6 * c_mu[6][5];
                        BEGIN_LEVEL(5, d_6, j_6)
                            float s4_5 = s4_in - zf_9 * c_mu[9][4] - zf_8 * c_mu[8][4] - zf_7 * c_mu[7][4] - zf_6 * c_mu[6][4] - zf_5 * c_mu[5][4];
                            float s3_5 = s3_in - zf_9 * c_mu[9][3] - zf_8 * c_mu[8][3] - zf_7 * c_mu[7][3] - zf_6 * c_mu[6][3] - zf_5 * c_mu[5][3];
                            float s2_5 = s2_in - zf_9 * c_mu[9][2] - zf_8 * c_mu[8][2] - zf_7 * c_mu[7][2] - zf_6 * c_mu[6][2] - zf_5 * c_mu[5][2];
                            float s1_5 = s1_in - zf_9 * c_mu[9][1] - zf_8 * c_mu[8][1] - zf_7 * c_mu[7][1] - zf_6 * c_mu[6][1] - zf_5 * c_mu[5][1];
                            float s0_5 = s0_in - zf_9 * c_mu[9][0] - zf_8 * c_mu[8][0] - zf_7 * c_mu[7][0] - zf_6 * c_mu[6][0] - zf_5 * c_mu[5][0];
                            __int128 cx5 = cx10
                                + (__int128)z_9 * (((__int128)c_xcol_hi[9] << 64) | (unsigned __int128)c_xcol_lo[9])
                                + (__int128)z_8 * (((__int128)c_xcol_hi[8] << 64) | (unsigned __int128)c_xcol_lo[8])
                                + (__int128)z_7 * (((__int128)c_xcol_hi[7] << 64) | (unsigned __int128)c_xcol_lo[7])
                                + (__int128)z_6 * (((__int128)c_xcol_hi[6] << 64) | (unsigned __int128)c_xcol_lo[6])
                                + (__int128)z_5 * (((__int128)c_xcol_hi[5] << 64) | (unsigned __int128)c_xcol_lo[5]);

                            float c4 = s4_5;
                            BEGIN_LEVEL(4, d_5, j_5)
                                float c3 = s3_5 - zf_4 * c_mu[4][3];
                                BEGIN_LEVEL(3, d_4, j_4)
                                    float c2 = s2_5 - zf_4 * c_mu[4][2] - zf_3 * c_mu[3][2];
                                    BEGIN_LEVEL(2, d_3, j_3)
                                        (void)j_2;
                                        float dist1 = d_2;
                                        float bnd1 = c_bounds[1];
                                        if (dist1 <= bnd1) {
                                            float c1 = s1_5 - zf_4 * c_mu[4][1] - zf_3 * c_mu[3][1] - zf_2 * c_mu[2][1];
                                            float rad = sqrtf((bnd1 - dist1) * c_inv_rs1);
                                            int32_t gso_lo = __float2int_ru(c1 - rad);
                                            int32_t gso_hi = __float2int_rd(c1 + rad);
                                            if (gso_lo <= gso_hi) {
                                                __int128 cur_x = cx5
                                                    + (__int128)z_4 * (((__int128)c_xcol_hi[4] << 64) | (unsigned __int128)c_xcol_lo[4])
                                                    + (__int128)z_3 * (((__int128)c_xcol_hi[3] << 64) | (unsigned __int128)c_xcol_lo[3])
                                                    + (__int128)z_2 * (((__int128)c_xcol_hi[2] << 64) | (unsigned __int128)c_xcol_lo[2]);
                                                __int128 xcol1 = ((__int128)c_xcol_hi[1] << 64) | (unsigned __int128)c_xcol_lo[1];
                                                float c0_base = s0_5 - zf_4 * c_mu[4][0] - zf_3 * c_mu[3][0] - zf_2 * c_mu[2][0];
                                                float rs1 = c_rs[1];
                                                float mu10 = c_mu[1][0];

                                                for (int32_t xx = gso_hi; xx >= gso_lo; --xx) {
                                                    __int128 nx = cur_x + (__int128)xx * xcol1;
                                                    if (nx <= win_lo || nx >= win_hi) continue;
                                                    float a1 = (float)xx - c1;
                                                    if (dist1 + a1 * a1 * rs1 <= bnd1) {
                                                        int32_t x0_leaf = __float2int_rn(c0_base - (float)xx * mu10);
                                                        __int128 full_x = nx + (__int128)x0_leaf;
                                                        if (full_x > 0 && full_x < max_x) {
                                                            local_cands++;
                                                            unsigned __int128 ux = (unsigned __int128)full_x;
                                                            uint64_t fx_lo = (uint64_t)ux;
                                                            uint64_t fx_hi = (uint64_t)(ux >> 64);

                                                            uint64_t u0 = fx_lo & 0x7FFFFFULL;
                                                            uint64_t u1 = (fx_lo >> 23) & 0x7FFFFFULL;
                                                            uint64_t u2 = ((fx_lo >> 46) | (fx_hi << 18)) & 0x7FFFFFULL;
                                                            uint64_t u3 = (fx_hi >> 5);

                                                            float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f, acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f;
                                                            #pragma unroll 4
                                                            for (int i = 0; i < 64; ++i) {
                                                                uint64_t ph64 = u0 * c_G0[i] + u1 * c_G1[i] + u2 * c_G2[i] + u3 * c_G3[i] - c_PSI64[i];
                                                                uint32_t idx0 = (uint32_t)(ph64 >> 52);
                                                                uint32_t st = c_step_idx[i];
                                                                float w = c_two_astar[i];
                                                                acc0 += w * s_cos_lut[(idx0 - 3 * st) & 4095u];
                                                                acc1 += w * s_cos_lut[(idx0 - 2 * st) & 4095u];
                                                                acc2 += w * s_cos_lut[(idx0 - st) & 4095u];
                                                                acc3 += w * s_cos_lut[idx0];
                                                                acc4 += w * s_cos_lut[(idx0 + st) & 4095u];
                                                                acc5 += w * s_cos_lut[(idx0 + 2 * st) & 4095u];
                                                                acc6 += w * s_cos_lut[(idx0 + 3 * st) & 4095u];
                                                            }
                                                            float m64 = fmaxf(fabsf(acc0), fmaxf(fabsf(acc1), fmaxf(fabsf(acc2), fmaxf(fabsf(acc3), fmaxf(fabsf(acc4), fmaxf(fabsf(acc5), fabsf(acc6)))))));
                                                            if (m64 >= m64_thresh) {
                                                                #pragma unroll 4
                                                                for (int i = 64; i < 128; ++i) {
                                                                    uint64_t ph64 = u0 * c_G0[i] + u1 * c_G1[i] + u2 * c_G2[i] + u3 * c_G3[i] - c_PSI64[i];
                                                                    uint32_t idx0 = (uint32_t)(ph64 >> 52);
                                                                    uint32_t st = c_step_idx[i];
                                                                    float w = c_two_astar[i];
                                                                    acc0 += w * s_cos_lut[(idx0 - 3 * st) & 4095u];
                                                                    acc1 += w * s_cos_lut[(idx0 - 2 * st) & 4095u];
                                                                    acc2 += w * s_cos_lut[(idx0 - st) & 4095u];
                                                                    acc3 += w * s_cos_lut[idx0];
                                                                    acc4 += w * s_cos_lut[(idx0 + st) & 4095u];
                                                                    acc5 += w * s_cos_lut[(idx0 + 2 * st) & 4095u];
                                                                    acc6 += w * s_cos_lut[(idx0 + 3 * st) & 4095u];
                                                                }
                                                                float m128 = fmaxf(fabsf(acc0), fmaxf(fabsf(acc1), fmaxf(fabsf(acc2), fmaxf(fabsf(acc3), fmaxf(fabsf(acc4), fmaxf(fabsf(acc5), fabsf(acc6)))))));
                                                                if (m128 >= m128_thresh) {
                                                                    #pragma unroll 4
                                                                    for (int i = 128; i < 256; ++i) {
                                                                        uint64_t ph64 = u0 * c_G0[i] + u1 * c_G1[i] + u2 * c_G2[i] + u3 * c_G3[i] - c_PSI64[i];
                                                                        uint32_t idx0 = (uint32_t)(ph64 >> 52);
                                                                        uint32_t st = c_step_idx[i];
                                                                        float w = c_two_astar[i];
                                                                        acc0 += w * s_cos_lut[(idx0 - 3 * st) & 4095u];
                                                                        acc1 += w * s_cos_lut[(idx0 - 2 * st) & 4095u];
                                                                        acc2 += w * s_cos_lut[(idx0 - st) & 4095u];
                                                                        acc3 += w * s_cos_lut[idx0];
                                                                        acc4 += w * s_cos_lut[(idx0 + st) & 4095u];
                                                                        acc5 += w * s_cos_lut[(idx0 + 2 * st) & 4095u];
                                                                        acc6 += w * s_cos_lut[(idx0 + 3 * st) & 4095u];
                                                                    }
                                                                    float m256 = fmaxf(fabsf(acc0), fmaxf(fabsf(acc1), fmaxf(fabsf(acc2), fmaxf(fabsf(acc3), fmaxf(fabsf(acc4), fmaxf(fabsf(acc5), fabsf(acc6)))))));
                                                                    if (m256 >= 0.892f) {
                                                                        uint32_t h_idx = atomicAdd(d_hit_count, 1u);
                                                                        if (h_idx < MAX_HITS_BATCH) {
                                                                            d_hits[h_idx] = {fx_lo, fx_hi, m128, m64};
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    END_LEVEL(2)
                                END_LEVEL(3)
                            END_LEVEL(4)
                        END_LEVEL(5)
                    END_LEVEL(6)
                END_LEVEL(7)
            END_LEVEL(8)
        END_LEVEL(9)
    }

    // Block-level reduction for counters
    __shared__ unsigned long long s_cands[256];
    __shared__ unsigned long long s_nodes[256];
    s_cands[threadIdx.x] = local_cands;
    s_nodes[threadIdx.x] = local_nodes;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_cands[threadIdx.x] += s_cands[threadIdx.x + stride];
            s_nodes[threadIdx.x] += s_nodes[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(d_leaf_cands, s_cands[0]);
        atomicAdd(d_tree_nodes, s_nodes[0]);
    }
}

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

        // Upload 256-zero constants to GPU constant memory
        std::vector<uint64_t> G0(256), G1(256), G2(256), G3(256), PSI64(256);
        std::vector<float> two_astar_f(256);
        std::vector<uint32_t> step_idx(256);
        for (int i = 0; i < 256; ++i) {
            G0[i] = (uint64_t)((glo[i] + ((unsigned __int128)1 << 63)) >> 64);
            G1[i] = (uint64_t)(((glo[i] << 23) + ((unsigned __int128)1 << 63)) >> 64);
            G2[i] = (uint64_t)((((ghi[i] >> 2) & (((unsigned __int128)1 << 126) - 1)) | (glo[i] << 46)) >> 64);
            G3[i] = (uint64_t)(((ghi[i] << 21) + ((unsigned __int128)1 << 63)) >> 64);
            PSI64[i] = (uint64_t)((psi[i] + ((unsigned __int128)1 << 63)) >> 64);
            two_astar_f[i] = (float)two_astar[i];
            step_idx[i] = ((uint32_t)std::llround(gamma[i] * 0.0014 * (4096.0 / (2.0 * M_PI)))) & 4095u;
        }
        CUDA_CHECK(cudaMemcpyToSymbol(c_G0, G0.data(), 256 * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemcpyToSymbol(c_G1, G1.data(), 256 * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemcpyToSymbol(c_G2, G2.data(), 256 * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemcpyToSymbol(c_G3, G3.data(), 256 * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemcpyToSymbol(c_PSI64, PSI64.data(), 256 * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemcpyToSymbol(c_two_astar, two_astar_f.data(), 256 * sizeof(float)));
        CUDA_CHECK(cudaMemcpyToSymbol(c_step_idx, step_idx.data(), 256 * sizeof(uint32_t)));
        return true;
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
            for (int i = 0; i < K0; ++i) {
                s += A0[i] * cp[i] - B0[i] * sp[i];
            }
            if (std::abs(s) > best_abs) {
                best_abs = std::abs(s);
                best_h = s;
                best_dy = (idx - M0) * 2.5e-5;
            }
        }
        if (best_abs < 0.918) return {best_h, best_dy};

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
            for (int i = 0; i < K1; ++i) {
                s += A1[i] * cp[i] - B1[i] * sp[i];
            }
            if (std::abs(s) > best_abs) {
                best_abs = std::abs(s);
                best_h = s;
                best_dy = center + (idx - M1) * 1.0e-5;
            }
        }
        if (best_abs < 0.948) return {best_h, best_dy};

        std::vector<double> full_ang(n_zeros);
        for (int i = 0; i < K1; ++i) full_ang[i] = ang1[i];
        for (int i = K1; i < n_zeros; ++i) {
            unsigned __int128 ph = (unsigned __int128)x_hi * ghi[i] + (unsigned __int128)x_lo * glo[i] - psi[i];
            int64_t top = (int64_t)(ph >> 64);
            full_ang[i] = (double)top * SCALE;
        }
        center = best_dy;
        best_abs = 0.0;
        const int K2 = std::min(15000, n_zeros);
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
    const char *fname_arg = (argc > 1) ? argv[1] : "gpu_m36_00_pref_M36_P36_pos.gso";
    double gh_top = (argc > 2) ? atof(argv[2]) : 1.06;
    const char *yc_arg = (argc > 3) ? argv[3] : "0.3:0.5:3.3";
    float m64_thresh = (argc > 4) ? (float)atof(argv[4]) : 0.780f;
    float m128_thresh = (argc > 5) ? (float)atof(argv[5]) : 0.855f;

    omp_set_num_threads(16);

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
    __int128 win_lo = -margin_x;
    __int128 win_hi = max_x + margin_x;

    uint64_t max_x_lo = (uint64_t)((unsigned __int128)max_x);
    int64_t max_x_hi = (int64_t)(max_x >> 64);
    uint64_t win_lo_lo = (uint64_t)((unsigned __int128)win_lo);
    int64_t win_lo_hi = (int64_t)(win_lo >> 64);
    uint64_t win_hi_lo = (uint64_t)((unsigned __int128)win_hi);
    int64_t win_hi_hi = (int64_t)(win_hi >> 64);

    CUDA_CHECK(cudaMemcpyToSymbol(c_max_x_lo, &max_x_lo, sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_max_x_hi, &max_x_hi, sizeof(int64_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_win_lo_lo, &win_lo_lo, sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_win_lo_hi, &win_lo_hi, sizeof(int64_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_win_hi_lo, &win_hi_lo, sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_win_hi_hi, &win_hi_hi, sizeof(int64_t)));

    // Allocate double-buffered pinned host memory and device buffers
    GpuNode *h_batch[2];
    GpuNode *d_batch[2];
    GpuHit *h_hits[2];
    GpuHit *d_hits[2];
    uint32_t *h_hit_cnt[2];
    uint32_t *d_hit_cnt[2];
    cudaStream_t streams[2];
    bool stream_busy[2] = {false, false};

    for (int s = 0; s < 2; ++s) {
        CUDA_CHECK(cudaMallocHost(&h_batch[s], GPU_BATCH * sizeof(GpuNode)));
        CUDA_CHECK(cudaMalloc(&d_batch[s], GPU_BATCH * sizeof(GpuNode)));
        CUDA_CHECK(cudaMallocHost(&h_hits[s], MAX_HITS_BATCH * sizeof(GpuHit)));
        CUDA_CHECK(cudaMalloc(&d_hits[s], MAX_HITS_BATCH * sizeof(GpuHit)));
        CUDA_CHECK(cudaMallocHost(&h_hit_cnt[s], sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_hit_cnt[s], sizeof(uint32_t)));
        CUDA_CHECK(cudaStreamCreate(&streams[s]));
    }

    unsigned long long *d_leaf_cands = nullptr;
    unsigned long long *d_tree_nodes = nullptr;
    CUDA_CHECK(cudaMalloc(&d_leaf_cands, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_tree_nodes, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_leaf_cands, 0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_tree_nodes, 0, sizeof(unsigned long long)));

    auto t_start = std::chrono::steady_clock::now();
    std::atomic<uint64_t> total_cpu_nodes{0};
    std::atomic<uint64_t> total_lvl10_nodes{0};
    std::atomic<uint64_t> total_clusters{0};
    std::mutex gpu_mtx;
    std::mutex out_mtx;
    ConcurrentBitset seen_y_buckets;
    double global_best_abs = 0.0;

    int cur_stream = 0;
    int cur_batch_len = 0;
    std::vector<GpuHit> pending_hits;
    pending_hits.reserve(65536);

    auto drain_stream_fast = [&](int s) {
        if (!stream_busy[s]) return;
        CUDA_CHECK(cudaStreamSynchronize(streams[s]));
        stream_busy[s] = false;
        uint32_t n_hits = std::min(*h_hit_cnt[s], (uint32_t)MAX_HITS_BATCH);
        for (uint32_t i = 0; i < n_hits; ++i) {
            unsigned __int128 full_x = ((unsigned __int128)h_hits[s][i].x_hi << 64) | (unsigned __int128)h_hits[s][i].x_lo;
            uint64_t y_bkt = (uint64_t)(full_x >> 32);
            if (!seen_y_buckets.test_and_set(y_bkt)) continue;
            pending_hits.push_back(h_hits[s][i]);
        }
    };

    auto launch_current_batch = [&]() {
        if (cur_batch_len == 0) return;
        int s = cur_stream;
        drain_stream_fast(s);
        CUDA_CHECK(cudaMemsetAsync(d_hit_cnt[s], 0, sizeof(uint32_t), streams[s]));
        CUDA_CHECK(cudaMemcpyAsync(d_batch[s], h_batch[s], cur_batch_len * sizeof(GpuNode), cudaMemcpyHostToDevice, streams[s]));
        int blocks = (cur_batch_len + 255) / 256;
        gpu_leaf_sweep_kernel<<<blocks, 256, 0, streams[s]>>>(
            d_batch[s], cur_batch_len, d_hits[s], d_hit_cnt[s],
            d_leaf_cands, d_tree_nodes, m64_thresh, m128_thresh
        );
        CUDA_CHECK(cudaMemcpyAsync(h_hit_cnt[s], d_hit_cnt[s], sizeof(uint32_t), cudaMemcpyDeviceToHost, streams[s]));
        CUDA_CHECK(cudaMemcpyAsync(h_hits[s], d_hits[s], MAX_HITS_BATCH * sizeof(GpuHit), cudaMemcpyDeviceToHost, streams[s]));
        stream_busy[s] = true;
        cur_stream ^= 1;
        cur_batch_len = 0;
    };

    auto polish_pending_hits = [&]() {
        if (pending_hits.empty()) return;
        #pragma omp parallel for schedule(dynamic, 1) num_threads(16)
        for (size_t i = 0; i < pending_hits.size(); ++i) {
            unsigned __int128 full_x = ((unsigned __int128)pending_hits[i].x_hi << 64) | (unsigned __int128)pending_hits[i].x_lo;
            uint64_t cid = ++total_clusters;
            auto [h_val, dy] = zd.polish(full_x);
            double ah = std::abs(h_val);
            if (ah >= 0.955 || ah > global_best_abs) {
                std::lock_guard<std::mutex> lk(out_mtx);
                if (ah > global_best_abs) global_best_abs = ah;
                char xbuf[64];
                print_int128((__int128)full_x, xbuf);
                double y_approx = (double)full_x / (double)(1ULL << 30);
                printf("CLUSTER #%llu |h|=%.8f h=%+.8f y=%.6f (x=%s, dy=%+.7f)%s\n",
                       (unsigned long long)cid, ah, h_val, y_approx, xbuf, dy,
                       (ah > 1.00000006) ? " *** RECORD BREAKING HIT ***" : "");
                fflush(stdout);
            }
        }
        pending_hits.clear();
    };

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
        if (k_y_cut != 1 || n <= K_GPU + 2) {
            fprintf(stderr, "Skipping incompatible GSO %s (n=%d, k_y_cut=%d)\n", fname, n, k_y_cut);
            continue;
        }

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
        double dim_scale = 1.2935 * std::pow(59.0 / (double)m_act, 0.38);
        double gh_eff = (m_act >= 38) ? (gh_top - 0.022) : gh_top;
        double r2_cut = (dim_scale * gh_act * gh_eff) * (dim_scale * gh_act * gh_eff);
        double inv_r2_cut = 1.0 / r2_cut;

        std::vector<double> bounds(n, 0.0);
        std::vector<double> max_inc(n, 0.0);
        std::vector<int> max_jumps(n, 0);
        std::vector<int> floor_jumps(n, 0);
        for (int k = 1; k < n; ++k) {
            int i_lvl = n - k;
            double u = (double)i_lvl / (double)m_act;
            double p_deep = (u <= 0.30)
                ? (0.040 + 0.68 * u + 0.30 * u * u)
                : std::min(1.0, 0.062 + 0.88 * u + 0.12 * u * u);
            bounds[k] = r2_cut * p_deep;

            double max_a = (k <= K_GPU) ? 2.15 :
                           (u <= 0.12)  ? 2.60 :
                           (u <= 0.42)  ? 1.85 :
                           (u <= 0.72)  ? 1.35 : 1.15;
            max_inc[k] = max_a * max_a * rs[k];

            max_jumps[k] = (k <= K_GPU) ? 14 :
                           (i_lvl <= 3)  ? 3 :
                           (i_lvl <= 6)  ? 5 :
                           (i_lvl <= 12) ? 7 :
                           (i_lvl <= 18) ? 9 :
                           (i_lvl <= 26) ? 10 : 10;

            floor_jumps[k] = (k <= K_GPU) ? 0 :
                             (u >= 0.60) ? 8 : 0;
        }

        drain_stream_fast(0);
        drain_stream_fast(1);
        polish_pending_hits();

        float h_rs10[10], h_bounds10[10], h_max_inc10[10];
        float h_mu10[10][10];
        uint64_t h_xcol_lo10[10];
        int64_t h_xcol_hi10[10];
        int h_mj10[10], h_fj10[10];
        for (int k = 0; k < 10; ++k) {
            double gpu_mult = 1.35;
            h_rs10[k] = (float)(rs[k] * inv_r2_cut);
            h_bounds10[k] = (float)(gpu_mult * bounds[k] * inv_r2_cut);
            h_max_inc10[k] = (float)(gpu_mult * max_inc[k] * inv_r2_cut);
            h_xcol_lo10[k] = lo[k];
            h_xcol_hi10[k] = (int64_t)hi[k];
            h_mj10[k] = max_jumps[k];
            h_fj10[k] = floor_jumps[k];
            for (int j = 0; j < 10; ++j) {
                h_mu10[k][j] = (float)mu[k * n + j];
            }
        }
        float inv_rs1 = 1.0f / h_rs10[1];

        CUDA_CHECK(cudaMemcpyToSymbol(c_rs, h_rs10, 10 * sizeof(float)));
        CUDA_CHECK(cudaMemcpyToSymbol(c_inv_rs1, &inv_rs1, sizeof(float)));
        CUDA_CHECK(cudaMemcpyToSymbol(c_bounds, h_bounds10, 10 * sizeof(float)));
        CUDA_CHECK(cudaMemcpyToSymbol(c_max_inc, h_max_inc10, 10 * sizeof(float)));
        CUDA_CHECK(cudaMemcpyToSymbol(c_mu, h_mu10, 100 * sizeof(float)));
        CUDA_CHECK(cudaMemcpyToSymbol(c_xcol_lo, h_xcol_lo10, 10 * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemcpyToSymbol(c_xcol_hi, h_xcol_hi10, 10 * sizeof(int64_t)));
        CUDA_CHECK(cudaMemcpyToSymbol(c_max_jumps, h_mj10, 10 * sizeof(int)));
        CUDA_CHECK(cudaMemcpyToSymbol(c_floor_jumps, h_fj10, 10 * sizeof(int)));

        int cur_k_split = n - 8;

        for (size_t yc_idx = 0; yc_idx < yc_list.size(); ++yc_idx) {
            double y_center_e18 = yc_list[yc_idx];
            std::vector<double> raw_sigma = base_sigma;
            if (y_center_e18 != 0.0) {
                double yc = y_center_e18 * 1e18;
                for (int i = 0; i < n; ++i) raw_sigma[i] += yc * dsigma[i];
            }

            std::vector<double> sigma(n, 0.0);
            std::vector<int64_t> z_babai(n, 0);
            __int128 root_x = 0;
            for (int k = n - 1; k >= 0; --k) {
                double c = raw_sigma[k];
                const double *mt = &mu_T[k * STRIDE];
                for (int i = k + 1; i < n; ++i) c -= (double)z_babai[i] * mt[i];
                int64_t zb = (int64_t)std::llround(c);
                z_babai[k] = zb;
                sigma[k] = c - (double)zb;
                root_x += (__int128)zb * xcol[k];
            }

            auto t0 = std::chrono::steady_clock::now();
            std::vector<Prefix> prefixes;
            Prefix init_p;
            init_p.jumps = 0;
            init_p.dist = 0.0;
            init_p.cur_x = root_x;
            memset(init_p.x, 0, sizeof(init_p.x));

            auto gen_prefixes = [&](auto &self, int k, Prefix &p) -> void {
                if (prefixes.size() >= 500000 || (k <= cur_k_split && (k <= cur_k_split - 4 || p.dist >= 0.60 * bounds[k + 1]))) {
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
            uint64_t prev_lvl10 = total_lvl10_nodes.load();

            #pragma omp parallel num_threads(16)
            {
                uint64_t local_cpu_nodes = 0;
                uint64_t local_lvl10 = 0;
                double x_dbl[MAXN];
                std::vector<GpuNode> local_q;
                local_q.reserve(8192);

                auto flush_local_q = [&]() {
                    if (local_q.empty()) return;
                    std::lock_guard<std::mutex> lk(gpu_mtx);
                    size_t idx = 0;
                    while (idx < local_q.size()) {
                        size_t space = (size_t)(GPU_BATCH - cur_batch_len);
                        size_t take = std::min(space, local_q.size() - idx);
                        memcpy(&h_batch[cur_stream][cur_batch_len], &local_q[idx], take * sizeof(GpuNode));
                        cur_batch_len += (int)take;
                        idx += take;
                        if (cur_batch_len == GPU_BATCH) {
                            launch_current_batch();
                        }
                    }
                    local_q.clear();
                };

                #pragma omp for schedule(dynamic, 1)
                for (size_t idx = 0; idx < prefixes.size(); ++idx) {
                    const Prefix &pf = prefixes[idx];
                    for (int i = pf.k_start + 1; i < n; ++i) {
                        x_dbl[i] = (double)pf.x[i];
                    }

                    double sigma_pf[MAXN];
                    double sigma_mid[MAXN];
                    const int k_mid = 18;
                    for (int j = 0; j <= pf.k_start; ++j) {
                        double s = sigma[j];
                        const double *mt = &mu_T[j * STRIDE];
                        for (int i = pf.k_start + 1; i < n; ++i) s -= x_dbl[i] * mt[i];
                        sigma_pf[j] = s;
                    }

                    auto rec_cpu = [&](auto &self, int k, double dist, __int128 cur_x, int jumps, bool have_mid) -> void {
                        if (k == K_GPU) {
                            GpuNode gn;
                            if (have_mid) {
                                for (int j = 0; j <= K_GPU; ++j) {
                                    double s = sigma_mid[j];
                                    const double *mt = &mu_T[j * STRIDE];
                                    for (int i = K_GPU + 1; i < k_mid; ++i) s -= x_dbl[i] * mt[i];
                                    gn.s[j] = (float)s;
                                }
                            } else {
                                for (int j = 0; j <= K_GPU; ++j) {
                                    double s = sigma_pf[j];
                                    const double *mt = &mu_T[j * STRIDE];
                                    for (int i = K_GPU + 1; i <= pf.k_start; ++i) s -= x_dbl[i] * mt[i];
                                    gn.s[j] = (float)s;
                                }
                            }
                            gn.dist = (float)(dist * inv_r2_cut);
                            gn.jumps = jumps;
                            gn.cx_lo = (uint64_t)((unsigned __int128)cur_x);
                            gn.cx_hi = (int64_t)(cur_x >> 64);
                            local_q.push_back(gn);
                            local_lvl10++;
                            if (local_q.size() >= 8192) flush_local_q();
                            return;
                        }

                        local_cpu_nodes++;
                        double c;
                        const double *mt = &mu_T[k * STRIDE];
                        if (have_mid && k < k_mid) {
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
                            x_dbl[k] = (double)xx;
                            int next_j = (nj < fj) ? fj : nj;
                            bool nm = have_mid;
                            if (k == k_mid) {
                                for (int j = 0; j < k_mid; ++j) {
                                    double s = sigma_pf[j];
                                    const double *mtj = &mu_T[j * STRIDE];
                                    for (int i = k_mid; i <= pf.k_start; ++i) s -= x_dbl[i] * mtj[i];
                                    sigma_mid[j] = s;
                                }
                                nm = true;
                            }
                            self(self, k - 1, nd, cur_x + (__int128)xx * xcol[k], next_j, nm);
                        }
                    };

                    rec_cpu(rec_cpu, pf.k_start, pf.dist, pf.cur_x, pf.jumps, false);
                }
                flush_local_q();
                total_cpu_nodes += local_cpu_nodes;
                total_lvl10_nodes += local_lvl10;
            }
            {
                std::lock_guard<std::mutex> lk(gpu_mtx);
                launch_current_batch();
                drain_stream_fast(0);
                drain_stream_fast(1);
            }
            polish_pending_hits();

            unsigned long long gpu_cands = 0, gpu_nodes = 0;
            CUDA_CHECK(cudaMemcpy(&gpu_cands, d_leaf_cands, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&gpu_nodes, d_tree_nodes, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

            auto t1 = std::chrono::steady_clock::now();
            double sec = std::chrono::duration<double>(t1 - t0).count();
            double tot_sec = std::chrono::duration<double>(t1 - t_start).count();
            fprintf(stderr, "[%s (%zu/%zu) yc=%.2fe18 (%zu/%zu)] %.2fs (tot %.1fs) | lvl10=%llu | gpu_cands=%llu (%.2f Gcands/s, %.2f Gnodes/s) | new_clust=%llu | tot_clust=%llu | best=%.8f\n",
                    fname, f_idx + 1, fnames.size(), y_center_e18, yc_idx + 1, yc_list.size(), sec, tot_sec,
                    (unsigned long long)(total_lvl10_nodes.load() - prev_lvl10),
                    gpu_cands,
                    (double)gpu_cands / (tot_sec * 1e9),
                    (double)(gpu_nodes + total_cpu_nodes.load()) / (tot_sec * 1e9),
                    (unsigned long long)(total_clusters.load() - prev_clusters),
                    (unsigned long long)total_clusters.load(), global_best_abs);
            fflush(stderr);
        }
    }

    drain_stream_fast(0);
    drain_stream_fast(1);
    polish_pending_hits();

    unsigned long long final_gpu_cands = 0, final_gpu_nodes = 0;
    CUDA_CHECK(cudaMemcpy(&final_gpu_cands, d_leaf_cands, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&final_gpu_nodes, d_tree_nodes, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    auto t_end = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(t_end - t_start).count();
    fprintf(stderr, "ALL DONE in %.2f s | cpu_nodes=%llu | lvl10_nodes=%llu | gpu_tree_nodes=%llu (%.2f Gnodes/s) | gpu_leaf_cands=%llu (%.2f Gcands/s) | unique_clusters=%llu | best_abs=%.8f\n",
            sec,
            (unsigned long long)total_cpu_nodes.load(),
            (unsigned long long)total_lvl10_nodes.load(),
            final_gpu_nodes,
            (double)(final_gpu_nodes + total_cpu_nodes.load()) / (sec * 1e9),
            final_gpu_cands,
            (double)final_gpu_cands / (sec * 1e9),
            (unsigned long long)total_clusters.load(),
            global_best_abs);

    return 0;
}
