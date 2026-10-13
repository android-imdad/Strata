// src/prefill/gu_exact.cu - see include/strata/prefill/gu_exact.hpp.
#include "strata/prefill/gu_exact.hpp"

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

namespace strata::prefill {
namespace {

constexpr int kN = 1280, kK = 2560;

// 64 outputs (one StaggerU group: one start for the block) x BT tokens; each thread TN x TT outputs; DU k a stage.
// Every output is acc = fma(w[k], x[k], acc) over k = start, start + 1, ... (mod 2560): rocBLAS's order.
template <int BN, int BT, int TN, int TT, int DU>
__global__ void __launch_bounds__((BN / TN) * (BT / TT)) gu_exact_kernel(const GuExactGroup g) {
    static_assert(BN <= 64 && 64 % BN == 0, "a block stays inside one 64-output StaggerU group");
    constexpr int NT = (BN / TN) * (BT / TT), TXN = BN / TN, P = DU + 4;
    __shared__ __attribute__((aligned(16))) float sw[BN * P];
    __shared__ __attribute__((aligned(16))) float sx[BT * P];
    int e = 0;
    while (e + 1 < g.n && (int) blockIdx.y >= g.b0[e + 1]) ++e;
    const __half* __restrict__ W = (const __half*) g.w[e];
    const __half* __restrict__ X = (const __half*) g.x[e];
    float* __restrict__ Y = g.y[e];
    const int T = g.t[e];
    const int tid = threadIdx.x, tx = tid % TXN, ty = tid / TXN;
    const int n0 = blockIdx.x * BN, t0 = ((int) blockIdx.y - g.b0[e]) * BT;
    const int start = ((n0 / 64) % 16) * 128;
    constexpr int WV = BN * DU / 8, XV = BT * DU / 8;
    constexpr int WL = (WV + NT - 1) / NT, XL = (XV + NT - 1) / NT;
    uint4 rw[WL], rx[XL];
    auto gload = [&](int kk) {
        int k0 = start + kk;
        if (k0 >= kK) k0 -= kK;
#pragma unroll
        for (int q = 0; q < WL; ++q) {
            const int i = tid + NT * q;
            if (WV % NT == 0 || i < WV) {
                const int r = i / (DU / 8), c = (i % (DU / 8)) * 8;
                rw[q] = *(const uint4*) (W + (size_t) (n0 + r) * kK + k0 + c);
            }
        }
#pragma unroll
        for (int q = 0; q < XL; ++q) {
            const int i = tid + NT * q;
            if (XV % NT == 0 || i < XV) {
                const int r = i / (DU / 8), c = (i % (DU / 8)) * 8, t = t0 + r;
                rx[q] = t < T ? *(const uint4*) (X + (size_t) t * kK + k0 + c) : make_uint4(0, 0, 0, 0);
            }
        }
    };
    auto st8 = [](float* d, const uint4& v) {
        const __half* h = (const __half*) &v;
        *(float4*) d = make_float4(__half2float(h[0]), __half2float(h[1]), __half2float(h[2]), __half2float(h[3]));
        *(float4*) (d + 4) = make_float4(__half2float(h[4]), __half2float(h[5]), __half2float(h[6]), __half2float(h[7]));
    };
    float acc[TN][TT];
#pragma unroll
    for (int i = 0; i < TN; ++i)
#pragma unroll
        for (int j = 0; j < TT; ++j) acc[i][j] = 0.f;
    gload(0);
    for (int kk = 0; kk < kK; kk += DU) {
        __syncthreads();
#pragma unroll
        for (int q = 0; q < WL; ++q) {
            const int i = tid + NT * q;
            if (WV % NT == 0 || i < WV) st8(&sw[(i / (DU / 8)) * P + (i % (DU / 8)) * 8], rw[q]);
        }
#pragma unroll
        for (int q = 0; q < XL; ++q) {
            const int i = tid + NT * q;
            if (XV % NT == 0 || i < XV) st8(&sx[(i / (DU / 8)) * P + (i % (DU / 8)) * 8], rx[q]);
        }
        __syncthreads();
        if (kk + DU < kK) gload(kk + DU);
#pragma unroll
        for (int k = 0; k < DU; k += 4) {
            float4 a[TN], b[TT];
#pragma unroll
            for (int i = 0; i < TN; ++i) a[i] = *(const float4*) &sw[(tx + TXN * i) * P + k];
#pragma unroll
            for (int j = 0; j < TT; ++j) b[j] = *(const float4*) &sx[(ty + (NT / TXN) * j) * P + k];
#pragma unroll
            for (int i = 0; i < TN; ++i)
#pragma unroll
                for (int j = 0; j < TT; ++j) {   // k, k+1, k+2, k+3 in order: the chain's order
                    acc[i][j] = __builtin_fmaf(a[i].x, b[j].x, acc[i][j]);
                    acc[i][j] = __builtin_fmaf(a[i].y, b[j].y, acc[i][j]);
                    acc[i][j] = __builtin_fmaf(a[i].z, b[j].z, acc[i][j]);
                    acc[i][j] = __builtin_fmaf(a[i].w, b[j].w, acc[i][j]);
                }
        }
    }
#pragma unroll
    for (int i = 0; i < TN; ++i)
#pragma unroll
        for (int j = 0; j < TT; ++j) {
            const int n = n0 + tx + TXN * i, t = t0 + ty + (NT / TXN) * j;
            if (t < T) Y[(size_t) t * kN + n] = acc[i][j];
        }
}

}  // namespace

bool gu_exact_run(GuExactGroup& g, void* stream) {
    constexpr int BN = 64, BT = 64, TN = 8, TT = 4, DU = 32;
    if (g.n <= 0 || g.n > kGuExactMax) return false;
    g.b0[0] = 0;
    for (int e = 0; e < g.n; ++e) g.b0[e + 1] = g.b0[e] + (g.t[e] + BT - 1) / BT;
    hipLaunchKernelGGL((gu_exact_kernel<BN, BT, TN, TT, DU>), dim3(kN / BN, g.b0[g.n]), dim3((BN / TN) * (BT / TT)), 0,
                       (hipStream_t) stream, g);
    return hipGetLastError() == hipSuccess;
}

}  // namespace strata::prefill

#else

namespace strata::prefill {
bool gu_exact_run(GuExactGroup&, void*) { return false; }
}  // namespace strata::prefill

#endif
