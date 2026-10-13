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


// The 3-split chains (2..64 tokens): blockIdx.z = split g; each stage is one 16-wide k step of the split's list,
// starting at list position ((n / 128) % 4) * 8.  Partials go to ws[g][row][n] (rows: the group's 3-split experts in
// order), reduced by gu_exact_reduce.
template <int BN, int BT, int TN, int TT>
__global__ void __launch_bounds__((BN / TN) * (BT / TT)) gu_exact_split_kernel(const GuExactGroup g, float* __restrict__ ws,
                                                                             int ws_rows) {
    constexpr int DU = 16, NT = (BN / TN) * (BT / TT), TXN = BN / TN, P = DU + 4, IT = kK / DU;
    __shared__ __attribute__((aligned(16))) float sw[BN * P];
    __shared__ __attribute__((aligned(16))) float sx[BT * P];
    int e = 0;
    while (e + 1 < g.n && (int) blockIdx.y >= g.b0[e + 1]) ++e;
    const __half* __restrict__ W = (const __half*) g.w[e];
    const __half* __restrict__ X = (const __half*) g.x[e];
    const int T = g.t[e];
    const int split = blockIdx.z, len = (IT - split + 2) / 3;
    const int tid = threadIdx.x, tx = tid % TXN, ty = tid / TXN;
    const int n0 = blockIdx.x * BN, t0 = ((int) blockIdx.y - g.b0[e]) * BT;
    const int rot = ((n0 / 128) % 4) * 8;
    float* __restrict__ out = ws + ((size_t) split * ws_rows + (size_t) g.row0[e]) * kN;
    constexpr int WV = BN * DU / 8, XV = BT * DU / 8;
    constexpr int WL = (WV + NT - 1) / NT, XL = (XV + NT - 1) / NT;
    uint4 rw[WL], rx[XL];
    auto gload = [&](int i) {
        const int k0 = (split + 3 * ((rot + i) % len)) * DU;
#pragma unroll
        for (int q = 0; q < WL; ++q) {
            const int v = tid + NT * q;
            if (WV % NT == 0 || v < WV) {
                const int r = v / (DU / 8), c = (v % (DU / 8)) * 8;
                rw[q] = *(const uint4*) (W + (size_t) (n0 + r) * kK + k0 + c);
            }
        }
#pragma unroll
        for (int q = 0; q < XL; ++q) {
            const int v = tid + NT * q;
            if (XV % NT == 0 || v < XV) {
                const int r = v / (DU / 8), c = (v % (DU / 8)) * 8, t = t0 + r;
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
    for (int i = 0; i < len; ++i) {
        __syncthreads();
#pragma unroll
        for (int q = 0; q < WL; ++q) {
            const int v = tid + NT * q;
            if (WV % NT == 0 || v < WV) st8(&sw[(v / (DU / 8)) * P + (v % (DU / 8)) * 8], rw[q]);
        }
#pragma unroll
        for (int q = 0; q < XL; ++q) {
            const int v = tid + NT * q;
            if (XV % NT == 0 || v < XV) st8(&sx[(v / (DU / 8)) * P + (v % (DU / 8)) * 8], rx[q]);
        }
        __syncthreads();
        if (i + 1 < len) gload(i + 1);
#pragma unroll
        for (int k = 0; k < DU; k += 4) {
            float4 a[TN], b[TT];
#pragma unroll
            for (int ii = 0; ii < TN; ++ii) a[ii] = *(const float4*) &sw[(tx + TXN * ii) * P + k];
#pragma unroll
            for (int j = 0; j < TT; ++j) b[j] = *(const float4*) &sx[(ty + (NT / TXN) * j) * P + k];
#pragma unroll
            for (int ii = 0; ii < TN; ++ii)
#pragma unroll
                for (int j = 0; j < TT; ++j) {
                    acc[ii][j] = __builtin_fmaf(a[ii].x, b[j].x, acc[ii][j]);
                    acc[ii][j] = __builtin_fmaf(a[ii].y, b[j].y, acc[ii][j]);
                    acc[ii][j] = __builtin_fmaf(a[ii].z, b[j].z, acc[ii][j]);
                    acc[ii][j] = __builtin_fmaf(a[ii].w, b[j].w, acc[ii][j]);
                }
        }
    }
#pragma unroll
    for (int ii = 0; ii < TN; ++ii)
#pragma unroll
        for (int j = 0; j < TT; ++j) {
            const int n = n0 + tx + TXN * ii, t = t0 + ty + (NT / TXN) * j;
            if (t < T) out[(size_t) t * kN + n] = acc[ii][j];
        }
}

// PostGSU3: y = (p0 + p1) + p2 for every row of the group's 3-split experts
__global__ void gu_exact_reduce(const GuExactGroup g, const float* __restrict__ ws, int ws_rows) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) ws_rows * kN) return;
    const int row = (int) (i / kN), n = (int) (i % kN);
    int e = 0;
    while (e + 1 < g.n && row >= g.row0[e + 1]) ++e;
    const size_t s = (size_t) ws_rows * kN;
    g.y[e][(size_t) (row - g.row0[e]) * kN + n] = (ws[i] + ws[s + i]) + ws[2 * s + i];
}
}  // namespace

bool gu_exact_run(GuExactGroup& g, float* ws, void* stream) {
    if (g.n <= 0 || g.n > kGuExactMax) return false;
    // two sub-groups: the chains (>= 65 tokens) and the 3-split chains (2..64), each a group of its own
    GuExactGroup big, small;
    for (int e = 0; e < g.n; ++e) {
        GuExactGroup& d = g.t[e] >= kGuExactSplitT ? big : small;
        d.x[d.n] = g.x[e];
        d.w[d.n] = g.w[e];
        d.y[d.n] = g.y[e];
        d.t[d.n] = g.t[e];
        ++d.n;
    }
    const hipStream_t st = (hipStream_t) stream;
    if (big.n > 0) {
        constexpr int BN = 64, BT = 64, TN = 8, TT = 4, DU = 32;
        big.b0[0] = 0;
        for (int e = 0; e < big.n; ++e) big.b0[e + 1] = big.b0[e] + (big.t[e] + BT - 1) / BT;
        hipLaunchKernelGGL((gu_exact_kernel<BN, BT, TN, TT, DU>), dim3(kN / BN, big.b0[big.n]), dim3((BN / TN) * (BT / TT)), 0,
                           st, big);
    }
    if (small.n > 0) {
        if (ws == nullptr) return false;
        constexpr int BN = 64, BT = 32, TN = 4, TT = 4;
        small.b0[0] = 0;
        small.row0[0] = 0;
        for (int e = 0; e < small.n; ++e) {
            small.b0[e + 1] = small.b0[e] + (small.t[e] + BT - 1) / BT;
            small.row0[e + 1] = small.row0[e] + small.t[e];
        }
        const int rows = small.row0[small.n];
        hipLaunchKernelGGL((gu_exact_split_kernel<BN, BT, TN, TT>), dim3(kN / BN, small.b0[small.n], 3),
                           dim3((BN / TN) * (BT / TT)), 0, st, small, ws, rows);
        const int64_t total = (int64_t) rows * kN;
        hipLaunchKernelGGL(gu_exact_reduce, dim3((unsigned) ((total + 255) / 256)), dim3(256), 0, st, (const GuExactGroup) small,
                           (const float*) ws, rows);
    }
    return hipGetLastError() == hipSuccess;
}

}  // namespace strata::prefill

#else

namespace strata::prefill {
bool gu_exact_run(GuExactGroup&, float*, void*) { return false; }
}  // namespace strata::prefill

#endif
