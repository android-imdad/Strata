// include/strata/prefill/gu_exact.hpp - the experts' gate/up product for many experts in one launch, bit for bit what
// rocBLAS's FP16 fallback gives each expert alone (STRATA_PF_GU_EXACT, HIP gfx1100).
//
// The prompt path's gate/up product (Y[T, 1280] = X[T, 2560] . W[1280, 2560]^T, FP16 in, FP32 out) has no hipBLASLt
// calibration, so it runs on rocBLAS's fallback Tensile kernels.  From 65 tokens on rocBLAS takes
// Cijk_Alik_Bljk_HSS_BH_MT64x32x8 (GSU1, StaggerU 32 / 256 bytes): every output is one FMA chain over k in order,
// starting at k = ((n / 64) % 16) * 128 and wrapping.  This kernel computes the same chain for every output of a group
// of experts, so its products are the same bits (checked against Gemm::f16 at start-up; any difference turns it off).
#pragma once

#include <cstdint>

namespace strata::prefill {

constexpr int kGuExactMax = 64;      // experts in one launch
constexpr int64_t kGuExactMinT = 65; // below: rocBLAS's split-K kernel, another order
constexpr int64_t kGuExactMaxT = 16384;

struct GuExactGroup {
    const uint16_t* x[kGuExactMax];   // FP16 activations, row stride 2560
    const uint16_t* w[kGuExactMax];   // FP16 gate/up weights [1280, 2560]
    float* y[kGuExactMax];            // FP32 out, row stride 1280
    int t[kGuExactMax];               // tokens of each expert
    int b0[kGuExactMax + 1];          // filled by the launcher
    int n = 0;
};

/// Y_e = X_e . W_e^T for every expert of the group, in one launch (HIP; false elsewhere or on a launch error).
bool gu_exact_run(GuExactGroup& g, void* stream);

}  // namespace strata::prefill
