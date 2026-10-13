// src/kernels/cpu/replica_avx512.hpp - prototype: the GPU expert arithmetic in AVX-512 (see replica_avx512.cpp).
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu::replica {

/// One token's q8_1 activations as the GPU makes them: the int8 values in order and each block's fp16 scale as a float
/// (d8x2: each scale twice, for the down rows' two parts per block).
struct alignas(64) Act {
    int8_t qs[2560];
    int8_t qp[2560];   // the same values in the down rows' part order (per 32: 0-7, 16-23, 8-15, 24-31)
    float d8[80];
    float d8x2[160];
};
void quant(const float* x, int n, Act& a);

bool gu_supported(int type);
/// h = SwiGLU(gate, up) rows [r0, r1) of nt tokens (sexp), the GPU's bits
void gu_rows(int type, int nt, const uint8_t* blob, size_t gu_row, size_t up_off, const Act* const* act, float* const* h,
             int r0, int r1);
/// IQ4_NL down rows [r0, r1) (n_ff 640) of nt tokens' q8_1 h, the GPU's bits
void down_iq4nl(int nt, const uint8_t* w, size_t row_bytes, const Act* const* h, float* const* out, int r0, int r1);

}  // namespace strata::kernels::cpu::replica
