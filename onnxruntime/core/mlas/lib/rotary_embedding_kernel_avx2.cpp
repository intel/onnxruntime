/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    rotary_embedding_kernel_avx2.cpp

Abstract:

    This module implements the rotary embedding kernels for AVX2 supported h/w.

--*/

#include <cassert>
#include <algorithm>
#include <cstring>
#include "rotary_embedding.h"
#include "rotary_embedding_kernel_avx2.h"

namespace rope_avx2 {

namespace {

typedef __m256 float32x8_t;

// Production-grade helper functions
inline float32x8_t load_fp16_to_fp32(const MLAS_FP16* src) {
    if (!src) return _mm256_setzero_ps();
    __m128i fp16_data = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
    return _mm256_cvtph_ps(fp16_data);
}

inline void store_fp32_to_fp16(MLAS_FP16* dst, const float32x8_t src) {
    if (!dst) return;
    __m128i fp16_data = _mm256_cvtps_ph(src, _MM_FROUND_TO_NEAREST_INT);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), fp16_data);
}

// NVIDIA/CUDA-inspired non-interleaved RoPE kernel for FP32
void rope_fp32_non_interleaved(const float* input, const float* sin_data,
                               const float* cos_data, size_t dim, float* output) {
    // Input validation - critical for avoiding segfaults
    if (!input || !sin_data || !cos_data || !output || dim == 0 || (dim & 1)) {
        return; // Invalid inputs
    }

    const size_t half_dim = dim >> 1;
    constexpr size_t vector_width = 8;

    // Mathematical analysis for safe SIMD bounds:
    // SIMD load reads 8 consecutive elements: [i, i+1, ..., i+7]
    // For real part: need i+7 < half_dim, so i <= half_dim-8
    // For imag part: j = half_dim + i, need j+7 < dim, so half_dim+i+7 < dim
    // Since dim = 2*half_dim: half_dim+i+7 < 2*half_dim, so i+7 < half_dim (same constraint)
    // Therefore: valid range is i ∈ [0, half_dim-8] when half_dim >= 8

    size_t i = 0;

    // Process SIMD chunks only if we have at least 8 elements in half_dim
    if (half_dim >= vector_width) {
        const size_t simd_iterations = (half_dim - vector_width + 1) / vector_width;
        const size_t simd_end = simd_iterations * vector_width;

        for (; i < simd_end; i += vector_width) {
            const size_t j = half_dim + i;

            // Additional runtime safety checks (can be removed in production after validation)
            if (i + vector_width > half_dim || j + vector_width > dim) {
                break; // Safety exit - should not happen with correct math
            }

            // Mathematical guarantee: i+7 <= simd_end-1+7 <= half_dim-1, so safe
            const float32x8_t real = _mm256_loadu_ps(input + i);
            const float32x8_t imag = _mm256_loadu_ps(input + j);
            const float32x8_t sin_val = _mm256_loadu_ps(sin_data + i);
            const float32x8_t cos_val = _mm256_loadu_ps(cos_data + i);

            // RoPE computation: complex rotation
            const float32x8_t real_out = _mm256_fmsub_ps(real, cos_val, _mm256_mul_ps(imag, sin_val));
            const float32x8_t imag_out = _mm256_fmadd_ps(real, sin_val, _mm256_mul_ps(imag, cos_val));

            _mm256_storeu_ps(output + i, real_out);
            _mm256_storeu_ps(output + j, imag_out);
        }
    }

    // Scalar cleanup for remaining elements
    for (; i < half_dim; ++i) {
        const size_t j = half_dim + i;
        const float real = input[i];
        const float imag = input[j];
        const float sin_val = sin_data[i];
        const float cos_val = cos_data[i];

        output[i] = real * cos_val - imag * sin_val;
        output[j] = real * sin_val + imag * cos_val;
    }
}

// ROCm-inspired interleaved RoPE kernel for FP32
void rope_fp32_interleaved(const float* input, const float* sin_data,
                          const float* cos_data, size_t dim, float* output) {
    // Input validation - critical for avoiding segfaults
    if (!input || !sin_data || !cos_data || !output || dim == 0 || (dim & 1)) {
        return; // Invalid inputs
    }

    constexpr size_t vector_width = 16;
    const __m256i shuffle_mask = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);

    // Mathematical analysis for interleaved SIMD bounds:
    // We process 16 elements: input[i] to input[i+15]
    // We need sin/cos data: sin_data[i/2] to sin_data[i/2+7] = sin_data[(i+14)/2]
    // Constraints: i+15 < dim AND (i+14)/2 < dim/2
    // Second constraint: i+14 < dim (more restrictive)
    // Therefore: valid range is i ∈ [0, dim-16] when dim >= 16

    size_t i = 0;

    // Process SIMD chunks only if we have at least 16 elements
    if (dim >= vector_width) {
        const size_t simd_iterations = (dim - vector_width + 1) / vector_width;
        const size_t simd_end = simd_iterations * vector_width;

        for (; i < simd_end; i += vector_width) {
            const size_t sin_cos_idx = i >> 1;

            // Additional runtime safety checks (can be removed in production after validation)
            if (i + vector_width > dim || sin_cos_idx + 8 > dim / 2) {
                break; // Safety exit - should not happen with correct math
            }

            // Mathematical guarantee: i+15 <= simd_end-1+15 <= dim-1, so safe
            const float32x8_t x0 = _mm256_loadu_ps(input + i);
            const float32x8_t x1 = _mm256_loadu_ps(input + i + 8);

            // Deinterleave real and imaginary parts
            const float32x8_t real_s = _mm256_shuffle_ps(x0, x1, 0b10001000);
            const float32x8_t imag_s = _mm256_shuffle_ps(x0, x1, 0b11011101);
            const float32x8_t real = _mm256_permutevar8x32_ps(real_s, shuffle_mask);
            const float32x8_t imag = _mm256_permutevar8x32_ps(imag_s, shuffle_mask);

            // sin_cos_idx+7 <= (simd_end-1)/2+7 <= (dim-16)/2+7 = dim/2-1, so safe
            const float32x8_t sin_val = _mm256_loadu_ps(sin_data + sin_cos_idx);
            const float32x8_t cos_val = _mm256_loadu_ps(cos_data + sin_cos_idx);

            // RoPE computation
            const float32x8_t real_out = _mm256_fmsub_ps(real, cos_val, _mm256_mul_ps(imag, sin_val));
            const float32x8_t imag_out = _mm256_fmadd_ps(real, sin_val, _mm256_mul_ps(imag, cos_val));

            // Reinterleave and store
            const float32x8_t real_out_s = _mm256_permutevar8x32_ps(real_out, shuffle_mask);
            const float32x8_t imag_out_s = _mm256_permutevar8x32_ps(imag_out, shuffle_mask);
            const float32x8_t y0 = _mm256_unpacklo_ps(real_out_s, imag_out_s);
            const float32x8_t y1 = _mm256_unpackhi_ps(real_out_s, imag_out_s);

            _mm256_storeu_ps(output + i, y0);
            _mm256_storeu_ps(output + i + 8, y1);
        }
    }

    // Scalar cleanup
    for (; i + 1 < dim; i += 2) {
        const size_t sin_cos_idx = i >> 1;
        const float real = input[i];
        const float imag = input[i + 1];
        const float cos_val = cos_data[sin_cos_idx];
        const float sin_val = sin_data[sin_cos_idx];

        output[i] = real * cos_val - imag * sin_val;
        output[i + 1] = real * sin_val + imag * cos_val;
    }
}

// QNN-inspired non-interleaved RoPE kernel for FP16
void rope_fp16_non_interleaved(const MLAS_FP16* input, const MLAS_FP16* sin_data,
                               const MLAS_FP16* cos_data, size_t dim, MLAS_FP16* output) {
    // Input validation - critical for avoiding segfaults
    if (!input || !sin_data || !cos_data || !output || dim == 0 || (dim & 1)) {
        return; // Invalid inputs
    }

    const size_t half_dim = dim >> 1;
    constexpr size_t vector_width = 8;

    // Mathematical analysis: same as FP32 non-interleaved
    // Each FP16 SIMD load reads 8 consecutive elements
    // Valid range: i ∈ [0, half_dim-8] when half_dim >= 8

    size_t i = 0;

    // Process SIMD chunks only if we have at least 8 elements in half_dim
    if (half_dim >= vector_width) {
        const size_t simd_iterations = (half_dim - vector_width + 1) / vector_width;
        const size_t simd_end = simd_iterations * vector_width;

        for (; i < simd_end; i += vector_width) {
            const size_t j = half_dim + i;

            // Additional runtime safety checks (can be removed in production after validation)
            if (i + vector_width > half_dim || j + vector_width > dim) {
                break; // Safety exit - should not happen with correct math
            }

            // Mathematical guarantee: all accesses within bounds
            const float32x8_t real = load_fp16_to_fp32(input + i);
            const float32x8_t imag = load_fp16_to_fp32(input + j);
            const float32x8_t sin_val = load_fp16_to_fp32(sin_data + i);
            const float32x8_t cos_val = load_fp16_to_fp32(cos_data + i);

            // RoPE computation
            const float32x8_t real_out = _mm256_fmsub_ps(real, cos_val, _mm256_mul_ps(imag, sin_val));
            const float32x8_t imag_out = _mm256_fmadd_ps(real, sin_val, _mm256_mul_ps(imag, cos_val));

            // Convert back to FP16 and store
            store_fp32_to_fp16(output + i, real_out);
            store_fp32_to_fp16(output + j, imag_out);
        }
    }

    // Scalar cleanup
    for (; i < half_dim; ++i) {
        const size_t j = half_dim + i;
        const float real = input[i].ToFloat();
        const float imag = input[j].ToFloat();
        const float sin_val = sin_data[i].ToFloat();
        const float cos_val = cos_data[i].ToFloat();

        output[i] = MLAS_FP16(real * cos_val - imag * sin_val);
        output[j] = MLAS_FP16(real * sin_val + imag * cos_val);
    }
}

// Production interleaved RoPE kernel for FP16
void rope_fp16_interleaved(const MLAS_FP16* input, const MLAS_FP16* sin_data,
                          const MLAS_FP16* cos_data, size_t dim, MLAS_FP16* output) {
    // Input validation - critical for avoiding segfaults
    if (!input || !sin_data || !cos_data || !output || dim == 0 || (dim & 1)) {
        return; // Invalid inputs
    }

    constexpr size_t vector_width = 16;
    const __m256i shuffle_mask = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);

    // Mathematical analysis: same as FP32 interleaved
    // Process 16 FP16 elements, need sin/cos for 8 complex pairs
    // Valid range: i ∈ [0, dim-16] when dim >= 16

    size_t i = 0;

    // Process SIMD chunks only if we have at least 16 elements
    if (dim >= vector_width) {
        const size_t simd_iterations = (dim - vector_width + 1) / vector_width;
        const size_t simd_end = simd_iterations * vector_width;

        for (; i < simd_end; i += vector_width) {
            const size_t sin_cos_idx = i >> 1;

            // Additional runtime safety checks (can be removed in production after validation)
            if (i + vector_width > dim || sin_cos_idx + 8 > dim / 2) {
                break; // Safety exit - should not happen with correct math
            }

            // Mathematical guarantee: all accesses within bounds
            const float32x8_t x0 = load_fp16_to_fp32(input + i);
            const float32x8_t x1 = load_fp16_to_fp32(input + i + 8);

            // Deinterleave
            const float32x8_t real_s = _mm256_shuffle_ps(x0, x1, 0b10001000);
            const float32x8_t imag_s = _mm256_shuffle_ps(x0, x1, 0b11011101);
            const float32x8_t real = _mm256_permutevar8x32_ps(real_s, shuffle_mask);
            const float32x8_t imag = _mm256_permutevar8x32_ps(imag_s, shuffle_mask);

            // Load sin/cos
            const float32x8_t sin_val = load_fp16_to_fp32(sin_data + sin_cos_idx);
            const float32x8_t cos_val = load_fp16_to_fp32(cos_data + sin_cos_idx);

            // RoPE computation
            const float32x8_t real_out = _mm256_fmsub_ps(real, cos_val, _mm256_mul_ps(imag, sin_val));
            const float32x8_t imag_out = _mm256_fmadd_ps(real, sin_val, _mm256_mul_ps(imag, cos_val));

            // Reinterleave
            const float32x8_t real_out_s = _mm256_permutevar8x32_ps(real_out, shuffle_mask);
            const float32x8_t imag_out_s = _mm256_permutevar8x32_ps(imag_out, shuffle_mask);
            const float32x8_t y0 = _mm256_unpacklo_ps(real_out_s, imag_out_s);
            const float32x8_t y1 = _mm256_unpackhi_ps(real_out_s, imag_out_s);

            // Convert back to FP16 and store
            store_fp32_to_fp16(output + i, y0);
            store_fp32_to_fp16(output + i + 8, y1);
        }
    }

    // Scalar cleanup
    for (; i < dim; ++i) {
        const size_t sin_cos_idx = i >> 1;
        const bool is_imag = i & 1;
        const size_t pair_idx = is_imag ? i - 1 : i + 1;

        if (pair_idx < dim && sin_cos_idx < (dim >> 1)) {
            const float input_i = input[i].ToFloat();
            const float input_pair = input[pair_idx].ToFloat();
            const float cos_val = cos_data[sin_cos_idx].ToFloat();
            const float sin_val = sin_data[sin_cos_idx].ToFloat();

            const float result = is_imag ?
                (input_pair * sin_val + input_i * cos_val) :
                (input_i * cos_val - input_pair * sin_val);

            output[i] = MLAS_FP16(result);
        } else {
            output[i] = input[i];
        }
    }
}

}  // anonymous namespace

void RopeKernel_Avx2_fp32(const float* input, const float* sin_data, const float* cos_data,
                           size_t dim, bool interleaved, float* output) {
    // Basic validation only - no excessive checks
    if (!input || !output || !sin_data || !cos_data || dim == 0 || dim % 2 != 0) {
        return;
    }

    if (interleaved) {
        rope_fp32_interleaved(input, sin_data, cos_data, dim, output);
    } else {
        rope_fp32_non_interleaved(input, sin_data, cos_data, dim, output);
    }
}

void RopeKernel_Avx2_fp16(const MLAS_FP16* input, const MLAS_FP16* sin_data, const MLAS_FP16* cos_data,
                           size_t dim, bool interleaved, MLAS_FP16* output) {
    // Basic validation only - no excessive checks
    if (!input || !output || !sin_data || !cos_data || dim == 0 || dim % 2 != 0) {
        return;
    }

    if (interleaved) {
        rope_fp16_interleaved(input, sin_data, cos_data, dim, output);
    } else {
        rope_fp16_non_interleaved(input, sin_data, cos_data, dim, output);
    }
}

}  // namespace rope_avx2

const MLAS_ROPE_DISPATCH MlasRopeDispatchAvx2 = []() {
    MLAS_ROPE_DISPATCH d;
    d.SRope = rope_avx2::RopeKernel_Avx2_fp32;
    d.HRope = rope_avx2::RopeKernel_Avx2_fp16;
    return d;
}();
