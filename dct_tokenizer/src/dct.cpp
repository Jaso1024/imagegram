#include "dct.hpp"
#include <cstring>
#include <cstdlib>
#include <omp.h>

namespace dct {

// Singleton initialization
const DCTCoefficients& DCTCoefficients::instance() {
    static DCTCoefficients inst;
    return inst;
}

DCTCoefficients::DCTCoefficients() {
    constexpr float PI = 3.14159265358979323846f;
    
    // Precompute cosine table
    // cos_table[k][n] = cos(π * k * (2n + 1) / 16)
    for (int k = 0; k < BLOCK_SIZE; k++) {
        for (int n = 0; n < BLOCK_SIZE; n++) {
            cos_table[k][n] = std::cos(PI * k * (2 * n + 1) / (2 * BLOCK_SIZE));
        }
    }
    
    // Precompute alpha values (normalization factors)
    // alpha[0] = 1/sqrt(N), alpha[k>0] = sqrt(2/N)
    alpha[0] = 1.0f / std::sqrt(static_cast<float>(BLOCK_SIZE));
    for (int k = 1; k < BLOCK_SIZE; k++) {
        alpha[k] = std::sqrt(2.0f / BLOCK_SIZE);
    }
    
    // Zigzag order (standard JPEG zigzag pattern)
    // This naturally orders coefficients from low to high frequency
    static const int zz[64] = {
         0,  1,  8, 16,  9,  2,  3, 10,
        17, 24, 32, 25, 18, 11,  4,  5,
        12, 19, 26, 33, 40, 48, 41, 34,
        27, 20, 13,  6,  7, 14, 21, 28,
        35, 42, 49, 56, 57, 50, 43, 36,
        29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46,
        53, 60, 61, 54, 47, 55, 62, 63
    };
    
    std::memcpy(zigzag, zz, sizeof(zigzag));
    
    // Build inverse zigzag
    for (int i = 0; i < BLOCK_SIZE_SQ; i++) {
        inverse_zigzag[zigzag[i]] = i;
    }
}

// Optimized 8x8 DCT using separable 1D transforms
// This is O(N^2) per dimension instead of O(N^4) for naive 2D DCT
void dct_8x8(const float* input, float* output) {
    const auto& coeff = DCTCoefficients::instance();
    
    alignas(32) float temp[64];
    
    // 1D DCT on rows
    for (int row = 0; row < 8; row++) {
        for (int k = 0; k < 8; k++) {
            float sum = 0.0f;
            for (int n = 0; n < 8; n++) {
                sum += input[row * 8 + n] * coeff.cos_table[k][n];
            }
            temp[row * 8 + k] = sum * coeff.alpha[k];
        }
    }
    
    // 1D DCT on columns
    for (int col = 0; col < 8; col++) {
        for (int k = 0; k < 8; k++) {
            float sum = 0.0f;
            for (int n = 0; n < 8; n++) {
                sum += temp[n * 8 + col] * coeff.cos_table[k][n];
            }
            output[k * 8 + col] = sum * coeff.alpha[k];
        }
    }
}

// Inverse DCT
void idct_8x8(const float* input, float* output) {
    const auto& coeff = DCTCoefficients::instance();
    
    alignas(32) float temp[64];
    
    // 1D IDCT on columns
    for (int col = 0; col < 8; col++) {
        for (int n = 0; n < 8; n++) {
            float sum = 0.0f;
            for (int k = 0; k < 8; k++) {
                sum += coeff.alpha[k] * input[k * 8 + col] * coeff.cos_table[k][n];
            }
            temp[n * 8 + col] = sum;
        }
    }
    
    // 1D IDCT on rows
    for (int row = 0; row < 8; row++) {
        for (int n = 0; n < 8; n++) {
            float sum = 0.0f;
            for (int k = 0; k < 8; k++) {
                sum += coeff.alpha[k] * temp[row * 8 + k] * coeff.cos_table[k][n];
            }
            output[row * 8 + n] = sum;
        }
    }
}

// Batch processing for better cache utilization
void dct_8x8_batch(const float* inputs, float* outputs, int num_blocks) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_blocks; i++) {
        dct_8x8(inputs + i * 64, outputs + i * 64);
    }
}

#if defined(__AVX2__) && !defined(__clang__)
#include <x86intrin.h>

// AVX2-optimized 8x8 DCT
// Processes 8 values at once using 256-bit vectors
void dct_8x8_avx2(const float* input, float* output) {
    const auto& coeff = DCTCoefficients::instance();
    
    alignas(32) float temp[64];
    
    // Load cosine table rows into AVX registers
    __m256 cos_rows[8];
    for (int k = 0; k < 8; k++) {
        cos_rows[k] = _mm256_load_ps(coeff.cos_table[k]);
    }
    
    // 1D DCT on rows (8 rows, each row is 8 floats = 1 AVX register)
    for (int row = 0; row < 8; row++) {
        __m256 in_row = _mm256_loadu_ps(input + row * 8);
        
        for (int k = 0; k < 8; k++) {
            // Multiply input by cosine values
            __m256 prod = _mm256_mul_ps(in_row, cos_rows[k]);
            
            // Horizontal sum (reduce 8 floats to 1)
            __m128 hi = _mm256_extractf128_ps(prod, 1);
            __m128 lo = _mm256_castps256_ps128(prod);
            __m128 sum4 = _mm_add_ps(lo, hi);
            __m128 sum2 = _mm_add_ps(sum4, _mm_movehl_ps(sum4, sum4));
            __m128 sum1 = _mm_add_ss(sum2, _mm_shuffle_ps(sum2, sum2, 1));
            
            temp[row * 8 + k] = _mm_cvtss_f32(sum1) * coeff.alpha[k];
        }
    }
    
    // 1D DCT on columns
    for (int col = 0; col < 8; col++) {
        // Gather column values
        alignas(32) float col_vals[8];
        for (int n = 0; n < 8; n++) {
            col_vals[n] = temp[n * 8 + col];
        }
        __m256 in_col = _mm256_load_ps(col_vals);
        
        for (int k = 0; k < 8; k++) {
            __m256 prod = _mm256_mul_ps(in_col, cos_rows[k]);
            
            __m128 hi = _mm256_extractf128_ps(prod, 1);
            __m128 lo = _mm256_castps256_ps128(prod);
            __m128 sum4 = _mm_add_ps(lo, hi);
            __m128 sum2 = _mm_add_ps(sum4, _mm_movehl_ps(sum4, sum4));
            __m128 sum1 = _mm_add_ss(sum2, _mm_shuffle_ps(sum2, sum2, 1));
            
            output[k * 8 + col] = _mm_cvtss_f32(sum1) * coeff.alpha[k];
        }
    }
}

void dct_8x8_batch_avx2(const float* inputs, float* outputs, int num_blocks) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_blocks; i++) {
        dct_8x8_avx2(inputs + i * 64, outputs + i * 64);
    }
}
#endif

} // namespace dct
