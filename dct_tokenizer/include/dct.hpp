#pragma once

#include <array>
#include <cstdint>
#include <vector>
#include <cmath>

namespace dct {

// Precomputed DCT coefficients for 8x8 blocks
// DCT-II: C[k,n] = cos(π * k * (2n + 1) / 16)
class DCTCoefficients {
public:
    static constexpr int BLOCK_SIZE = 8;
    static constexpr int BLOCK_SIZE_SQ = 64;
    
    // Singleton access
    static const DCTCoefficients& instance();
    
    // Precomputed cosine table for DCT
    alignas(64) float cos_table[BLOCK_SIZE][BLOCK_SIZE];
    
    // Precomputed alpha values (normalization)
    alignas(64) float alpha[BLOCK_SIZE];
    
    // Zigzag order indices for coarse->fine ordering
    alignas(64) int zigzag[BLOCK_SIZE_SQ];
    
    // Inverse zigzag for reconstruction
    alignas(64) int inverse_zigzag[BLOCK_SIZE_SQ];
    
private:
    DCTCoefficients();
};

// Fast 8x8 DCT using separable 1D transforms
// Input: 8x8 block of floats (row-major)
// Output: 8x8 DCT coefficients (row-major)
void dct_8x8(const float* input, float* output);

// Inverse DCT for reconstruction
void idct_8x8(const float* input, float* output);

// Batch DCT for multiple blocks (better cache utilization)
void dct_8x8_batch(const float* inputs, float* outputs, int num_blocks);

// SIMD-optimized version (AVX2)
#ifdef __AVX2__
void dct_8x8_avx2(const float* input, float* output);
void dct_8x8_batch_avx2(const float* inputs, float* outputs, int num_blocks);
#endif

} // namespace dct
