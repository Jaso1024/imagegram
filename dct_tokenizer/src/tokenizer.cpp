#include "tokenizer.hpp"
#include "dct.hpp"
#include "image_loader.hpp"
#include <cstring>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <omp.h>

namespace dct {

// Thread-local context for parallel processing
struct Tokenizer::ThreadContext {
    std::vector<float> y_channel;
    std::vector<float> cb_channel;
    std::vector<float> cr_channel;
    std::vector<float> blocks;
    std::vector<float> dct_coeffs;
    std::vector<uint16_t> block_tokens;
    std::vector<uint8_t> resized_image;
    
    void resize(int image_size) {
        int pixels = image_size * image_size;
        int num_blocks = (image_size / 8) * (image_size / 8);
        
        y_channel.resize(pixels);
        cb_channel.resize(pixels);
        cr_channel.resize(pixels);
        blocks.resize(num_blocks * 64);
        dct_coeffs.resize(num_blocks * 64);
        block_tokens.resize(num_blocks * 64);
        resized_image.resize(pixels * 3);
    }
};

Tokenizer::Tokenizer(const TokenizerConfig& config) : config_(config) {
    int num_threads = config_.num_threads;
    if (num_threads <= 0) {
        num_threads = omp_get_max_threads();
    }
    config_.num_threads = num_threads;
    
    // Pre-allocate thread contexts
    thread_contexts_.resize(num_threads);
    for (int i = 0; i < num_threads; i++) {
        thread_contexts_[i] = std::make_unique<ThreadContext>();
        thread_contexts_[i]->resize(config_.image_size);
    }
}

Tokenizer::~Tokenizer() = default;

Tokenizer::ThreadContext& Tokenizer::get_context(int thread_id) const {
    return *thread_contexts_[thread_id % thread_contexts_.size()];
}

// RGB to YCbCr conversion (JPEG standard)
void Tokenizer::rgb_to_ycbcr(const uint8_t* rgb, float* y, float* cb, float* cr,
                             int width, int height) const {
    const int pixels = width * height;
    
    #pragma omp parallel for simd schedule(static)
    for (int i = 0; i < pixels; i++) {
        float r = rgb[i * 3 + 0];
        float g = rgb[i * 3 + 1];
        float b = rgb[i * 3 + 2];
        
        // JPEG conversion formulas (shifted to center around 0)
        y[i]  =  0.299f * r + 0.587f * g + 0.114f * b - 128.0f;
        cb[i] = -0.168736f * r - 0.331264f * g + 0.5f * b;
        cr[i] =  0.5f * r - 0.418688f * g - 0.081312f * b;
    }
}

// YCbCr to RGB conversion
void Tokenizer::ycbcr_to_rgb(const float* y, const float* cb, const float* cr,
                             uint8_t* rgb, int width, int height) const {
    const int pixels = width * height;
    
    #pragma omp parallel for simd schedule(static)
    for (int i = 0; i < pixels; i++) {
        float yv = y[i] + 128.0f;
        float cbv = cb[i];
        float crv = cr[i];
        
        float r = yv + 1.402f * crv;
        float g = yv - 0.344136f * cbv - 0.714136f * crv;
        float b = yv + 1.772f * cbv;
        
        rgb[i * 3 + 0] = static_cast<uint8_t>(std::clamp(r, 0.0f, 255.0f));
        rgb[i * 3 + 1] = static_cast<uint8_t>(std::clamp(g, 0.0f, 255.0f));
        rgb[i * 3 + 2] = static_cast<uint8_t>(std::clamp(b, 0.0f, 255.0f));
    }
}

// Extract 8x8 blocks from a channel (row-major order)
void Tokenizer::extract_blocks(const float* channel, float* blocks,
                               int width, int height) const {
    const int blocks_x = width / 8;
    const int blocks_y = height / 8;
    
    #pragma omp parallel for collapse(2) schedule(static)
    for (int by = 0; by < blocks_y; by++) {
        for (int bx = 0; bx < blocks_x; bx++) {
            float* block = blocks + (by * blocks_x + bx) * 64;
            
            for (int y = 0; y < 8; y++) {
                for (int x = 0; x < 8; x++) {
                    int px = bx * 8 + x;
                    int py = by * 8 + y;
                    block[y * 8 + x] = channel[py * width + px];
                }
            }
        }
    }
}

// Reconstruct channel from blocks
void Tokenizer::reconstruct_from_blocks(const float* blocks, float* channel,
                                        int width, int height) const {
    const int blocks_x = width / 8;
    const int blocks_y = height / 8;
    
    #pragma omp parallel for collapse(2) schedule(static)
    for (int by = 0; by < blocks_y; by++) {
        for (int bx = 0; bx < blocks_x; bx++) {
            const float* block = blocks + (by * blocks_x + bx) * 64;
            
            for (int y = 0; y < 8; y++) {
                for (int x = 0; x < 8; x++) {
                    int px = bx * 8 + x;
                    int py = by * 8 + y;
                    channel[py * width + px] = block[y * 8 + x];
                }
            }
        }
    }
}

// Quantize DCT coefficients to tokens
void Tokenizer::quantize_coefficients(const float* coeffs, uint16_t* tokens,
                                      int num_blocks) const {
    const auto& dct_coeff = DCTCoefficients::instance();
    const int vocab_size = config_.vocab_size();
    const float half_vocab = vocab_size / 2.0f;
    
    // Quantization ranges for different frequencies
    // DC coefficients have larger range, AC coefficients smaller
    // These values are tuned for typical image statistics
    static const float quant_scale[64] = {
        // Lower frequencies (larger range)
        16, 11, 10, 16, 24, 40, 51, 61,
        12, 12, 14, 19, 26, 58, 60, 55,
        14, 13, 16, 24, 40, 57, 69, 56,
        14, 17, 22, 29, 51, 87, 80, 62,
        18, 22, 37, 56, 68, 109, 103, 77,
        24, 35, 55, 64, 81, 104, 113, 92,
        49, 64, 78, 87, 103, 121, 120, 101,
        72, 92, 95, 98, 112, 100, 103, 99
    };
    
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < num_blocks; b++) {
        const float* block_coeffs = coeffs + b * 64;
        uint16_t* block_tokens = tokens + b * 64;
        
        for (int i = 0; i < 64; i++) {
            // Get zigzag index for this coefficient
            int zz_idx = dct_coeff.zigzag[i];
            float coeff = block_coeffs[zz_idx];
            
            // Quantize: map [-range, range] to [0, vocab_size-1]
            float scale = quant_scale[i] * 0.5f;  // Adjustable quality factor
            float normalized = (coeff / scale) + half_vocab;
            
            // Clamp to valid range
            int token = static_cast<int>(std::round(normalized));
            token = std::clamp(token, 0, vocab_size - 1);
            
            block_tokens[i] = static_cast<uint16_t>(token);
        }
    }
}

// Dequantize tokens back to DCT coefficients
void Tokenizer::dequantize_coefficients(const uint16_t* tokens, float* coeffs,
                                        int num_blocks) const {
    const auto& dct_coeff = DCTCoefficients::instance();
    const int vocab_size = config_.vocab_size();
    const float half_vocab = vocab_size / 2.0f;
    
    static const float quant_scale[64] = {
        16, 11, 10, 16, 24, 40, 51, 61,
        12, 12, 14, 19, 26, 58, 60, 55,
        14, 13, 16, 24, 40, 57, 69, 56,
        14, 17, 22, 29, 51, 87, 80, 62,
        18, 22, 37, 56, 68, 109, 103, 77,
        24, 35, 55, 64, 81, 104, 113, 92,
        49, 64, 78, 87, 103, 121, 120, 101,
        72, 92, 95, 98, 112, 100, 103, 99
    };
    
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < num_blocks; b++) {
        const uint16_t* block_tokens = tokens + b * 64;
        float* block_coeffs = coeffs + b * 64;
        
        // Initialize to zero
        std::memset(block_coeffs, 0, 64 * sizeof(float));
        
        for (int i = 0; i < 64; i++) {
            int zz_idx = dct_coeff.zigzag[i];
            float scale = quant_scale[i] * 0.5f;
            
            float normalized = static_cast<float>(block_tokens[i]) - half_vocab;
            block_coeffs[zz_idx] = normalized * scale;
        }
    }
}

// Reorder from block-first to frequency-first layout
// Input:  [block0_freq0, block0_freq1, ..., block0_freq63, block1_freq0, ...]
// Output: [block0_freq0, block1_freq0, ..., blockN_freq0, block0_freq1, ...]
void Tokenizer::reorder_frequency_first(const uint16_t* block_order, uint16_t* freq_order,
                                        int num_blocks) const {
    const int num_freqs = config_.num_frequencies;
    
    #pragma omp parallel for collapse(2) schedule(static)
    for (int f = 0; f < num_freqs; f++) {
        for (int b = 0; b < num_blocks; b++) {
            freq_order[f * num_blocks + b] = block_order[b * 64 + f];
        }
    }
}

// Reorder from frequency-first back to block-first
void Tokenizer::reorder_block_first(const uint16_t* freq_order, uint16_t* block_order,
                                    int num_blocks) const {
    const int num_freqs = config_.num_frequencies;
    
    #pragma omp parallel for collapse(2) schedule(static)
    for (int b = 0; b < num_blocks; b++) {
        for (int f = 0; f < num_freqs; f++) {
            block_order[b * 64 + f] = freq_order[f * num_blocks + b];
        }
    }
}

TokenizedImage Tokenizer::tokenize(const uint8_t* rgb_data, int width, int height) const {
    TokenizedImage result;
    result.config = config_;
    result.width = width;
    result.height = height;
    
    int thread_id = omp_get_thread_num();
    auto& ctx = get_context(thread_id);
    
    const int target_size = config_.image_size;
    const int num_blocks = config_.total_blocks();
    
    // Resize if needed
    const uint8_t* image_data = rgb_data;
    if (width != target_size || height != target_size) {
        resize_image(rgb_data, width, height,
                     ctx.resized_image.data(), target_size, target_size, 3);
        image_data = ctx.resized_image.data();
        width = height = target_size;
    }
    
    // Convert to YCbCr
    if (config_.use_ycbcr) {
        rgb_to_ycbcr(image_data, ctx.y_channel.data(), 
                     ctx.cb_channel.data(), ctx.cr_channel.data(),
                     width, height);
    } else {
        // Use RGB channels directly
        const int pixels = width * height;
        for (int i = 0; i < pixels; i++) {
            ctx.y_channel[i] = image_data[i * 3 + 0] - 128.0f;
            ctx.cb_channel[i] = image_data[i * 3 + 1] - 128.0f;
            ctx.cr_channel[i] = image_data[i * 3 + 2] - 128.0f;
        }
    }
    
    // Process each channel
    result.tokens.resize(config_.tokens_per_image());
    uint16_t* output_ptr = result.tokens.data();
    
    auto process_channel = [&](float* channel) {
        // Extract blocks
        extract_blocks(channel, ctx.blocks.data(), width, height);
        
        // DCT transform all blocks
        #ifdef __AVX2__
        dct_8x8_batch_avx2(ctx.blocks.data(), ctx.dct_coeffs.data(), num_blocks);
        #else
        dct_8x8_batch(ctx.blocks.data(), ctx.dct_coeffs.data(), num_blocks);
        #endif
        
        // Quantize to tokens (in zigzag order within each block)
        quantize_coefficients(ctx.dct_coeffs.data(), ctx.block_tokens.data(), num_blocks);
        
        // Reorder to frequency-first layout
        reorder_frequency_first(ctx.block_tokens.data(), output_ptr, num_blocks);
        
        output_ptr += num_blocks * config_.num_frequencies;
    };
    
    process_channel(ctx.y_channel.data());
    process_channel(ctx.cb_channel.data());
    process_channel(ctx.cr_channel.data());
    
    return result;
}

TokenizedImage Tokenizer::tokenize_file(const std::string& path) const {
    Image img = load_image_resized(path, config_.image_size);
    if (img.empty()) {
        return TokenizedImage{};  // Return empty on failure
    }
    return tokenize(img.data.data(), img.width, img.height);
}

std::vector<TokenizedImage> Tokenizer::tokenize_batch(
    const std::vector<std::string>& paths,
    int num_threads
) const {
    if (num_threads <= 0) {
        num_threads = config_.num_threads;
    }
    
    std::vector<TokenizedImage> results(paths.size());
    
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 8)
    for (size_t i = 0; i < paths.size(); i++) {
        results[i] = tokenize_file(paths[i]);
    }
    
    return results;
}

std::vector<uint8_t> Tokenizer::detokenize(const TokenizedImage& tokens) const {
    return detokenize(tokens.tokens.data(), tokens.tokens.size());
}

std::vector<uint8_t> Tokenizer::detokenize(const uint16_t* tokens, int num_tokens) const {
    int thread_id = omp_get_thread_num();
    auto& ctx = get_context(thread_id);
    
    const int target_size = config_.image_size;
    const int num_blocks = config_.total_blocks();
    const int tokens_per_channel = num_blocks * config_.num_frequencies;
    
    auto process_channel = [&](const uint16_t* channel_tokens, float* channel) {
        // Reorder from frequency-first to block-first
        reorder_block_first(channel_tokens, ctx.block_tokens.data(), num_blocks);
        
        // Pad with zeros if we only kept some frequencies
        if (config_.num_frequencies < 64) {
            for (int b = 0; b < num_blocks; b++) {
                for (int f = config_.num_frequencies; f < 64; f++) {
                    ctx.block_tokens[b * 64 + f] = config_.vocab_size() / 2;  // Zero after dequant
                }
            }
        }
        
        // Dequantize
        dequantize_coefficients(ctx.block_tokens.data(), ctx.dct_coeffs.data(), num_blocks);
        
        // Inverse DCT
        for (int b = 0; b < num_blocks; b++) {
            idct_8x8(ctx.dct_coeffs.data() + b * 64, ctx.blocks.data() + b * 64);
        }
        
        // Reconstruct channel from blocks
        reconstruct_from_blocks(ctx.blocks.data(), channel, target_size, target_size);
    };
    
    process_channel(tokens, ctx.y_channel.data());
    process_channel(tokens + tokens_per_channel, ctx.cb_channel.data());
    process_channel(tokens + 2 * tokens_per_channel, ctx.cr_channel.data());
    
    // Convert back to RGB
    std::vector<uint8_t> rgb(target_size * target_size * 3);
    
    if (config_.use_ycbcr) {
        ycbcr_to_rgb(ctx.y_channel.data(), ctx.cb_channel.data(), ctx.cr_channel.data(),
                     rgb.data(), target_size, target_size);
    } else {
        const int pixels = target_size * target_size;
        for (int i = 0; i < pixels; i++) {
            rgb[i * 3 + 0] = static_cast<uint8_t>(std::clamp(ctx.y_channel[i] + 128.0f, 0.0f, 255.0f));
            rgb[i * 3 + 1] = static_cast<uint8_t>(std::clamp(ctx.cb_channel[i] + 128.0f, 0.0f, 255.0f));
            rgb[i * 3 + 2] = static_cast<uint8_t>(std::clamp(ctx.cr_channel[i] + 128.0f, 0.0f, 255.0f));
        }
    }
    
    return rgb;
}

void Tokenizer::save_tokens(const std::string& path,
                            const std::vector<TokenizedImage>& images,
                            bool flat,
                            uint16_t separator) {
    std::ofstream file(path, std::ios::binary);
    if (!file) return;
    
    if (flat) {
        // Write all tokens in a flat array with separators
        for (const auto& img : images) {
            file.write(reinterpret_cast<const char*>(img.tokens.data()),
                       img.tokens.size() * sizeof(uint16_t));
            file.write(reinterpret_cast<const char*>(&separator), sizeof(uint16_t));
        }
    } else {
        // Write header
        uint32_t num_images = images.size();
        uint32_t tokens_per_image = images.empty() ? 0 : images[0].tokens.size();
        file.write(reinterpret_cast<const char*>(&num_images), sizeof(uint32_t));
        file.write(reinterpret_cast<const char*>(&tokens_per_image), sizeof(uint32_t));
        
        // Write all tokens
        for (const auto& img : images) {
            file.write(reinterpret_cast<const char*>(img.tokens.data()),
                       img.tokens.size() * sizeof(uint16_t));
        }
    }
}

std::vector<TokenizedImage> Tokenizer::load_tokens(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return {};
    
    uint32_t num_images, tokens_per_image;
    file.read(reinterpret_cast<char*>(&num_images), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&tokens_per_image), sizeof(uint32_t));
    
    std::vector<TokenizedImage> images(num_images);
    
    for (auto& img : images) {
        img.tokens.resize(tokens_per_image);
        file.read(reinterpret_cast<char*>(img.tokens.data()),
                  tokens_per_image * sizeof(uint16_t));
    }
    
    return images;
}

} // namespace dct
