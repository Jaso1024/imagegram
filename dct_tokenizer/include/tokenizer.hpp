#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <memory>

namespace dct {

// Tokenizer configuration
struct TokenizerConfig {
    // Image size (will be resized to this)
    int image_size = 256;
    
    // DCT block size (always 8 for now)
    int block_size = 8;
    
    // Number of frequency bands to keep (1-64)
    // Lower = fewer tokens, coarser representation
    int num_frequencies = 64;
    
    // Quantization bits per coefficient (determines vocab size)
    // 8 = 256 vocab, 10 = 1024 vocab, 12 = 4096 vocab
    int quant_bits = 10;
    
    // Whether to use YCbCr color space (better compression, like JPEG)
    bool use_ycbcr = true;
    
    // Subsample chroma channels (4:2:0 like JPEG)
    bool subsample_chroma = false;
    
    // Compact mode: use larger blocks and fewer frequencies for ~256-1024 tokens
    // block_size becomes 16 or 32, aiming for FlexTok-comparable token counts
    bool compact_mode = false;
    int compact_block_size = 16;  // 16x16 or 32x32 blocks
    
    // Number of threads for parallel processing
    int num_threads = 0;  // 0 = auto-detect
    
    // Computed values
    int vocab_size() const { return 1 << quant_bits; }
    int effective_block_size() const { return compact_mode ? compact_block_size : block_size; }
    int blocks_per_side() const { return image_size / effective_block_size(); }
    int total_blocks() const { return blocks_per_side() * blocks_per_side(); }
    int freqs_per_block() const { 
        return compact_mode ? std::min(num_frequencies, compact_block_size * compact_block_size) : num_frequencies;
    }
    int tokens_per_channel() const { return total_blocks() * freqs_per_block(); }
    int tokens_per_image() const { 
        if (subsample_chroma) {
            // Y full res, Cb/Cr quarter res
            return tokens_per_channel() + 2 * (tokens_per_channel() / 4);
        }
        return tokens_per_channel() * 3; 
    }
};

// Result of tokenization
struct TokenizedImage {
    std::vector<uint16_t> tokens;
    int width;
    int height;
    TokenizerConfig config;
};

// Main tokenizer class
class Tokenizer {
public:
    explicit Tokenizer(const TokenizerConfig& config = TokenizerConfig());
    ~Tokenizer();
    
    // Tokenize a single image
    // Input: RGB image data (row-major, 3 channels, uint8)
    TokenizedImage tokenize(const uint8_t* rgb_data, int width, int height) const;
    
    // Tokenize from file
    TokenizedImage tokenize_file(const std::string& path) const;
    
    // Batch tokenize multiple images (parallelized)
    std::vector<TokenizedImage> tokenize_batch(
        const std::vector<std::string>& paths,
        int num_threads = 0  // 0 = use config default
    ) const;
    
    // Detokenize back to image
    std::vector<uint8_t> detokenize(const TokenizedImage& tokens) const;
    std::vector<uint8_t> detokenize(const uint16_t* tokens, int num_tokens) const;
    
    // Save tokens to binary file (efficient format for Fastgram)
    static void save_tokens(const std::string& path, 
                           const std::vector<TokenizedImage>& images,
                           bool flat = true,  // flat = single array with separators
                           uint16_t separator = 65535);
    
    // Load tokens from binary file
    static std::vector<TokenizedImage> load_tokens(const std::string& path);
    
    const TokenizerConfig& config() const { return config_; }
    
private:
    TokenizerConfig config_;
    
    // Internal buffers (per-thread in parallel mode)
    struct ThreadContext;
    mutable std::vector<std::unique_ptr<ThreadContext>> thread_contexts_;
    
    ThreadContext& get_context(int thread_id) const;
    
    // Core processing functions
    void rgb_to_ycbcr(const uint8_t* rgb, float* y, float* cb, float* cr, 
                      int width, int height) const;
    void ycbcr_to_rgb(const float* y, const float* cb, const float* cr,
                      uint8_t* rgb, int width, int height) const;
    
    void extract_blocks(const float* channel, float* blocks, 
                        int width, int height) const;
    void reconstruct_from_blocks(const float* blocks, float* channel,
                                 int width, int height) const;
    
    void quantize_coefficients(const float* coeffs, uint16_t* tokens,
                               int num_blocks) const;
    void dequantize_coefficients(const uint16_t* tokens, float* coeffs,
                                 int num_blocks) const;
    
    void reorder_frequency_first(const uint16_t* block_order, uint16_t* freq_order,
                                 int num_blocks) const;
    void reorder_block_first(const uint16_t* freq_order, uint16_t* block_order,
                             int num_blocks) const;
};

// Utility: resize image using bilinear interpolation
void resize_image(const uint8_t* src, int src_w, int src_h,
                  uint8_t* dst, int dst_w, int dst_h, int channels);

} // namespace dct
