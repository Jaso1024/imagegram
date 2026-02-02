// Compact tokenizer designed for n-gram image generation
// Aims for ~256-1024 tokens per image (similar to FlexTok)
// Uses large blocks and aggressive frequency truncation

#include "tokenizer.hpp"
#include "dct.hpp"
#include "image_loader.hpp"
#include <cstring>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <iostream>

namespace dct {

// Compact configuration presets
struct CompactConfig {
    int image_size;
    int block_size;
    int num_frequencies;  // Per block
    int quant_bits;
    bool luma_only;       // Only Y channel (grayscale effectively)
    
    int blocks_per_side() const { return image_size / block_size; }
    int total_blocks() const { return blocks_per_side() * blocks_per_side(); }
    int tokens_per_image() const {
        int channels = luma_only ? 1 : 3;
        return total_blocks() * num_frequencies * channels;
    }
    int vocab_size() const { return 1 << quant_bits; }
};

// Preset configurations
namespace presets {
    // ~256 tokens - extremely coarse grayscale
    constexpr CompactConfig TINY = {256, 32, 4, 10, true};    // 64 blocks * 4 freqs = 256 tokens
    
    // ~768 tokens - coarse color (like FlexTok but coarser)
    constexpr CompactConfig SMALL = {256, 32, 4, 10, false};  // 64 blocks * 4 freqs * 3 = 768 tokens
    
    // ~3072 tokens - decent color
    constexpr CompactConfig MEDIUM = {256, 16, 4, 10, false}; // 256 blocks * 4 freqs * 3 = 3072 tokens
    
    // ~12k tokens - high detail color
    constexpr CompactConfig LARGE = {256, 8, 16, 10, false};  // 1024 * 4 * 3 = 12288 tokens
    
    // Custom: ~256 tokens with color via chroma subsampling
    // Y: 64 blocks * 4 freqs = 256
    // Cb/Cr: subsample 2x2, so 16 blocks * 2 freqs * 2 = 64
    // Total: ~320 tokens
}

class CompactTokenizer {
public:
    CompactConfig config;
    
    CompactTokenizer(const CompactConfig& cfg) : config(cfg) {
        // Precompute DCT basis for arbitrary block sizes
        precompute_dct_basis();
    }
    
    std::vector<uint16_t> tokenize(const uint8_t* rgb, int width, int height) {
        // Resize to target size
        std::vector<uint8_t> resized;
        const uint8_t* img_data = rgb;
        
        if (width != config.image_size || height != config.image_size) {
            resized.resize(config.image_size * config.image_size * 3);
            resize_image(rgb, width, height, resized.data(), 
                        config.image_size, config.image_size, 3);
            img_data = resized.data();
        }
        
        const int img_size = config.image_size;
        const int block_size = config.block_size;
        const int blocks_per_side = config.blocks_per_side();
        const int num_blocks = config.total_blocks();
        const int num_freqs = config.num_frequencies;
        
        // Convert to YCbCr (or just extract Y for luma_only)
        std::vector<float> y_channel(img_size * img_size);
        std::vector<float> cb_channel, cr_channel;
        
        if (!config.luma_only) {
            cb_channel.resize(img_size * img_size);
            cr_channel.resize(img_size * img_size);
        }
        
        #pragma omp parallel for simd
        for (int i = 0; i < img_size * img_size; i++) {
            float r = img_data[i * 3 + 0];
            float g = img_data[i * 3 + 1];
            float b = img_data[i * 3 + 2];
            
            y_channel[i] = 0.299f * r + 0.587f * g + 0.114f * b - 128.0f;
            
            if (!config.luma_only) {
                cb_channel[i] = -0.168736f * r - 0.331264f * g + 0.5f * b;
                cr_channel[i] = 0.5f * r - 0.418688f * g - 0.081312f * b;
            }
        }
        
        // Allocate output tokens (frequency-first order)
        int tokens_per_channel = num_blocks * num_freqs;
        int total_tokens = config.tokens_per_image();
        std::vector<uint16_t> tokens(total_tokens);
        
        auto process_channel = [&](const std::vector<float>& channel, uint16_t* out) {
            // Temporary storage for all block DCT coefficients
            std::vector<float> all_coeffs(num_blocks * block_size * block_size);
            
            // DCT each block
            #pragma omp parallel for collapse(2)
            for (int by = 0; by < blocks_per_side; by++) {
                for (int bx = 0; bx < blocks_per_side; bx++) {
                    int block_idx = by * blocks_per_side + bx;
                    float* block_out = all_coeffs.data() + block_idx * block_size * block_size;
                    
                    // Extract block
                    std::vector<float> block(block_size * block_size);
                    for (int y = 0; y < block_size; y++) {
                        for (int x = 0; x < block_size; x++) {
                            int px = bx * block_size + x;
                            int py = by * block_size + y;
                            block[y * block_size + x] = channel[py * img_size + px];
                        }
                    }
                    
                    // DCT
                    dct_nd(block.data(), block_out, block_size);
                }
            }
            
            // Extract frequencies in zigzag order, reorder to frequency-first
            const auto& zz = get_zigzag(block_size);
            
            for (int f = 0; f < num_freqs; f++) {
                for (int b = 0; b < num_blocks; b++) {
                    float coeff = all_coeffs[b * block_size * block_size + zz[f]];
                    
                    // Quantize
                    float scale = get_quant_scale(f, block_size);
                    float normalized = (coeff / scale) + (config.vocab_size() / 2.0f);
                    int token = static_cast<int>(std::round(normalized));
                    token = std::clamp(token, 0, config.vocab_size() - 1);
                    
                    out[f * num_blocks + b] = static_cast<uint16_t>(token);
                }
            }
        };
        
        process_channel(y_channel, tokens.data());
        
        if (!config.luma_only) {
            process_channel(cb_channel, tokens.data() + tokens_per_channel);
            process_channel(cr_channel, tokens.data() + 2 * tokens_per_channel);
        }
        
        return tokens;
    }
    
    std::vector<uint8_t> detokenize(const uint16_t* tokens) {
        const int img_size = config.image_size;
        const int block_size = config.block_size;
        const int blocks_per_side = config.blocks_per_side();
        const int num_blocks = config.total_blocks();
        const int num_freqs = config.num_frequencies;
        int tokens_per_channel = num_blocks * num_freqs;
        
        auto reconstruct_channel = [&](const uint16_t* channel_tokens) -> std::vector<float> {
            std::vector<float> channel(img_size * img_size);
            std::vector<float> all_coeffs(num_blocks * block_size * block_size, 0.0f);
            
            const auto& zz = get_zigzag(block_size);
            
            // Dequantize and place in block order
            for (int f = 0; f < num_freqs; f++) {
                for (int b = 0; b < num_blocks; b++) {
                    uint16_t token = channel_tokens[f * num_blocks + b];
                    float scale = get_quant_scale(f, block_size);
                    float coeff = (static_cast<float>(token) - config.vocab_size() / 2.0f) * scale;
                    all_coeffs[b * block_size * block_size + zz[f]] = coeff;
                }
            }
            
            // IDCT each block
            #pragma omp parallel for collapse(2)
            for (int by = 0; by < blocks_per_side; by++) {
                for (int bx = 0; bx < blocks_per_side; bx++) {
                    int block_idx = by * blocks_per_side + bx;
                    const float* block_coeffs = all_coeffs.data() + block_idx * block_size * block_size;
                    
                    std::vector<float> block(block_size * block_size);
                    idct_nd(block_coeffs, block.data(), block_size);
                    
                    // Place block in image
                    for (int y = 0; y < block_size; y++) {
                        for (int x = 0; x < block_size; x++) {
                            int px = bx * block_size + x;
                            int py = by * block_size + y;
                            channel[py * img_size + px] = block[y * block_size + x];
                        }
                    }
                }
            }
            
            return channel;
        };
        
        auto y_channel = reconstruct_channel(tokens);
        
        std::vector<uint8_t> rgb(img_size * img_size * 3);
        
        if (config.luma_only) {
            // Grayscale output
            #pragma omp parallel for simd
            for (int i = 0; i < img_size * img_size; i++) {
                uint8_t val = static_cast<uint8_t>(std::clamp(y_channel[i] + 128.0f, 0.0f, 255.0f));
                rgb[i * 3 + 0] = val;
                rgb[i * 3 + 1] = val;
                rgb[i * 3 + 2] = val;
            }
        } else {
            auto cb_channel = reconstruct_channel(tokens + tokens_per_channel);
            auto cr_channel = reconstruct_channel(tokens + 2 * tokens_per_channel);
            
            #pragma omp parallel for simd
            for (int i = 0; i < img_size * img_size; i++) {
                float y = y_channel[i] + 128.0f;
                float cb = cb_channel[i];
                float cr = cr_channel[i];
                
                float r = y + 1.402f * cr;
                float g = y - 0.344136f * cb - 0.714136f * cr;
                float b = y + 1.772f * cb;
                
                rgb[i * 3 + 0] = static_cast<uint8_t>(std::clamp(r, 0.0f, 255.0f));
                rgb[i * 3 + 1] = static_cast<uint8_t>(std::clamp(g, 0.0f, 255.0f));
                rgb[i * 3 + 2] = static_cast<uint8_t>(std::clamp(b, 0.0f, 255.0f));
            }
        }
        
        return rgb;
    }
    
private:
    std::vector<std::vector<float>> dct_basis_cache;
    std::vector<std::vector<int>> zigzag_cache;
    
    void precompute_dct_basis() {
        // Support block sizes 8, 16, 32
        for (int bs : {8, 16, 32}) {
            std::vector<float> basis(bs * bs);
            for (int k = 0; k < bs; k++) {
                float alpha = (k == 0) ? std::sqrt(1.0f / bs) : std::sqrt(2.0f / bs);
                for (int n = 0; n < bs; n++) {
                    basis[k * bs + n] = alpha * std::cos(M_PI * k * (2 * n + 1) / (2.0f * bs));
                }
            }
            dct_basis_cache.push_back(basis);
            
            // Compute zigzag for this block size
            zigzag_cache.push_back(compute_zigzag(bs));
        }
    }
    
    std::vector<int> compute_zigzag(int size) {
        std::vector<int> zz(size * size);
        int idx = 0;
        
        for (int sum = 0; sum < 2 * size - 1; sum++) {
            if (sum % 2 == 0) {
                for (int i = std::min(sum, size - 1); i >= std::max(0, sum - size + 1); i--) {
                    int j = sum - i;
                    zz[idx++] = i * size + j;
                }
            } else {
                for (int i = std::max(0, sum - size + 1); i <= std::min(sum, size - 1); i++) {
                    int j = sum - i;
                    zz[idx++] = i * size + j;
                }
            }
        }
        return zz;
    }
    
    const std::vector<float>& get_basis(int block_size) {
        if (block_size == 8) return dct_basis_cache[0];
        if (block_size == 16) return dct_basis_cache[1];
        return dct_basis_cache[2];
    }
    
    const std::vector<int>& get_zigzag(int block_size) {
        if (block_size == 8) return zigzag_cache[0];
        if (block_size == 16) return zigzag_cache[1];
        return zigzag_cache[2];
    }
    
    float get_quant_scale(int freq_idx, int block_size) {
        // Scale based on frequency - lower frequencies get larger ranges
        // This is simplified; JPEG uses separate tables
        float base_scale = 16.0f * (block_size / 8.0f);
        float freq_factor = 1.0f + freq_idx * 0.5f;
        return base_scale * freq_factor;
    }
    
    void dct_nd(const float* input, float* output, int size) {
        const auto& basis = get_basis(size);
        std::vector<float> temp(size * size);
        
        // Row DCT
        for (int row = 0; row < size; row++) {
            for (int k = 0; k < size; k++) {
                float sum = 0;
                for (int n = 0; n < size; n++) {
                    sum += input[row * size + n] * basis[k * size + n];
                }
                temp[row * size + k] = sum;
            }
        }
        
        // Column DCT
        for (int col = 0; col < size; col++) {
            for (int k = 0; k < size; k++) {
                float sum = 0;
                for (int n = 0; n < size; n++) {
                    sum += temp[n * size + col] * basis[k * size + n];
                }
                output[k * size + col] = sum;
            }
        }
    }
    
    void idct_nd(const float* input, float* output, int size) {
        const auto& basis = get_basis(size);
        std::vector<float> temp(size * size);
        
        // Column IDCT
        for (int col = 0; col < size; col++) {
            for (int n = 0; n < size; n++) {
                float sum = 0;
                for (int k = 0; k < size; k++) {
                    sum += input[k * size + col] * basis[k * size + n];
                }
                temp[n * size + col] = sum;
            }
        }
        
        // Row IDCT
        for (int row = 0; row < size; row++) {
            for (int n = 0; n < size; n++) {
                float sum = 0;
                for (int k = 0; k < size; k++) {
                    sum += temp[row * size + k] * basis[k * size + n];
                }
                output[row * size + n] = sum;
            }
        }
    }
};

} // namespace dct

// Standalone tool for compact tokenization
int main(int argc, char* argv[]) {
    using namespace dct;
    
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <preset> <input> <output>\n"
                  << "       " << argv[0] << " --test <preset> <image>   (test roundtrip)\n"
                  << "\nPresets:\n"
                  << "  tiny   - 256 tokens/image (grayscale)\n"
                  << "  small  - 768 tokens/image (color)\n"
                  << "  medium - 3072 tokens/image (color)\n"
                  << "  large  - 12288 tokens/image (color)\n";
        return 1;
    }
    
    std::string arg1 = argv[1];
    
    // Test mode: roundtrip a single image
    if (arg1 == "--test") {
        if (argc < 4) {
            std::cerr << "Usage: " << argv[0] << " --test <preset> <image>\n";
            return 1;
        }
        
        std::string preset_name = argv[2];
        std::string image_path = argv[3];
        
        CompactConfig config;
        if (preset_name == "tiny") config = presets::TINY;
        else if (preset_name == "small") config = presets::SMALL;
        else if (preset_name == "medium") config = presets::MEDIUM;
        else if (preset_name == "large") config = presets::LARGE;
        else {
            std::cerr << "Unknown preset: " << preset_name << "\n";
            return 1;
        }
        
        std::cout << "Testing roundtrip with " << preset_name << " preset\n";
        std::cout << "Tokens per image: " << config.tokens_per_image() << "\n";
        
        CompactTokenizer tokenizer(config);
        
        auto img = load_image_resized(image_path, config.image_size);
        if (img.empty()) {
            std::cerr << "Failed to load image\n";
            return 1;
        }
        
        auto tokens = tokenizer.tokenize(img.ptr(), img.width, img.height);
        std::cout << "Tokenized: " << tokens.size() << " tokens\n";
        
        // Print first 20 tokens
        std::cout << "First 20 tokens: ";
        for (int i = 0; i < std::min(20, (int)tokens.size()); i++) {
            std::cout << tokens[i] << " ";
        }
        std::cout << "\n";
        
        auto reconstructed = tokenizer.detokenize(tokens.data());
        
        // Save output
        std::string out_path = image_path.substr(0, image_path.find_last_of('.')) 
                              + "_" + preset_name + "_recon.png";
        save_image(out_path, reconstructed.data(), config.image_size, config.image_size, 3);
        std::cout << "Saved: " << out_path << "\n";
        
        return 0;
    }
    
    // Batch mode
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <preset> <input_dir> <output_file>\n";
        return 1;
    }
    
    std::string preset_name = argv[1];
    std::string input_dir = argv[2];
    std::string output_file = argv[3];
    
    CompactConfig config;
    if (preset_name == "tiny") config = presets::TINY;
    else if (preset_name == "small") config = presets::SMALL;
    else if (preset_name == "medium") config = presets::MEDIUM;
    else if (preset_name == "large") config = presets::LARGE;
    else {
        std::cerr << "Unknown preset: " << preset_name << "\n";
        return 1;
    }
    
    std::cout << "Config: block_size=" << config.block_size
              << ", frequencies=" << config.num_frequencies
              << ", tokens/image=" << config.tokens_per_image()
              << ", vocab=" << config.vocab_size()
              << ", luma_only=" << config.luma_only << "\n";
    
    CompactTokenizer tokenizer(config);
    
    auto paths = list_images(input_dir, true);
    std::cout << "Found " << paths.size() << " images\n";
    
    std::ofstream out(output_file, std::ios::binary);
    uint16_t separator = 65535;
    
    auto start = std::chrono::high_resolution_clock::now();
    size_t processed = 0;
    
    #pragma omp parallel for schedule(dynamic, 16)
    for (size_t i = 0; i < paths.size(); i++) {
        auto img = load_image_resized(paths[i], config.image_size);
        if (img.empty()) continue;
        
        auto tokens = tokenizer.tokenize(img.ptr(), img.width, img.height);
        
        #pragma omp critical
        {
            out.write(reinterpret_cast<const char*>(tokens.data()), 
                     tokens.size() * sizeof(uint16_t));
            out.write(reinterpret_cast<const char*>(&separator), sizeof(uint16_t));
            processed++;
            
            if (processed % 1000 == 0) {
                auto now = std::chrono::high_resolution_clock::now();
                double elapsed = std::chrono::duration<double>(now - start).count();
                std::cout << "\r" << processed << "/" << paths.size() 
                          << " (" << int(processed / elapsed) << " img/s)" << std::flush;
            }
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double total = std::chrono::duration<double>(end - start).count();
    
    std::cout << "\n\nProcessed " << processed << " images in " << total << "s\n";
    std::cout << "Rate: " << int(processed / total) << " img/s\n";
    
    return 0;
}
