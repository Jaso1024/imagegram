#include "tokenizer.hpp"
#include "image_loader.hpp"
#include <iostream>
#include <chrono>
#include <cstring>

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options] <input> <output>\n"
              << "\nOptions:\n"
              << "  -s, --size SIZE       Image size (default: 256)\n"
              << "  -f, --frequencies N   Number of frequency bands to keep (1-64, default: 64)\n"
              << "  -q, --quant-bits N    Quantization bits (8, 10, 12; default: 10)\n"
              << "  -t, --threads N       Number of threads (default: auto)\n"
              << "  -d, --detokenize      Detokenize instead of tokenize\n"
              << "  -h, --help            Show this help\n"
              << "\nExamples:\n"
              << "  " << prog << " image.jpg tokens.bin\n"
              << "  " << prog << " -d tokens.bin reconstructed.png\n";
}

int main(int argc, char* argv[]) {
    dct::TokenizerConfig config;
    std::string input_path, output_path;
    bool detokenize = false;
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "-s" || arg == "--size") {
            if (++i >= argc) { std::cerr << "Missing size argument\n"; return 1; }
            config.image_size = std::stoi(argv[i]);
        } else if (arg == "-f" || arg == "--frequencies") {
            if (++i >= argc) { std::cerr << "Missing frequencies argument\n"; return 1; }
            config.num_frequencies = std::stoi(argv[i]);
        } else if (arg == "-q" || arg == "--quant-bits") {
            if (++i >= argc) { std::cerr << "Missing quant-bits argument\n"; return 1; }
            config.quant_bits = std::stoi(argv[i]);
        } else if (arg == "-t" || arg == "--threads") {
            if (++i >= argc) { std::cerr << "Missing threads argument\n"; return 1; }
            config.num_threads = std::stoi(argv[i]);
        } else if (arg == "-d" || arg == "--detokenize") {
            detokenize = true;
        } else if (input_path.empty()) {
            input_path = arg;
        } else if (output_path.empty()) {
            output_path = arg;
        }
    }
    
    if (input_path.empty() || output_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }
    
    dct::Tokenizer tokenizer(config);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    if (detokenize) {
        // Load tokens and convert back to image
        auto images = dct::Tokenizer::load_tokens(input_path);
        if (images.empty()) {
            std::cerr << "Failed to load tokens from " << input_path << "\n";
            return 1;
        }
        
        auto rgb = tokenizer.detokenize(images[0]);
        if (!dct::save_image(output_path, rgb.data(), config.image_size, config.image_size)) {
            std::cerr << "Failed to save image to " << output_path << "\n";
            return 1;
        }
    } else {
        // Tokenize image
        auto result = tokenizer.tokenize_file(input_path);
        if (result.tokens.empty()) {
            std::cerr << "Failed to tokenize " << input_path << "\n";
            return 1;
        }
        
        std::vector<dct::TokenizedImage> images = {result};
        dct::Tokenizer::save_tokens(output_path, images, false);
        
        std::cout << "Tokens: " << result.tokens.size() << "\n";
        std::cout << "Vocab size: " << config.vocab_size() << "\n";
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    
    std::cout << "Time: " << elapsed << " ms\n";
    
    return 0;
}
