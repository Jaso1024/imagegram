#include "tokenizer.hpp"
#include "image_loader.hpp"
#include <iostream>
#include <fstream>
#include <chrono>
#include <cstring>
#include <atomic>
#include <iomanip>
#include <omp.h>

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options] <input_dir> <output_file>\n"
              << "\nOptions:\n"
              << "  -s, --size SIZE       Image size (default: 256)\n"
              << "  -f, --frequencies N   Number of frequency bands (1-64, default: 64)\n"
              << "  -q, --quant-bits N    Quantization bits (8, 10, 12; default: 10)\n"
              << "  -t, --threads N       Number of threads (default: auto)\n"
              << "  -r, --recursive       Search directories recursively\n"
              << "  -b, --batch-size N    Batch size for processing (default: 1000)\n"
              << "  --flat                Output flat format with separators (for Fastgram)\n"
              << "  --separator N         Separator token for flat format (default: 65535)\n"
              << "  -v, --verbose         Verbose output\n"
              << "  -h, --help            Show this help\n"
              << "\nExamples:\n"
              << "  " << prog << " /data/images tokens.bin\n"
              << "  " << prog << " -r --flat /data/imagenet tokens_flat.bin\n";
}

int main(int argc, char* argv[]) {
    dct::TokenizerConfig config;
    std::string input_dir, output_path;
    bool recursive = false;
    bool flat = false;
    bool verbose = false;
    int batch_size = 1000;
    uint16_t separator = 65535;
    
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
        } else if (arg == "-r" || arg == "--recursive") {
            recursive = true;
        } else if (arg == "-b" || arg == "--batch-size") {
            if (++i >= argc) { std::cerr << "Missing batch-size argument\n"; return 1; }
            batch_size = std::stoi(argv[i]);
        } else if (arg == "--flat") {
            flat = true;
        } else if (arg == "--separator") {
            if (++i >= argc) { std::cerr << "Missing separator argument\n"; return 1; }
            separator = static_cast<uint16_t>(std::stoi(argv[i]));
        } else if (arg == "-v" || arg == "--verbose") {
            verbose = true;
        } else if (input_dir.empty()) {
            input_dir = arg;
        } else if (output_path.empty()) {
            output_path = arg;
        }
    }
    
    if (input_dir.empty() || output_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }
    
    // Discover images
    std::cout << "Scanning for images in " << input_dir << "...\n";
    auto paths = dct::list_images(input_dir, recursive);
    
    if (paths.empty()) {
        std::cerr << "No images found in " << input_dir << "\n";
        return 1;
    }
    
    std::cout << "Found " << paths.size() << " images\n";
    std::cout << "Config: size=" << config.image_size 
              << ", frequencies=" << config.num_frequencies
              << ", quant_bits=" << config.quant_bits
              << ", vocab_size=" << config.vocab_size()
              << ", tokens_per_image=" << config.tokens_per_image()
              << ", threads=" << (config.num_threads ? config.num_threads : omp_get_max_threads())
              << "\n";
    
    // Initialize tokenizer
    dct::Tokenizer tokenizer(config);
    
    // Open output file
    std::ofstream outfile(output_path, std::ios::binary);
    if (!outfile) {
        std::cerr << "Failed to open output file: " << output_path << "\n";
        return 1;
    }
    
    // Write header for non-flat format
    if (!flat) {
        uint32_t num_images = static_cast<uint32_t>(paths.size());
        uint32_t tokens_per_image = static_cast<uint32_t>(config.tokens_per_image());
        outfile.write(reinterpret_cast<const char*>(&num_images), sizeof(uint32_t));
        outfile.write(reinterpret_cast<const char*>(&tokens_per_image), sizeof(uint32_t));
    }
    
    // Process in batches
    auto total_start = std::chrono::high_resolution_clock::now();
    std::atomic<size_t> total_processed{0};
    std::atomic<size_t> total_failed{0};
    
    size_t num_batches = (paths.size() + batch_size - 1) / batch_size;
    
    for (size_t batch_idx = 0; batch_idx < num_batches; batch_idx++) {
        size_t start_idx = batch_idx * batch_size;
        size_t end_idx = std::min(start_idx + batch_size, paths.size());
        
        std::vector<std::string> batch_paths(paths.begin() + start_idx, 
                                              paths.begin() + end_idx);
        
        auto batch_start = std::chrono::high_resolution_clock::now();
        
        // Tokenize batch
        auto results = tokenizer.tokenize_batch(batch_paths);
        
        // Write results
        for (size_t i = 0; i < results.size(); i++) {
            if (results[i].tokens.empty()) {
                total_failed++;
                if (verbose) {
                    std::cerr << "Failed: " << batch_paths[i] << "\n";
                }
                continue;
            }
            
            outfile.write(reinterpret_cast<const char*>(results[i].tokens.data()),
                         results[i].tokens.size() * sizeof(uint16_t));
            
            if (flat) {
                outfile.write(reinterpret_cast<const char*>(&separator), sizeof(uint16_t));
            }
            
            total_processed++;
        }
        
        auto batch_end = std::chrono::high_resolution_clock::now();
        double batch_time = std::chrono::duration<double>(batch_end - batch_start).count();
        double imgs_per_sec = batch_paths.size() / batch_time;
        
        // Progress update
        double progress = 100.0 * (batch_idx + 1) / num_batches;
        auto total_elapsed = std::chrono::duration<double>(batch_end - total_start).count();
        double avg_rate = total_processed / total_elapsed;
        double eta = (paths.size() - end_idx) / avg_rate;
        
        std::cout << "\r[" << std::fixed << std::setprecision(1) << progress << "%] "
                  << total_processed << "/" << paths.size() << " images | "
                  << std::setprecision(0) << imgs_per_sec << " img/s (batch) | "
                  << std::setprecision(0) << avg_rate << " img/s (avg) | "
                  << "ETA: " << std::setprecision(1) << eta << "s     " << std::flush;
    }
    
    outfile.close();
    
    auto total_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(total_end - total_start).count();
    
    std::cout << "\n\n=== Summary ===\n";
    std::cout << "Processed: " << total_processed << " images\n";
    std::cout << "Failed: " << total_failed << " images\n";
    std::cout << "Total time: " << std::fixed << std::setprecision(2) << total_time << " seconds\n";
    std::cout << "Average rate: " << std::setprecision(0) << (total_processed / total_time) << " images/sec\n";
    std::cout << "Output: " << output_path << "\n";
    
    // Calculate output size
    size_t expected_size = total_processed * config.tokens_per_image() * sizeof(uint16_t);
    if (flat) {
        expected_size += total_processed * sizeof(uint16_t);  // separators
    } else {
        expected_size += 8;  // header
    }
    std::cout << "Output size: " << (expected_size / (1024.0 * 1024.0)) << " MB\n";
    
    return 0;
}
