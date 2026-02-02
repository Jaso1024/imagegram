// Fast parallel suffix array index builder using libdivsufsort
// Compile: g++ -O3 -march=native -fopenmp -o fast_build_index fast_build_index.cpp -ldivsufsort -ldivsufsort64

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <filesystem>
#include <algorithm>
#include <omp.h>

extern "C" {
#include <divsufsort.h>
#include <divsufsort64.h>
}

namespace fs = std::filesystem;

std::vector<uint8_t> read_file(const fs::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    f.seekg(0, std::ios::end);
    size_t size = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> data(size);
    f.read(reinterpret_cast<char*>(data.data()), size);
    return data;
}

bool write_file(const fs::path& path, const void* data, size_t size) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write(reinterpret_cast<const char*>(data), size);
    return true;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <input_dir> <output_dir> <token_width>\n";
        std::cerr << "  token_width: 1, 2, or 4\n";
        return 1;
    }
    
    fs::path input_dir = argv[1];
    fs::path output_dir = argv[2];
    int token_width = std::stoi(argv[3]);
    
    if (token_width != 1 && token_width != 2 && token_width != 4) {
        std::cerr << "token_width must be 1, 2, or 4\n";
        return 1;
    }
    
    int num_threads = omp_get_max_threads();
    std::cout << "Using " << num_threads << " threads\n";
    
    // Read tokenized data
    auto t0 = std::chrono::steady_clock::now();
    
    fs::path token_path = input_dir / "tokenized.0";
    auto data = read_file(token_path);
    if (data.empty()) {
        std::cerr << "Failed to read " << token_path << "\n";
        return 1;
    }
    
    size_t num_tokens = data.size() / token_width;
    std::cout << "Loaded " << num_tokens << " tokens (" << data.size() / 1e9 << " GB)\n";
    
    auto t1 = std::chrono::steady_clock::now();
    double read_time = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "Read time: " << read_time << "s\n";
    
    // For suffix array, we need to treat the data as a byte sequence
    // libdivsufsort works on uint8_t array
    // For token_width > 1, we need to handle endianness
    
    std::cout << "Building suffix array...\n";
    auto t2 = std::chrono::steady_clock::now();
    
    // Use 64-bit suffix array for large inputs
    std::vector<int64_t> sa(data.size());
    
    // libdivsufsort64 for large arrays
    int result = divsufsort64(data.data(), sa.data(), data.size());
    
    if (result != 0) {
        std::cerr << "Suffix array construction failed\n";
        return 1;
    }
    
    auto t3 = std::chrono::steady_clock::now();
    double sa_time = std::chrono::duration<double>(t3 - t2).count();
    std::cout << "Suffix array time: " << sa_time << "s (" << data.size() / sa_time / 1e6 << " M elem/s)\n";
    
    // Filter to only keep positions that are token-aligned
    std::cout << "Filtering to token boundaries...\n";
    auto t4 = std::chrono::steady_clock::now();
    
    std::vector<int64_t> sa_filtered;
    sa_filtered.reserve(num_tokens);
    
    for (size_t i = 0; i < sa.size(); i++) {
        if (sa[i] % token_width == 0) {
            sa_filtered.push_back(sa[i]);
        }
    }
    
    auto t5 = std::chrono::steady_clock::now();
    double filter_time = std::chrono::duration<double>(t5 - t4).count();
    std::cout << "Filter time: " << filter_time << "s, kept " << sa_filtered.size() << " positions\n";
    
    // Encode table (convert suffix array to byte pointers)
    std::cout << "Encoding table...\n";
    auto t6 = std::chrono::steady_clock::now();
    
    // Determine pointer size needed
    int64_t max_ptr = sa_filtered.empty() ? 0 : *std::max_element(sa_filtered.begin(), sa_filtered.end());
    int ptr_size = 1;
    while ((max_ptr >> (ptr_size * 8)) != 0) ptr_size++;
    
    std::cout << "Pointer size: " << ptr_size << " bytes\n";
    
    std::vector<uint8_t> table(sa_filtered.size() * ptr_size);
    
    #pragma omp parallel for
    for (size_t i = 0; i < sa_filtered.size(); i++) {
        int64_t ptr = sa_filtered[i];
        for (int b = 0; b < ptr_size; b++) {
            table[i * ptr_size + b] = static_cast<uint8_t>((ptr >> (8 * b)) & 0xFF);
        }
    }
    
    auto t7 = std::chrono::steady_clock::now();
    double encode_time = std::chrono::duration<double>(t7 - t6).count();
    std::cout << "Encode time: " << encode_time << "s\n";
    
    // Write output
    fs::create_directories(output_dir);
    
    std::cout << "Writing table (" << table.size() / 1e9 << " GB)...\n";
    auto t8 = std::chrono::steady_clock::now();
    
    if (!write_file(output_dir / "table.0", table.data(), table.size())) {
        std::cerr << "Failed to write table\n";
        return 1;
    }
    
    // Copy tokenized data
    fs::copy_file(token_path, output_dir / "tokenized.0", fs::copy_options::overwrite_existing);
    
    auto t9 = std::chrono::steady_clock::now();
    double write_time = std::chrono::duration<double>(t9 - t8).count();
    
    double total_time = std::chrono::duration<double>(t9 - t0).count();
    
    std::cout << "\n=== Summary ===\n";
    std::cout << "Tokens: " << num_tokens << "\n";
    std::cout << "Table entries: " << sa_filtered.size() << "\n";
    std::cout << "Total time: " << total_time << "s\n";
    std::cout << "  Read: " << read_time << "s\n";
    std::cout << "  Suffix array: " << sa_time << "s\n";
    std::cout << "  Filter: " << filter_time << "s\n";
    std::cout << "  Encode: " << encode_time << "s\n";
    std::cout << "  Write: " << write_time << "s\n";
    
    return 0;
}
