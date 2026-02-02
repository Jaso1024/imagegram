#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <memory>

namespace dct {

// Simple image container
struct Image {
    std::vector<uint8_t> data;  // RGB, row-major
    int width = 0;
    int height = 0;
    int channels = 3;
    
    bool empty() const { return data.empty(); }
    size_t size() const { return data.size(); }
    uint8_t* ptr() { return data.data(); }
    const uint8_t* ptr() const { return data.data(); }
};

// Load image from file (supports JPEG, PNG, BMP, etc.)
Image load_image(const std::string& path);

// Load image and resize in one step (more efficient)
Image load_image_resized(const std::string& path, int target_size);

// Batch load with parallel I/O
std::vector<Image> load_images_parallel(
    const std::vector<std::string>& paths,
    int target_size = 0,  // 0 = no resize
    int num_threads = 0   // 0 = auto
);

// Save image to file
bool save_image(const std::string& path, const Image& image);
bool save_image(const std::string& path, const uint8_t* data, 
                int width, int height, int channels = 3);

// List image files in directory
std::vector<std::string> list_images(const std::string& directory, 
                                      bool recursive = false);

// Supported formats
bool is_supported_format(const std::string& path);

} // namespace dct
