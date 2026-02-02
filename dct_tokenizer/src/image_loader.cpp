#include "image_loader.hpp"
#include <algorithm>
#include <filesystem>
#include <cstring>
#include <omp.h>

// Use stb_image for image loading (header-only, no dependencies)
#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include "../third_party/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../third_party/stb_image_write.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "../third_party/stb_image_resize2.h"

#ifdef USE_TURBOJPEG
#include <turbojpeg.h>
#endif

namespace fs = std::filesystem;

namespace dct {

Image load_image(const std::string& path) {
    Image img;
    
    int width, height, channels;
    uint8_t* data = stbi_load(path.c_str(), &width, &height, &channels, 3);
    
    if (!data) {
        return img;  // Return empty image on failure
    }
    
    img.width = width;
    img.height = height;
    img.channels = 3;
    img.data.assign(data, data + width * height * 3);
    
    stbi_image_free(data);
    return img;
}

Image load_image_resized(const std::string& path, int target_size) {
    Image img = load_image(path);
    
    if (img.empty() || target_size <= 0) {
        return img;
    }
    
    if (img.width == target_size && img.height == target_size) {
        return img;
    }
    
    // Resize to target_size x target_size (center crop approach)
    // First resize so smaller dimension = target_size, then crop
    
    int new_width, new_height;
    if (img.width > img.height) {
        new_height = target_size;
        new_width = (img.width * target_size) / img.height;
    } else {
        new_width = target_size;
        new_height = (img.height * target_size) / img.width;
    }
    
    // Resize
    std::vector<uint8_t> resized(new_width * new_height * 3);
    stbir_resize_uint8_linear(
        img.data.data(), img.width, img.height, img.width * 3,
        resized.data(), new_width, new_height, new_width * 3,
        STBIR_RGB
    );
    
    // Center crop to target_size x target_size
    Image result;
    result.width = target_size;
    result.height = target_size;
    result.channels = 3;
    result.data.resize(target_size * target_size * 3);
    
    int offset_x = (new_width - target_size) / 2;
    int offset_y = (new_height - target_size) / 2;
    
    for (int y = 0; y < target_size; y++) {
        const uint8_t* src_row = resized.data() + ((y + offset_y) * new_width + offset_x) * 3;
        uint8_t* dst_row = result.data.data() + y * target_size * 3;
        memcpy(dst_row, src_row, target_size * 3);
    }
    
    return result;
}

std::vector<Image> load_images_parallel(
    const std::vector<std::string>& paths,
    int target_size,
    int num_threads
) {
    if (num_threads <= 0) {
        num_threads = omp_get_max_threads();
    }
    
    std::vector<Image> images(paths.size());
    
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 16)
    for (size_t i = 0; i < paths.size(); i++) {
        if (target_size > 0) {
            images[i] = load_image_resized(paths[i], target_size);
        } else {
            images[i] = load_image(paths[i]);
        }
    }
    
    return images;
}

bool save_image(const std::string& path, const Image& image) {
    return save_image(path, image.data.data(), image.width, image.height, image.channels);
}

bool save_image(const std::string& path, const uint8_t* data, 
                int width, int height, int channels) {
    std::string ext = fs::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
    if (ext == ".png") {
        return stbi_write_png(path.c_str(), width, height, channels, 
                             data, width * channels) != 0;
    } else if (ext == ".jpg" || ext == ".jpeg") {
        return stbi_write_jpg(path.c_str(), width, height, channels, data, 95) != 0;
    } else if (ext == ".bmp") {
        return stbi_write_bmp(path.c_str(), width, height, channels, data) != 0;
    }
    
    // Default to PNG
    return stbi_write_png(path.c_str(), width, height, channels,
                         data, width * channels) != 0;
}

std::vector<std::string> list_images(const std::string& directory, bool recursive) {
    std::vector<std::string> paths;
    
    auto add_if_image = [&](const fs::path& p) {
        if (is_supported_format(p.string())) {
            paths.push_back(p.string());
        }
    };
    
    if (recursive) {
        for (const auto& entry : fs::recursive_directory_iterator(directory)) {
            if (entry.is_regular_file()) {
                add_if_image(entry.path());
            }
        }
    } else {
        for (const auto& entry : fs::directory_iterator(directory)) {
            if (entry.is_regular_file()) {
                add_if_image(entry.path());
            }
        }
    }
    
    std::sort(paths.begin(), paths.end());
    return paths;
}

bool is_supported_format(const std::string& path) {
    std::string ext = fs::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" ||
           ext == ".bmp" || ext == ".gif" || ext == ".tga" ||
           ext == ".psd" || ext == ".hdr" || ext == ".pic";
}

// Image resizing utility
void resize_image(const uint8_t* src, int src_w, int src_h,
                  uint8_t* dst, int dst_w, int dst_h, int channels) {
    stbir_pixel_layout layout = (channels == 3) ? STBIR_RGB : STBIR_RGBA;
    stbir_resize_uint8_linear(
        src, src_w, src_h, src_w * channels,
        dst, dst_w, dst_h, dst_w * channels,
        layout
    );
}

} // namespace dct
