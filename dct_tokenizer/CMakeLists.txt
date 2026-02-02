cmake_minimum_required(VERSION 3.16)
project(dct_tokenizer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math -funroll-loops -fomit-frame-pointer")

# Find required packages
find_package(OpenMP REQUIRED)

# Option to use libjpeg-turbo for faster JPEG decoding
option(USE_TURBOJPEG "Use libjpeg-turbo for JPEG decoding" ON)

# Main library
add_library(dct_tokenizer STATIC
    src/dct.cpp
    src/tokenizer.cpp
    src/image_loader.cpp
)

target_include_directories(dct_tokenizer PUBLIC include)
target_link_libraries(dct_tokenizer PUBLIC OpenMP::OpenMP_CXX)

if(USE_TURBOJPEG)
    find_library(TURBOJPEG_LIB turbojpeg)
    if(TURBOJPEG_LIB)
        target_compile_definitions(dct_tokenizer PUBLIC USE_TURBOJPEG)
        target_link_libraries(dct_tokenizer PUBLIC ${TURBOJPEG_LIB})
        message(STATUS "Using libjpeg-turbo: ${TURBOJPEG_LIB}")
    else()
        message(WARNING "libjpeg-turbo not found, falling back to stb_image")
    endif()
endif()

# CLI tool
add_executable(tokenize src/main.cpp)
target_link_libraries(tokenize dct_tokenizer)

# Batch processing tool
add_executable(tokenize_batch src/batch.cpp)
target_link_libraries(tokenize_batch dct_tokenizer)

# Compact tokenizer (FlexTok-comparable token counts)
add_executable(tokenize_compact src/compact_tokenizer.cpp)
target_link_libraries(tokenize_compact dct_tokenizer)

# Python bindings (optional)
option(BUILD_PYTHON "Build Python bindings" OFF)
if(BUILD_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    find_package(pybind11 CONFIG)
    if(pybind11_FOUND)
        pybind11_add_module(pydct_tokenizer src/python_bindings.cpp)
        target_link_libraries(pydct_tokenizer PRIVATE dct_tokenizer)
    endif()
endif()

# Install
install(TARGETS tokenize tokenize_batch DESTINATION bin)
install(TARGETS dct_tokenizer DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
