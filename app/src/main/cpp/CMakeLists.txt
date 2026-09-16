cmake_minimum_required(VERSION 3.22.1)

project(kiraenhance_native LANGUAGES CXX)

if(NOT DEFINED ncnn_DIR OR ncnn_DIR STREQUAL "")
    message(FATAL_ERROR "ncnn_DIR must point to the ABI-specific ncnn CMake package")
endif()

find_package(ncnn REQUIRED CONFIG)
find_library(log-lib log)

add_library(
    kiraenhance
    SHARED
    kira_native.cpp
)

target_compile_features(kiraenhance PRIVATE cxx_std_17)

target_link_libraries(
    kiraenhance
    PRIVATE
    ncnn
    ${log-lib}
)
