cmake_minimum_required(VERSION 3.22.1)

project(kiraenhance_native LANGUAGES CXX)

if(NOT DEFINED ncnn_DIR OR ncnn_DIR STREQUAL "")
    message(FATAL_ERROR "ncnn_DIR must point to the ABI-specific ncnn CMake package")
endif()

find_package(ncnn REQUIRED CONFIG)
find_library(log-lib log)

set(KIRA_HAS_MNN 0)
set(MNN_CONFIGURED 0)
if(DEFINED MNN_INCLUDE_DIR AND NOT MNN_INCLUDE_DIR STREQUAL "")
    if(NOT DEFINED MNN_LIB_DIR OR MNN_LIB_DIR STREQUAL "")
        message(FATAL_ERROR "MNN_LIB_DIR must be set when MNN_INCLUDE_DIR is set")
    endif()
    set(MNN_CONFIGURED 1)
elseif(DEFINED MNN_LIB_DIR AND NOT MNN_LIB_DIR STREQUAL "")
    message(FATAL_ERROR "MNN_INCLUDE_DIR must be set when MNN_LIB_DIR is set")
endif()

if(MNN_CONFIGURED)
    if(NOT EXISTS "${MNN_INCLUDE_DIR}/MNN/Interpreter.hpp")
        message(FATAL_ERROR "MNN_INCLUDE_DIR does not contain MNN/Interpreter.hpp")
    endif()
    if(NOT EXISTS "${MNN_LIB_DIR}/libMNN.so")
        message(FATAL_ERROR "MNN_LIB_DIR does not contain libMNN.so")
    endif()

    add_library(mnn_runtime SHARED IMPORTED)
    set_target_properties(
        mnn_runtime
        PROPERTIES
        IMPORTED_LOCATION "${MNN_LIB_DIR}/libMNN.so"
        INTERFACE_INCLUDE_DIRECTORIES "${MNN_INCLUDE_DIR}"
    )
    set(KIRA_HAS_MNN 1)
endif()

add_library(
    kiraenhance
    SHARED
    kira_native.cpp
    mnn_runtime_probe.cpp
    mnn_pisa_loader.cpp
    mnn_pisa_tensor_io.cpp
    pisa_latent_math.cpp
    pisa_gaussian_noise.cpp
    pisa_image_tensor.cpp
    pisa_pillow_resize.cpp
    pisa_preprocess.cpp
    pisa_resize_plan.cpp
    pisa_tile_plan.cpp
    pisa_tile_tensor.cpp
    pisa_tiled_inference.cpp
    pisa_vae_tile_plan.cpp
    pisa_vae_tile_tensor.cpp
    pisa_group_norm_stats.cpp
    pisa_adain.cpp
    pisa_denoise_math.cpp
    pisa_half.cpp
)

target_compile_features(kiraenhance PRIVATE cxx_std_17)
target_compile_definitions(kiraenhance PRIVATE KIRA_HAS_MNN=${KIRA_HAS_MNN})

target_link_libraries(
    kiraenhance
    PRIVATE
    ncnn
    ${log-lib}
)

if(KIRA_HAS_MNN)
    target_link_libraries(kiraenhance PRIVATE mnn_runtime)
endif()
