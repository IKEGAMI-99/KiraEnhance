#pragma once

#include <cstddef>
#include <cstdint>

namespace kira::pisa {

enum class PillowResizeFilter {
    BICUBIC,
    LANCZOS,
};

bool resizeRgba8888PillowRgb(
    const std::uint8_t* input,
    int inputWidth,
    int inputHeight,
    int inputRowStrideBytes,
    std::uint8_t* output,
    int outputWidth,
    int outputHeight,
    int outputRowStrideBytes,
    std::size_t outputByteCount,
    PillowResizeFilter filter
);

}  // namespace kira::pisa
