#pragma once

#include <cstddef>
#include <cstdint>

namespace kira::pisa {

bool rgba8888ToNormalizedNchw(
    const std::uint8_t* input,
    int width,
    int height,
    int rowStrideBytes,
    float* output,
    std::size_t outputFloatCount
);

bool normalizedNchwToRgba8888(
    const float* input,
    int width,
    int height,
    std::uint8_t* output,
    int rowStrideBytes,
    std::size_t outputByteCount
);

}  // namespace kira::pisa
