#pragma once

#include <cstddef>
#include <cstdint>

namespace kira::pisa {

std::uint16_t floatToHalf(float value);
float halfToFloat(std::uint16_t value);

bool floatsToHalfs(
    const float* input,
    std::uint16_t* output,
    std::size_t count
);

bool halfsToFloats(
    const std::uint16_t* input,
    float* output,
    std::size_t count
);

}  // namespace kira::pisa
