#pragma once

#include <cstddef>

namespace kira::pisa {

bool adainColorFixRgbPlanar(
    const float* target,
    std::size_t targetPixels,
    const float* source,
    std::size_t sourcePixels,
    float* output,
    float epsilon = 1.0e-5f
);

}  // namespace kira::pisa
