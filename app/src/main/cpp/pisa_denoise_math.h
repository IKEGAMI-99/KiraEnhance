#pragma once

#include <cstddef>

namespace kira::pisa {

bool buildDecoderLatent(
    const float* encodedControl,
    const float* modelPrediction,
    float* output,
    std::size_t count,
    float scalingFactor
);

}  // namespace kira::pisa
