#include "pisa_denoise_math.h"

#include <cmath>
#include <cstddef>

namespace kira::pisa {

bool buildDecoderLatent(
    const float* encodedControl,
    const float* modelPrediction,
    float* output,
    std::size_t count,
    float scalingFactor
) {
    if (
        encodedControl == nullptr ||
        modelPrediction == nullptr ||
        output == nullptr ||
        count == 0 ||
        !std::isfinite(scalingFactor) ||
        scalingFactor <= 0.0f
    ) {
        return false;
    }

    const float inverseScale = 1.0f / scalingFactor;
    for (std::size_t index = 0; index < count; ++index) {
        output[index] =
            (encodedControl[index] - modelPrediction[index]) *
            inverseScale;
    }
    return true;
}

}  // namespace kira::pisa
