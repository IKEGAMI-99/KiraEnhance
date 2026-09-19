#include "pisa_latent_math.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace kira::pisa {
namespace {

bool checkedMultiply(
    std::size_t left,
    std::size_t right,
    std::size_t& output
) {
    if (
        left != 0 &&
        right > std::numeric_limits<std::size_t>::max() / left
    ) {
        return false;
    }
    output = left * right;
    return true;
}

}  // namespace

bool sampleLatentFromMoments(
    const float* moments,
    const float* noise,
    float* output,
    int batch,
    int latentChannels,
    int height,
    int width,
    float scalingFactor
) {
    if (
        moments == nullptr ||
        noise == nullptr ||
        output == nullptr ||
        batch <= 0 ||
        latentChannels <= 0 ||
        height <= 0 ||
        width <= 0 ||
        !std::isfinite(scalingFactor) ||
        scalingFactor <= 0.0f
    ) {
        return false;
    }

    std::size_t plane = 0;
    std::size_t latentPerBatch = 0;
    std::size_t momentsPerBatch = 0;
    std::size_t totalLatents = 0;

    if (
        !checkedMultiply(
            static_cast<std::size_t>(height),
            static_cast<std::size_t>(width),
            plane
        ) ||
        !checkedMultiply(
            static_cast<std::size_t>(latentChannels),
            plane,
            latentPerBatch
        ) ||
        !checkedMultiply(
            latentPerBatch,
            static_cast<std::size_t>(2),
            momentsPerBatch
        ) ||
        !checkedMultiply(
            static_cast<std::size_t>(batch),
            latentPerBatch,
            totalLatents
        )
    ) {
        return false;
    }

    for (int batchIndex = 0; batchIndex < batch; ++batchIndex) {
        const auto batchOffset =
            static_cast<std::size_t>(batchIndex) * momentsPerBatch;
        const auto outputOffset =
            static_cast<std::size_t>(batchIndex) * latentPerBatch;

        const float* mean = moments + batchOffset;
        const float* logVariance = mean + latentPerBatch;
        const float* batchNoise = noise + outputOffset;
        float* batchOutput = output + outputOffset;

        for (std::size_t index = 0; index < latentPerBatch; ++index) {
            const float clampedLogVariance = std::clamp(
                logVariance[index],
                -30.0f,
                20.0f
            );
            const float standardDeviation = std::exp(
                0.5f * clampedLogVariance
            );
            const float sample =
                mean[index] +
                standardDeviation * batchNoise[index];

            batchOutput[index] = sample * scalingFactor;
        }
    }

    return true;
}

}  // namespace kira::pisa
