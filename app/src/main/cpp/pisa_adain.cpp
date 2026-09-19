#include "pisa_adain.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace kira::pisa {
namespace {

bool channelMeanStd(
    const float* values,
    std::size_t count,
    float epsilon,
    double& mean,
    double& stddev
) {
    if (
        values == nullptr ||
        count < 2 ||
        !std::isfinite(epsilon) ||
        epsilon <= 0.0f
    ) {
        return false;
    }

    double sum = 0.0;
    for (std::size_t index = 0; index < count; ++index) {
        const double value = static_cast<double>(values[index]);
        if (!std::isfinite(value)) {
            return false;
        }
        sum += value;
    }
    mean = sum / static_cast<double>(count);

    double squaredError = 0.0;
    for (std::size_t index = 0; index < count; ++index) {
        const double delta =
            static_cast<double>(values[index]) - mean;
        squaredError += delta * delta;
    }

    // torch.var(dim=...) uses Bessel's correction (correction=1)
    // by default. PiSA-SR's reference AdaIN helper relies on that.
    const double variance =
        squaredError / static_cast<double>(count - 1);
    stddev = std::sqrt(
        variance + static_cast<double>(epsilon)
    );
    return std::isfinite(stddev) && stddev > 0.0;
}

}  // namespace

bool adainColorFixRgbPlanar(
    const float* target,
    std::size_t targetPixels,
    const float* source,
    std::size_t sourcePixels,
    float* output,
    float epsilon
) {
    if (
        target == nullptr ||
        source == nullptr ||
        output == nullptr ||
        targetPixels < 2 ||
        sourcePixels < 2
    ) {
        return false;
    }

    for (std::size_t channel = 0; channel < 3; ++channel) {
        const float* targetChannel =
            target + channel * targetPixels;
        const float* sourceChannel =
            source + channel * sourcePixels;
        float* outputChannel =
            output + channel * targetPixels;

        double targetMean = 0.0;
        double targetStd = 0.0;
        double sourceMean = 0.0;
        double sourceStd = 0.0;
        if (
            !channelMeanStd(
                targetChannel,
                targetPixels,
                epsilon,
                targetMean,
                targetStd
            ) ||
            !channelMeanStd(
                sourceChannel,
                sourcePixels,
                epsilon,
                sourceMean,
                sourceStd
            )
        ) {
            return false;
        }

        for (
            std::size_t index = 0;
            index < targetPixels;
            ++index
        ) {
            const double normalized =
                (
                    static_cast<double>(
                        targetChannel[index]
                    ) - targetMean
                ) / targetStd;
            const double aligned =
                normalized * sourceStd + sourceMean;
            if (!std::isfinite(aligned)) {
                return false;
            }
            outputChannel[index] = static_cast<float>(
                std::clamp(aligned, 0.0, 1.0)
            );
        }
    }

    return true;
}

}  // namespace kira::pisa
