#include "pisa_vae_tile_tensor.h"

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
        right >
            std::numeric_limits<std::size_t>::max() / left
    ) {
        return false;
    }
    output = left * right;
    return true;
}

bool validRegion(
    const TileRegion& region,
    int width,
    int height
) {
    return
        width > 0 &&
        height > 0 &&
        region.x >= 0 &&
        region.y >= 0 &&
        region.width > 0 &&
        region.height > 0 &&
        region.width <= width &&
        region.height <= height &&
        region.x <= width - region.width &&
        region.y <= height - region.height;
}

}  // namespace

bool copyCroppedPlanarTile(
    const float* tileValues,
    int channels,
    int tileWidth,
    int tileHeight,
    const TileRegion& localCrop,
    const TileRegion& destination,
    int outputWidth,
    int outputHeight,
    float* output,
    std::size_t outputCount,
    std::uint8_t* coverage,
    std::size_t coverageCount
) {
    if (
        tileValues == nullptr ||
        output == nullptr ||
        coverage == nullptr ||
        channels <= 0 ||
        !validRegion(
            localCrop,
            tileWidth,
            tileHeight
        ) ||
        !validRegion(
            destination,
            outputWidth,
            outputHeight
        ) ||
        localCrop.width != destination.width ||
        localCrop.height != destination.height
    ) {
        return false;
    }

    std::size_t tilePixels = 0;
    std::size_t outputPixels = 0;
    std::size_t requiredOutput = 0;
    if (
        !checkedMultiply(
            static_cast<std::size_t>(tileWidth),
            static_cast<std::size_t>(tileHeight),
            tilePixels
        ) ||
        !checkedMultiply(
            static_cast<std::size_t>(outputWidth),
            static_cast<std::size_t>(outputHeight),
            outputPixels
        ) ||
        !checkedMultiply(
            static_cast<std::size_t>(channels),
            outputPixels,
            requiredOutput
        ) ||
        outputCount < requiredOutput ||
        coverageCount < outputPixels
    ) {
        return false;
    }

    const std::size_t tileWidthSize =
        static_cast<std::size_t>(tileWidth);
    const std::size_t outputWidthSize =
        static_cast<std::size_t>(outputWidth);

    for (int y = 0; y < destination.height; ++y) {
        for (int x = 0; x < destination.width; ++x) {
            const std::size_t tilePixel =
                static_cast<std::size_t>(
                    localCrop.y + y
                ) * tileWidthSize +
                static_cast<std::size_t>(
                    localCrop.x + x
                );
            const std::size_t outputPixel =
                static_cast<std::size_t>(
                    destination.y + y
                ) * outputWidthSize +
                static_cast<std::size_t>(
                    destination.x + x
                );
            if (coverage[outputPixel] != 0U) {
                return false;
            }

            for (int channel = 0; channel < channels; ++channel) {
                const std::size_t tileIndex =
                    static_cast<std::size_t>(channel) *
                        tilePixels +
                    tilePixel;
                if (!std::isfinite(tileValues[tileIndex])) {
                    return false;
                }
            }
        }
    }

    for (int y = 0; y < destination.height; ++y) {
        for (int x = 0; x < destination.width; ++x) {
            const std::size_t tilePixel =
                static_cast<std::size_t>(
                    localCrop.y + y
                ) * tileWidthSize +
                static_cast<std::size_t>(
                    localCrop.x + x
                );
            const std::size_t outputPixel =
                static_cast<std::size_t>(
                    destination.y + y
                ) * outputWidthSize +
                static_cast<std::size_t>(
                    destination.x + x
                );

            for (int channel = 0; channel < channels; ++channel) {
                const std::size_t tileIndex =
                    static_cast<std::size_t>(channel) *
                        tilePixels +
                    tilePixel;
                const std::size_t outputIndex =
                    static_cast<std::size_t>(channel) *
                        outputPixels +
                    outputPixel;
                output[outputIndex] = tileValues[tileIndex];
            }
            coverage[outputPixel] = 1U;
        }
    }

    return true;
}

bool validateExactCoverage(
    const std::uint8_t* coverage,
    std::size_t coverageCount
) {
    if (coverage == nullptr || coverageCount == 0) {
        return false;
    }

    for (std::size_t index = 0; index < coverageCount; ++index) {
        if (coverage[index] != 1U) {
            return false;
        }
    }
    return true;
}

}  // namespace kira::pisa
