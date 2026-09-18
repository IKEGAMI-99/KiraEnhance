#include "pisa_tile_tensor.h"

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

bool validateTile(
    int imageWidth,
    int imageHeight,
    const TileRegion& tile
) {
    return imageWidth > 0 &&
        imageHeight > 0 &&
        tile.x >= 0 &&
        tile.y >= 0 &&
        tile.width > 0 &&
        tile.height > 0 &&
        tile.width <= imageWidth &&
        tile.height <= imageHeight &&
        tile.x <= imageWidth - tile.width &&
        tile.y <= imageHeight - tile.height;
}

bool planarCounts(
    int channels,
    int imageWidth,
    int imageHeight,
    const TileRegion& tile,
    std::size_t& imagePixels,
    std::size_t& tilePixels,
    std::size_t& imageValues,
    std::size_t& tileValues
) {
    if (
        channels <= 0 ||
        !validateTile(imageWidth, imageHeight, tile)
    ) {
        return false;
    }

    if (
        !checkedMultiply(
            static_cast<std::size_t>(imageWidth),
            static_cast<std::size_t>(imageHeight),
            imagePixels
        ) ||
        !checkedMultiply(
            static_cast<std::size_t>(tile.width),
            static_cast<std::size_t>(tile.height),
            tilePixels
        ) ||
        !checkedMultiply(
            static_cast<std::size_t>(channels),
            imagePixels,
            imageValues
        ) ||
        !checkedMultiply(
            static_cast<std::size_t>(channels),
            tilePixels,
            tileValues
        )
    ) {
        return false;
    }

    return true;
}

}  // namespace

bool extractPlanarTile(
    const float* input,
    int channels,
    int imageWidth,
    int imageHeight,
    const TileRegion& tile,
    float* output,
    std::size_t outputCount
) {
    if (input == nullptr || output == nullptr) {
        return false;
    }

    std::size_t imagePixels = 0;
    std::size_t tilePixels = 0;
    std::size_t imageValues = 0;
    std::size_t requiredValues = 0;
    if (
        !planarCounts(
            channels,
            imageWidth,
            imageHeight,
            tile,
            imagePixels,
            tilePixels,
            imageValues,
            requiredValues
        ) ||
        outputCount < requiredValues
    ) {
        return false;
    }
    (void)imageValues;

    const std::size_t imageWidthSize =
        static_cast<std::size_t>(imageWidth);
    const std::size_t tileWidthSize =
        static_cast<std::size_t>(tile.width);

    for (int channel = 0; channel < channels; ++channel) {
        const std::size_t inputChannelOffset =
            static_cast<std::size_t>(channel) *
            imagePixels;
        const std::size_t outputChannelOffset =
            static_cast<std::size_t>(channel) *
            tilePixels;

        for (int y = 0; y < tile.height; ++y) {
            const std::size_t inputRow =
                static_cast<std::size_t>(tile.y + y) *
                    imageWidthSize +
                static_cast<std::size_t>(tile.x);
            const std::size_t outputRow =
                static_cast<std::size_t>(y) *
                tileWidthSize;

            for (int x = 0; x < tile.width; ++x) {
                output[
                    outputChannelOffset +
                    outputRow +
                    static_cast<std::size_t>(x)
                ] = input[
                    inputChannelOffset +
                    inputRow +
                    static_cast<std::size_t>(x)
                ];
            }
        }
    }

    return true;
}

bool accumulateWeightedPlanarTile(
    const float* tileValues,
    int channels,
    const TileRegion& tile,
    int imageWidth,
    int imageHeight,
    const float* weights,
    std::size_t weightCount,
    float* accumulator,
    std::size_t accumulatorCount,
    float* contributors,
    std::size_t contributorCount
) {
    if (
        tileValues == nullptr ||
        weights == nullptr ||
        accumulator == nullptr ||
        contributors == nullptr
    ) {
        return false;
    }

    std::size_t imagePixels = 0;
    std::size_t tilePixels = 0;
    std::size_t requiredAccumulator = 0;
    std::size_t requiredTileValues = 0;
    if (
        !planarCounts(
            channels,
            imageWidth,
            imageHeight,
            tile,
            imagePixels,
            tilePixels,
            requiredAccumulator,
            requiredTileValues
        ) ||
        weightCount < tilePixels ||
        accumulatorCount < requiredAccumulator ||
        contributorCount < imagePixels
    ) {
        return false;
    }

    const std::size_t imageWidthSize =
        static_cast<std::size_t>(imageWidth);
    const std::size_t tileWidthSize =
        static_cast<std::size_t>(tile.width);

    for (int y = 0; y < tile.height; ++y) {
        for (int x = 0; x < tile.width; ++x) {
            const std::size_t localIndex =
                static_cast<std::size_t>(y) *
                    tileWidthSize +
                static_cast<std::size_t>(x);
            const std::size_t imageIndex =
                static_cast<std::size_t>(tile.y + y) *
                    imageWidthSize +
                static_cast<std::size_t>(tile.x + x);
            const float weight = weights[localIndex];
            if (!std::isfinite(weight) || weight <= 0.0f) {
                return false;
            }

            const float nextContributor =
                contributors[imageIndex] + weight;
            if (!std::isfinite(nextContributor)) {
                return false;
            }
            contributors[imageIndex] = nextContributor;

            for (
                int channel = 0;
                channel < channels;
                ++channel
            ) {
                const std::size_t tileIndex =
                    static_cast<std::size_t>(channel) *
                        tilePixels +
                    localIndex;
                const std::size_t accumulatorIndex =
                    static_cast<std::size_t>(channel) *
                        imagePixels +
                    imageIndex;
                const float value = tileValues[tileIndex];
                if (!std::isfinite(value)) {
                    return false;
                }

                const float next =
                    accumulator[accumulatorIndex] +
                    value * weight;
                if (!std::isfinite(next)) {
                    return false;
                }
                accumulator[accumulatorIndex] = next;
            }
        }
    }

    return true;
}

bool normalizePlanarAccumulation(
    float* accumulator,
    int channels,
    int imageWidth,
    int imageHeight,
    const float* contributors,
    std::size_t contributorCount
) {
    if (
        accumulator == nullptr ||
        contributors == nullptr ||
        channels <= 0 ||
        imageWidth <= 0 ||
        imageHeight <= 0
    ) {
        return false;
    }

    std::size_t imagePixels = 0;
    std::size_t accumulatorCount = 0;
    if (
        !checkedMultiply(
            static_cast<std::size_t>(imageWidth),
            static_cast<std::size_t>(imageHeight),
            imagePixels
        ) ||
        !checkedMultiply(
            static_cast<std::size_t>(channels),
            imagePixels,
            accumulatorCount
        ) ||
        contributorCount < imagePixels
    ) {
        return false;
    }

    for (
        std::size_t index = 0;
        index < imagePixels;
        ++index
    ) {
        if (
            !std::isfinite(contributors[index]) ||
            contributors[index] <= 0.0f
        ) {
            return false;
        }
    }
    for (
        std::size_t index = 0;
        index < accumulatorCount;
        ++index
    ) {
        if (!std::isfinite(accumulator[index])) {
            return false;
        }
    }

    for (int channel = 0; channel < channels; ++channel) {
        const std::size_t channelOffset =
            static_cast<std::size_t>(channel) *
            imagePixels;
        for (
            std::size_t index = 0;
            index < imagePixels;
            ++index
        ) {
            const float value =
                accumulator[channelOffset + index] /
                contributors[index];
            if (!std::isfinite(value)) {
                return false;
            }
            accumulator[channelOffset + index] = value;
        }
    }

    return true;
}

}  // namespace kira::pisa
