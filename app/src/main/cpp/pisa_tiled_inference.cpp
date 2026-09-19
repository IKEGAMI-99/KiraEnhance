#include "pisa_tiled_inference.h"

#include "pisa_tile_plan.h"
#include "pisa_tile_tensor.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

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

bool checkedScaleDimension(
    int value,
    int numerator,
    int denominator,
    int& output
) {
    if (
        value <= 0 ||
        numerator <= 0 ||
        denominator <= 0
    ) {
        return false;
    }

    const std::int64_t scaled =
        static_cast<std::int64_t>(value) *
        static_cast<std::int64_t>(numerator);
    if (
        scaled % denominator != 0 ||
        scaled / denominator <= 0 ||
        scaled / denominator >
            std::numeric_limits<int>::max()
    ) {
        return false;
    }

    output = static_cast<int>(scaled / denominator);
    return true;
}

}  // namespace

bool runScaledTiledPlanarTransform(
    const float* input,
    int inputChannels,
    int imageWidth,
    int imageHeight,
    int tileSize,
    int overlap,
    int outputChannels,
    int scaleNumerator,
    int scaleDenominator,
    PlanarScaledTileTransform transform,
    void* context,
    float* output,
    std::size_t outputCount
) {
    if (
        input == nullptr ||
        output == nullptr ||
        input == output ||
        transform == nullptr ||
        inputChannels <= 0 ||
        outputChannels <= 0 ||
        imageWidth <= 0 ||
        imageHeight <= 0
    ) {
        return false;
    }

    int outputWidth = 0;
    int outputHeight = 0;
    if (
        !checkedScaleDimension(
            imageWidth,
            scaleNumerator,
            scaleDenominator,
            outputWidth
        ) ||
        !checkedScaleDimension(
            imageHeight,
            scaleNumerator,
            scaleDenominator,
            outputHeight
        )
    ) {
        return false;
    }

    std::size_t outputPixels = 0;
    std::size_t fullOutputCount = 0;
    if (
        !checkedMultiply(
            static_cast<std::size_t>(outputWidth),
            static_cast<std::size_t>(outputHeight),
            outputPixels
        ) ||
        !checkedMultiply(
            static_cast<std::size_t>(outputChannels),
            outputPixels,
            fullOutputCount
        ) ||
        outputCount < fullOutputCount
    ) {
        return false;
    }

    TilePlan plan;
    if (
        !buildTilePlan(
            imageWidth,
            imageHeight,
            tileSize,
            overlap,
            plan
        )
    ) {
        return false;
    }

    TileRegion firstOutputTile;
    if (
        plan.tiles.empty() ||
        !scaleTileRegionExact(
            plan.tiles.front(),
            scaleNumerator,
            scaleDenominator,
            outputWidth,
            outputHeight,
            firstOutputTile
        )
    ) {
        return false;
    }

    std::vector<float> weights;
    if (
        !buildGaussianTileWeights(
            firstOutputTile.width,
            firstOutputTile.height,
            weights
        )
    ) {
        return false;
    }

    std::size_t inputTilePixels = 0;
    std::size_t inputTileCount = 0;
    std::size_t outputTilePixels = 0;
    std::size_t outputTileCount = 0;
    if (
        !checkedMultiply(
            static_cast<std::size_t>(plan.tileSize),
            static_cast<std::size_t>(plan.tileSize),
            inputTilePixels
        ) ||
        !checkedMultiply(
            static_cast<std::size_t>(inputChannels),
            inputTilePixels,
            inputTileCount
        ) ||
        !checkedMultiply(
            static_cast<std::size_t>(firstOutputTile.width),
            static_cast<std::size_t>(firstOutputTile.height),
            outputTilePixels
        ) ||
        !checkedMultiply(
            static_cast<std::size_t>(outputChannels),
            outputTilePixels,
            outputTileCount
        )
    ) {
        return false;
    }

    std::vector<float> tileInput(inputTileCount);
    std::vector<float> tileOutput(outputTileCount);
    std::vector<float> contributors(outputPixels, 0.0f);
    std::fill(
        output,
        output + fullOutputCount,
        0.0f
    );

    for (const TileRegion& tile : plan.tiles) {
        TileRegion outputTile;
        if (
            !scaleTileRegionExact(
                tile,
                scaleNumerator,
                scaleDenominator,
                outputWidth,
                outputHeight,
                outputTile
            ) ||
            outputTile.width != firstOutputTile.width ||
            outputTile.height != firstOutputTile.height ||
            !extractPlanarTile(
                input,
                inputChannels,
                imageWidth,
                imageHeight,
                tile,
                tileInput.data(),
                tileInput.size()
            ) ||
            !transform(
                tileInput.data(),
                tileInput.size(),
                inputChannels,
                tile.width,
                tile.height,
                tileOutput.data(),
                tileOutput.size(),
                outputChannels,
                outputTile.width,
                outputTile.height,
                context
            ) ||
            !accumulateWeightedPlanarTile(
                tileOutput.data(),
                outputChannels,
                outputTile,
                outputWidth,
                outputHeight,
                weights.data(),
                weights.size(),
                output,
                fullOutputCount,
                contributors.data(),
                contributors.size()
            )
        ) {
            return false;
        }
    }

    return normalizePlanarAccumulation(
        output,
        outputChannels,
        outputWidth,
        outputHeight,
        contributors.data(),
        contributors.size()
    );
}

bool runTiledPlanarTransform(
    const float* input,
    int channels,
    int imageWidth,
    int imageHeight,
    int tileSize,
    int overlap,
    PlanarTileTransform transform,
    void* context,
    float* output,
    std::size_t outputCount
) {
    if (
        input == nullptr ||
        output == nullptr ||
        input == output ||
        transform == nullptr ||
        channels <= 0 ||
        imageWidth <= 0 ||
        imageHeight <= 0
    ) {
        return false;
    }

    std::size_t imagePixels = 0;
    std::size_t fullValueCount = 0;
    if (
        !checkedMultiply(
            static_cast<std::size_t>(imageWidth),
            static_cast<std::size_t>(imageHeight),
            imagePixels
        ) ||
        !checkedMultiply(
            static_cast<std::size_t>(channels),
            imagePixels,
            fullValueCount
        ) ||
        outputCount < fullValueCount
    ) {
        return false;
    }

    TilePlan plan;
    if (
        !buildTilePlan(
            imageWidth,
            imageHeight,
            tileSize,
            overlap,
            plan
        )
    ) {
        return false;
    }

    std::vector<float> weights;
    if (
        !buildGaussianTileWeights(
            plan.tileSize,
            plan.tileSize,
            weights
        )
    ) {
        return false;
    }

    std::size_t tilePixels = 0;
    std::size_t tileValueCount = 0;
    if (
        !checkedMultiply(
            static_cast<std::size_t>(plan.tileSize),
            static_cast<std::size_t>(plan.tileSize),
            tilePixels
        ) ||
        !checkedMultiply(
            static_cast<std::size_t>(channels),
            tilePixels,
            tileValueCount
        )
    ) {
        return false;
    }

    std::vector<float> tileInput(tileValueCount);
    std::vector<float> tileOutput(tileValueCount);
    std::vector<float> contributors(imagePixels, 0.0f);
    std::fill(
        output,
        output + fullValueCount,
        0.0f
    );

    for (const TileRegion& tile : plan.tiles) {
        if (
            !extractPlanarTile(
                input,
                channels,
                imageWidth,
                imageHeight,
                tile,
                tileInput.data(),
                tileInput.size()
            ) ||
            !transform(
                tileInput.data(),
                tileInput.size(),
                channels,
                tile.width,
                tile.height,
                tileOutput.data(),
                tileOutput.size(),
                context
            ) ||
            !accumulateWeightedPlanarTile(
                tileOutput.data(),
                channels,
                tile,
                imageWidth,
                imageHeight,
                weights.data(),
                weights.size(),
                output,
                fullValueCount,
                contributors.data(),
                contributors.size()
            )
        ) {
            return false;
        }
    }

    return normalizePlanarAccumulation(
        output,
        channels,
        imageWidth,
        imageHeight,
        contributors.data(),
        contributors.size()
    );
}

}  // namespace kira::pisa
