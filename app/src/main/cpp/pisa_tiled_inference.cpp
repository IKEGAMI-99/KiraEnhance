#include "pisa_tiled_inference.h"

#include "pisa_tile_plan.h"
#include "pisa_tile_tensor.h"

#include <algorithm>
#include <cstddef>
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

}  // namespace

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
