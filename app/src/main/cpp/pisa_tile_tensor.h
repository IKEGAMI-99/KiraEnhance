#pragma once

#include "pisa_tile_plan.h"

#include <cstddef>

namespace kira::pisa {

bool extractPlanarTile(
    const float* input,
    int channels,
    int imageWidth,
    int imageHeight,
    const TileRegion& tile,
    float* output,
    std::size_t outputCount
);

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
);

bool normalizePlanarAccumulation(
    float* accumulator,
    int channels,
    int imageWidth,
    int imageHeight,
    const float* contributors,
    std::size_t contributorCount
);

}  // namespace kira::pisa
