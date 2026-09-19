#pragma once

#include "pisa_tile_plan.h"

#include <cstddef>
#include <cstdint>

namespace kira::pisa {

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
);

bool validateExactCoverage(
    const std::uint8_t* coverage,
    std::size_t coverageCount
);

}  // namespace kira::pisa
