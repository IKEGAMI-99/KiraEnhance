#pragma once

#include <cstddef>

namespace kira::pisa {

using PlanarTileTransform = bool (*)(
    const float* input,
    std::size_t inputCount,
    int channels,
    int tileWidth,
    int tileHeight,
    float* output,
    std::size_t outputCount,
    void* context
);

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
);

}  // namespace kira::pisa
