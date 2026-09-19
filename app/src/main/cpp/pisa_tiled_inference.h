#pragma once

#include <cstddef>

namespace kira::pisa {

using PlanarScaledTileTransform = bool (*)(
    const float* input,
    std::size_t inputCount,
    int inputChannels,
    int inputTileWidth,
    int inputTileHeight,
    float* output,
    std::size_t outputCount,
    int outputChannels,
    int outputTileWidth,
    int outputTileHeight,
    void* context
);

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
