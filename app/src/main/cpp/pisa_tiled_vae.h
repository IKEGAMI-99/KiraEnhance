#pragma once

#include <cstddef>

namespace kira::pisa {

using VaeTileTransform = bool (*)(
    const float* input,
    std::size_t inputCount,
    int inputChannels,
    int inputWidth,
    int inputHeight,
    float* output,
    std::size_t outputCount,
    int outputChannels,
    int outputWidth,
    int outputHeight,
    void* context
);

bool runTiledVaePlanarTransform(
    const float* input,
    int inputChannels,
    int inputWidth,
    int inputHeight,
    int tileSize,
    int padding,
    bool decoder,
    int outputChannels,
    VaeTileTransform transform,
    void* context,
    float* output,
    std::size_t outputCount
);

}  // namespace kira::pisa
