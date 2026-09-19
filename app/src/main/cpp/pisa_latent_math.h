#pragma once

namespace kira::pisa {

bool sampleLatentFromMoments(
    const float* moments,
    const float* noise,
    float* output,
    int batch,
    int latentChannels,
    int height,
    int width,
    float scalingFactor
);

}  // namespace kira::pisa
