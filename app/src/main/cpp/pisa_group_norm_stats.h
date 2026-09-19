#pragma once

#include <cstddef>
#include <vector>

namespace kira::pisa {

struct GroupNormTileStats {
    int batch = 0;
    int groups = 0;
    std::size_t spatialPixels = 0;
    std::vector<float> means;
    std::vector<float> variances;
};

bool computeGroupNormTileStats(
    const float* input,
    int batch,
    int channels,
    int height,
    int width,
    int groups,
    GroupNormTileStats& output
);

bool summarizeGroupNormTileStats(
    const std::vector<GroupNormTileStats>& tiles,
    std::vector<float>& means,
    std::vector<float>& variances
);

bool applyGroupNormWithStats(
    const float* input,
    int batch,
    int channels,
    int height,
    int width,
    int groups,
    const float* means,
    const float* variances,
    const float* weight,
    const float* bias,
    float epsilon,
    float* output,
    std::size_t outputCount
);

}  // namespace kira::pisa
