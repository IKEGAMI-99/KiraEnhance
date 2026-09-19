#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace kira::pisa {

struct VaeGroupNormAffine {
    int groups = 0;
    int channels = 0;
    float epsilon = 0.0f;
    std::vector<float> weight;
    std::vector<float> bias;
};

struct VaeSegmentPack {
    std::vector<VaeGroupNormAffine> encoderAffine;
    std::vector<VaeGroupNormAffine> decoderAffine;
    std::vector<std::vector<std::uint8_t>> encoderModels;
    std::vector<std::vector<std::uint8_t>> decoderModels;
};

bool parseVaeSegmentPack(
    const std::uint8_t* data,
    std::size_t size,
    VaeSegmentPack& output
);

bool loadVaeSegmentPackFile(
    const std::string& path,
    VaeSegmentPack& output
);

}  // namespace kira::pisa
