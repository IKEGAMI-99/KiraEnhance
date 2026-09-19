#pragma once

#include "pisa_vae_segment_pack.h"

#include <cstddef>
#include <vector>

#if KIRA_HAS_MNN
#include <MNN/Interpreter.hpp>
#endif

namespace kira::pisa {

#if KIRA_HAS_MNN

using SegmentedVaeCancelProbe = bool (*)(void* context);

bool runMnnSegmentedVae(
    const std::vector<MNN::Interpreter*>& segments,
    const std::vector<VaeGroupNormAffine>& affine,
    MNNForwardType backend,
    const float* input,
    int inputChannels,
    int inputWidth,
    int inputHeight,
    int tileSize,
    int padding,
    bool decoder,
    int outputChannels,
    SegmentedVaeCancelProbe cancelProbe,
    void* cancelContext,
    float* output,
    std::size_t outputCount
);

#endif

}  // namespace kira::pisa
