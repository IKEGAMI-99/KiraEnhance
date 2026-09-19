#pragma once

#include "pisa_resize_plan.h"

#include <cstddef>
#include <cstdint>

namespace kira::pisa {

bool preparePisaModelRgba(
    const std::uint8_t* input,
    int inputRowStrideBytes,
    const ResizePlan& plan,
    std::uint8_t* output,
    int outputRowStrideBytes,
    std::size_t outputByteCount
);

}  // namespace kira::pisa
