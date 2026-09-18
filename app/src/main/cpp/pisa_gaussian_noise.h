#pragma once

#include <cstddef>
#include <cstdint>

namespace kira::pisa {

bool fillGaussianNoise(
    float* output,
    std::size_t count,
    std::uint64_t seed
);

}  // namespace kira::pisa
