#include "pisa_gaussian_noise.h"

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace kira::pisa {
namespace {

constexpr std::uint64_t SPLITMIX_GAMMA = 0x9E3779B97F4A7C15ULL;
constexpr std::uint64_t SPLITMIX_MUL_1 = 0xBF58476D1CE4E5B9ULL;
constexpr std::uint64_t SPLITMIX_MUL_2 = 0x94D049BB133111EBULL;
constexpr double TWO_PI = 6.283185307179586476925286766559;
constexpr double TWO_POW_53 = 9007199254740992.0;

std::uint64_t nextSplitMix64(std::uint64_t& state) {
    state += SPLITMIX_GAMMA;
    std::uint64_t value = state;
    value = (value ^ (value >> 30U)) * SPLITMIX_MUL_1;
    value = (value ^ (value >> 27U)) * SPLITMIX_MUL_2;
    return value ^ (value >> 31U);
}

double uniformOpen01(std::uint64_t bits) {
    const std::uint64_t top53 = bits >> 11U;
    return (static_cast<double>(top53) + 0.5) / TWO_POW_53;
}

}  // namespace

bool fillGaussianNoise(
    float* output,
    std::size_t count,
    std::uint64_t seed
) {
    if (output == nullptr || count == 0) {
        return false;
    }

    std::uint64_t state = seed;
    std::size_t index = 0;

    while (index < count) {
        const double uniformRadius = uniformOpen01(
            nextSplitMix64(state)
        );
        const double uniformAngle = uniformOpen01(
            nextSplitMix64(state)
        );

        const double magnitude = std::sqrt(
            -2.0 * std::log(uniformRadius)
        );
        const double angle = TWO_PI * uniformAngle;

        output[index++] = static_cast<float>(
            magnitude * std::cos(angle)
        );
        if (index < count) {
            output[index++] = static_cast<float>(
                magnitude * std::sin(angle)
            );
        }
    }

    return true;
}

}  // namespace kira::pisa
