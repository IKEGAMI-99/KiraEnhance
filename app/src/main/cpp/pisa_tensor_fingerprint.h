#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace kira::pisa {

struct TensorFingerprint {
    std::size_t elementCount = 0;
    std::size_t finiteCount = 0;
    std::size_t nanCount = 0;
    std::size_t positiveInfinityCount = 0;
    std::size_t negativeInfinityCount = 0;
    float minimumFinite = 0.0f;
    float maximumFinite = 0.0f;
    double meanFinite = 0.0;
    double rmsFinite = 0.0;
    std::uint64_t fnv1a64 = 14695981039346656037ULL;
};

inline TensorFingerprint makeTensorFingerprint(
    const float* values,
    std::size_t count
) {
    TensorFingerprint result;
    result.elementCount = count;
    if (values == nullptr || count == 0) {
        return result;
    }

    constexpr std::uint64_t kFnvPrime = 1099511628211ULL;
    double sum = 0.0;
    double sumSquares = 0.0;
    bool hasFinite = false;

    for (std::size_t index = 0; index < count; ++index) {
        const float value = values[index];

        std::uint32_t bits = 0;
        static_assert(sizeof(bits) == sizeof(value));
        std::memcpy(&bits, &value, sizeof(bits));
        for (unsigned int shift = 0; shift < 32; shift += 8) {
            const auto byte = static_cast<std::uint8_t>(
                (bits >> shift) & 0xffU
            );
            result.fnv1a64 ^= byte;
            result.fnv1a64 *= kFnvPrime;
        }

        if (std::isnan(value)) {
            ++result.nanCount;
            continue;
        }
        if (std::isinf(value)) {
            if (value > 0.0f) {
                ++result.positiveInfinityCount;
            } else {
                ++result.negativeInfinityCount;
            }
            continue;
        }

        if (!hasFinite) {
            result.minimumFinite = value;
            result.maximumFinite = value;
            hasFinite = true;
        } else {
            if (value < result.minimumFinite) {
                result.minimumFinite = value;
            }
            if (value > result.maximumFinite) {
                result.maximumFinite = value;
            }
        }

        ++result.finiteCount;
        const double promoted = static_cast<double>(value);
        sum += promoted;
        sumSquares += promoted * promoted;
    }

    if (result.finiteCount != 0) {
        const double divisor =
            static_cast<double>(result.finiteCount);
        result.meanFinite = sum / divisor;
        result.rmsFinite = std::sqrt(sumSquares / divisor);
    }

    return result;
}

}  // namespace kira::pisa
