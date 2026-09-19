#include "pisa_half.h"

#include <cstdint>
#include <cstring>

namespace kira::pisa {

namespace {

std::uint32_t floatBits(float value) {
    std::uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

float bitsFloat(std::uint32_t bits) {
    float value = 0.0f;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

}  // namespace

std::uint16_t floatToHalf(float value) {
    const std::uint32_t bits = floatBits(value);
    const std::uint32_t sign = (bits >> 16U) & 0x8000U;
    const std::uint32_t exponent = (bits >> 23U) & 0xffU;
    const std::uint32_t mantissa = bits & 0x007fffffU;

    if (exponent == 0xffU) {
        if (mantissa == 0U) {
            return static_cast<std::uint16_t>(sign | 0x7c00U);
        }
        const std::uint32_t payload = (mantissa >> 13U) | 0x0200U;
        return static_cast<std::uint16_t>(
            sign | 0x7c00U | (payload & 0x03ffU)
        );
    }

    int halfExponent =
        static_cast<int>(exponent) - 127 + 15;

    if (halfExponent >= 31) {
        return static_cast<std::uint16_t>(sign | 0x7c00U);
    }

    if (halfExponent <= 0) {
        if (halfExponent < -10) {
            return static_cast<std::uint16_t>(sign);
        }

        std::uint32_t normalizedMantissa = mantissa | 0x00800000U;
        const int shift = 14 - halfExponent;
        std::uint32_t halfMantissa =
            normalizedMantissa >> static_cast<unsigned int>(shift);

        const std::uint32_t remainderMask =
            (1U << static_cast<unsigned int>(shift)) - 1U;
        const std::uint32_t remainder =
            normalizedMantissa & remainderMask;
        const std::uint32_t halfway =
            1U << static_cast<unsigned int>(shift - 1);

        if (
            remainder > halfway ||
            (remainder == halfway && (halfMantissa & 1U) != 0U)
        ) {
            ++halfMantissa;
        }

        return static_cast<std::uint16_t>(
            sign | (halfMantissa & 0x03ffU)
        );
    }

    std::uint32_t halfMantissa = mantissa >> 13U;
    const std::uint32_t remainder = mantissa & 0x1fffU;
    if (
        remainder > 0x1000U ||
        (remainder == 0x1000U && (halfMantissa & 1U) != 0U)
    ) {
        ++halfMantissa;
        if (halfMantissa == 0x0400U) {
            halfMantissa = 0U;
            ++halfExponent;
            if (halfExponent >= 31) {
                return static_cast<std::uint16_t>(sign | 0x7c00U);
            }
        }
    }

    return static_cast<std::uint16_t>(
        sign |
        (static_cast<std::uint32_t>(halfExponent) << 10U) |
        (halfMantissa & 0x03ffU)
    );
}

float halfToFloat(std::uint16_t value) {
    const std::uint32_t sign =
        (static_cast<std::uint32_t>(value) & 0x8000U) << 16U;
    std::uint32_t exponent =
        (static_cast<std::uint32_t>(value) >> 10U) & 0x1fU;
    std::uint32_t mantissa =
        static_cast<std::uint32_t>(value) & 0x03ffU;

    if (exponent == 0U) {
        if (mantissa == 0U) {
            return bitsFloat(sign);
        }

        int adjustedExponent = -14;
        while ((mantissa & 0x0400U) == 0U) {
            mantissa <<= 1U;
            --adjustedExponent;
        }
        mantissa &= 0x03ffU;

        const std::uint32_t floatExponent =
            static_cast<std::uint32_t>(adjustedExponent + 127);
        return bitsFloat(
            sign |
            (floatExponent << 23U) |
            (mantissa << 13U)
        );
    }

    if (exponent == 0x1fU) {
        return bitsFloat(
            sign |
            0x7f800000U |
            (mantissa << 13U)
        );
    }

    exponent = exponent - 15U + 127U;
    return bitsFloat(
        sign |
        (exponent << 23U) |
        (mantissa << 13U)
    );
}

bool floatsToHalfs(
    const float* input,
    std::uint16_t* output,
    std::size_t count
) {
    if (input == nullptr || output == nullptr) {
        return false;
    }
    for (std::size_t index = 0; index < count; ++index) {
        output[index] = floatToHalf(input[index]);
    }
    return true;
}

bool halfsToFloats(
    const std::uint16_t* input,
    float* output,
    std::size_t count
) {
    if (input == nullptr || output == nullptr) {
        return false;
    }
    for (std::size_t index = 0; index < count; ++index) {
        output[index] = halfToFloat(input[index]);
    }
    return true;
}

}  // namespace kira::pisa
