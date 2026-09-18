#include "pisa_image_tensor.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace kira::pisa {
namespace {

bool imageElementCount(int width, int height, std::size_t& pixels) {
    if (width <= 0 || height <= 0) {
        return false;
    }

    const auto w = static_cast<std::size_t>(width);
    const auto h = static_cast<std::size_t>(height);
    if (w > std::numeric_limits<std::size_t>::max() / h) {
        return false;
    }
    pixels = w * h;
    return true;
}

std::uint8_t normalizedToByte(float value) {
    if (!std::isfinite(value)) {
        value = 0.0f;
    }
    const float clamped = std::clamp(value, -1.0f, 1.0f);
    const float scaled = (clamped + 1.0f) * 127.5f;
    return static_cast<std::uint8_t>(std::lround(scaled));
}

}  // namespace

bool rgba8888ToNormalizedNchw(
    const std::uint8_t* input,
    int width,
    int height,
    int rowStrideBytes,
    float* output,
    std::size_t outputFloatCount
) {
    std::size_t pixels = 0;
    if (
        input == nullptr ||
        output == nullptr ||
        !imageElementCount(width, height, pixels) ||
        rowStrideBytes < width * 4 ||
        outputFloatCount < pixels * 3
    ) {
        return false;
    }

    float* red = output;
    float* green = output + pixels;
    float* blue = output + pixels * 2;

    for (int y = 0; y < height; ++y) {
        const auto* row =
            input + static_cast<std::size_t>(y) * rowStrideBytes;
        for (int x = 0; x < width; ++x) {
            const auto pixelIndex =
                static_cast<std::size_t>(y) * width + x;
            const auto byteIndex = static_cast<std::size_t>(x) * 4;
            red[pixelIndex] =
                static_cast<float>(row[byteIndex]) / 127.5f - 1.0f;
            green[pixelIndex] =
                static_cast<float>(row[byteIndex + 1]) / 127.5f - 1.0f;
            blue[pixelIndex] =
                static_cast<float>(row[byteIndex + 2]) / 127.5f - 1.0f;
        }
    }

    return true;
}

bool normalizedNchwToRgba8888(
    const float* input,
    int width,
    int height,
    std::uint8_t* output,
    int rowStrideBytes,
    std::size_t outputByteCount
) {
    std::size_t pixels = 0;
    if (
        input == nullptr ||
        output == nullptr ||
        !imageElementCount(width, height, pixels) ||
        rowStrideBytes < width * 4
    ) {
        return false;
    }

    const auto requiredBytes =
        static_cast<std::size_t>(rowStrideBytes) *
        static_cast<std::size_t>(height);
    if (outputByteCount < requiredBytes) {
        return false;
    }

    const float* red = input;
    const float* green = input + pixels;
    const float* blue = input + pixels * 2;

    for (int y = 0; y < height; ++y) {
        auto* row = output + static_cast<std::size_t>(y) * rowStrideBytes;
        for (int x = 0; x < width; ++x) {
            const auto pixelIndex =
                static_cast<std::size_t>(y) * width + x;
            const auto byteIndex = static_cast<std::size_t>(x) * 4;
            row[byteIndex] = normalizedToByte(red[pixelIndex]);
            row[byteIndex + 1] = normalizedToByte(green[pixelIndex]);
            row[byteIndex + 2] = normalizedToByte(blue[pixelIndex]);
            row[byteIndex + 3] = 255;
        }
    }

    return true;
}

}  // namespace kira::pisa
