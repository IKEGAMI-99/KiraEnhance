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

bool resizePlanarBilinear(
    const float* input,
    int channels,
    int inputWidth,
    int inputHeight,
    float* output,
    int outputWidth,
    int outputHeight,
    std::size_t outputFloatCount
) {
    if (
        input == nullptr ||
        output == nullptr ||
        channels <= 0 ||
        inputWidth <= 0 ||
        inputHeight <= 0 ||
        outputWidth <= 0 ||
        outputHeight <= 0
    ) {
        return false;
    }

    std::size_t inputPixels = 0;
    std::size_t outputPixels = 0;
    if (
        !imageElementCount(
            inputWidth,
            inputHeight,
            inputPixels
        ) ||
        !imageElementCount(
            outputWidth,
            outputHeight,
            outputPixels
        )
    ) {
        return false;
    }

    const auto channelCount =
        static_cast<std::size_t>(channels);
    if (
        outputPixels >
            std::numeric_limits<std::size_t>::max() /
                channelCount ||
        outputFloatCount < outputPixels * channelCount
    ) {
        return false;
    }

    const double scaleX =
        outputWidth > 1
            ? static_cast<double>(inputWidth - 1) /
                static_cast<double>(outputWidth - 1)
            : 0.0;
    const double scaleY =
        outputHeight > 1
            ? static_cast<double>(inputHeight - 1) /
                static_cast<double>(outputHeight - 1)
            : 0.0;

    for (int channel = 0; channel < channels; ++channel) {
        const std::size_t inputChannelOffset =
            static_cast<std::size_t>(channel) *
            inputPixels;
        const std::size_t outputChannelOffset =
            static_cast<std::size_t>(channel) *
            outputPixels;

        for (int y = 0; y < outputHeight; ++y) {
            const double sourceY =
                static_cast<double>(y) * scaleY;
            const int y0 =
                static_cast<int>(std::floor(sourceY));
            const int y1 = std::min(y0 + 1, inputHeight - 1);
            const float fy =
                static_cast<float>(
                    sourceY - static_cast<double>(y0)
                );

            for (int x = 0; x < outputWidth; ++x) {
                const double sourceX =
                    static_cast<double>(x) * scaleX;
                const int x0 =
                    static_cast<int>(std::floor(sourceX));
                const int x1 =
                    std::min(x0 + 1, inputWidth - 1);
                const float fx =
                    static_cast<float>(
                        sourceX - static_cast<double>(x0)
                    );

                const std::size_t row0 =
                    static_cast<std::size_t>(y0) *
                    static_cast<std::size_t>(inputWidth);
                const std::size_t row1 =
                    static_cast<std::size_t>(y1) *
                    static_cast<std::size_t>(inputWidth);
                const float topLeft = input[
                    inputChannelOffset +
                    row0 +
                    static_cast<std::size_t>(x0)
                ];
                const float topRight = input[
                    inputChannelOffset +
                    row0 +
                    static_cast<std::size_t>(x1)
                ];
                const float bottomLeft = input[
                    inputChannelOffset +
                    row1 +
                    static_cast<std::size_t>(x0)
                ];
                const float bottomRight = input[
                    inputChannelOffset +
                    row1 +
                    static_cast<std::size_t>(x1)
                ];
                if (
                    !std::isfinite(topLeft) ||
                    !std::isfinite(topRight) ||
                    !std::isfinite(bottomLeft) ||
                    !std::isfinite(bottomRight)
                ) {
                    return false;
                }

                const float top =
                    topLeft +
                    (topRight - topLeft) * fx;
                const float bottom =
                    bottomLeft +
                    (bottomRight - bottomLeft) * fx;
                const float value =
                    top + (bottom - top) * fy;
                if (!std::isfinite(value)) {
                    return false;
                }

                output[
                    outputChannelOffset +
                    static_cast<std::size_t>(y) *
                        static_cast<std::size_t>(outputWidth) +
                    static_cast<std::size_t>(x)
                ] = value;
            }
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
