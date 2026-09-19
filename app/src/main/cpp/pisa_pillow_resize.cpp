#include "pisa_pillow_resize.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace kira::pisa {
namespace {

constexpr int PRECISION_BITS = 22;
constexpr std::int64_t ROUNDING_OFFSET =
    static_cast<std::int64_t>(1) << (PRECISION_BITS - 1);
constexpr double PI =
    3.141592653589793238462643383279502884;

struct AxisCoefficients {
    int outputSize = 0;
    int kernelSize = 0;
    std::vector<int> starts;
    std::vector<int> counts;
    std::vector<std::int32_t> weights;
};

bool checkedMultiply(
    std::size_t left,
    std::size_t right,
    std::size_t& output
) {
    if (
        left != 0 &&
        right >
            std::numeric_limits<std::size_t>::max() / left
    ) {
        return false;
    }
    output = left * right;
    return true;
}

double bicubicFilter(double value) {
    constexpr double a = -0.5;
    value = std::abs(value);
    if (value < 1.0) {
        return (
            (a + 2.0) * value -
            (a + 3.0)
        ) * value * value + 1.0;
    }
    if (value < 2.0) {
        return (
            (
                (value - 5.0) * value +
                8.0
            ) * value -
            4.0
        ) * a;
    }
    return 0.0;
}

double sincFilter(double value) {
    if (value == 0.0) {
        return 1.0;
    }
    const double angle = value * PI;
    return std::sin(angle) / angle;
}

double lanczosFilter(double value) {
    if (value >= -3.0 && value < 3.0) {
        return
            sincFilter(value) *
            sincFilter(value / 3.0);
    }
    return 0.0;
}

double filterSupport(PillowResizeFilter filter) {
    switch (filter) {
        case PillowResizeFilter::BICUBIC:
            return 2.0;
        case PillowResizeFilter::LANCZOS:
            return 3.0;
    }
    return 0.0;
}

double filterValue(
    PillowResizeFilter filter,
    double value
) {
    switch (filter) {
        case PillowResizeFilter::BICUBIC:
            return bicubicFilter(value);
        case PillowResizeFilter::LANCZOS:
            return lanczosFilter(value);
    }
    return 0.0;
}

bool buildAxisCoefficients(
    int inputSize,
    int outputSize,
    PillowResizeFilter filter,
    AxisCoefficients& output
) {
    output = AxisCoefficients{};
    if (inputSize <= 0 || outputSize <= 0) {
        return false;
    }

    const double baseSupport = filterSupport(filter);
    if (baseSupport <= 0.0) {
        return false;
    }

    const double scale =
        static_cast<double>(inputSize) /
        static_cast<double>(outputSize);
    const double filterScale = std::max(scale, 1.0);
    const double support = baseSupport * filterScale;
    const double inverseFilterScale = 1.0 / filterScale;
    const double kernelSizeDouble =
        std::ceil(support) * 2.0 + 1.0;
    if (
        !std::isfinite(kernelSizeDouble) ||
        kernelSizeDouble <= 0.0 ||
        kernelSizeDouble >
            static_cast<double>(
                std::numeric_limits<int>::max()
            )
    ) {
        return false;
    }

    const int kernelSize =
        static_cast<int>(kernelSizeDouble);
    std::size_t weightCount = 0;
    if (
        !checkedMultiply(
            static_cast<std::size_t>(outputSize),
            static_cast<std::size_t>(kernelSize),
            weightCount
        )
    ) {
        return false;
    }

    AxisCoefficients result;
    result.outputSize = outputSize;
    result.kernelSize = kernelSize;
    result.starts.resize(
        static_cast<std::size_t>(outputSize)
    );
    result.counts.resize(
        static_cast<std::size_t>(outputSize)
    );
    result.weights.assign(weightCount, 0);

    std::vector<double> raw(
        static_cast<std::size_t>(kernelSize),
        0.0
    );
    constexpr double fixedScale =
        static_cast<double>(
            static_cast<std::uint64_t>(1) <<
            PRECISION_BITS
        );

    for (int index = 0; index < outputSize; ++index) {
        const double center =
            (static_cast<double>(index) + 0.5) *
            scale;

        int start = static_cast<int>(
            center - support + 0.5
        );
        if (start < 0) {
            start = 0;
        }

        int end = static_cast<int>(
            center + support + 0.5
        );
        if (end > inputSize) {
            end = inputSize;
        }

        const int count = end - start;
        if (count <= 0 || count > kernelSize) {
            return false;
        }

        double sum = 0.0;
        for (int tap = 0; tap < count; ++tap) {
            const double value =
                (
                    static_cast<double>(tap + start) -
                    center +
                    0.5
                ) * inverseFilterScale;
            const double weight =
                filterValue(filter, value);
            if (!std::isfinite(weight)) {
                return false;
            }
            raw[static_cast<std::size_t>(tap)] =
                weight;
            sum += weight;
        }
        if (!std::isfinite(sum) || sum == 0.0) {
            return false;
        }

        const std::size_t offset =
            static_cast<std::size_t>(index) *
            static_cast<std::size_t>(kernelSize);
        for (int tap = 0; tap < count; ++tap) {
            const double normalized =
                raw[static_cast<std::size_t>(tap)] /
                sum;
            const double fixed =
                normalized * fixedScale;
            if (
                !std::isfinite(fixed) ||
                fixed <
                    static_cast<double>(
                        std::numeric_limits<std::int32_t>::min()
                    ) ||
                fixed >
                    static_cast<double>(
                        std::numeric_limits<std::int32_t>::max()
                    )
            ) {
                return false;
            }

            result.weights[
                offset +
                static_cast<std::size_t>(tap)
            ] = static_cast<std::int32_t>(
                fixed < 0.0
                    ? fixed - 0.5
                    : fixed + 0.5
            );
        }

        result.starts[
            static_cast<std::size_t>(index)
        ] = start;
        result.counts[
            static_cast<std::size_t>(index)
        ] = count;
    }

    output = std::move(result);
    return true;
}

std::uint8_t fixedToByte(std::int64_t value) {
    if (value <= 0) {
        return 0;
    }

    const std::int64_t rounded =
        value >> PRECISION_BITS;
    if (rounded >= 255) {
        return 255;
    }
    return static_cast<std::uint8_t>(rounded);
}

bool resampleHorizontalRgb(
    const std::vector<std::uint8_t>& input,
    int inputWidth,
    int inputHeight,
    const AxisCoefficients& coefficients,
    std::vector<std::uint8_t>& output
) {
    if (
        inputWidth <= 0 ||
        inputHeight <= 0 ||
        coefficients.outputSize <= 0 ||
        coefficients.kernelSize <= 0
    ) {
        return false;
    }

    std::size_t inputPixels = 0;
    std::size_t inputBytes = 0;
    std::size_t outputPixels = 0;
    std::size_t outputBytes = 0;
    if (
        !checkedMultiply(
            static_cast<std::size_t>(inputWidth),
            static_cast<std::size_t>(inputHeight),
            inputPixels
        ) ||
        !checkedMultiply(inputPixels, 3U, inputBytes) ||
        input.size() < inputBytes ||
        !checkedMultiply(
            static_cast<std::size_t>(
                coefficients.outputSize
            ),
            static_cast<std::size_t>(inputHeight),
            outputPixels
        ) ||
        !checkedMultiply(outputPixels, 3U, outputBytes)
    ) {
        return false;
    }

    output.resize(outputBytes);
    for (int y = 0; y < inputHeight; ++y) {
        for (
            int outputX = 0;
            outputX < coefficients.outputSize;
            ++outputX
        ) {
            const int start = coefficients.starts[
                static_cast<std::size_t>(outputX)
            ];
            const int count = coefficients.counts[
                static_cast<std::size_t>(outputX)
            ];
            const std::size_t weightOffset =
                static_cast<std::size_t>(outputX) *
                static_cast<std::size_t>(
                    coefficients.kernelSize
                );

            for (int channel = 0; channel < 3; ++channel) {
                std::int64_t sum = ROUNDING_OFFSET;
                for (int tap = 0; tap < count; ++tap) {
                    const std::size_t sourceIndex =
                        (
                            static_cast<std::size_t>(y) *
                                static_cast<std::size_t>(
                                    inputWidth
                                ) +
                            static_cast<std::size_t>(
                                start + tap
                            )
                        ) * 3U +
                        static_cast<std::size_t>(channel);
                    sum +=
                        static_cast<std::int64_t>(
                            input[sourceIndex]
                        ) *
                        coefficients.weights[
                            weightOffset +
                            static_cast<std::size_t>(tap)
                        ];
                }

                const std::size_t destinationIndex =
                    (
                        static_cast<std::size_t>(y) *
                            static_cast<std::size_t>(
                                coefficients.outputSize
                            ) +
                        static_cast<std::size_t>(outputX)
                    ) * 3U +
                    static_cast<std::size_t>(channel);
                output[destinationIndex] =
                    fixedToByte(sum);
            }
        }
    }
    return true;
}

bool resampleVerticalRgb(
    const std::vector<std::uint8_t>& input,
    int width,
    int inputHeight,
    const AxisCoefficients& coefficients,
    std::vector<std::uint8_t>& output
) {
    if (
        width <= 0 ||
        inputHeight <= 0 ||
        coefficients.outputSize <= 0 ||
        coefficients.kernelSize <= 0
    ) {
        return false;
    }

    std::size_t inputPixels = 0;
    std::size_t inputBytes = 0;
    std::size_t outputPixels = 0;
    std::size_t outputBytes = 0;
    if (
        !checkedMultiply(
            static_cast<std::size_t>(width),
            static_cast<std::size_t>(inputHeight),
            inputPixels
        ) ||
        !checkedMultiply(inputPixels, 3U, inputBytes) ||
        input.size() < inputBytes ||
        !checkedMultiply(
            static_cast<std::size_t>(width),
            static_cast<std::size_t>(
                coefficients.outputSize
            ),
            outputPixels
        ) ||
        !checkedMultiply(outputPixels, 3U, outputBytes)
    ) {
        return false;
    }

    output.resize(outputBytes);
    for (
        int outputY = 0;
        outputY < coefficients.outputSize;
        ++outputY
    ) {
        const int start = coefficients.starts[
            static_cast<std::size_t>(outputY)
        ];
        const int count = coefficients.counts[
            static_cast<std::size_t>(outputY)
        ];
        const std::size_t weightOffset =
            static_cast<std::size_t>(outputY) *
            static_cast<std::size_t>(
                coefficients.kernelSize
            );

        for (int x = 0; x < width; ++x) {
            for (int channel = 0; channel < 3; ++channel) {
                std::int64_t sum = ROUNDING_OFFSET;
                for (int tap = 0; tap < count; ++tap) {
                    const std::size_t sourceIndex =
                        (
                            static_cast<std::size_t>(
                                start + tap
                            ) *
                                static_cast<std::size_t>(width) +
                            static_cast<std::size_t>(x)
                        ) * 3U +
                        static_cast<std::size_t>(channel);
                    sum +=
                        static_cast<std::int64_t>(
                            input[sourceIndex]
                        ) *
                        coefficients.weights[
                            weightOffset +
                            static_cast<std::size_t>(tap)
                        ];
                }

                const std::size_t destinationIndex =
                    (
                        static_cast<std::size_t>(outputY) *
                            static_cast<std::size_t>(width) +
                        static_cast<std::size_t>(x)
                    ) * 3U +
                    static_cast<std::size_t>(channel);
                output[destinationIndex] =
                    fixedToByte(sum);
            }
        }
    }
    return true;
}

}  // namespace

bool resizeRgba8888PillowRgb(
    const std::uint8_t* input,
    int inputWidth,
    int inputHeight,
    int inputRowStrideBytes,
    std::uint8_t* output,
    int outputWidth,
    int outputHeight,
    int outputRowStrideBytes,
    std::size_t outputByteCount,
    PillowResizeFilter filter
) {
    if (
        input == nullptr ||
        output == nullptr ||
        inputWidth <= 0 ||
        inputHeight <= 0 ||
        outputWidth <= 0 ||
        outputHeight <= 0 ||
        inputWidth >
            std::numeric_limits<int>::max() / 4 ||
        outputWidth >
            std::numeric_limits<int>::max() / 4 ||
        inputRowStrideBytes < inputWidth * 4 ||
        outputRowStrideBytes < outputWidth * 4
    ) {
        return false;
    }

    std::size_t inputPixels = 0;
    std::size_t inputRgbBytes = 0;
    std::size_t requiredOutputBytes = 0;
    if (
        !checkedMultiply(
            static_cast<std::size_t>(inputWidth),
            static_cast<std::size_t>(inputHeight),
            inputPixels
        ) ||
        !checkedMultiply(inputPixels, 3U, inputRgbBytes) ||
        !checkedMultiply(
            static_cast<std::size_t>(outputRowStrideBytes),
            static_cast<std::size_t>(outputHeight),
            requiredOutputBytes
        ) ||
        outputByteCount < requiredOutputBytes
    ) {
        return false;
    }

    std::vector<std::uint8_t> source(inputRgbBytes);
    for (int y = 0; y < inputHeight; ++y) {
        const std::uint8_t* sourceRow =
            input +
            static_cast<std::size_t>(y) *
                static_cast<std::size_t>(
                    inputRowStrideBytes
                );
        for (int x = 0; x < inputWidth; ++x) {
            const std::size_t sourceOffset =
                static_cast<std::size_t>(x) * 4U;
            const std::size_t destinationOffset =
                (
                    static_cast<std::size_t>(y) *
                        static_cast<std::size_t>(
                            inputWidth
                        ) +
                    static_cast<std::size_t>(x)
                ) * 3U;
            source[destinationOffset] =
                sourceRow[sourceOffset];
            source[destinationOffset + 1U] =
                sourceRow[sourceOffset + 1U];
            source[destinationOffset + 2U] =
                sourceRow[sourceOffset + 2U];
        }
    }

    std::vector<std::uint8_t> horizontal;
    const std::vector<std::uint8_t>* verticalInput =
        &source;
    int verticalWidth = inputWidth;

    if (outputWidth != inputWidth) {
        AxisCoefficients horizontalCoefficients;
        if (
            !buildAxisCoefficients(
                inputWidth,
                outputWidth,
                filter,
                horizontalCoefficients
            ) ||
            !resampleHorizontalRgb(
                source,
                inputWidth,
                inputHeight,
                horizontalCoefficients,
                horizontal
            )
        ) {
            return false;
        }
        verticalInput = &horizontal;
        verticalWidth = outputWidth;
    }

    std::vector<std::uint8_t> resized;
    const std::vector<std::uint8_t>* finalRgb =
        verticalInput;

    if (outputHeight != inputHeight) {
        AxisCoefficients verticalCoefficients;
        if (
            !buildAxisCoefficients(
                inputHeight,
                outputHeight,
                filter,
                verticalCoefficients
            ) ||
            !resampleVerticalRgb(
                *verticalInput,
                verticalWidth,
                inputHeight,
                verticalCoefficients,
                resized
            )
        ) {
            return false;
        }
        finalRgb = &resized;
    }

    std::size_t expectedRgbPixels = 0;
    std::size_t expectedRgbBytes = 0;
    if (
        !checkedMultiply(
            static_cast<std::size_t>(outputWidth),
            static_cast<std::size_t>(outputHeight),
            expectedRgbPixels
        ) ||
        !checkedMultiply(
            expectedRgbPixels,
            3U,
            expectedRgbBytes
        ) ||
        finalRgb->size() < expectedRgbBytes
    ) {
        return false;
    }

    for (int y = 0; y < outputHeight; ++y) {
        std::uint8_t* destinationRow =
            output +
            static_cast<std::size_t>(y) *
                static_cast<std::size_t>(
                    outputRowStrideBytes
                );
        for (int x = 0; x < outputWidth; ++x) {
            const std::size_t sourceOffset =
                (
                    static_cast<std::size_t>(y) *
                        static_cast<std::size_t>(
                            outputWidth
                        ) +
                    static_cast<std::size_t>(x)
                ) * 3U;
            const std::size_t destinationOffset =
                static_cast<std::size_t>(x) * 4U;
            destinationRow[destinationOffset] =
                (*finalRgb)[sourceOffset];
            destinationRow[destinationOffset + 1U] =
                (*finalRgb)[sourceOffset + 1U];
            destinationRow[destinationOffset + 2U] =
                (*finalRgb)[sourceOffset + 2U];
            destinationRow[destinationOffset + 3U] = 255U;
        }
    }
    return true;
}

}  // namespace kira::pisa
