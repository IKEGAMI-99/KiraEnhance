#include "pisa_tiled_inference.h"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

namespace {

int failures = 0;

void expectTrue(bool value, const char* label) {
    if (!value) {
        std::cerr << "FAIL: " << label << "\n";
        ++failures;
    }
}

void expectFalse(bool value, const char* label) {
    expectTrue(!value, label);
}

void expectNear(
    float actual,
    float expected,
    float tolerance,
    const char* label
) {
    if (std::fabs(actual - expected) > tolerance) {
        std::cerr
            << "FAIL: " << label
            << " expected=" << expected
            << " actual=" << actual
            << "\n";
        ++failures;
    }
}

struct TransformContext {
    float scale = 1.0f;
    int calls = 0;
    int expectedChannels = 0;
    int expectedTileSize = 0;
    bool fail = false;
};

bool scaleTransform(
    const float* input,
    std::size_t inputCount,
    int channels,
    int tileWidth,
    int tileHeight,
    float* output,
    std::size_t outputCount,
    void* rawContext
) {
    auto* context =
        static_cast<TransformContext*>(rawContext);
    if (
        context == nullptr ||
        input == nullptr ||
        output == nullptr ||
        inputCount != outputCount ||
        channels != context->expectedChannels ||
        tileWidth != context->expectedTileSize ||
        tileHeight != context->expectedTileSize
    ) {
        return false;
    }

    context->calls += 1;
    if (context->fail) {
        return false;
    }

    for (
        std::size_t index = 0;
        index < inputCount;
        ++index
    ) {
        output[index] = input[index] * context->scale;
    }
    return true;
}

struct ScaledTransformContext {
    int numerator = 1;
    int denominator = 1;
    int calls = 0;
};

bool nearestScaledTransform(
    const float* input,
    std::size_t inputCount,
    int inputChannels,
    int inputTileWidth,
    int inputTileHeight,
    float* output,
    std::size_t outputCount,
    int outputChannels,
    int outputTileWidth,
    int outputTileHeight,
    void* rawContext
) {
    auto* context =
        static_cast<ScaledTransformContext*>(rawContext);
    if (
        context == nullptr ||
        input == nullptr ||
        output == nullptr ||
        inputChannels != 1 ||
        outputChannels != 1 ||
        context->numerator <= 0 ||
        context->denominator <= 0 ||
        inputCount !=
            static_cast<std::size_t>(
                inputTileWidth * inputTileHeight
            ) ||
        outputCount !=
            static_cast<std::size_t>(
                outputTileWidth * outputTileHeight
            )
    ) {
        return false;
    }

    context->calls += 1;
    for (int y = 0; y < outputTileHeight; ++y) {
        const int sourceY =
            y * context->denominator /
            context->numerator;
        if (sourceY < 0 || sourceY >= inputTileHeight) {
            return false;
        }
        for (int x = 0; x < outputTileWidth; ++x) {
            const int sourceX =
                x * context->denominator /
                context->numerator;
            if (sourceX < 0 || sourceX >= inputTileWidth) {
                return false;
            }
            output[
                static_cast<std::size_t>(y) *
                    outputTileWidth +
                x
            ] = input[
                static_cast<std::size_t>(sourceY) *
                    inputTileWidth +
                sourceX
            ];
        }
    }
    return true;
}

void testDownscaledTiledTransformReconstructsAlignedGrid() {
    constexpr int width = 8;
    constexpr int height = 8;
    std::vector<float> input(width * height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            input[
                static_cast<std::size_t>(y) * width + x
            ] = static_cast<float>(y * 100 + x);
        }
    }

    std::vector<float> output(4 * 4, 0.0f);
    ScaledTransformContext context{1, 2, 0};
    expectTrue(
        kira::pisa::runScaledTiledPlanarTransform(
            input.data(),
            1,
            width,
            height,
            4,
            2,
            1,
            1,
            2,
            nearestScaledTransform,
            &context,
            output.data(),
            output.size()
        ),
        "downscaled tiled transform succeeds"
    );
    expectTrue(
        context.calls > 1,
        "downscaled path uses multiple tiles"
    );

    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) {
            expectNear(
                output[
                    static_cast<std::size_t>(y) * 4 + x
                ],
                input[
                    static_cast<std::size_t>(y * 2) *
                        width +
                    x * 2
                ],
                1.0e-5f,
                "downscaled tiled output"
            );
        }
    }
}

void testUpscaledTiledTransformReconstructsNearestGrid() {
    constexpr int width = 4;
    constexpr int height = 4;
    std::vector<float> input(width * height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            input[
                static_cast<std::size_t>(y) * width + x
            ] = static_cast<float>(y * 10 + x);
        }
    }

    std::vector<float> output(8 * 8, 0.0f);
    ScaledTransformContext context{2, 1, 0};
    expectTrue(
        kira::pisa::runScaledTiledPlanarTransform(
            input.data(),
            1,
            width,
            height,
            2,
            1,
            1,
            2,
            1,
            nearestScaledTransform,
            &context,
            output.data(),
            output.size()
        ),
        "upscaled tiled transform succeeds"
    );

    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
            expectNear(
                output[
                    static_cast<std::size_t>(y) * 8 + x
                ],
                input[
                    static_cast<std::size_t>(y / 2) *
                        width +
                    x / 2
                ],
                1.0e-5f,
                "upscaled tiled output"
            );
        }
    }
}

void testScaledRunnerRejectsInexactTileMapping() {
    const float input[8 * 8] = {};
    float output[4 * 4] = {};
    ScaledTransformContext context{1, 2, 0};

    expectFalse(
        kira::pisa::runScaledTiledPlanarTransform(
            input,
            1,
            8,
            8,
            4,
            1,
            1,
            1,
            2,
            nearestScaledTransform,
            &context,
            output,
            16
        ),
        "inexact scaled tile mapping rejected"
    );
}

void testIdentityTransformReconstructsFullTensor() {
    constexpr int channels = 2;
    constexpr int width = 6;
    constexpr int height = 4;
    std::vector<float> input(channels * width * height);
    for (
        std::size_t index = 0;
        index < input.size();
        ++index
    ) {
        input[index] = static_cast<float>(index) * 0.25f;
    }

    std::vector<float> output(input.size(), 0.0f);
    TransformContext context{
        1.0f,
        0,
        channels,
        4,
        false,
    };

    expectTrue(
        kira::pisa::runTiledPlanarTransform(
            input.data(),
            channels,
            width,
            height,
            4,
            2,
            scaleTransform,
            &context,
            output.data(),
            output.size()
        ),
        "identity tiled transform succeeds"
    );

    expectTrue(
        context.calls > 1,
        "identity path actually uses multiple tiles"
    );
    for (
        std::size_t index = 0;
        index < input.size();
        ++index
    ) {
        expectNear(
            output[index],
            input[index],
            1.0e-5f,
            "identity tiled output"
        );
    }
}

void testScaledTransformBlendsWithoutSeams() {
    constexpr int channels = 1;
    constexpr int width = 7;
    constexpr int height = 5;
    std::vector<float> input(channels * width * height);
    for (
        std::size_t index = 0;
        index < input.size();
        ++index
    ) {
        input[index] =
            0.5f + static_cast<float>(index) * 0.1f;
    }

    std::vector<float> output(input.size(), 0.0f);
    TransformContext context{
        2.0f,
        0,
        channels,
        4,
        false,
    };

    expectTrue(
        kira::pisa::runTiledPlanarTransform(
            input.data(),
            channels,
            width,
            height,
            4,
            1,
            scaleTransform,
            &context,
            output.data(),
            output.size()
        ),
        "scaled tiled transform succeeds"
    );

    for (
        std::size_t index = 0;
        index < input.size();
        ++index
    ) {
        expectNear(
            output[index],
            input[index] * 2.0f,
            1.0e-5f,
            "scaled tiled output"
        );
    }
}

void testPropagatesTileTransformFailure() {
    constexpr int channels = 1;
    constexpr int width = 4;
    constexpr int height = 4;
    const float input[channels * width * height] = {};
    float output[channels * width * height] = {};
    TransformContext context{
        1.0f,
        0,
        channels,
        2,
        true,
    };

    expectFalse(
        kira::pisa::runTiledPlanarTransform(
            input,
            channels,
            width,
            height,
            2,
            1,
            scaleTransform,
            &context,
            output,
            channels * width * height
        ),
        "tile transform failure propagates"
    );
    expectTrue(
        context.calls == 1,
        "runner stops after first transform failure"
    );
}

void testRejectsInvalidBuffersAndParameters() {
    const float input[16] = {};
    float output[16] = {};
    TransformContext context{
        1.0f,
        0,
        1,
        2,
        false,
    };

    expectFalse(
        kira::pisa::runTiledPlanarTransform(
            nullptr,
            1,
            4,
            4,
            2,
            1,
            scaleTransform,
            &context,
            output,
            16
        ),
        "null tiled input rejected"
    );
    expectFalse(
        kira::pisa::runTiledPlanarTransform(
            input,
            1,
            4,
            4,
            2,
            1,
            nullptr,
            &context,
            output,
            16
        ),
        "null transform rejected"
    );
    expectFalse(
        kira::pisa::runTiledPlanarTransform(
            input,
            1,
            4,
            4,
            2,
            1,
            scaleTransform,
            &context,
            output,
            15
        ),
        "short tiled output rejected"
    );
}

}  // namespace

int main() {
    testDownscaledTiledTransformReconstructsAlignedGrid();
    testUpscaledTiledTransformReconstructsNearestGrid();
    testScaledRunnerRejectsInexactTileMapping();
    testIdentityTransformReconstructsFullTensor();
    testScaledTransformBlendsWithoutSeams();
    testPropagatesTileTransformFailure();
    testRejectsInvalidBuffersAndParameters();

    if (failures != 0) {
        std::cerr
            << failures
            << " PiSA tiled inference test(s) failed\n";
        return 1;
    }

    std::cout << "PiSA tiled inference tests passed\n";
    return 0;
}
