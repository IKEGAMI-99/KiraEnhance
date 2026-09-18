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
