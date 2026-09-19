#include "pisa_tiled_vae.h"

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

struct TransformContext {
    bool decoder = false;
    int calls = 0;
};

bool fillScaledTile(
    const float* input,
    std::size_t inputCount,
    int inputChannels,
    int inputWidth,
    int inputHeight,
    float* output,
    std::size_t outputCount,
    int outputChannels,
    int outputWidth,
    int outputHeight,
    void* rawContext
) {
    auto* context =
        static_cast<TransformContext*>(rawContext);
    if (
        context == nullptr ||
        input == nullptr ||
        output == nullptr ||
        inputCount == 0 ||
        inputChannels <= 0 ||
        outputChannels <= 0
    ) {
        return false;
    }

    const int expectedWidth =
        context->decoder
            ? inputWidth * 8
            : inputWidth / 8;
    const int expectedHeight =
        context->decoder
            ? inputHeight * 8
            : inputHeight / 8;
    if (
        outputWidth != expectedWidth ||
        outputHeight != expectedHeight
    ) {
        return false;
    }

    ++context->calls;
    const float value =
        static_cast<float>(context->calls);
    for (std::size_t index = 0; index < outputCount; ++index) {
        output[index] = value;
    }
    return true;
}

void testRunsEncoderTilesAndStitchesExactCoverage() {
    constexpr int width = 128;
    constexpr int height = 64;
    std::vector<float> input(
        static_cast<std::size_t>(width) * height,
        0.25f
    );
    constexpr int outputWidth = width / 8;
    constexpr int outputHeight = height / 8;
    std::vector<float> output(
        static_cast<std::size_t>(2) *
            outputWidth *
            outputHeight,
        0.0f
    );

    TransformContext context;
    context.decoder = false;
    expectTrue(
        kira::pisa::runTiledVaePlanarTransform(
            input.data(),
            1,
            width,
            height,
            64,
            16,
            false,
            2,
            fillScaledTile,
            &context,
            output.data(),
            output.size()
        ),
        "encoder tiled VAE succeeds"
    );
    expectTrue(context.calls > 1, "encoder uses multiple tiles");
    for (float value : output) {
        expectTrue(
            std::isfinite(value) && value >= 1.0f,
            "encoder output written exactly"
        );
    }
}

void testRunsDecoderTilesAndStitchesExactCoverage() {
    constexpr int width = 8;
    constexpr int height = 4;
    std::vector<float> input(
        static_cast<std::size_t>(4) * width * height,
        0.5f
    );
    constexpr int outputWidth = width * 8;
    constexpr int outputHeight = height * 8;
    std::vector<float> output(
        static_cast<std::size_t>(3) *
            outputWidth *
            outputHeight,
        0.0f
    );

    TransformContext context;
    context.decoder = true;
    expectTrue(
        kira::pisa::runTiledVaePlanarTransform(
            input.data(),
            4,
            width,
            height,
            4,
            1,
            true,
            3,
            fillScaledTile,
            &context,
            output.data(),
            output.size()
        ),
        "decoder tiled VAE succeeds"
    );
    expectTrue(context.calls > 1, "decoder uses multiple tiles");
    for (float value : output) {
        expectTrue(
            std::isfinite(value) && value >= 1.0f,
            "decoder output written exactly"
        );
    }
}

void testRejectsSmallOutputBuffer() {
    constexpr float input[64] = {};
    float output[1] = {};
    TransformContext context;

    expectFalse(
        kira::pisa::runTiledVaePlanarTransform(
            input,
            1,
            8,
            8,
            8,
            0,
            false,
            1,
            fillScaledTile,
            &context,
            output,
            0
        ),
        "small output buffer rejected"
    );
}

}  // namespace

int main() {
    testRunsEncoderTilesAndStitchesExactCoverage();
    testRunsDecoderTilesAndStitchesExactCoverage();
    testRejectsSmallOutputBuffer();

    if (failures != 0) {
        std::cerr
            << failures
            << " PiSA tiled VAE test(s) failed\n";
        return 1;
    }

    std::cout << "PiSA tiled VAE tests passed\n";
    return 0;
}
