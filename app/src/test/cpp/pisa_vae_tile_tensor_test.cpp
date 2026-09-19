#include "pisa_vae_tile_plan.h"
#include "pisa_vae_tile_tensor.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
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

void fillTile(
    std::vector<float>& values,
    int channels,
    int width,
    int height,
    float base
) {
    const std::size_t pixels =
        static_cast<std::size_t>(width) * height;
    values.resize(
        static_cast<std::size_t>(channels) * pixels
    );
    for (int channel = 0; channel < channels; ++channel) {
        for (std::size_t index = 0; index < pixels; ++index) {
            values[
                static_cast<std::size_t>(channel) *
                    pixels +
                index
            ] =
                base +
                static_cast<float>(channel * 1000) +
                static_cast<float>(index);
        }
    }
}

void testDecoderTilesFillOutputExactlyOnce() {
    kira::pisa::VaeTilePlan plan;
    expectTrue(
        kira::pisa::buildVaeTilePlan(
            8,
            4,
            4,
            1,
            true,
            plan
        ),
        "small decoder tile plan succeeds"
    );

    constexpr int channels = 2;
    const std::size_t outputPixels =
        static_cast<std::size_t>(plan.outputWidth) *
        plan.outputHeight;
    std::vector<float> output(
        static_cast<std::size_t>(channels) *
            outputPixels,
        -1.0f
    );
    std::vector<std::uint8_t> coverage(
        outputPixels,
        0U
    );

    float base = 10.0f;
    for (const auto& tile : plan.tiles) {
        const int tileWidth = tile.input.width * 8;
        const int tileHeight = tile.input.height * 8;
        kira::pisa::TileRegion crop;
        expectTrue(
            kira::pisa::localOutputCropForVaeTile(
                tile,
                true,
                crop
            ),
            "decoder local crop succeeds"
        );

        std::vector<float> tileValues;
        fillTile(
            tileValues,
            channels,
            tileWidth,
            tileHeight,
            base
        );
        expectTrue(
            kira::pisa::copyCroppedPlanarTile(
                tileValues.data(),
                channels,
                tileWidth,
                tileHeight,
                crop,
                tile.output,
                plan.outputWidth,
                plan.outputHeight,
                output.data(),
                output.size(),
                coverage.data(),
                coverage.size()
            ),
            "decoder cropped tile copy succeeds"
        );
        base += 100.0f;
    }

    expectTrue(
        kira::pisa::validateExactCoverage(
            coverage.data(),
            coverage.size()
        ),
        "decoder output has exact coverage"
    );
    for (float value : output) {
        expectTrue(
            std::isfinite(value) && value >= 0.0f,
            "decoder output pixel written"
        );
    }
}

void testEncoderTilesFillOutputExactlyOnce() {
    kira::pisa::VaeTilePlan plan;
    expectTrue(
        kira::pisa::buildVaeTilePlan(
            128,
            64,
            64,
            16,
            false,
            plan
        ),
        "small encoder tile plan succeeds"
    );

    constexpr int channels = 1;
    const std::size_t outputPixels =
        static_cast<std::size_t>(plan.outputWidth) *
        plan.outputHeight;
    std::vector<float> output(outputPixels, -1.0f);
    std::vector<std::uint8_t> coverage(
        outputPixels,
        0U
    );

    float base = 1.0f;
    for (const auto& tile : plan.tiles) {
        const int tileWidth =
            (tile.input.x + tile.input.width) / 8 -
            tile.input.x / 8;
        const int tileHeight =
            (tile.input.y + tile.input.height) / 8 -
            tile.input.y / 8;
        kira::pisa::TileRegion crop;
        expectTrue(
            kira::pisa::localOutputCropForVaeTile(
                tile,
                false,
                crop
            ),
            "encoder local crop succeeds"
        );

        std::vector<float> tileValues;
        fillTile(
            tileValues,
            channels,
            tileWidth,
            tileHeight,
            base
        );
        expectTrue(
            kira::pisa::copyCroppedPlanarTile(
                tileValues.data(),
                channels,
                tileWidth,
                tileHeight,
                crop,
                tile.output,
                plan.outputWidth,
                plan.outputHeight,
                output.data(),
                output.size(),
                coverage.data(),
                coverage.size()
            ),
            "encoder cropped tile copy succeeds"
        );
        base += 100.0f;
    }

    expectTrue(
        kira::pisa::validateExactCoverage(
            coverage.data(),
            coverage.size()
        ),
        "encoder output has exact coverage"
    );
}

void testRejectsOverlapBeforeWritingSecondTile() {
    const float tileValues[4] = {1, 2, 3, 4};
    float output[4] = {};
    std::uint8_t coverage[4] = {};

    const kira::pisa::TileRegion full{0, 0, 2, 2};
    expectTrue(
        kira::pisa::copyCroppedPlanarTile(
            tileValues,
            1,
            2,
            2,
            full,
            full,
            2,
            2,
            output,
            4,
            coverage,
            4
        ),
        "first exact tile copy succeeds"
    );

    expectFalse(
        kira::pisa::copyCroppedPlanarTile(
            tileValues,
            1,
            2,
            2,
            full,
            full,
            2,
            2,
            output,
            4,
            coverage,
            4
        ),
        "overlapping second tile rejected"
    );
}

void testRejectsNonFiniteTileValue() {
    const float tileValues[4] = {
        1.0f,
        2.0f,
        3.0f,
        std::nanf(""),
    };
    float output[4] = {};
    std::uint8_t coverage[4] = {};
    const kira::pisa::TileRegion full{0, 0, 2, 2};

    expectFalse(
        kira::pisa::copyCroppedPlanarTile(
            tileValues,
            1,
            2,
            2,
            full,
            full,
            2,
            2,
            output,
            4,
            coverage,
            4
        ),
        "non-finite VAE tile rejected"
    );
}

void testCoverageValidatorRejectsHoles() {
    const std::uint8_t incomplete[4] = {1, 1, 0, 1};
    expectFalse(
        kira::pisa::validateExactCoverage(
            incomplete,
            4
        ),
        "coverage hole rejected"
    );
}

}  // namespace

int main() {
    testDecoderTilesFillOutputExactlyOnce();
    testEncoderTilesFillOutputExactlyOnce();
    testRejectsOverlapBeforeWritingSecondTile();
    testRejectsNonFiniteTileValue();
    testCoverageValidatorRejectsHoles();

    if (failures != 0) {
        std::cerr
            << failures
            << " PiSA VAE tile tensor test(s) failed\n";
        return 1;
    }

    std::cout << "PiSA VAE tile tensor tests passed\n";
    return 0;
}
