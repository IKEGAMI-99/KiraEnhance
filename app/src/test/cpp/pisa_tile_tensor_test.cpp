#include "pisa_tile_plan.h"
#include "pisa_tile_tensor.h"

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

void testExtractsPlanarTileAcrossChannels() {
    const float input[] = {
        0, 1, 2, 3,
        4, 5, 6, 7,
        8, 9, 10, 11,

        100, 101, 102, 103,
        104, 105, 106, 107,
        108, 109, 110, 111,
    };
    const kira::pisa::TileRegion tile{
        1,
        1,
        2,
        2,
    };
    float output[8] = {};

    expectTrue(
        kira::pisa::extractPlanarTile(
            input,
            2,
            4,
            3,
            tile,
            output,
            8
        ),
        "planar tile extraction succeeds"
    );

    const float expected[] = {
        5, 6,
        9, 10,
        105, 106,
        109, 110,
    };
    for (std::size_t index = 0; index < 8; ++index) {
        expectNear(
            output[index],
            expected[index],
            0.0f,
            "extracted tile value"
        );
    }
}

void testWeightedStitchReconstructsOriginalTensor() {
    constexpr int channels = 2;
    constexpr int width = 4;
    constexpr int height = 2;
    const float input[] = {
        1, 2, 3, 4,
        5, 6, 7, 8,

        11, 12, 13, 14,
        15, 16, 17, 18,
    };

    kira::pisa::TilePlan plan;
    expectTrue(
        kira::pisa::buildTilePlan(
            width,
            height,
            2,
            1,
            plan
        ),
        "reconstruction tile plan succeeds"
    );

    std::vector<float> weights;
    expectTrue(
        kira::pisa::buildGaussianTileWeights(
            2,
            2,
            weights
        ),
        "reconstruction gaussian weights succeed"
    );

    std::vector<float> accumulator(
        channels * width * height,
        0.0f
    );
    std::vector<float> contributors(
        width * height,
        0.0f
    );
    std::vector<float> tileValues(
        channels * plan.tileSize * plan.tileSize
    );

    for (const auto& tile : plan.tiles) {
        expectTrue(
            kira::pisa::extractPlanarTile(
                input,
                channels,
                width,
                height,
                tile,
                tileValues.data(),
                tileValues.size()
            ),
            "tile extraction for reconstruction succeeds"
        );
        expectTrue(
            kira::pisa::accumulateWeightedPlanarTile(
                tileValues.data(),
                channels,
                tile,
                width,
                height,
                weights.data(),
                weights.size(),
                accumulator.data(),
                accumulator.size(),
                contributors.data(),
                contributors.size()
            ),
            "weighted tile accumulation succeeds"
        );
    }

    expectTrue(
        kira::pisa::normalizePlanarAccumulation(
            accumulator.data(),
            channels,
            width,
            height,
            contributors.data(),
            contributors.size()
        ),
        "weighted accumulation normalization succeeds"
    );

    for (
        std::size_t index = 0;
        index < accumulator.size();
        ++index
    ) {
        expectNear(
            accumulator[index],
            input[index],
            1.0e-5f,
            "stitched value reconstructs input"
        );
    }
    for (float contributor : contributors) {
        expectTrue(
            std::isfinite(contributor) &&
                contributor > 0.0f,
            "every pixel receives contribution"
        );
    }
}

void testRejectsOutOfBoundsTileAndShortBuffers() {
    const float input[8] = {};
    float output[8] = {};
    const kira::pisa::TileRegion outOfBounds{
        3,
        0,
        2,
        2,
    };

    expectFalse(
        kira::pisa::extractPlanarTile(
            input,
            1,
            4,
            2,
            outOfBounds,
            output,
            8
        ),
        "out-of-bounds tile rejected"
    );

    const kira::pisa::TileRegion valid{
        0,
        0,
        2,
        2,
    };
    expectFalse(
        kira::pisa::extractPlanarTile(
            input,
            1,
            4,
            2,
            valid,
            output,
            3
        ),
        "short tile output rejected"
    );

    float accumulator[8] = {};
    float contributors[8] = {};
    const float weights[4] = {1, 1, 1, 1};
    expectFalse(
        kira::pisa::accumulateWeightedPlanarTile(
            output,
            1,
            valid,
            4,
            2,
            weights,
            3,
            accumulator,
            8,
            contributors,
            8
        ),
        "short weight buffer rejected"
    );
}

void testNormalizationRejectsUncoveredPixels() {
    float accumulator[] = {
        1, 2,
        3, 4,
    };
    const float contributors[] = {
        1, 0,
        1, 1,
    };

    expectFalse(
        kira::pisa::normalizePlanarAccumulation(
            accumulator,
            1,
            2,
            2,
            contributors,
            4
        ),
        "zero contributor rejected"
    );
}

}  // namespace

int main() {
    testExtractsPlanarTileAcrossChannels();
    testWeightedStitchReconstructsOriginalTensor();
    testRejectsOutOfBoundsTileAndShortBuffers();
    testNormalizationRejectsUncoveredPixels();

    if (failures != 0) {
        std::cerr
            << failures
            << " PiSA tile tensor test(s) failed\n";
        return 1;
    }

    std::cout << "PiSA tile tensor tests passed\n";
    return 0;
}
