#include "pisa_tile_plan.h"

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

void expectEqual(int actual, int expected, const char* label) {
    if (actual != expected) {
        std::cerr
            << "FAIL: " << label
            << " expected=" << expected
            << " actual=" << actual
            << "\n";
        ++failures;
    }
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

void testBuildsOverlappingPlanWithPinnedEdges() {
    kira::pisa::TilePlan plan;
    expectTrue(
        kira::pisa::buildTilePlan(
            256,
            256,
            96,
            32,
            plan
        ),
        "square tile plan succeeds"
    );

    expectEqual(plan.tileSize, 96, "effective tile size");
    expectEqual(plan.overlap, 32, "overlap");
    expectEqual(
        static_cast<int>(plan.tiles.size()),
        16,
        "square tile count"
    );

    const auto& first = plan.tiles.front();
    expectEqual(first.x, 0, "first x");
    expectEqual(first.y, 0, "first y");
    expectEqual(first.width, 96, "first width");
    expectEqual(first.height, 96, "first height");

    const auto& last = plan.tiles.back();
    expectEqual(last.x, 160, "last x pinned to right edge");
    expectEqual(last.y, 160, "last y pinned to bottom edge");
}

void testBuildsRectangularPlan() {
    kira::pisa::TilePlan plan;
    expectTrue(
        kira::pisa::buildTilePlan(
            224,
            160,
            96,
            32,
            plan
        ),
        "rectangular tile plan succeeds"
    );

    expectEqual(
        static_cast<int>(plan.tiles.size()),
        6,
        "rectangular tile count"
    );

    const int expectedX[] = {0, 64, 128};
    const int expectedY[] = {0, 64};
    std::size_t index = 0;
    for (int y : expectedY) {
        for (int x : expectedX) {
            const auto& tile = plan.tiles[index++];
            expectEqual(tile.x, x, "rectangular x");
            expectEqual(tile.y, y, "rectangular y");
        }
    }
}

void testClampsTileToSmallerImageSide() {
    kira::pisa::TilePlan plan;
    expectTrue(
        kira::pisa::buildTilePlan(
            80,
            200,
            96,
            16,
            plan
        ),
        "small-side tile plan succeeds"
    );

    expectEqual(plan.tileSize, 80, "tile clamped to min side");
    expectEqual(
        static_cast<int>(plan.tiles.size()),
        3,
        "small-side tile count"
    );
    expectEqual(plan.tiles[0].y, 0, "small-side first y");
    expectEqual(plan.tiles[1].y, 64, "small-side middle y");
    expectEqual(plan.tiles[2].y, 120, "small-side last y");
}

void testRejectsNonProgressingOverlap() {
    kira::pisa::TilePlan plan;
    expectFalse(
        kira::pisa::buildTilePlan(
            64,
            64,
            96,
            64,
            plan
        ),
        "overlap equal to effective tile rejected"
    );
    expectFalse(
        kira::pisa::buildTilePlan(
            64,
            64,
            96,
            80,
            plan
        ),
        "overlap larger than effective tile rejected"
    );
}

void testScalesTileRegionsExactlyForVaeStages() {
    const kira::pisa::TileRegion encoderInput{
        64,
        32,
        512,
        512,
    };
    kira::pisa::TileRegion latentTile;

    expectTrue(
        kira::pisa::scaleTileRegionExact(
            encoderInput,
            1,
            8,
            256,
            192,
            latentTile
        ),
        "encoder tile scales to latent space"
    );
    expectEqual(latentTile.x, 8, "encoder scaled x");
    expectEqual(latentTile.y, 4, "encoder scaled y");
    expectEqual(latentTile.width, 64, "encoder scaled width");
    expectEqual(latentTile.height, 64, "encoder scaled height");

    kira::pisa::TileRegion decodedTile;
    expectTrue(
        kira::pisa::scaleTileRegionExact(
            latentTile,
            8,
            1,
            2048,
            1536,
            decodedTile
        ),
        "decoder tile scales to image space"
    );
    expectEqual(decodedTile.x, 64, "decoder scaled x");
    expectEqual(decodedTile.y, 32, "decoder scaled y");
    expectEqual(decodedTile.width, 512, "decoder scaled width");
    expectEqual(decodedTile.height, 512, "decoder scaled height");
}

void testRejectsInexactOrOutOfBoundsTileScaling() {
    kira::pisa::TileRegion output;

    expectFalse(
        kira::pisa::scaleTileRegionExact(
            kira::pisa::TileRegion{1, 0, 512, 512},
            1,
            8,
            256,
            256,
            output
        ),
        "unaligned encoder tile rejected"
    );
    expectFalse(
        kira::pisa::scaleTileRegionExact(
            kira::pisa::TileRegion{192, 0, 96, 96},
            1,
            1,
            256,
            256,
            output
        ),
        "scaled tile outside output bounds rejected"
    );
    expectFalse(
        kira::pisa::scaleTileRegionExact(
            kira::pisa::TileRegion{0, 0, 96, 96},
            1,
            0,
            256,
            256,
            output
        ),
        "zero scale denominator rejected"
    );
}

void testGaussianWeightsArePositiveSymmetricAndCenterWeighted() {
    std::vector<float> weights;
    expectTrue(
        kira::pisa::buildGaussianTileWeights(
            5,
            5,
            weights
        ),
        "gaussian weights succeed"
    );

    expectEqual(
        static_cast<int>(weights.size()),
        25,
        "gaussian weight count"
    );

    for (float value : weights) {
        expectTrue(std::isfinite(value), "weight finite");
        expectTrue(value > 0.0f, "weight positive");
    }

    expectNear(
        weights[0],
        weights[4],
        1.0e-6f,
        "top corners symmetric"
    );
    expectNear(
        weights[0],
        weights[20],
        1.0e-6f,
        "left corners symmetric"
    );
    expectTrue(
        weights[12] > weights[0],
        "center weight exceeds corner"
    );
}

void testTilingPolicyUsesTileThreshold() {
    expectFalse(
        kira::pisa::requiresTiling(
            96,
            96,
            96
        ),
        "exact tile size remains monolithic"
    );
    expectFalse(
        kira::pisa::requiresTiling(
            80,
            96,
            96
        ),
        "smaller latent remains monolithic"
    );
    expectFalse(
        kira::pisa::requiresTiling(
            64,
            128,
            96
        ),
        "narrow latent below tile area remains monolithic"
    );
    expectTrue(
        kira::pisa::requiresTiling(
            80,
            120,
            96
        ),
        "rectangular latent above tile area uses tiling"
    );
    expectTrue(
        kira::pisa::requiresTiling(
            97,
            96,
            96
        ),
        "width above tile threshold uses tiling"
    );
    expectTrue(
        kira::pisa::requiresTiling(
            96,
            97,
            96
        ),
        "height above tile threshold uses tiling"
    );
    expectFalse(
        kira::pisa::requiresTiling(
            0,
            96,
            96
        ),
        "invalid dimensions do not request tiling"
    );
}

void testRejectsInvalidGaussianDimensions() {
    std::vector<float> weights;
    expectFalse(
        kira::pisa::buildGaussianTileWeights(
            0,
            5,
            weights
        ),
        "zero gaussian width rejected"
    );
    expectFalse(
        kira::pisa::buildGaussianTileWeights(
            5,
            -1,
            weights
        ),
        "negative gaussian height rejected"
    );
}

}  // namespace

int main() {
    testBuildsOverlappingPlanWithPinnedEdges();
    testBuildsRectangularPlan();
    testClampsTileToSmallerImageSide();
    testRejectsNonProgressingOverlap();
    testScalesTileRegionsExactlyForVaeStages();
    testRejectsInexactOrOutOfBoundsTileScaling();
    testGaussianWeightsArePositiveSymmetricAndCenterWeighted();
    testTilingPolicyUsesTileThreshold();
    testRejectsInvalidGaussianDimensions();

    if (failures != 0) {
        std::cerr
            << failures
            << " PiSA tile plan test(s) failed\n";
        return 1;
    }

    std::cout << "PiSA tile plan tests passed\n";
    return 0;
}
