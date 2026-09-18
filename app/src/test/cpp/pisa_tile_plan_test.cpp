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
    testGaussianWeightsArePositiveSymmetricAndCenterWeighted();
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
