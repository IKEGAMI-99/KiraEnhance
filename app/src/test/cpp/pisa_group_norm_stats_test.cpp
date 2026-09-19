#include "pisa_group_norm_stats.h"

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

void testComputesPopulationStatsLikeUpstreamVarMean() {
    const float tile[] = {
        1.0f,
        3.0f,
        10.0f,
        14.0f,
    };

    kira::pisa::GroupNormTileStats stats;
    expectTrue(
        kira::pisa::computeGroupNormTileStats(
            tile,
            1,
            4,
            1,
            1,
            2,
            stats
        ),
        "tile stats succeed"
    );
    expectTrue(stats.means.size() == 2, "two group means");
    expectTrue(stats.variances.size() == 2, "two group variances");
    expectNear(stats.means[0], 2.0f, 1.0e-6f, "group zero mean");
    expectNear(stats.variances[0], 1.0f, 1.0e-6f, "group zero variance");
    expectNear(stats.means[1], 12.0f, 1.0e-6f, "group one mean");
    expectNear(stats.variances[1], 4.0f, 1.0e-6f, "group one variance");
    expectTrue(stats.spatialPixels == 1U, "tile pixel count");
}

void testSummarizesTilesUsingUpstreamPixelWeights() {
    kira::pisa::GroupNormTileStats first;
    first.batch = 1;
    first.groups = 2;
    first.spatialPixels = 1;
    first.means = {2.0f, 12.0f};
    first.variances = {1.0f, 4.0f};

    kira::pisa::GroupNormTileStats second;
    second.batch = 1;
    second.groups = 2;
    second.spatialPixels = 2;
    second.means = {8.0f, 26.0f};
    second.variances = {5.0f, 20.0f};

    std::vector<float> means;
    std::vector<float> variances;
    expectTrue(
        kira::pisa::summarizeGroupNormTileStats(
            {first, second},
            means,
            variances
        ),
        "summary succeeds"
    );

    expectNear(means[0], 6.0f, 1.0e-6f, "weighted group zero mean");
    expectNear(
        variances[0],
        11.0f / 3.0f,
        1.0e-6f,
        "weighted group zero variance"
    );
    expectNear(
        means[1],
        64.0f / 3.0f,
        1.0e-6f,
        "weighted group one mean"
    );
    expectNear(
        variances[1],
        44.0f / 3.0f,
        1.0e-6f,
        "weighted group one variance"
    );
}

void testAppliesSharedStatsAndAffineTransform() {
    const float tile[] = {
        1.0f,
        3.0f,
        10.0f,
        14.0f,
    };
    const float means[] = {2.0f, 12.0f};
    const float variances[] = {1.0f, 4.0f};
    const float weight[] = {2.0f, 2.0f, 0.5f, 0.5f};
    const float bias[] = {1.0f, 1.0f, -1.0f, -1.0f};
    float output[4] = {};

    expectTrue(
        kira::pisa::applyGroupNormWithStats(
            tile,
            1,
            4,
            1,
            1,
            2,
            means,
            variances,
            weight,
            bias,
            1.0e-6f,
            output,
            4
        ),
        "group norm apply succeeds"
    );

    expectNear(output[0], -0.999999f, 2.0e-6f, "channel zero");
    expectNear(output[1], 2.999999f, 2.0e-6f, "channel one");
    expectNear(output[2], -1.5f, 2.0e-6f, "channel two");
    expectNear(output[3], -0.5f, 2.0e-6f, "channel three");
}

void testNormalizesTilesUsingOneSharedDistribution() {
    const float first[] = {1.0f, 3.0f};
    const float second[] = {5.0f, 7.0f};
    float firstOutput[2] = {};
    float secondOutput[2] = {};
    const float weight[] = {2.0f, 2.0f};
    const float bias[] = {1.0f, 1.0f};

    const std::vector<kira::pisa::GroupNormTileBuffer> tiles = {
        {first, firstOutput, 1, 2, 1, 1, 2},
        {second, secondOutput, 1, 2, 1, 1, 2},
    };

    expectTrue(
        kira::pisa::normalizeGroupNormTiles(
            tiles,
            1,
            weight,
            bias,
            1.0e-6f
        ),
        "shared tile group norm succeeds"
    );

    const float denominator = std::sqrt(5.0f + 1.0e-6f);
    expectNear(
        firstOutput[0],
        ((1.0f - 4.0f) / denominator) * 2.0f + 1.0f,
        2.0e-6f,
        "shared norm first tile channel zero"
    );
    expectNear(
        firstOutput[1],
        ((3.0f - 4.0f) / denominator) * 2.0f + 1.0f,
        2.0e-6f,
        "shared norm first tile channel one"
    );
    expectNear(
        secondOutput[0],
        ((5.0f - 4.0f) / denominator) * 2.0f + 1.0f,
        2.0e-6f,
        "shared norm second tile channel zero"
    );
    expectNear(
        secondOutput[1],
        ((7.0f - 4.0f) / denominator) * 2.0f + 1.0f,
        2.0e-6f,
        "shared norm second tile channel one"
    );
}

void testRejectsInvalidContracts() {
    const float input[] = {1.0f, 2.0f, 3.0f};
    kira::pisa::GroupNormTileStats stats;

    expectFalse(
        kira::pisa::computeGroupNormTileStats(
            input,
            1,
            3,
            1,
            1,
            2,
            stats
        ),
        "channels must divide groups"
    );

    std::vector<float> means;
    std::vector<float> variances;
    expectFalse(
        kira::pisa::summarizeGroupNormTileStats(
            {},
            means,
            variances
        ),
        "empty summary rejected"
    );
}

}  // namespace

int main() {
    testComputesPopulationStatsLikeUpstreamVarMean();
    testSummarizesTilesUsingUpstreamPixelWeights();
    testAppliesSharedStatsAndAffineTransform();
    testNormalizesTilesUsingOneSharedDistribution();
    testRejectsInvalidContracts();

    if (failures != 0) {
        std::cerr
            << failures
            << " PiSA group norm stats test(s) failed\n";
        return 1;
    }

    std::cout << "PiSA group norm stats tests passed\n";
    return 0;
}
