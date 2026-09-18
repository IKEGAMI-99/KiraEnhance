#include "pisa_gaussian_noise.h"

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

void expectNear(float actual, float expected, float tolerance, const char* label) {
    if (std::fabs(actual - expected) > tolerance) {
        std::cerr
            << "FAIL: " << label
            << " expected=" << expected
            << " actual=" << actual
            << "\n";
        ++failures;
    }
}

void testSeed42PrefixIsStable() {
    float values[6] = {};
    expectTrue(
        kira::pisa::fillGaussianNoise(values, 6, 42),
        "seed 42 generation succeeds"
    );

    const float expected[6] = {
        0.4147197604f,
        0.6526812315f,
        -0.8918862343f,
        1.3268336058f,
        1.7295930386f,
        -1.8834167719f,
    };

    for (std::size_t index = 0; index < 6; ++index) {
        expectNear(
            values[index],
            expected[index],
            1.0e-6f,
            "seed 42 stable prefix"
        );
    }
}

void testSameSeedRepeatsAndDifferentSeedChanges() {
    float first[9] = {};
    float second[9] = {};
    float different[9] = {};

    expectTrue(
        kira::pisa::fillGaussianNoise(first, 9, 123456789ULL),
        "first deterministic generation succeeds"
    );
    expectTrue(
        kira::pisa::fillGaussianNoise(second, 9, 123456789ULL),
        "second deterministic generation succeeds"
    );
    expectTrue(
        kira::pisa::fillGaussianNoise(different, 9, 987654321ULL),
        "different seed generation succeeds"
    );

    bool anyDifferent = false;
    for (std::size_t index = 0; index < 9; ++index) {
        expectTrue(first[index] == second[index], "same seed repeats exactly");
        if (first[index] != different[index]) {
            anyDifferent = true;
        }
    }
    expectTrue(anyDifferent, "different seed changes output");
}

void testDistributionIsReasonablyStandardNormal() {
    constexpr std::size_t count = 20000;
    std::vector<float> values(count);

    expectTrue(
        kira::pisa::fillGaussianNoise(values.data(), values.size(), 42),
        "distribution generation succeeds"
    );

    double sum = 0.0;
    double squaredSum = 0.0;
    for (float value : values) {
        expectTrue(std::isfinite(value), "gaussian output is finite");
        sum += value;
        squaredSum += static_cast<double>(value) * value;
    }

    const double mean = sum / static_cast<double>(count);
    const double variance =
        squaredSum / static_cast<double>(count) - mean * mean;

    expectTrue(std::fabs(mean) < 0.03, "gaussian mean near zero");
    expectTrue(variance > 0.95 && variance < 1.05, "gaussian variance near one");
}

void testRejectsInvalidArguments() {
    float output = 0.0f;
    expectFalse(
        kira::pisa::fillGaussianNoise(nullptr, 1, 42),
        "null output rejected"
    );
    expectFalse(
        kira::pisa::fillGaussianNoise(&output, 0, 42),
        "zero count rejected"
    );
}

}  // namespace

int main() {
    testSeed42PrefixIsStable();
    testSameSeedRepeatsAndDifferentSeedChanges();
    testDistributionIsReasonablyStandardNormal();
    testRejectsInvalidArguments();

    if (failures != 0) {
        std::cerr << failures << " PiSA Gaussian noise test(s) failed\n";
        return 1;
    }

    std::cout << "PiSA Gaussian noise tests passed\n";
    return 0;
}
