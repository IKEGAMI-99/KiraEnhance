#include "pisa_tensor_fingerprint.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>

namespace {

int failures = 0;

void expectTrue(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void expectNear(
    double actual,
    double expected,
    double tolerance,
    const char* message
) {
    if (std::abs(actual - expected) > tolerance) {
        std::cerr
            << "FAIL: " << message
            << " actual=" << actual
            << " expected=" << expected
            << '\n';
        ++failures;
    }
}

void testFiniteStatisticsAndStableHash() {
    const float values[] = {0.0f, 1.0f, -2.5f};
    const auto fingerprint =
        kira::pisa::makeTensorFingerprint(values, 3);

    expectTrue(
        fingerprint.elementCount == 3,
        "element count"
    );
    expectTrue(
        fingerprint.finiteCount == 3,
        "finite count"
    );
    expectTrue(
        fingerprint.nanCount == 0,
        "nan count"
    );
    expectNear(
        fingerprint.minimumFinite,
        -2.5,
        1.0e-7,
        "minimum finite"
    );
    expectNear(
        fingerprint.maximumFinite,
        1.0,
        1.0e-7,
        "maximum finite"
    );
    expectNear(
        fingerprint.meanFinite,
        -0.5,
        1.0e-12,
        "mean finite"
    );
    expectNear(
        fingerprint.rmsFinite,
        std::sqrt(7.25 / 3.0),
        1.0e-12,
        "rms finite"
    );
    expectTrue(
        fingerprint.fnv1a64 == 0xe1c67ad3bea76428ULL,
        "canonical float-byte FNV-1a hash"
    );
}

void testSpecialValueCounts() {
    const float values[] = {
        2.0f,
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
    };
    const auto fingerprint =
        kira::pisa::makeTensorFingerprint(values, 4);

    expectTrue(
        fingerprint.elementCount == 4,
        "special element count"
    );
    expectTrue(
        fingerprint.finiteCount == 1,
        "special finite count"
    );
    expectTrue(
        fingerprint.nanCount == 1,
        "special nan count"
    );
    expectTrue(
        fingerprint.positiveInfinityCount == 1,
        "positive infinity count"
    );
    expectTrue(
        fingerprint.negativeInfinityCount == 1,
        "negative infinity count"
    );
    expectNear(
        fingerprint.meanFinite,
        2.0,
        1.0e-12,
        "special mean"
    );
    expectNear(
        fingerprint.rmsFinite,
        2.0,
        1.0e-12,
        "special rms"
    );
}

void testNullAndEmptyInput() {
    const auto empty =
        kira::pisa::makeTensorFingerprint(nullptr, 0);
    expectTrue(empty.elementCount == 0, "empty count");
    expectTrue(empty.finiteCount == 0, "empty finite count");
    expectTrue(
        empty.fnv1a64 == 14695981039346656037ULL,
        "empty hash offset"
    );
}

}  // namespace

int main() {
    testFiniteStatisticsAndStableHash();
    testSpecialValueCounts();
    testNullAndEmptyInput();

    if (failures != 0) {
        std::cerr << failures << " test(s) failed\n";
        return 1;
    }

    std::cout << "PiSA tensor fingerprint tests passed\n";
    return 0;
}
