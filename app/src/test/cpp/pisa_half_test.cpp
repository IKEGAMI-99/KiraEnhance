#include "pisa_half.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>

namespace {

int failures = 0;

void expectTrue(bool value, const char* label) {
    if (!value) {
        std::cerr << "FAIL: " << label << "\n";
        ++failures;
    }
}

void expectEqual(
    std::uint16_t actual,
    std::uint16_t expected,
    const char* label
) {
    if (actual != expected) {
        std::cerr
            << "FAIL: " << label
            << " expected=0x" << std::hex << expected
            << " actual=0x" << actual
            << std::dec << "\n";
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

void testKnownEncodings() {
    expectEqual(kira::pisa::floatToHalf(0.0f), 0x0000U, "positive zero");
    expectEqual(kira::pisa::floatToHalf(-0.0f), 0x8000U, "negative zero");
    expectEqual(kira::pisa::floatToHalf(1.0f), 0x3c00U, "one");
    expectEqual(kira::pisa::floatToHalf(-2.0f), 0xc000U, "negative two");
    expectEqual(kira::pisa::floatToHalf(65504.0f), 0x7bffU, "max finite");
    expectEqual(
        kira::pisa::floatToHalf(std::numeric_limits<float>::infinity()),
        0x7c00U,
        "positive infinity"
    );
}

void testKnownDecodings() {
    expectNear(kira::pisa::halfToFloat(0x3c00U), 1.0f, 0.0f, "decode one");
    expectNear(kira::pisa::halfToFloat(0xc000U), -2.0f, 0.0f, "decode negative two");
    expectNear(
        kira::pisa::halfToFloat(0x0400U),
        0.00006103515625f,
        0.0f,
        "minimum normal"
    );
    expectNear(
        kira::pisa::halfToFloat(0x0001U),
        5.960464477539063e-8f,
        1.0e-12f,
        "minimum subnormal"
    );
}

void testRoundTripRepresentativeValues() {
    const float values[] = {
        -10.25f,
        -1.0f,
        -0.33325f,
        0.0f,
        0.125f,
        1.0f,
        12.75f,
        1000.0f,
    };

    for (float value : values) {
        const float roundTrip =
            kira::pisa::halfToFloat(kira::pisa::floatToHalf(value));
        const float tolerance =
            std::max(1.0e-4f, std::fabs(value) * 0.001f);
        expectNear(roundTrip, value, tolerance, "representative round trip");
    }
}

void testArrayConversion() {
    const float input[] = {1.0f, -2.0f, 0.5f};
    std::uint16_t halfs[3] = {};
    float output[3] = {};

    expectTrue(
        kira::pisa::floatsToHalfs(input, halfs, 3),
        "float array to half"
    );
    expectTrue(
        kira::pisa::halfsToFloats(halfs, output, 3),
        "half array to float"
    );

    expectNear(output[0], 1.0f, 0.0f, "array first");
    expectNear(output[1], -2.0f, 0.0f, "array second");
    expectNear(output[2], 0.5f, 0.0f, "array third");
}

void testRejectsNullPointers() {
    float value = 1.0f;
    std::uint16_t half = 0;

    expectTrue(
        !kira::pisa::floatsToHalfs(nullptr, &half, 1),
        "null float input rejected"
    );
    expectTrue(
        !kira::pisa::floatsToHalfs(&value, nullptr, 1),
        "null half output rejected"
    );
    expectTrue(
        !kira::pisa::halfsToFloats(nullptr, &value, 1),
        "null half input rejected"
    );
    expectTrue(
        !kira::pisa::halfsToFloats(&half, nullptr, 1),
        "null float output rejected"
    );
}

}  // namespace

int main() {
    testKnownEncodings();
    testKnownDecodings();
    testRoundTripRepresentativeValues();
    testArrayConversion();
    testRejectsNullPointers();

    if (failures != 0) {
        std::cerr << failures << " PiSA half conversion test(s) failed\n";
        return 1;
    }

    std::cout << "PiSA half conversion tests passed\n";
    return 0;
}
