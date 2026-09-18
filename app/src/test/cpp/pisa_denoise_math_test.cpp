#include "pisa_denoise_math.h"

#include <cmath>
#include <cstddef>
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

void testSubtractsPredictionThenUnscalesForVaeDecode() {
    const float encoded[] = {0.5f, -0.25f, 1.0f};
    const float predicted[] = {0.1f, 0.25f, -0.5f};
    float output[3] = {};
    constexpr float scale = 0.18215f;

    expectTrue(
        kira::pisa::buildDecoderLatent(
            encoded,
            predicted,
            output,
            3,
            scale
        ),
        "decoder latent math succeeds"
    );

    expectNear(output[0], 0.4f / scale, 1.0e-5f, "first latent");
    expectNear(output[1], -0.5f / scale, 1.0e-5f, "second latent");
    expectNear(output[2], 1.5f / scale, 1.0e-5f, "third latent");
}

void testSupportsInPlaceOutput() {
    float encoded[] = {0.3f, 0.7f};
    const float predicted[] = {0.1f, 0.2f};

    expectTrue(
        kira::pisa::buildDecoderLatent(
            encoded,
            predicted,
            encoded,
            2,
            0.5f
        ),
        "in-place decoder latent succeeds"
    );

    expectNear(encoded[0], 0.4f, 1.0e-6f, "in-place first");
    expectNear(encoded[1], 1.0f, 1.0e-6f, "in-place second");
}

void testRejectsInvalidArguments() {
    const float value = 1.0f;
    float output = 0.0f;

    expectFalse(
        kira::pisa::buildDecoderLatent(
            nullptr, &value, &output, 1, 1.0f
        ),
        "null encoded rejected"
    );
    expectFalse(
        kira::pisa::buildDecoderLatent(
            &value, nullptr, &output, 1, 1.0f
        ),
        "null prediction rejected"
    );
    expectFalse(
        kira::pisa::buildDecoderLatent(
            &value, &value, nullptr, 1, 1.0f
        ),
        "null output rejected"
    );
    expectFalse(
        kira::pisa::buildDecoderLatent(
            &value, &value, &output, 0, 1.0f
        ),
        "zero count rejected"
    );
    expectFalse(
        kira::pisa::buildDecoderLatent(
            &value, &value, &output, 1, 0.0f
        ),
        "zero scale rejected"
    );
    expectFalse(
        kira::pisa::buildDecoderLatent(
            &value,
            &value,
            &output,
            1,
            std::numeric_limits<float>::infinity()
        ),
        "non-finite scale rejected"
    );
}

}  // namespace

int main() {
    testSubtractsPredictionThenUnscalesForVaeDecode();
    testSupportsInPlaceOutput();
    testRejectsInvalidArguments();

    if (failures != 0) {
        std::cerr << failures << " PiSA denoise math test(s) failed\n";
        return 1;
    }

    std::cout << "PiSA denoise math tests passed\n";
    return 0;
}
