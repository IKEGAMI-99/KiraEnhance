#include "pisa_latent_math.h"

#include <cmath>
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

void testBasicSamplingAndScaling() {
    const float logFour = std::log(4.0f);
    const float moments[] = {
        1.0f, -1.0f,
        0.0f, logFour,
    };
    const float noise[] = {0.5f, -0.25f};
    float output[] = {0.0f, 0.0f};
    constexpr float scale = 0.18215f;

    expectTrue(
        kira::pisa::sampleLatentFromMoments(
            moments,
            noise,
            output,
            1,
            1,
            1,
            2,
            scale
        ),
        "basic sampling succeeds"
    );

    expectNear(output[0], 1.5f * scale, 1.0e-6f, "mean + unit std noise");
    expectNear(output[1], -1.5f * scale, 1.0e-6f, "std uses exp(0.5 logvar)");
}

void testLogVarianceClampMatchesDiffusers() {
    const float moments[] = {
        0.0f, 0.0f,
        -100.0f, 100.0f,
    };
    const float noise[] = {1.0f, 1.0f};
    float output[] = {0.0f, 0.0f};

    expectTrue(
        kira::pisa::sampleLatentFromMoments(
            moments,
            noise,
            output,
            1,
            1,
            1,
            2,
            1.0f
        ),
        "clamp sampling succeeds"
    );

    expectNear(output[0], std::exp(-15.0f), 1.0e-9f, "logvar lower clamp -30");
    expectNear(output[1], std::exp(10.0f), 1.0e-2f, "logvar upper clamp 20");
}

void testBatchLayoutSplitsMomentsAlongChannels() {
    const float moments[] = {
        10.0f, 0.0f,
        20.0f, 0.0f,
    };
    const float noise[] = {1.0f, -1.0f};
    float output[] = {0.0f, 0.0f};

    expectTrue(
        kira::pisa::sampleLatentFromMoments(
            moments,
            noise,
            output,
            2,
            1,
            1,
            1,
            1.0f
        ),
        "batched sampling succeeds"
    );

    expectNear(output[0], 11.0f, 1.0e-6f, "batch zero layout");
    expectNear(output[1], 19.0f, 1.0e-6f, "batch one layout");
}

void testRejectsInvalidArguments() {
    const float value = 0.0f;
    float output = 0.0f;

    expectFalse(
        kira::pisa::sampleLatentFromMoments(
            nullptr, &value, &output, 1, 1, 1, 1, 1.0f
        ),
        "null moments rejected"
    );
    expectFalse(
        kira::pisa::sampleLatentFromMoments(
            &value, nullptr, &output, 1, 1, 1, 1, 1.0f
        ),
        "null noise rejected"
    );
    expectFalse(
        kira::pisa::sampleLatentFromMoments(
            &value, &value, nullptr, 1, 1, 1, 1, 1.0f
        ),
        "null output rejected"
    );
    expectFalse(
        kira::pisa::sampleLatentFromMoments(
            &value, &value, &output, 0, 1, 1, 1, 1.0f
        ),
        "zero batch rejected"
    );
    expectFalse(
        kira::pisa::sampleLatentFromMoments(
            &value, &value, &output, 1, 0, 1, 1, 1.0f
        ),
        "zero channels rejected"
    );
    expectFalse(
        kira::pisa::sampleLatentFromMoments(
            &value, &value, &output, 1, 1, 0, 1, 1.0f
        ),
        "zero height rejected"
    );
    expectFalse(
        kira::pisa::sampleLatentFromMoments(
            &value, &value, &output, 1, 1, 1, 0, 1.0f
        ),
        "zero width rejected"
    );
    expectFalse(
        kira::pisa::sampleLatentFromMoments(
            &value, &value, &output, 1, 1, 1, 1, 0.0f
        ),
        "zero scale rejected"
    );
    expectFalse(
        kira::pisa::sampleLatentFromMoments(
            &value,
            &value,
            &output,
            1,
            1,
            1,
            1,
            std::numeric_limits<float>::infinity()
        ),
        "non-finite scale rejected"
    );
}

}  // namespace

int main() {
    testBasicSamplingAndScaling();
    testLogVarianceClampMatchesDiffusers();
    testBatchLayoutSplitsMomentsAlongChannels();
    testRejectsInvalidArguments();

    if (failures != 0) {
        std::cerr << failures << " PiSA latent math test(s) failed\n";
        return 1;
    }

    std::cout << "PiSA latent math tests passed\n";
    return 0;
}
