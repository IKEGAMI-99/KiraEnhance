#include "pisa_adain.h"

#include <cmath>
#include <cstddef>
#include <iostream>

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

void meanStd(
    const float* values,
    std::size_t count,
    float& mean,
    float& stddev
) {
    double sum = 0.0;
    for (std::size_t index = 0; index < count; ++index) {
        sum += values[index];
    }
    const double m = sum / static_cast<double>(count);

    double squared = 0.0;
    for (std::size_t index = 0; index < count; ++index) {
        const double delta = values[index] - m;
        squared += delta * delta;
    }

    mean = static_cast<float>(m);
    stddev = static_cast<float>(
        std::sqrt(squared / static_cast<double>(count - 1))
    );
}

void testIdentityWhenTargetMatchesSource() {
    const float input[] = {
        0.10f, 0.25f, 0.55f, 0.80f,
        0.20f, 0.35f, 0.50f, 0.65f,
        0.15f, 0.30f, 0.60f, 0.90f,
    };
    float output[12] = {};

    expectTrue(
        kira::pisa::adainColorFixRgbPlanar(
            input,
            4,
            input,
            4,
            output
        ),
        "identity color fix succeeds"
    );

    for (std::size_t index = 0; index < 12; ++index) {
        expectNear(
            output[index],
            input[index],
            1.0e-6f,
            "identity preserves value"
        );
    }
}

void testMatchesSourceChannelStatistics() {
    const float target[] = {
        0.20f, 0.30f, 0.40f, 0.50f,
        0.30f, 0.40f, 0.50f, 0.60f,
        0.15f, 0.35f, 0.55f, 0.75f,
    };
    const float source[] = {
        0.35f, 0.40f, 0.45f, 0.50f, 0.55f,
        0.20f, 0.30f, 0.40f, 0.50f, 0.60f,
        0.45f, 0.50f, 0.55f, 0.60f, 0.65f,
    };
    float output[12] = {};

    expectTrue(
        kira::pisa::adainColorFixRgbPlanar(
            target,
            4,
            source,
            5,
            output
        ),
        "different-size color fix succeeds"
    );

    for (std::size_t channel = 0; channel < 3; ++channel) {
        float outputMean = 0.0f;
        float outputStd = 0.0f;
        float sourceMean = 0.0f;
        float sourceStd = 0.0f;
        meanStd(
            output + channel * 4,
            4,
            outputMean,
            outputStd
        );
        meanStd(
            source + channel * 5,
            5,
            sourceMean,
            sourceStd
        );

        expectNear(
            outputMean,
            sourceMean,
            1.0e-5f,
            "aligned mean matches source"
        );
        expectNear(
            outputStd,
            sourceStd,
            2.0e-4f,
            "aligned std matches source"
        );
    }
}

void testSupportsInPlaceTargetOutput() {
    float target[] = {
        0.25f, 0.50f, 0.75f,
        0.20f, 0.50f, 0.80f,
        0.10f, 0.50f, 0.90f,
    };
    const float source[] = {
        0.40f, 0.50f, 0.60f,
        0.30f, 0.50f, 0.70f,
        0.20f, 0.50f, 0.80f,
    };

    expectTrue(
        kira::pisa::adainColorFixRgbPlanar(
            target,
            3,
            source,
            3,
            target
        ),
        "in-place target output succeeds"
    );

    expectNear(target[1], 0.50f, 1.0e-5f, "red center aligned");
    expectNear(target[4], 0.50f, 1.0e-5f, "green center aligned");
    expectNear(target[7], 0.50f, 1.0e-5f, "blue center aligned");
}

void testRejectsInvalidInputs() {
    const float values[6] = {
        0.0f, 1.0f,
        0.0f, 1.0f,
        0.0f, 1.0f,
    };
    float output[6] = {};

    expectFalse(
        kira::pisa::adainColorFixRgbPlanar(
            nullptr,
            2,
            values,
            2,
            output
        ),
        "null target rejected"
    );
    expectFalse(
        kira::pisa::adainColorFixRgbPlanar(
            values,
            1,
            values,
            2,
            output
        ),
        "single target pixel rejected"
    );
    expectFalse(
        kira::pisa::adainColorFixRgbPlanar(
            values,
            2,
            values,
            2,
            output,
            0.0f
        ),
        "non-positive epsilon rejected"
    );
}

}  // namespace

int main() {
    testIdentityWhenTargetMatchesSource();
    testMatchesSourceChannelStatistics();
    testSupportsInPlaceTargetOutput();
    testRejectsInvalidInputs();

    if (failures != 0) {
        std::cerr
            << failures
            << " PiSA AdaIN color test(s) failed\n";
        return 1;
    }

    std::cout << "PiSA AdaIN color tests passed\n";
    return 0;
}
