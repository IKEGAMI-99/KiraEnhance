#include "pisa_preprocess.h"

#include <cstddef>
#include <cstdint>
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

std::uint64_t fnv1a64(
    const std::uint8_t* bytes,
    std::size_t count
) {
    std::uint64_t hash = 14695981039346656037ULL;
    for (std::size_t index = 0; index < count; ++index) {
        hash ^= static_cast<std::uint64_t>(bytes[index]);
        hash *= 1099511628211ULL;
    }
    return hash;
}

constexpr std::uint8_t SOURCE[] = {
    0, 20, 255, 1,
    80, 40, 200, 2,
    255, 60, 0, 3,

    30, 220, 10, 4,
    128, 128, 128, 5,
    240, 250, 64, 6,
};

void testSmallInputChainMatchesPillowFixture() {
    const kira::pisa::ResizePlan plan{
        3,
        2,
        9,
        6,
        18,
        12,
        16,
        12,
        6,
        4,
        true,
    };

    std::uint8_t output[16 * 12 * 4] = {};
    expectTrue(
        kira::pisa::preparePisaModelRgba(
            SOURCE,
            3 * 4,
            plan,
            output,
            16 * 4,
            sizeof(output)
        ),
        "small PiSA preprocessing succeeds"
    );

    expectTrue(
        fnv1a64(output, sizeof(output)) ==
            0xa38ba71ff181f1bdULL,
        "small PiSA preprocessing matches Pillow bytes"
    );
}

void testAlignedInputSkipsLanczosCorrection() {
    const kira::pisa::ResizePlan plan{
        3,
        2,
        3,
        2,
        6,
        4,
        6,
        4,
        6,
        4,
        false,
    };

    std::uint8_t output[6 * 4 * 4] = {};
    expectTrue(
        kira::pisa::preparePisaModelRgba(
            SOURCE,
            3 * 4,
            plan,
            output,
            6 * 4,
            sizeof(output)
        ),
        "aligned PiSA preprocessing succeeds"
    );

    for (std::size_t index = 3; index < sizeof(output); index += 4) {
        expectTrue(
            output[index] == 255U,
            "preprocessed alpha is opaque"
        );
    }
}

void testRejectsInvalidPlanAndShortOutput() {
    const kira::pisa::ResizePlan invalid{
        3,
        2,
        3,
        2,
        6,
        4,
        8,
        4,
        6,
        4,
        false,
    };
    std::uint8_t output[8 * 4 * 4] = {};

    expectFalse(
        kira::pisa::preparePisaModelRgba(
            SOURCE,
            3 * 4,
            invalid,
            output,
            8 * 4,
            sizeof(output)
        ),
        "model wider than raw resize rejected"
    );

    const kira::pisa::ResizePlan valid{
        3,
        2,
        3,
        2,
        6,
        4,
        6,
        4,
        6,
        4,
        false,
    };
    expectFalse(
        kira::pisa::preparePisaModelRgba(
            SOURCE,
            3 * 4,
            valid,
            output,
            6 * 4,
            static_cast<std::size_t>(6 * 4 * 4) - 1U
        ),
        "short preprocessing output rejected"
    );
}

}  // namespace

int main() {
    testSmallInputChainMatchesPillowFixture();
    testAlignedInputSkipsLanczosCorrection();
    testRejectsInvalidPlanAndShortOutput();

    if (failures != 0) {
        std::cerr
            << failures
            << " PiSA preprocessing test(s) failed\n";
        return 1;
    }

    std::cout << "PiSA preprocessing tests passed\n";
    return 0;
}
