#include "pisa_image_tensor.h"

#include <cmath>
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

void expectByte(std::uint8_t actual, int expected, const char* label) {
    if (static_cast<int>(actual) != expected) {
        std::cerr
            << "FAIL: " << label
            << " expected=" << expected
            << " actual=" << static_cast<int>(actual)
            << "\n";
        ++failures;
    }
}

void testRgbaToNchwUsesRgbAndRespectsStride() {
    const std::uint8_t rgba[] = {
        0, 127, 255, 12,
        255, 128, 0, 34,
        99, 99, 99, 99,
    };
    float output[6] = {};

    expectTrue(
        kira::pisa::rgba8888ToNormalizedNchw(
            rgba, 2, 1, 12, output, 6
        ),
        "RGBA to NCHW conversion succeeds"
    );

    expectNear(output[0], -1.0f, 1.0e-6f, "red pixel zero");
    expectNear(output[1], 1.0f, 1.0e-6f, "red pixel one");
    expectNear(output[2], 127.0f / 127.5f - 1.0f, 1.0e-6f, "green 127");
    expectNear(output[3], 128.0f / 127.5f - 1.0f, 1.0e-6f, "green 128");
    expectNear(output[4], 1.0f, 1.0e-6f, "blue pixel zero");
    expectNear(output[5], -1.0f, 1.0e-6f, "blue pixel one");
}

void testNchwToRgbaClampsAndWritesOpaqueAlpha() {
    const float input[] = {
        -2.0f, 1.0f,
        0.0f, -1.0f,
        2.0f, 0.0f,
    };
    std::uint8_t rgba[12] = {};

    expectTrue(
        kira::pisa::normalizedNchwToRgba8888(
            input, 2, 1, rgba, 12, sizeof(rgba)
        ),
        "NCHW to RGBA conversion succeeds"
    );

    expectByte(rgba[0], 0, "clamped red low");
    expectByte(rgba[1], 128, "green midpoint");
    expectByte(rgba[2], 255, "clamped blue high");
    expectByte(rgba[3], 255, "opaque alpha zero");

    expectByte(rgba[4], 255, "red high");
    expectByte(rgba[5], 0, "green low");
    expectByte(rgba[6], 128, "blue midpoint");
    expectByte(rgba[7], 255, "opaque alpha one");
}

void testRejectsInvalidBuffersAndDimensions() {
    const std::uint8_t pixel[4] = {};
    float tensor[3] = {};
    std::uint8_t output[4] = {};

    expectFalse(
        kira::pisa::rgba8888ToNormalizedNchw(
            nullptr, 1, 1, 4, tensor, 3
        ),
        "null input rejected"
    );
    expectFalse(
        kira::pisa::rgba8888ToNormalizedNchw(
            pixel, 1, 1, 3, tensor, 3
        ),
        "short RGBA stride rejected"
    );
    expectFalse(
        kira::pisa::rgba8888ToNormalizedNchw(
            pixel, 1, 1, 4, tensor, 2
        ),
        "short tensor rejected"
    );
    expectFalse(
        kira::pisa::normalizedNchwToRgba8888(
            tensor, 0, 1, output, 4, 4
        ),
        "zero width rejected"
    );
    expectFalse(
        kira::pisa::normalizedNchwToRgba8888(
            tensor, 1, 1, output, 4, 3
        ),
        "short output rejected"
    );
}

}  // namespace

int main() {
    testRgbaToNchwUsesRgbAndRespectsStride();
    testNchwToRgbaClampsAndWritesOpaqueAlpha();
    testRejectsInvalidBuffersAndDimensions();

    if (failures != 0) {
        std::cerr << failures << " PiSA image tensor test(s) failed\n";
        return 1;
    }

    std::cout << "PiSA image tensor tests passed\n";
    return 0;
}
