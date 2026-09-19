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

void testRgbaToUnitNchwMatchesTorchvisionToTensor() {
    const std::uint8_t rgba[] = {
        1, 127, 255, 9,
        2, 128, 0, 8,
    };
    float output[6] = {};

    expectTrue(
        kira::pisa::rgba8888ToUnitNchw(
            rgba,
            2,
            1,
            8,
            output,
            6
        ),
        "RGBA to unit NCHW succeeds"
    );

    expectNear(
        output[0],
        1.0f / 255.0f,
        0.0f,
        "unit red one"
    );
    expectNear(
        output[1],
        2.0f / 255.0f,
        0.0f,
        "unit red two"
    );
    expectNear(
        output[2],
        127.0f / 255.0f,
        0.0f,
        "unit green 127"
    );
    expectNear(
        output[3],
        128.0f / 255.0f,
        0.0f,
        "unit green 128"
    );
    expectNear(output[4], 1.0f, 0.0f, "unit blue max");
    expectNear(output[5], 0.0f, 0.0f, "unit blue zero");
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

void testTorchvisionQuantizationTruncatesToByteGrid() {
    float values[] = {
        0.0f,
        0.5f,
        1.0f,
        0.501f,
        -1.0f,
        2.0f,
    };

    expectTrue(
        kira::pisa::quantizeUnitPlanarLikeTorchvision(
            values,
            6
        ),
        "torchvision quantization succeeds"
    );

    expectNear(values[0], 0.0f, 0.0f, "quantized zero");
    expectNear(
        values[1],
        127.0f / 255.0f,
        1.0e-7f,
        "half truncates to 127"
    );
    expectNear(values[2], 1.0f, 0.0f, "quantized one");
    expectNear(
        values[3],
        127.0f / 255.0f,
        1.0e-7f,
        "fraction truncates"
    );
    expectNear(values[4], 0.0f, 0.0f, "low clamp");
    expectNear(values[5], 1.0f, 0.0f, "high clamp");
}

void testTorchvisionUnitTensorToRgbaTruncates() {
    const float input[] = {
        0.0f, 1.0f,
        0.5f, 0.501f,
        1.5f, -0.5f,
    };
    std::uint8_t rgba[8] = {};

    expectTrue(
        kira::pisa::unitNchwToRgba8888LikeTorchvision(
            input,
            2,
            1,
            rgba,
            8,
            sizeof(rgba)
        ),
        "torchvision RGBA conversion succeeds"
    );

    expectByte(rgba[0], 0, "torchvision red zero");
    expectByte(rgba[1], 127, "torchvision green half");
    expectByte(rgba[2], 255, "torchvision blue high");
    expectByte(rgba[3], 255, "torchvision alpha zero");

    expectByte(rgba[4], 255, "torchvision red one");
    expectByte(rgba[5], 127, "torchvision green fraction");
    expectByte(rgba[6], 0, "torchvision blue low");
    expectByte(rgba[7], 255, "torchvision alpha one");
}

void testTorchvisionQuantizationRejectsNonFinite() {
    float values[] = {0.0f, std::nanf("")};
    expectFalse(
        kira::pisa::quantizeUnitPlanarLikeTorchvision(
            values,
            2
        ),
        "non-finite torchvision quantization rejected"
    );
}

void testPlanarBilinearResizePreservesIdentity() {
    const float input[] = {
        0.0f, 1.0f,
        2.0f, 3.0f,

        10.0f, 11.0f,
        12.0f, 13.0f,
    };
    float output[8] = {};

    expectTrue(
        kira::pisa::resizePlanarBilinear(
            input,
            2,
            2,
            2,
            output,
            2,
            2,
            8
        ),
        "identity planar resize succeeds"
    );

    for (std::size_t index = 0; index < 8; ++index) {
        expectNear(
            output[index],
            input[index],
            1.0e-6f,
            "identity planar resize value"
        );
    }
}

void testPlanarBilinearResizeInterpolatesCornersAndCenter() {
    const float input[] = {
        0.0f, 2.0f,
        4.0f, 6.0f,
    };
    float output[9] = {};

    expectTrue(
        kira::pisa::resizePlanarBilinear(
            input,
            1,
            2,
            2,
            output,
            3,
            3,
            9
        ),
        "interpolated planar resize succeeds"
    );

    expectNear(output[0], 0.0f, 1.0e-6f, "top-left preserved");
    expectNear(output[2], 2.0f, 1.0e-6f, "top-right preserved");
    expectNear(output[6], 4.0f, 1.0e-6f, "bottom-left preserved");
    expectNear(output[8], 6.0f, 1.0e-6f, "bottom-right preserved");
    expectNear(output[4], 3.0f, 1.0e-6f, "center interpolated");
}

void testPlanarBilinearResizeRejectsShortOutput() {
    const float input[4] = {};
    float output[8] = {};

    expectFalse(
        kira::pisa::resizePlanarBilinear(
            input,
            1,
            2,
            2,
            output,
            3,
            3,
            8
        ),
        "short planar resize output rejected"
    );
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
    testRgbaToUnitNchwMatchesTorchvisionToTensor();
    testNchwToRgbaClampsAndWritesOpaqueAlpha();
    testTorchvisionQuantizationTruncatesToByteGrid();
    testTorchvisionUnitTensorToRgbaTruncates();
    testTorchvisionQuantizationRejectsNonFinite();
    testPlanarBilinearResizePreservesIdentity();
    testPlanarBilinearResizeInterpolatesCornersAndCenter();
    testPlanarBilinearResizeRejectsShortOutput();
    testRejectsInvalidBuffersAndDimensions();

    if (failures != 0) {
        std::cerr << failures << " PiSA image tensor test(s) failed\n";
        return 1;
    }

    std::cout << "PiSA image tensor tests passed\n";
    return 0;
}
