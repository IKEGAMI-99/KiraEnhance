#include "pisa_pillow_resize.h"

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

void expectPixel(
    const std::uint8_t* image,
    int width,
    int x,
    int y,
    int red,
    int green,
    int blue,
    const char* label
) {
    const std::size_t offset =
        (
            static_cast<std::size_t>(y) *
                static_cast<std::size_t>(width) +
            static_cast<std::size_t>(x)
        ) * 4U;
    const bool matches =
        static_cast<int>(image[offset]) == red &&
        static_cast<int>(image[offset + 1U]) == green &&
        static_cast<int>(image[offset + 2U]) == blue &&
        image[offset + 3U] == 255U;
    expectTrue(matches, label);
}

constexpr std::uint8_t SOURCE[] = {
    0, 20, 255, 1,
    80, 40, 200, 2,
    255, 60, 0, 3,

    30, 220, 10, 4,
    128, 128, 128, 5,
    240, 250, 64, 6,
};

void testBicubicMatchesPillowFixture() {
    std::uint8_t output[7 * 5 * 4] = {};
    expectTrue(
        kira::pisa::resizeRgba8888PillowRgb(
            SOURCE,
            3,
            2,
            3 * 4,
            output,
            7,
            5,
            7 * 4,
            sizeof(output),
            kira::pisa::PillowResizeFilter::BICUBIC
        ),
        "Pillow bicubic resize succeeds"
    );

    const int expected[5][7][3] = {
        {
            {0, 0, 255},
            {3, 3, 255},
            {29, 17, 254},
            {75, 31, 207},
            {160, 38, 108},
            {240, 40, 13},
            {255, 43, 0},
        },
        {
            {1, 32, 238},
            {8, 34, 236},
            {36, 39, 228},
            {83, 46, 195},
            {163, 58, 108},
            {238, 70, 21},
            {255, 75, 4},
        },
        {
            {11, 124, 128},
            {22, 116, 137},
            {55, 95, 159},
            {104, 84, 164},
            {173, 113, 108},
            {235, 148, 45},
            {253, 159, 29},
        },
        {
            {20, 215, 17},
            {36, 198, 37},
            {74, 151, 90},
            {125, 122, 133},
            {183, 168, 108},
            {231, 226, 68},
            {251, 242, 54},
        },
        {
            {23, 250, 0},
            {41, 229, 0},
            {81, 173, 64},
            {133, 137, 121},
            {186, 188, 108},
            {229, 255, 76},
            {251, 255, 64},
        },
    };

    for (int y = 0; y < 5; ++y) {
        for (int x = 0; x < 7; ++x) {
            expectPixel(
                output,
                7,
                x,
                y,
                expected[y][x][0],
                expected[y][x][1],
                expected[y][x][2],
                "bicubic pixel matches Pillow"
            );
        }
    }
}

void testLanczosMatchesPillowFixture() {
    std::uint8_t bicubic[7 * 5 * 4] = {};
    expectTrue(
        kira::pisa::resizeRgba8888PillowRgb(
            SOURCE,
            3,
            2,
            3 * 4,
            bicubic,
            7,
            5,
            7 * 4,
            sizeof(bicubic),
            kira::pisa::PillowResizeFilter::BICUBIC
        ),
        "fixture bicubic pre-resize succeeds"
    );

    std::uint8_t output[6 * 4 * 4] = {};
    expectTrue(
        kira::pisa::resizeRgba8888PillowRgb(
            bicubic,
            7,
            5,
            7 * 4,
            output,
            6,
            4,
            6 * 4,
            sizeof(output),
            kira::pisa::PillowResizeFilter::LANCZOS
        ),
        "Pillow Lanczos resize succeeds"
    );

    const int expected[4][6][3] = {
        {
            {0, 1, 255},
            {8, 7, 255},
            {44, 25, 241},
            {121, 36, 154},
            {225, 41, 31},
            {255, 44, 0},
        },
        {
            {4, 61, 206},
            {19, 59, 206},
            {59, 55, 203},
            {132, 66, 145},
            {223, 91, 44},
            {255, 101, 10},
        },
        {
            {18, 187, 50},
            {39, 165, 76},
            {87, 117, 130},
            {154, 128, 129},
            {222, 192, 70},
            {251, 218, 46},
        },
        {
            {24, 249, 0},
            {49, 217, 12},
            {102, 148, 95},
            {164, 158, 121},
            {220, 242, 82},
            {250, 255, 64},
        },
    };

    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 6; ++x) {
            expectPixel(
                output,
                6,
                x,
                y,
                expected[y][x][0],
                expected[y][x][1],
                expected[y][x][2],
                "Lanczos pixel matches Pillow"
            );
        }
    }
}

void testIdentityCopiesRgbAndForcesOpaqueAlpha() {
    std::uint8_t output[sizeof(SOURCE)] = {};
    expectTrue(
        kira::pisa::resizeRgba8888PillowRgb(
            SOURCE,
            3,
            2,
            3 * 4,
            output,
            3,
            2,
            3 * 4,
            sizeof(output),
            kira::pisa::PillowResizeFilter::BICUBIC
        ),
        "identity Pillow resize succeeds"
    );

    for (int y = 0; y < 2; ++y) {
        for (int x = 0; x < 3; ++x) {
            const std::size_t offset =
                (
                    static_cast<std::size_t>(y) * 3U +
                    static_cast<std::size_t>(x)
                ) * 4U;
            expectPixel(
                output,
                3,
                x,
                y,
                SOURCE[offset],
                SOURCE[offset + 1U],
                SOURCE[offset + 2U],
                "identity RGB preserved"
            );
        }
    }
}

void testRejectsShortOutput() {
    std::uint8_t output[6 * 4 * 4] = {};
    expectFalse(
        kira::pisa::resizeRgba8888PillowRgb(
            SOURCE,
            3,
            2,
            3 * 4,
            output,
            6,
            4,
            6 * 4,
            sizeof(output) - 1U,
            kira::pisa::PillowResizeFilter::LANCZOS
        ),
        "short Pillow resize output rejected"
    );
}

}  // namespace

int main() {
    testBicubicMatchesPillowFixture();
    testLanczosMatchesPillowFixture();
    testIdentityCopiesRgbAndForcesOpaqueAlpha();
    testRejectsShortOutput();

    if (failures != 0) {
        std::cerr
            << failures
            << " PiSA Pillow resize test(s) failed\n";
        return 1;
    }

    std::cout << "PiSA Pillow resize tests passed\n";
    return 0;
}
