#include "pisa_vae_tile_plan.h"

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

void expectEqual(
    std::int64_t actual,
    std::int64_t expected,
    const char* label
) {
    if (actual != expected) {
        std::cerr
            << "FAIL: " << label
            << " expected=" << expected
            << " actual=" << actual
            << "\n";
        ++failures;
    }
}

std::int64_t outputArea(
    const kira::pisa::VaeTilePlan& plan
) {
    std::int64_t area = 0;
    for (const auto& tile : plan.tiles) {
        area +=
            static_cast<std::int64_t>(tile.output.width) *
            tile.output.height;
    }
    return area;
}

void testEncoderPlanMatchesOfficialDefaults() {
    kira::pisa::VaeTilePlan plan;
    expectTrue(
        kira::pisa::buildVaeTilePlan(
            2048,
            1536,
            1024,
            32,
            false,
            plan
        ),
        "encoder VAE plan succeeds"
    );

    expectEqual(plan.outputWidth, 256, "encoder output width");
    expectEqual(plan.outputHeight, 192, "encoder output height");
    expectEqual(
        static_cast<std::int64_t>(plan.tiles.size()),
        4,
        "encoder tile count"
    );
    expectEqual(
        outputArea(plan),
        static_cast<std::int64_t>(256) * 192,
        "encoder valid tiles cover output exactly"
    );

    const auto& first = plan.tiles.front();
    expectEqual(first.input.x, 0, "encoder first input x");
    expectEqual(first.input.y, 0, "encoder first input y");
    expectEqual(first.input.width, 1056, "encoder first input width");
    expectEqual(first.input.height, 800, "encoder first input height");
    expectEqual(first.output.x, 0, "encoder first output x");
    expectEqual(first.output.y, 0, "encoder first output y");
    expectEqual(first.output.width, 128, "encoder first output width");
    expectEqual(first.output.height, 96, "encoder first output height");

    const auto& last = plan.tiles.back();
    expectEqual(last.input.x, 992, "encoder last input x");
    expectEqual(last.input.y, 736, "encoder last input y");
    expectEqual(last.input.width, 1056, "encoder last input width");
    expectEqual(last.input.height, 800, "encoder last input height");
    expectEqual(last.output.x, 128, "encoder last output x");
    expectEqual(last.output.y, 96, "encoder last output y");
    expectEqual(last.output.width, 128, "encoder last output width");
    expectEqual(last.output.height, 96, "encoder last output height");
}

void testDecoderPlanMatchesOfficialDefaults() {
    kira::pisa::VaeTilePlan plan;
    expectTrue(
        kira::pisa::buildVaeTilePlan(
            256,
            192,
            224,
            11,
            true,
            plan
        ),
        "decoder VAE plan succeeds"
    );

    expectEqual(plan.outputWidth, 2048, "decoder output width");
    expectEqual(plan.outputHeight, 1536, "decoder output height");
    expectEqual(
        static_cast<std::int64_t>(plan.tiles.size()),
        2,
        "decoder tile count"
    );
    expectEqual(
        outputArea(plan),
        static_cast<std::int64_t>(2048) * 1536,
        "decoder valid tiles cover output exactly"
    );

    const auto& first = plan.tiles.front();
    expectEqual(first.input.x, 0, "decoder first input x");
    expectEqual(first.input.width, 150, "decoder first input width");
    expectEqual(first.output.x, 0, "decoder first output x");
    expectEqual(first.output.width, 1112, "decoder first output width");

    const auto& last = plan.tiles.back();
    expectEqual(last.input.x, 128, "decoder last input x");
    expectEqual(last.input.width, 128, "decoder last input width");
    expectEqual(last.output.x, 1112, "decoder last output x");
    expectEqual(last.output.width, 936, "decoder last output width");
}

void testComputesEncoderLocalCrops() {
    kira::pisa::VaeTilePlan plan;
    expectTrue(
        kira::pisa::buildVaeTilePlan(
            2048,
            1536,
            1024,
            32,
            false,
            plan
        ),
        "encoder crop plan succeeds"
    );

    kira::pisa::TileRegion firstCrop;
    expectTrue(
        kira::pisa::localOutputCropForVaeTile(
            plan.tiles.front(),
            false,
            firstCrop
        ),
        "encoder first local crop succeeds"
    );
    expectEqual(firstCrop.x, 0, "encoder first local crop x");
    expectEqual(firstCrop.y, 0, "encoder first local crop y");
    expectEqual(firstCrop.width, 128, "encoder first crop width");
    expectEqual(firstCrop.height, 96, "encoder first crop height");

    kira::pisa::TileRegion lastCrop;
    expectTrue(
        kira::pisa::localOutputCropForVaeTile(
            plan.tiles.back(),
            false,
            lastCrop
        ),
        "encoder last local crop succeeds"
    );
    expectEqual(lastCrop.x, 4, "encoder last local crop x");
    expectEqual(lastCrop.y, 4, "encoder last local crop y");
    expectEqual(lastCrop.width, 128, "encoder last crop width");
    expectEqual(lastCrop.height, 96, "encoder last crop height");
}

void testComputesDecoderLocalCrops() {
    kira::pisa::VaeTilePlan plan;
    expectTrue(
        kira::pisa::buildVaeTilePlan(
            256,
            192,
            224,
            11,
            true,
            plan
        ),
        "decoder crop plan succeeds"
    );

    kira::pisa::TileRegion firstCrop;
    expectTrue(
        kira::pisa::localOutputCropForVaeTile(
            plan.tiles.front(),
            true,
            firstCrop
        ),
        "decoder first local crop succeeds"
    );
    expectEqual(firstCrop.x, 0, "decoder first local crop x");
    expectEqual(firstCrop.y, 0, "decoder first local crop y");
    expectEqual(firstCrop.width, 1112, "decoder first crop width");
    expectEqual(firstCrop.height, 1536, "decoder first crop height");

    kira::pisa::TileRegion lastCrop;
    expectTrue(
        kira::pisa::localOutputCropForVaeTile(
            plan.tiles.back(),
            true,
            lastCrop
        ),
        "decoder last local crop succeeds"
    );
    expectEqual(lastCrop.x, 88, "decoder last local crop x");
    expectEqual(lastCrop.y, 0, "decoder last local crop y");
    expectEqual(lastCrop.width, 936, "decoder last crop width");
    expectEqual(lastCrop.height, 1536, "decoder last crop height");
}

void testRejectsInvalidLocalCrop() {
    kira::pisa::TileRegion output;
    expectFalse(
        kira::pisa::localOutputCropForVaeTile(
            kira::pisa::VaeTileRegion{
                kira::pisa::TileRegion{100, 0, 64, 64},
                kira::pisa::TileRegion{0, 0, 8, 8},
            },
            false,
            output
        ),
        "output preceding padded encoder tile rejected"
    );
}

void testPolicyMatchesUpstreamTinyThreshold() {
    expectFalse(
        kira::pisa::requiresVaeTiling(
            1088,
            512,
            1024,
            32
        ),
        "encoder threshold stays monolithic"
    );
    expectTrue(
        kira::pisa::requiresVaeTiling(
            1096,
            512,
            1024,
            32
        ),
        "encoder above threshold tiles"
    );
    expectFalse(
        kira::pisa::requiresVaeTiling(
            246,
            192,
            224,
            11
        ),
        "decoder threshold stays monolithic"
    );
    expectTrue(
        kira::pisa::requiresVaeTiling(
            247,
            192,
            224,
            11
        ),
        "decoder above threshold tiles"
    );
}

void testRejectsInvalidEncoderDimensions() {
    kira::pisa::VaeTilePlan plan;
    expectFalse(
        kira::pisa::buildVaeTilePlan(
            1025,
            1024,
            1024,
            32,
            false,
            plan
        ),
        "encoder width not divisible by eight rejected"
    );
    expectFalse(
        kira::pisa::buildVaeTilePlan(
            1024,
            1023,
            1024,
            32,
            false,
            plan
        ),
        "encoder height not divisible by eight rejected"
    );
}

void testLongThinInputStillCoversOutput() {
    kira::pisa::VaeTilePlan plan;
    expectTrue(
        kira::pisa::buildVaeTilePlan(
            4096,
            64,
            1024,
            32,
            false,
            plan
        ),
        "long thin encoder plan succeeds"
    );
    expectEqual(plan.outputWidth, 512, "long thin output width");
    expectEqual(plan.outputHeight, 8, "long thin output height");
    expectEqual(
        outputArea(plan),
        static_cast<std::int64_t>(512) * 8,
        "long thin plan covers output exactly"
    );
}

}  // namespace

int main() {
    testEncoderPlanMatchesOfficialDefaults();
    testDecoderPlanMatchesOfficialDefaults();
    testComputesEncoderLocalCrops();
    testComputesDecoderLocalCrops();
    testRejectsInvalidLocalCrop();
    testPolicyMatchesUpstreamTinyThreshold();
    testRejectsInvalidEncoderDimensions();
    testLongThinInputStillCoversOutput();

    if (failures != 0) {
        std::cerr
            << failures
            << " PiSA VAE tile plan test(s) failed\n";
        return 1;
    }

    std::cout << "PiSA VAE tile plan tests passed\n";
    return 0;
}
