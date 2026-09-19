#include "pisa_resize_plan.h"

#include <climits>
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

void expectEqual(int actual, int expected, const char* label) {
    if (actual != expected) {
        std::cerr
            << "FAIL: " << label
            << " expected=" << expected
            << " actual=" << actual
            << "\n";
        ++failures;
    }
}

void testRegularInputMatchesFourXAndEightMultiple() {
    kira::pisa::ResizePlan plan;
    expectTrue(
        kira::pisa::buildResizePlan(1920, 1080, plan),
        "regular plan succeeds"
    );

    expectFalse(plan.smallInputBoosted, "regular input is not boosted");
    expectEqual(plan.preUpscaleWidth, 1920, "regular pre width");
    expectEqual(plan.preUpscaleHeight, 1080, "regular pre height");
    expectEqual(plan.rawModelWidth, 7680, "regular raw model width");
    expectEqual(plan.rawModelHeight, 4320, "regular raw model height");
    expectEqual(plan.modelWidth, 7680, "regular model width");
    expectEqual(plan.modelHeight, 4320, "regular model height");
    expectEqual(plan.outputWidth, 7680, "regular output width");
    expectEqual(plan.outputHeight, 4320, "regular output height");
}

void testOddDimensionsFloorModelToEightButPreserveAppOutputContract() {
    kira::pisa::ResizePlan plan;
    expectTrue(
        kira::pisa::buildResizePlan(501, 333, plan),
        "odd plan succeeds"
    );

    expectEqual(plan.rawModelWidth, 2004, "odd raw model width");
    expectEqual(plan.rawModelHeight, 1332, "odd raw model height");
    expectEqual(plan.modelWidth, 2000, "odd model width floors to multiple of eight");
    expectEqual(plan.modelHeight, 1328, "odd model height floors to multiple of eight");
    expectEqual(plan.outputWidth, 2004, "app output remains exact 4x width");
    expectEqual(plan.outputHeight, 1332, "app output remains exact 4x height");
}

void testSmallInputMatchesUpstreamMinimumSideBoost() {
    kira::pisa::ResizePlan plan;
    expectTrue(
        kira::pisa::buildResizePlan(64, 96, plan),
        "small plan succeeds"
    );

    expectTrue(plan.smallInputBoosted, "small input is boosted");
    expectEqual(plan.preUpscaleWidth, 128, "small pre width reaches threshold");
    expectEqual(plan.preUpscaleHeight, 192, "small pre height scales proportionally");
    expectEqual(plan.rawModelWidth, 512, "small raw model width");
    expectEqual(plan.rawModelHeight, 768, "small raw model height");
    expectEqual(plan.modelWidth, 512, "small model width");
    expectEqual(plan.modelHeight, 768, "small model height");
    expectEqual(plan.outputWidth, 256, "small final output returns to exact 4x width");
    expectEqual(plan.outputHeight, 384, "small final output returns to exact 4x height");
}

void testSmallNonSquareInputUsesPythonStyleTruncation() {
    kira::pisa::ResizePlan plan;
    expectTrue(
        kira::pisa::buildResizePlan(100, 73, plan),
        "truncation plan succeeds"
    );

    expectTrue(plan.smallInputBoosted, "truncation input is boosted");
    expectEqual(plan.preUpscaleWidth, 175, "scaled width truncates like Python int");
    expectEqual(plan.preUpscaleHeight, 128, "minimum side becomes threshold");
    expectEqual(plan.rawModelWidth, 700, "truncation raw model width");
    expectEqual(plan.rawModelHeight, 512, "truncation raw model height");
    expectEqual(plan.modelWidth, 696, "model width floors to eight");
    expectEqual(plan.modelHeight, 512, "model height");
    expectEqual(plan.outputWidth, 400, "final output width");
    expectEqual(plan.outputHeight, 292, "final output height");
}

void testRejectsInvalidAndOverflowingDimensions() {
    kira::pisa::ResizePlan plan;
    expectFalse(
        kira::pisa::buildResizePlan(0, 100, plan),
        "zero width rejected"
    );
    expectFalse(
        kira::pisa::buildResizePlan(100, 0, plan),
        "zero height rejected"
    );
    expectFalse(
        kira::pisa::buildResizePlan(INT_MAX, INT_MAX, plan),
        "four-x overflow rejected"
    );
    expectFalse(
        kira::pisa::buildResizePlan(100, 100, plan, 3, 4, 8),
        "process size smaller than upscale rejected"
    );
}

}  // namespace

int main() {
    testRegularInputMatchesFourXAndEightMultiple();
    testOddDimensionsFloorModelToEightButPreserveAppOutputContract();
    testSmallInputMatchesUpstreamMinimumSideBoost();
    testSmallNonSquareInputUsesPythonStyleTruncation();
    testRejectsInvalidAndOverflowingDimensions();

    if (failures != 0) {
        std::cerr << failures << " PiSA resize plan test(s) failed\n";
        return 1;
    }

    std::cout << "PiSA resize plan tests passed\n";
    return 0;
}
