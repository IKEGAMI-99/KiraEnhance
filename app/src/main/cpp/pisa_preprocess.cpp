#include "pisa_preprocess.h"

#include "pisa_pillow_resize.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace kira::pisa {
namespace {

bool checkedImageBytes(
    int width,
    int height,
    std::size_t& output
) {
    if (
        width <= 0 ||
        height <= 0 ||
        width > std::numeric_limits<int>::max() / 4
    ) {
        return false;
    }

    const std::size_t rowBytes =
        static_cast<std::size_t>(width) * 4U;
    if (
        rowBytes >
        std::numeric_limits<std::size_t>::max() /
            static_cast<std::size_t>(height)
    ) {
        return false;
    }

    output =
        rowBytes * static_cast<std::size_t>(height);
    return true;
}

bool planIsUsable(const ResizePlan& plan) {
    return
        plan.sourceWidth > 0 &&
        plan.sourceHeight > 0 &&
        plan.preUpscaleWidth > 0 &&
        plan.preUpscaleHeight > 0 &&
        plan.rawModelWidth > 0 &&
        plan.rawModelHeight > 0 &&
        plan.modelWidth > 0 &&
        plan.modelHeight > 0 &&
        plan.outputWidth > 0 &&
        plan.outputHeight > 0 &&
        plan.modelWidth <= plan.rawModelWidth &&
        plan.modelHeight <= plan.rawModelHeight;
}

}  // namespace

bool preparePisaModelRgba(
    const std::uint8_t* input,
    int inputRowStrideBytes,
    const ResizePlan& plan,
    std::uint8_t* output,
    int outputRowStrideBytes,
    std::size_t outputByteCount
) {
    if (
        input == nullptr ||
        output == nullptr ||
        !planIsUsable(plan) ||
        plan.sourceWidth >
            std::numeric_limits<int>::max() / 4 ||
        plan.modelWidth >
            std::numeric_limits<int>::max() / 4 ||
        inputRowStrideBytes < plan.sourceWidth * 4 ||
        outputRowStrideBytes < plan.modelWidth * 4
    ) {
        return false;
    }

    std::size_t requiredOutputBytes = 0;
    if (
        !checkedImageBytes(
            plan.modelWidth,
            plan.modelHeight,
            requiredOutputBytes
        ) ||
        outputByteCount < requiredOutputBytes
    ) {
        return false;
    }

    const std::uint8_t* current = input;
    int currentWidth = plan.sourceWidth;
    int currentHeight = plan.sourceHeight;
    int currentStride = inputRowStrideBytes;

    std::vector<std::uint8_t> preUpscale;
    if (plan.smallInputBoosted) {
        std::size_t preBytes = 0;
        if (
            !checkedImageBytes(
                plan.preUpscaleWidth,
                plan.preUpscaleHeight,
                preBytes
            )
        ) {
            return false;
        }
        preUpscale.resize(preBytes);

        const int preStride =
            plan.preUpscaleWidth * 4;
        if (
            !resizeRgba8888PillowRgb(
                current,
                currentWidth,
                currentHeight,
                currentStride,
                preUpscale.data(),
                plan.preUpscaleWidth,
                plan.preUpscaleHeight,
                preStride,
                preUpscale.size(),
                PillowResizeFilter::BICUBIC
            )
        ) {
            return false;
        }

        current = preUpscale.data();
        currentWidth = plan.preUpscaleWidth;
        currentHeight = plan.preUpscaleHeight;
        currentStride = preStride;
    } else if (
        currentWidth != plan.preUpscaleWidth ||
        currentHeight != plan.preUpscaleHeight
    ) {
        return false;
    }

    std::size_t rawBytes = 0;
    if (
        !checkedImageBytes(
            plan.rawModelWidth,
            plan.rawModelHeight,
            rawBytes
        )
    ) {
        return false;
    }

    std::vector<std::uint8_t> rawModel(rawBytes);
    const int rawStride = plan.rawModelWidth * 4;
    if (
        !resizeRgba8888PillowRgb(
            current,
            currentWidth,
            currentHeight,
            currentStride,
            rawModel.data(),
            plan.rawModelWidth,
            plan.rawModelHeight,
            rawStride,
            rawModel.size(),
            PillowResizeFilter::BICUBIC
        )
    ) {
        return false;
    }

    if (
        plan.rawModelWidth == plan.modelWidth &&
        plan.rawModelHeight == plan.modelHeight
    ) {
        return resizeRgba8888PillowRgb(
            rawModel.data(),
            plan.rawModelWidth,
            plan.rawModelHeight,
            rawStride,
            output,
            plan.modelWidth,
            plan.modelHeight,
            outputRowStrideBytes,
            outputByteCount,
            PillowResizeFilter::BICUBIC
        );
    }

    return resizeRgba8888PillowRgb(
        rawModel.data(),
        plan.rawModelWidth,
        plan.rawModelHeight,
        rawStride,
        output,
        plan.modelWidth,
        plan.modelHeight,
        outputRowStrideBytes,
        outputByteCount,
        PillowResizeFilter::LANCZOS
    );
}

}  // namespace kira::pisa
