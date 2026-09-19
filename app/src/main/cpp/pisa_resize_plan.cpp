#include "pisa_resize_plan.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace kira::pisa {
namespace {

bool checkedInt(std::int64_t value, int& output) {
    if (value <= 0 || value > std::numeric_limits<int>::max()) {
        return false;
    }
    output = static_cast<int>(value);
    return true;
}

bool checkedScale(int value, int scale, int& output) {
    return checkedInt(
        static_cast<std::int64_t>(value) * static_cast<std::int64_t>(scale),
        output
    );
}

int floorToMultiple(int value, int multiple) {
    return value - value % multiple;
}

}  // namespace

bool buildResizePlan(
    int sourceWidth,
    int sourceHeight,
    ResizePlan& output,
    int processSize,
    int upscale,
    int modelMultiple
) {
    if (
        sourceWidth <= 0 ||
        sourceHeight <= 0 ||
        processSize <= 0 ||
        upscale <= 0 ||
        modelMultiple <= 0 ||
        processSize < upscale
    ) {
        return false;
    }

    const int minimumPreUpscale = processSize / upscale;
    if (minimumPreUpscale <= 0) {
        return false;
    }

    int preWidth = sourceWidth;
    int preHeight = sourceHeight;
    const bool boostSmallInput =
        sourceWidth < minimumPreUpscale ||
        sourceHeight < minimumPreUpscale;

    if (boostSmallInput) {
        const int minimumSide = std::min(sourceWidth, sourceHeight);
        const double scale =
            static_cast<double>(minimumPreUpscale) /
            static_cast<double>(minimumSide);

        const double scaledWidth =
            scale * static_cast<double>(sourceWidth);
        const double scaledHeight =
            scale * static_cast<double>(sourceHeight);
        if (
            !std::isfinite(scaledWidth) ||
            !std::isfinite(scaledHeight) ||
            scaledWidth > static_cast<double>(std::numeric_limits<int>::max()) ||
            scaledHeight > static_cast<double>(std::numeric_limits<int>::max())
        ) {
            return false;
        }

        // Match Python int(...) truncation from the upstream preprocessing.
        preWidth = static_cast<int>(scaledWidth);
        preHeight = static_cast<int>(scaledHeight);
        if (preWidth <= 0 || preHeight <= 0) {
            return false;
        }
    }

    int rawModelWidth = 0;
    int rawModelHeight = 0;
    if (
        !checkedScale(preWidth, upscale, rawModelWidth) ||
        !checkedScale(preHeight, upscale, rawModelHeight)
    ) {
        return false;
    }

    const int modelWidth = floorToMultiple(rawModelWidth, modelMultiple);
    const int modelHeight = floorToMultiple(rawModelHeight, modelMultiple);
    if (modelWidth <= 0 || modelHeight <= 0) {
        return false;
    }

    // Upstream only restores the exact original x upscale dimensions
    // when it first boosted a too-small input. For regular inputs, the
    // post-inference image stays on the 8-pixel-aligned model grid.
    int outputWidth = modelWidth;
    int outputHeight = modelHeight;
    if (
        boostSmallInput &&
        (
            !checkedScale(sourceWidth, upscale, outputWidth) ||
            !checkedScale(sourceHeight, upscale, outputHeight)
        )
    ) {
        return false;
    }

    output = ResizePlan{
        sourceWidth,
        sourceHeight,
        preWidth,
        preHeight,
        rawModelWidth,
        rawModelHeight,
        modelWidth,
        modelHeight,
        outputWidth,
        outputHeight,
        boostSmallInput,
    };
    return true;
}

}  // namespace kira::pisa
