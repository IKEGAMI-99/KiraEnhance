#pragma once

namespace kira::pisa {

struct ResizePlan {
    int sourceWidth = 0;
    int sourceHeight = 0;
    int preUpscaleWidth = 0;
    int preUpscaleHeight = 0;
    int modelWidth = 0;
    int modelHeight = 0;
    int outputWidth = 0;
    int outputHeight = 0;
    bool smallInputBoosted = false;
};

bool buildResizePlan(
    int sourceWidth,
    int sourceHeight,
    ResizePlan& output,
    int processSize = 512,
    int upscale = 4,
    int modelMultiple = 8
);

}  // namespace kira::pisa
