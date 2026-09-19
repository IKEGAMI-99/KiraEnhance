#pragma once

#include "pisa_tile_plan.h"

#include <vector>

namespace kira::pisa {

struct VaeTileRegion {
    TileRegion input;
    TileRegion output;
};

struct VaeTilePlan {
    int inputWidth = 0;
    int inputHeight = 0;
    int outputWidth = 0;
    int outputHeight = 0;
    int tileSize = 0;
    int padding = 0;
    bool decoder = false;
    std::vector<VaeTileRegion> tiles;
};

bool requiresVaeTiling(
    int inputWidth,
    int inputHeight,
    int tileSize,
    int padding
);

bool buildVaeTilePlan(
    int inputWidth,
    int inputHeight,
    int tileSize,
    int padding,
    bool decoder,
    VaeTilePlan& output
);

}  // namespace kira::pisa
