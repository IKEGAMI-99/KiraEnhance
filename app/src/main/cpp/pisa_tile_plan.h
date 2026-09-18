#pragma once

#include <vector>

namespace kira::pisa {

struct TileRegion {
    int x = 0;
    int y = 0;
    int width = 0;
    int height = 0;
};

struct TilePlan {
    int imageWidth = 0;
    int imageHeight = 0;
    int tileSize = 0;
    int overlap = 0;
    std::vector<TileRegion> tiles;
};

bool requiresTiling(
    int imageWidth,
    int imageHeight,
    int tileSize
);

bool buildTilePlan(
    int imageWidth,
    int imageHeight,
    int tileSize,
    int overlap,
    TilePlan& output
);

bool buildGaussianTileWeights(
    int tileWidth,
    int tileHeight,
    std::vector<float>& output
);

}  // namespace kira::pisa
