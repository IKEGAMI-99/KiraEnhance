#include "pisa_tile_plan.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace kira::pisa {
namespace {

bool buildAxisStarts(
    int length,
    int tileSize,
    int overlap,
    std::vector<int>& output
) {
    output.clear();
    if (
        length <= 0 ||
        tileSize <= 0 ||
        overlap < 0 ||
        overlap >= tileSize ||
        tileSize > length
    ) {
        return false;
    }

    const int step = tileSize - overlap;
    int position = 0;
    while (position + tileSize < length) {
        output.push_back(position);
        if (position > std::numeric_limits<int>::max() - step) {
            return false;
        }
        position += step;
    }

    const int last = length - tileSize;
    if (output.empty() || output.back() != last) {
        output.push_back(last);
    }
    return !output.empty();
}

}  // namespace

bool buildTilePlan(
    int imageWidth,
    int imageHeight,
    int tileSize,
    int overlap,
    TilePlan& output
) {
    output = TilePlan{};

    if (
        imageWidth <= 0 ||
        imageHeight <= 0 ||
        tileSize <= 0 ||
        overlap < 0
    ) {
        return false;
    }

    const int effectiveTile = std::min(
        tileSize,
        std::min(imageWidth, imageHeight)
    );
    if (effectiveTile <= 0 || overlap >= effectiveTile) {
        return false;
    }

    std::vector<int> xStarts;
    std::vector<int> yStarts;
    if (
        !buildAxisStarts(
            imageWidth,
            effectiveTile,
            overlap,
            xStarts
        ) ||
        !buildAxisStarts(
            imageHeight,
            effectiveTile,
            overlap,
            yStarts
        )
    ) {
        return false;
    }

    if (
        xStarts.size() >
        std::numeric_limits<std::size_t>::max() /
            yStarts.size()
    ) {
        return false;
    }

    output.imageWidth = imageWidth;
    output.imageHeight = imageHeight;
    output.tileSize = effectiveTile;
    output.overlap = overlap;
    output.tiles.reserve(xStarts.size() * yStarts.size());

    for (int y : yStarts) {
        for (int x : xStarts) {
            output.tiles.push_back(
                TileRegion{
                    x,
                    y,
                    effectiveTile,
                    effectiveTile,
                }
            );
        }
    }

    return !output.tiles.empty();
}

bool buildGaussianTileWeights(
    int tileWidth,
    int tileHeight,
    std::vector<float>& output
) {
    output.clear();
    if (tileWidth <= 0 || tileHeight <= 0) {
        return false;
    }

    const std::size_t width =
        static_cast<std::size_t>(tileWidth);
    const std::size_t height =
        static_cast<std::size_t>(tileHeight);
    if (
        width >
        std::numeric_limits<std::size_t>::max() / height
    ) {
        return false;
    }

    const std::size_t count = width * height;
    output.resize(count);

    constexpr double PI =
        3.141592653589793238462643383279502884;
    constexpr double VARIANCE_SCALE = 0.01;
    const double midpointX =
        (static_cast<double>(tileWidth) - 1.0) * 0.5;
    const double midpointY =
        (static_cast<double>(tileHeight) - 1.0) * 0.5;
    const double widthSquared =
        static_cast<double>(tileWidth) *
        static_cast<double>(tileWidth);
    const double heightSquared =
        static_cast<double>(tileHeight) *
        static_cast<double>(tileHeight);
    const double normalization =
        1.0 / std::sqrt(2.0 * PI * VARIANCE_SCALE);

    for (int y = 0; y < tileHeight; ++y) {
        const double deltaY =
            static_cast<double>(y) - midpointY;
        const double yProbability =
            std::exp(
                -(deltaY * deltaY) /
                (
                    2.0 *
                    heightSquared *
                    VARIANCE_SCALE
                )
            ) * normalization;

        for (int x = 0; x < tileWidth; ++x) {
            const double deltaX =
                static_cast<double>(x) - midpointX;
            const double xProbability =
                std::exp(
                    -(deltaX * deltaX) /
                    (
                        2.0 *
                        widthSquared *
                        VARIANCE_SCALE
                    )
                ) * normalization;
            const double weight =
                xProbability * yProbability;
            if (
                !std::isfinite(weight) ||
                weight <= 0.0 ||
                weight >
                    static_cast<double>(
                        std::numeric_limits<float>::max()
                    )
            ) {
                output.clear();
                return false;
            }

            output[
                static_cast<std::size_t>(y) * width +
                static_cast<std::size_t>(x)
            ] = static_cast<float>(weight);
        }
    }

    return true;
}

}  // namespace kira::pisa
