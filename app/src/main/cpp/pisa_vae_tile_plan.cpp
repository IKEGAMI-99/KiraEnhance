#include "pisa_vae_tile_plan.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace kira::pisa {
namespace {

constexpr int VAE_SPATIAL_SCALE = 8;

struct AxisTile {
    int inputStart = 0;
    int inputEnd = 0;
    int validStart = 0;
    int validEnd = 0;
};

bool checkedAdd(
    int left,
    int right,
    int& output
) {
    const std::int64_t value =
        static_cast<std::int64_t>(left) +
        static_cast<std::int64_t>(right);
    if (
        value < std::numeric_limits<int>::min() ||
        value > std::numeric_limits<int>::max()
    ) {
        return false;
    }
    output = static_cast<int>(value);
    return true;
}

bool checkedMultiply(
    int left,
    int right,
    int& output
) {
    const std::int64_t value =
        static_cast<std::int64_t>(left) *
        static_cast<std::int64_t>(right);
    if (
        value <= 0 ||
        value > std::numeric_limits<int>::max()
    ) {
        return false;
    }
    output = static_cast<int>(value);
    return true;
}

int ceilDividePositive(
    int numerator,
    int denominator
) {
    return numerator / denominator +
        (numerator % denominator == 0 ? 0 : 1);
}

int bestTileSize(
    int lowerBound,
    int upperBound
) {
    if (lowerBound <= 0 || upperBound <= 0) {
        return 0;
    }

    for (int divider = 32; divider >= 2; divider /= 2) {
        const int remainder = lowerBound % divider;
        if (remainder == 0) {
            return lowerBound;
        }

        const int candidate =
            lowerBound - remainder + divider;
        if (candidate <= upperBound) {
            return candidate;
        }
    }
    return lowerBound;
}

bool buildAxisTiles(
    int length,
    int tileSize,
    int padding,
    std::vector<AxisTile>& output
) {
    output.clear();
    if (
        length <= 0 ||
        tileSize <= 0 ||
        padding < 0
    ) {
        return false;
    }

    if (length <= padding * 2) {
        output.push_back(AxisTile{0, length, 0, length});
        return true;
    }

    const int innerLength = length - padding * 2;
    const int tileCount = std::max(
        ceilDividePositive(innerLength, tileSize),
        1
    );
    int realTileSize = ceilDividePositive(
        innerLength,
        tileCount
    );
    realTileSize = bestTileSize(realTileSize, tileSize);
    if (realTileSize <= 0) {
        return false;
    }

    output.reserve(static_cast<std::size_t>(tileCount));
    for (int index = 0; index < tileCount; ++index) {
        const std::int64_t coreStart64 =
            static_cast<std::int64_t>(padding) +
            static_cast<std::int64_t>(index) *
                realTileSize;
        const std::int64_t coreEnd64 =
            static_cast<std::int64_t>(padding) +
            static_cast<std::int64_t>(index + 1) *
                realTileSize;
        if (
            coreStart64 < 0 ||
            coreStart64 >= length ||
            coreEnd64 <= coreStart64
        ) {
            return false;
        }

        const int coreStart =
            static_cast<int>(coreStart64);
        const int coreEnd = static_cast<int>(
            std::min<std::int64_t>(
                coreEnd64,
                static_cast<std::int64_t>(length)
            )
        );

        const int validStart =
            coreStart > padding ? coreStart : 0;
        const int validEnd =
            coreEnd < length - padding
                ? coreEnd
                : length;

        int paddedEnd = 0;
        if (!checkedAdd(coreEnd, padding, paddedEnd)) {
            return false;
        }

        const int inputStart = std::max(
            0,
            coreStart - padding
        );
        const int inputEnd = std::min(
            length,
            paddedEnd
        );
        if (
            inputStart < 0 ||
            inputEnd <= inputStart ||
            validStart < 0 ||
            validEnd <= validStart ||
            validEnd > length
        ) {
            return false;
        }

        output.push_back(
            AxisTile{
                inputStart,
                inputEnd,
                validStart,
                validEnd,
            }
        );
    }

    return !output.empty();
}

bool scaleEndpoint(
    int value,
    bool decoder,
    int& output
) {
    if (value < 0) {
        return false;
    }

    if (decoder) {
        const std::int64_t scaled =
            static_cast<std::int64_t>(value) *
            VAE_SPATIAL_SCALE;
        if (scaled > std::numeric_limits<int>::max()) {
            return false;
        }
        output = static_cast<int>(scaled);
        return true;
    }

    output = value / VAE_SPATIAL_SCALE;
    return true;
}

bool outputDimension(
    int input,
    bool decoder,
    int& output
) {
    if (input <= 0) {
        return false;
    }

    if (decoder) {
        return checkedMultiply(
            input,
            VAE_SPATIAL_SCALE,
            output
        );
    }

    if (input % VAE_SPATIAL_SCALE != 0) {
        return false;
    }
    output = input / VAE_SPATIAL_SCALE;
    return output > 0;
}

}  // namespace

bool requiresVaeTiling(
    int inputWidth,
    int inputHeight,
    int tileSize,
    int padding
) {
    if (
        inputWidth <= 0 ||
        inputHeight <= 0 ||
        tileSize <= 0 ||
        padding < 0
    ) {
        return false;
    }

    const std::int64_t threshold =
        static_cast<std::int64_t>(tileSize) +
        static_cast<std::int64_t>(padding) * 2;
    return std::max(inputWidth, inputHeight) > threshold;
}

bool buildVaeTilePlan(
    int inputWidth,
    int inputHeight,
    int tileSize,
    int padding,
    bool decoder,
    VaeTilePlan& output
) {
    output = VaeTilePlan{};

    if (
        inputWidth <= 0 ||
        inputHeight <= 0 ||
        tileSize <= 0 ||
        padding < 0
    ) {
        return false;
    }

    int outputWidth = 0;
    int outputHeight = 0;
    if (
        !outputDimension(
            inputWidth,
            decoder,
            outputWidth
        ) ||
        !outputDimension(
            inputHeight,
            decoder,
            outputHeight
        )
    ) {
        return false;
    }

    std::vector<AxisTile> xTiles;
    std::vector<AxisTile> yTiles;
    if (
        !buildAxisTiles(
            inputWidth,
            tileSize,
            padding,
            xTiles
        ) ||
        !buildAxisTiles(
            inputHeight,
            tileSize,
            padding,
            yTiles
        )
    ) {
        return false;
    }

    if (
        xTiles.size() >
        std::numeric_limits<std::size_t>::max() /
            yTiles.size()
    ) {
        return false;
    }

    VaeTilePlan plan;
    plan.inputWidth = inputWidth;
    plan.inputHeight = inputHeight;
    plan.outputWidth = outputWidth;
    plan.outputHeight = outputHeight;
    plan.tileSize = tileSize;
    plan.padding = padding;
    plan.decoder = decoder;
    plan.tiles.reserve(xTiles.size() * yTiles.size());

    for (const AxisTile& y : yTiles) {
        for (const AxisTile& x : xTiles) {
            int outputX1 = 0;
            int outputX2 = 0;
            int outputY1 = 0;
            int outputY2 = 0;
            if (
                !scaleEndpoint(
                    x.validStart,
                    decoder,
                    outputX1
                ) ||
                !scaleEndpoint(
                    x.validEnd,
                    decoder,
                    outputX2
                ) ||
                !scaleEndpoint(
                    y.validStart,
                    decoder,
                    outputY1
                ) ||
                !scaleEndpoint(
                    y.validEnd,
                    decoder,
                    outputY2
                )
            ) {
                return false;
            }

            const TileRegion inputRegion{
                x.inputStart,
                y.inputStart,
                x.inputEnd - x.inputStart,
                y.inputEnd - y.inputStart,
            };
            const TileRegion outputRegion{
                outputX1,
                outputY1,
                outputX2 - outputX1,
                outputY2 - outputY1,
            };
            if (
                inputRegion.width <= 0 ||
                inputRegion.height <= 0 ||
                outputRegion.width <= 0 ||
                outputRegion.height <= 0 ||
                outputRegion.x < 0 ||
                outputRegion.y < 0 ||
                outputRegion.x >
                    outputWidth - outputRegion.width ||
                outputRegion.y >
                    outputHeight - outputRegion.height
            ) {
                return false;
            }

            plan.tiles.push_back(
                VaeTileRegion{
                    inputRegion,
                    outputRegion,
                }
            );
        }
    }

    if (plan.tiles.empty()) {
        return false;
    }

    output = std::move(plan);
    return true;
}

bool localOutputCropForVaeTile(
    const VaeTileRegion& tile,
    bool decoder,
    TileRegion& output
) {
    output = TileRegion{};

    if (
        tile.input.x < 0 ||
        tile.input.y < 0 ||
        tile.input.width <= 0 ||
        tile.input.height <= 0 ||
        tile.output.x < 0 ||
        tile.output.y < 0 ||
        tile.output.width <= 0 ||
        tile.output.height <= 0
    ) {
        return false;
    }

    int inputEndX = 0;
    int inputEndY = 0;
    if (
        !checkedAdd(
            tile.input.x,
            tile.input.width,
            inputEndX
        ) ||
        !checkedAdd(
            tile.input.y,
            tile.input.height,
            inputEndY
        )
    ) {
        return false;
    }

    int scaledInputX = 0;
    int scaledInputY = 0;
    int scaledInputEndX = 0;
    int scaledInputEndY = 0;
    if (
        !scaleEndpoint(
            tile.input.x,
            decoder,
            scaledInputX
        ) ||
        !scaleEndpoint(
            tile.input.y,
            decoder,
            scaledInputY
        ) ||
        !scaleEndpoint(
            inputEndX,
            decoder,
            scaledInputEndX
        ) ||
        !scaleEndpoint(
            inputEndY,
            decoder,
            scaledInputEndY
        )
    ) {
        return false;
    }

    const int localX = tile.output.x - scaledInputX;
    const int localY = tile.output.y - scaledInputY;
    const int fullWidth =
        scaledInputEndX - scaledInputX;
    const int fullHeight =
        scaledInputEndY - scaledInputY;
    if (
        localX < 0 ||
        localY < 0 ||
        fullWidth <= 0 ||
        fullHeight <= 0 ||
        tile.output.width > fullWidth - localX ||
        tile.output.height > fullHeight - localY
    ) {
        return false;
    }

    output = TileRegion{
        localX,
        localY,
        tile.output.width,
        tile.output.height,
    };
    return true;
}

}  // namespace kira::pisa
