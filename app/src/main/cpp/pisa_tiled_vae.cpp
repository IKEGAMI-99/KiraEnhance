#include "pisa_tiled_vae.h"

#include "pisa_tile_tensor.h"
#include "pisa_vae_tile_plan.h"
#include "pisa_vae_tile_tensor.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace kira::pisa {
namespace {

constexpr int VAE_SPATIAL_SCALE = 8;

bool checkedMultiply(
    std::size_t left,
    std::size_t right,
    std::size_t& output
) {
    if (
        left != 0 &&
        right >
            std::numeric_limits<std::size_t>::max() / left
    ) {
        return false;
    }
    output = left * right;
    return true;
}

bool scaledTileDimension(
    int input,
    bool decoder,
    int& output
) {
    if (input <= 0) {
        return false;
    }

    if (decoder) {
        const std::int64_t scaled =
            static_cast<std::int64_t>(input) *
            VAE_SPATIAL_SCALE;
        if (
            scaled <= 0 ||
            scaled > std::numeric_limits<int>::max()
        ) {
            return false;
        }
        output = static_cast<int>(scaled);
        return true;
    }

    if (input % VAE_SPATIAL_SCALE != 0) {
        return false;
    }
    output = input / VAE_SPATIAL_SCALE;
    return output > 0;
}

}  // namespace

bool runTiledVaePlanarTransform(
    const float* input,
    int inputChannels,
    int inputWidth,
    int inputHeight,
    int tileSize,
    int padding,
    bool decoder,
    int outputChannels,
    VaeTileTransform transform,
    void* context,
    float* output,
    std::size_t outputCount
) {
    if (
        input == nullptr ||
        output == nullptr ||
        input == output ||
        transform == nullptr ||
        inputChannels <= 0 ||
        outputChannels <= 0
    ) {
        return false;
    }

    VaeTilePlan plan;
    if (!buildVaeTilePlan(
        inputWidth,
        inputHeight,
        tileSize,
        padding,
        decoder,
        plan
    )) {
        return false;
    }

    std::size_t outputPixels = 0;
    std::size_t requiredOutput = 0;
    if (
        !checkedMultiply(
            static_cast<std::size_t>(plan.outputWidth),
            static_cast<std::size_t>(plan.outputHeight),
            outputPixels
        ) ||
        !checkedMultiply(
            static_cast<std::size_t>(outputChannels),
            outputPixels,
            requiredOutput
        ) ||
        outputCount < requiredOutput
    ) {
        return false;
    }

    std::vector<std::uint8_t> coverage(outputPixels, 0U);

    for (const VaeTileRegion& tile : plan.tiles) {
        std::size_t inputPixels = 0;
        std::size_t inputValues = 0;
        if (
            !checkedMultiply(
                static_cast<std::size_t>(tile.input.width),
                static_cast<std::size_t>(tile.input.height),
                inputPixels
            ) ||
            !checkedMultiply(
                static_cast<std::size_t>(inputChannels),
                inputPixels,
                inputValues
            )
        ) {
            return false;
        }

        int tileOutputWidth = 0;
        int tileOutputHeight = 0;
        if (
            !scaledTileDimension(
                tile.input.width,
                decoder,
                tileOutputWidth
            ) ||
            !scaledTileDimension(
                tile.input.height,
                decoder,
                tileOutputHeight
            )
        ) {
            return false;
        }

        std::size_t tileOutputPixels = 0;
        std::size_t tileOutputValues = 0;
        if (
            !checkedMultiply(
                static_cast<std::size_t>(tileOutputWidth),
                static_cast<std::size_t>(tileOutputHeight),
                tileOutputPixels
            ) ||
            !checkedMultiply(
                static_cast<std::size_t>(outputChannels),
                tileOutputPixels,
                tileOutputValues
            )
        ) {
            return false;
        }

        std::vector<float> tileInput(inputValues);
        std::vector<float> tileOutput(tileOutputValues);
        if (
            !extractPlanarTile(
                input,
                inputChannels,
                inputWidth,
                inputHeight,
                tile.input,
                tileInput.data(),
                tileInput.size()
            ) ||
            !transform(
                tileInput.data(),
                tileInput.size(),
                inputChannels,
                tile.input.width,
                tile.input.height,
                tileOutput.data(),
                tileOutput.size(),
                outputChannels,
                tileOutputWidth,
                tileOutputHeight,
                context
            )
        ) {
            return false;
        }

        TileRegion localCrop;
        if (
            !localOutputCropForVaeTile(
                tile,
                decoder,
                localCrop
            ) ||
            !copyCroppedPlanarTile(
                tileOutput.data(),
                outputChannels,
                tileOutputWidth,
                tileOutputHeight,
                localCrop,
                tile.output,
                plan.outputWidth,
                plan.outputHeight,
                output,
                outputCount,
                coverage.data(),
                coverage.size()
            )
        ) {
            return false;
        }
    }

    return validateExactCoverage(
        coverage.data(),
        coverage.size()
    );
}

}  // namespace kira::pisa
