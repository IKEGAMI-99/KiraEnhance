#include "pisa_group_norm_stats.h"

#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>

namespace kira::pisa {
namespace {

bool checkedElementCount(
    int batch,
    int channels,
    int height,
    int width,
    std::size_t& output
) {
    if (
        batch <= 0 ||
        channels <= 0 ||
        height <= 0 ||
        width <= 0
    ) {
        return false;
    }

    std::size_t count = static_cast<std::size_t>(batch);
    for (int value : {channels, height, width}) {
        const auto dimension = static_cast<std::size_t>(value);
        if (
            count >
            std::numeric_limits<std::size_t>::max() / dimension
        ) {
            return false;
        }
        count *= dimension;
    }
    output = count;
    return true;
}

}  // namespace

bool computeGroupNormTileStats(
    const float* input,
    int batch,
    int channels,
    int height,
    int width,
    int groups,
    GroupNormTileStats& output
) {
    output = GroupNormTileStats{};
    if (
        input == nullptr ||
        groups <= 0 ||
        channels <= 0 ||
        channels % groups != 0
    ) {
        return false;
    }

    std::size_t elementCount = 0;
    if (!checkedElementCount(
        batch,
        channels,
        height,
        width,
        elementCount
    )) {
        return false;
    }

    const std::size_t spatial =
        static_cast<std::size_t>(height) *
        static_cast<std::size_t>(width);
    const int channelsPerGroup = channels / groups;
    const std::size_t valuesPerGroup =
        static_cast<std::size_t>(channelsPerGroup) * spatial;
    const std::size_t statCount =
        static_cast<std::size_t>(batch) *
        static_cast<std::size_t>(groups);

    GroupNormTileStats result;
    result.batch = batch;
    result.groups = groups;
    result.spatialPixels = spatial;
    result.means.resize(statCount);
    result.variances.resize(statCount);

    for (int batchIndex = 0; batchIndex < batch; ++batchIndex) {
        const std::size_t batchOffset =
            static_cast<std::size_t>(batchIndex) *
            static_cast<std::size_t>(channels) * spatial;

        for (int group = 0; group < groups; ++group) {
            const std::size_t statIndex =
                static_cast<std::size_t>(batchIndex) *
                    static_cast<std::size_t>(groups) +
                static_cast<std::size_t>(group);
            const int firstChannel = group * channelsPerGroup;

            double sum = 0.0;
            for (
                int localChannel = 0;
                localChannel < channelsPerGroup;
                ++localChannel
            ) {
                const int channel = firstChannel + localChannel;
                const std::size_t channelOffset =
                    batchOffset +
                    static_cast<std::size_t>(channel) * spatial;
                for (std::size_t pixel = 0; pixel < spatial; ++pixel) {
                    const float value = input[channelOffset + pixel];
                    if (!std::isfinite(value)) {
                        return false;
                    }
                    sum += static_cast<double>(value);
                }
            }

            const double mean =
                sum / static_cast<double>(valuesPerGroup);
            double squaredError = 0.0;
            for (
                int localChannel = 0;
                localChannel < channelsPerGroup;
                ++localChannel
            ) {
                const int channel = firstChannel + localChannel;
                const std::size_t channelOffset =
                    batchOffset +
                    static_cast<std::size_t>(channel) * spatial;
                for (std::size_t pixel = 0; pixel < spatial; ++pixel) {
                    const double delta =
                        static_cast<double>(
                            input[channelOffset + pixel]
                        ) - mean;
                    squaredError += delta * delta;
                }
            }

            const double variance =
                squaredError /
                static_cast<double>(valuesPerGroup);
            if (!std::isfinite(mean) || !std::isfinite(variance)) {
                return false;
            }

            result.means[statIndex] = static_cast<float>(mean);
            result.variances[statIndex] =
                static_cast<float>(variance);
        }
    }

    output = std::move(result);
    return true;
}

bool summarizeGroupNormTileStats(
    const std::vector<GroupNormTileStats>& tiles,
    std::vector<float>& means,
    std::vector<float>& variances
) {
    means.clear();
    variances.clear();
    if (tiles.empty()) {
        return false;
    }

    const int batch = tiles.front().batch;
    const int groups = tiles.front().groups;
    if (batch <= 0 || groups <= 0) {
        return false;
    }
    const std::size_t statCount =
        static_cast<std::size_t>(batch) *
        static_cast<std::size_t>(groups);

    std::size_t maxPixels = 0;
    for (const auto& tile : tiles) {
        if (
            tile.batch != batch ||
            tile.groups != groups ||
            tile.spatialPixels == 0 ||
            tile.means.size() != statCount ||
            tile.variances.size() != statCount
        ) {
            return false;
        }
        if (tile.spatialPixels > maxPixels) {
            maxPixels = tile.spatialPixels;
        }
    }
    if (maxPixels == 0) {
        return false;
    }

    std::vector<float> normalizedPixels;
    normalizedPixels.reserve(tiles.size());
    float sumPixels = 0.0f;
    for (const auto& tile : tiles) {
        const float normalized =
            static_cast<float>(tile.spatialPixels) /
            static_cast<float>(maxPixels);
        if (!std::isfinite(normalized) || normalized <= 0.0f) {
            return false;
        }
        normalizedPixels.push_back(normalized);
        sumPixels += normalized;
    }
    if (!std::isfinite(sumPixels) || sumPixels <= 0.0f) {
        return false;
    }

    std::vector<double> combinedMeans(statCount, 0.0);
    std::vector<double> combinedVariances(statCount, 0.0);

    for (std::size_t tileIndex = 0; tileIndex < tiles.size(); ++tileIndex) {
        const double weight =
            static_cast<double>(normalizedPixels[tileIndex]) /
            static_cast<double>(sumPixels);
        const auto& tile = tiles[tileIndex];
        for (std::size_t stat = 0; stat < statCount; ++stat) {
            const float mean = tile.means[stat];
            const float variance = tile.variances[stat];
            if (
                !std::isfinite(mean) ||
                !std::isfinite(variance) ||
                variance < 0.0f
            ) {
                return false;
            }
            combinedMeans[stat] +=
                static_cast<double>(mean) * weight;
        }
    }

    for (std::size_t tileIndex = 0; tileIndex < tiles.size(); ++tileIndex) {
        const double weight =
            static_cast<double>(normalizedPixels[tileIndex]) /
            static_cast<double>(sumPixels);
        const auto& tile = tiles[tileIndex];
        for (std::size_t stat = 0; stat < statCount; ++stat) {
            const double delta =
                static_cast<double>(tile.means[stat]) -
                combinedMeans[stat];
            combinedVariances[stat] += weight * (
                static_cast<double>(tile.variances[stat]) +
                delta * delta
            );
        }
    }

    means.resize(statCount);
    variances.resize(statCount);
    for (std::size_t stat = 0; stat < statCount; ++stat) {
        if (
            !std::isfinite(combinedMeans[stat]) ||
            !std::isfinite(combinedVariances[stat]) ||
            combinedVariances[stat] < 0.0
        ) {
            means.clear();
            variances.clear();
            return false;
        }
        means[stat] = static_cast<float>(combinedMeans[stat]);
        variances[stat] =
            static_cast<float>(combinedVariances[stat]);
    }

    return true;
}

bool applyGroupNormWithStats(
    const float* input,
    int batch,
    int channels,
    int height,
    int width,
    int groups,
    const float* means,
    const float* variances,
    const float* weight,
    const float* bias,
    float epsilon,
    float* output,
    std::size_t outputCount
) {
    if (
        input == nullptr ||
        means == nullptr ||
        variances == nullptr ||
        output == nullptr ||
        groups <= 0 ||
        channels <= 0 ||
        channels % groups != 0 ||
        !std::isfinite(epsilon) ||
        epsilon <= 0.0f
    ) {
        return false;
    }

    std::size_t elementCount = 0;
    if (
        !checkedElementCount(
            batch,
            channels,
            height,
            width,
            elementCount
        ) ||
        outputCount < elementCount
    ) {
        return false;
    }

    const std::size_t spatial =
        static_cast<std::size_t>(height) *
        static_cast<std::size_t>(width);
    const int channelsPerGroup = channels / groups;

    for (int batchIndex = 0; batchIndex < batch; ++batchIndex) {
        const std::size_t batchOffset =
            static_cast<std::size_t>(batchIndex) *
            static_cast<std::size_t>(channels) * spatial;
        for (int channel = 0; channel < channels; ++channel) {
            const int group = channel / channelsPerGroup;
            const std::size_t statIndex =
                static_cast<std::size_t>(batchIndex) *
                    static_cast<std::size_t>(groups) +
                static_cast<std::size_t>(group);
            const float mean = means[statIndex];
            const float variance = variances[statIndex];
            if (
                !std::isfinite(mean) ||
                !std::isfinite(variance) ||
                variance < 0.0f
            ) {
                return false;
            }

            const float denominator =
                std::sqrt(variance + epsilon);
            if (
                !std::isfinite(denominator) ||
                denominator <= 0.0f
            ) {
                return false;
            }

            const float channelWeight =
                weight == nullptr ? 1.0f : weight[channel];
            const float channelBias =
                bias == nullptr ? 0.0f : bias[channel];
            if (
                !std::isfinite(channelWeight) ||
                !std::isfinite(channelBias)
            ) {
                return false;
            }

            const std::size_t channelOffset =
                batchOffset +
                static_cast<std::size_t>(channel) * spatial;
            for (std::size_t pixel = 0; pixel < spatial; ++pixel) {
                const float value = input[channelOffset + pixel];
                if (!std::isfinite(value)) {
                    return false;
                }
                output[channelOffset + pixel] =
                    ((value - mean) / denominator) *
                        channelWeight +
                    channelBias;
            }
        }
    }

    return true;
}

}  // namespace kira::pisa
