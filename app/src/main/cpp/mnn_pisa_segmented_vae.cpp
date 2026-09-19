#include "mnn_pisa_segmented_vae.h"

#if KIRA_HAS_MNN

#include "mnn_pisa_tensor_io.h"
#include "pisa_group_norm_stats.h"
#include "pisa_tile_tensor.h"
#include "pisa_vae_tile_plan.h"
#include "pisa_vae_tile_tensor.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

namespace kira::pisa {
namespace {

struct TileState {
    std::vector<float> activation;
    int activationChannels = 0;
    int activationHeight = 0;
    int activationWidth = 0;

    std::vector<float> residual;
    int residualChannels = 0;
    int residualHeight = 0;
    int residualWidth = 0;
    bool hasResidual = false;
};

bool checkedAdd(
    std::size_t left,
    std::size_t right,
    std::size_t& output
) {
    if (
        right >
        std::numeric_limits<std::size_t>::max() - left
    ) {
        return false;
    }
    output = left + right;
    return true;
}

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

bool nchwCount(
    int channels,
    int height,
    int width,
    std::size_t& output
) {
    if (channels <= 0 || height <= 0 || width <= 0) {
        return false;
    }

    std::size_t pixels = 0;
    return
        checkedMultiply(
            static_cast<std::size_t>(height),
            static_cast<std::size_t>(width),
            pixels
        ) &&
        checkedMultiply(
            static_cast<std::size_t>(channels),
            pixels,
            output
        );
}

bool floatVectorBytes(
    const std::vector<float>& values,
    std::size_t& output
) {
    return checkedMultiply(
        values.size(),
        sizeof(float),
        output
    );
}

bool tileStateBytes(
    const TileState& state,
    std::size_t& output
) {
    std::size_t activationBytes = 0;
    std::size_t residualBytes = 0;
    return
        floatVectorBytes(state.activation, activationBytes) &&
        floatVectorBytes(state.residual, residualBytes) &&
        checkedAdd(
            activationBytes,
            residualBytes,
            output
        );
}

bool trackedStateBytes(
    const std::vector<TileState>& states,
    std::size_t& output
) {
    output = 0;
    for (const TileState& state : states) {
        std::size_t stateBytes = 0;
        if (
            !tileStateBytes(state, stateBytes) ||
            !checkedAdd(output, stateBytes, output)
        ) {
            return false;
        }
    }
    return true;
}

void recordPeak(
    SegmentedVaeMetrics* metrics,
    std::size_t bytes
) {
    if (metrics != nullptr) {
        metrics->peakTrackedBytes =
            std::max(metrics->peakTrackedBytes, bytes);
    }
}

bool tensorShape(
    const MNN::Tensor* tensor,
    int& channels,
    int& height,
    int& width
) {
    if (tensor == nullptr) {
        return false;
    }
    const auto shape = tensor->shape();
    if (
        shape.size() != 4 ||
        shape[0] != 1 ||
        shape[1] <= 0 ||
        shape[2] <= 0 ||
        shape[3] <= 0
    ) {
        return false;
    }
    channels = shape[1];
    height = shape[2];
    width = shape[3];
    return true;
}

MNN::Session* createSegmentSession(
    MNN::Interpreter* interpreter,
    MNNForwardType backend
) {
    if (interpreter == nullptr) {
        return nullptr;
    }

    MNN::BackendConfig backendConfig;
    backendConfig.precision = MNN::BackendConfig::Precision_Low;
    backendConfig.memory = MNN::BackendConfig::Memory_Low;
    backendConfig.power = MNN::BackendConfig::Power_High;

    MNN::ScheduleConfig config;
    config.type = backend;
    config.backupType = MNN_FORWARD_CPU;
    config.backendConfig = &backendConfig;

    switch (backend) {
        case MNN_FORWARD_OPENCL:
            config.mode = MNN_GPU_TUNING_FAST;
            break;
        case MNN_FORWARD_VULKAN:
            config.mode = MNN_GPU_TUNING_NONE;
            break;
        case MNN_FORWARD_CPU:
            config.numThread = 4;
            break;
        default:
            return nullptr;
    }

    return interpreter->createSession(config);
}

bool cancelled(
    SegmentedVaeCancelProbe probe,
    void* context
) {
    return probe != nullptr && probe(context);
}

bool runSegmentForTile(
    MNN::Interpreter* interpreter,
    MNN::Session* session,
    TileState& state
) {
    if (
        interpreter == nullptr ||
        session == nullptr ||
        state.activation.empty()
    ) {
        return false;
    }

    MNN::Tensor* activationInput =
        interpreter->getSessionInput(session, "activation");
    MNN::Tensor* residualInput =
        interpreter->getSessionInput(session, "residual");
    if (activationInput == nullptr) {
        return false;
    }
    if ((residualInput != nullptr) != state.hasResidual) {
        return false;
    }

    interpreter->resizeTensor(
        activationInput,
        {
            1,
            state.activationChannels,
            state.activationHeight,
            state.activationWidth,
        }
    );
    if (residualInput != nullptr) {
        interpreter->resizeTensor(
            residualInput,
            {
                1,
                state.residualChannels,
                state.residualHeight,
                state.residualWidth,
            }
        );
    }
    interpreter->resizeSession(session);

    if (
        !writeFloatNchwTensor(
            activationInput,
            state.activation.data(),
            state.activation.size()
        ) ||
        (
            residualInput != nullptr &&
            !writeFloatNchwTensor(
                residualInput,
                state.residual.data(),
                state.residual.size()
            )
        )
    ) {
        return false;
    }

    if (interpreter->runSession(session) != MNN::NO_ERROR) {
        return false;
    }

    const MNN::Tensor* activationOutput =
        interpreter->getSessionOutput(session, "activation_out");
    const MNN::Tensor* residualOutput =
        interpreter->getSessionOutput(session, "residual_out");
    int activationChannels = 0;
    int activationHeight = 0;
    int activationWidth = 0;
    if (
        !tensorShape(
            activationOutput,
            activationChannels,
            activationHeight,
            activationWidth
        )
    ) {
        return false;
    }

    std::size_t activationCount = 0;
    if (
        !nchwCount(
            activationChannels,
            activationHeight,
            activationWidth,
            activationCount
        )
    ) {
        return false;
    }
    // The input vectors are no longer needed after runSession(). Reuse their
    // storage for the segment outputs instead of keeping old + new tile
    // buffers alive at the same time. This removes one avoidable per-tile
    // allocation peak when the output fits the existing capacity.
    state.activation.resize(activationCount);
    if (
        !readFloatNchwTensor(
            activationOutput,
            state.activation.data(),
            state.activation.size()
        )
    ) {
        return false;
    }
    state.activationChannels = activationChannels;
    state.activationHeight = activationHeight;
    state.activationWidth = activationWidth;

    int residualChannels = 0;
    int residualHeight = 0;
    int residualWidth = 0;
    if (residualOutput != nullptr) {
        if (
            !tensorShape(
                residualOutput,
                residualChannels,
                residualHeight,
                residualWidth
            )
        ) {
            return false;
        }
        std::size_t residualCount = 0;
        if (
            !nchwCount(
                residualChannels,
                residualHeight,
                residualWidth,
                residualCount
            )
        ) {
            return false;
        }
        state.residual.resize(residualCount);
        if (
            !readFloatNchwTensor(
                residualOutput,
                state.residual.data(),
                state.residual.size()
            )
        ) {
            return false;
        }
        state.hasResidual = true;
    } else {
        // Drop stale residual storage as soon as the segment consumes it.
        // swap() is used instead of clear() so the heap allocation is
        // released rather than merely changing the logical size.
        std::vector<float>().swap(state.residual);
        state.hasResidual = false;
    }
    state.residualChannels = residualChannels;
    state.residualHeight = residualHeight;
    state.residualWidth = residualWidth;
    return true;
}

bool normalizeStates(
    std::vector<TileState>& states,
    const VaeGroupNormAffine& affine
) {
    if (states.empty()) {
        return false;
    }

    std::vector<GroupNormTileBuffer> buffers;
    buffers.reserve(states.size());

    for (TileState& state : states) {
        if (
            state.activationChannels != affine.channels ||
            state.activation.empty()
        ) {
            return false;
        }
        buffers.push_back(
            GroupNormTileBuffer{
                state.activation.data(),
                state.activation.data(),
                1,
                state.activationChannels,
                state.activationHeight,
                state.activationWidth,
                state.activation.size(),
            }
        );
    }

    // normalizeGroupNormTiles gathers every tile's statistics before
    // applying the shared distribution, so the transform is safe in place.
    return normalizeGroupNormTiles(
        buffers,
        affine.groups,
        affine.weight.data(),
        affine.bias.data(),
        affine.epsilon
    );
}

}  // namespace

bool runMnnSegmentedVae(
    const std::vector<MNN::Interpreter*>& segments,
    const std::vector<VaeGroupNormAffine>& affine,
    MNNForwardType backend,
    const float* input,
    int inputChannels,
    int inputWidth,
    int inputHeight,
    int tileSize,
    int padding,
    bool decoder,
    int outputChannels,
    SegmentedVaeCancelProbe cancelProbe,
    void* cancelContext,
    float* output,
    std::size_t outputCount,
    SegmentedVaeMetrics* metrics
) {
    if (metrics != nullptr) {
        *metrics = SegmentedVaeMetrics{};
    }

    if (
        segments.empty() ||
        affine.size() + 1 != segments.size() ||
        input == nullptr ||
        output == nullptr ||
        input == output ||
        inputChannels <= 0 ||
        outputChannels <= 0
    ) {
        return false;
    }

    VaeTilePlan plan;
    if (
        !buildVaeTilePlan(
            inputWidth,
            inputHeight,
            tileSize,
            padding,
            decoder,
            plan
        ) ||
        plan.tiles.empty()
    ) {
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

    if (metrics != nullptr) {
        metrics->tileCount = plan.tiles.size();
        metrics->segmentCount = segments.size();
    }

    std::vector<TileState> states(plan.tiles.size());
    for (std::size_t index = 0; index < plan.tiles.size(); ++index) {
        if (cancelled(cancelProbe, cancelContext)) {
            return false;
        }

        const TileRegion& tile = plan.tiles[index].input;
        std::size_t tileCount = 0;
        if (
            !nchwCount(
                inputChannels,
                tile.height,
                tile.width,
                tileCount
            )
        ) {
            return false;
        }

        TileState& state = states[index];
        state.activation.resize(tileCount);
        state.activationChannels = inputChannels;
        state.activationHeight = tile.height;
        state.activationWidth = tile.width;
        if (
            !extractPlanarTile(
                input,
                inputChannels,
                inputWidth,
                inputHeight,
                tile,
                state.activation.data(),
                state.activation.size()
            )
        ) {
            return false;
        }

        std::size_t trackedBytes = 0;
        if (!trackedStateBytes(states, trackedBytes)) {
            return false;
        }
        recordPeak(metrics, trackedBytes);
    }

    for (
        std::size_t segmentIndex = 0;
        segmentIndex < segments.size();
        ++segmentIndex
    ) {
        if (cancelled(cancelProbe, cancelContext)) {
            return false;
        }

        MNN::Interpreter* interpreter = segments[segmentIndex];
        MNN::Session* session =
            createSegmentSession(interpreter, backend);
        if (session == nullptr) {
            return false;
        }

        bool success = true;
        for (TileState& state : states) {
            std::size_t beforeBytes = 0;
            if (!trackedStateBytes(states, beforeBytes)) {
                success = false;
                break;
            }
            recordPeak(metrics, beforeBytes);

            if (
                cancelled(cancelProbe, cancelContext) ||
                !runSegmentForTile(
                    interpreter,
                    session,
                    state
                )
            ) {
                success = false;
                break;
            }

            std::size_t afterBytes = 0;
            if (!trackedStateBytes(states, afterBytes)) {
                success = false;
                break;
            }
            // runSegmentForTile now reuses each tile's vectors in place, so
            // the old implementation's before + next-tile estimate would
            // double-count storage that no longer exists concurrently.
            recordPeak(metrics, afterBytes);
        }
        interpreter->releaseSession(session);
        if (!success) {
            return false;
        }

        if (segmentIndex < affine.size()) {
            std::size_t stateBytes = 0;
            if (!trackedStateBytes(states, stateBytes)) {
                return false;
            }
            recordPeak(metrics, stateBytes);

            if (
                !normalizeStates(
                    states,
                    affine[segmentIndex]
                )
            ) {
                return false;
            }
        }
    }

    std::fill(output, output + requiredOutput, 0.0f);
    std::vector<std::uint8_t> coverage(outputPixels, 0U);
    for (std::size_t index = 0; index < states.size(); ++index) {
        const TileState& state = states[index];
        if (
            state.hasResidual ||
            state.activationChannels != outputChannels
        ) {
            return false;
        }

        TileRegion crop;
        if (
            !localOutputCropForVaeTile(
                plan.tiles[index],
                decoder,
                crop
            ) ||
            !copyCroppedPlanarTile(
                state.activation.data(),
                state.activationChannels,
                state.activationWidth,
                state.activationHeight,
                crop,
                plan.tiles[index].output,
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

#endif
