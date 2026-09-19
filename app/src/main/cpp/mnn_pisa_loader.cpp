#include <jni.h>

#include "mnn_pisa_segmented_vae.h"
#include "mnn_pisa_tensor_io.h"
#include "pisa_adain.h"
#include "pisa_denoise_math.h"
#include "pisa_gaussian_noise.h"
#include "pisa_latent_math.h"
#include "pisa_image_tensor.h"
#include "pisa_pillow_resize.h"
#include "pisa_preprocess.h"
#include "pisa_resize_plan.h"
#include "pisa_tile_plan.h"
#include "pisa_tiled_inference.h"
#include "pisa_vae_segment_pack.h"
#include "pisa_vae_tile_plan.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

#if KIRA_HAS_MNN
#include <MNN/Interpreter.hpp>
#endif

namespace {

constexpr std::size_t EMPTY_PROMPT_FP16_BYTES =
    static_cast<std::size_t>(1) * 77 * 1024 * 2;
constexpr float VAE_SCALING_FACTOR = 0.18215f;
constexpr std::int64_t PISA_TIMESTEP = 1;
constexpr std::uint64_t SMOKE_NOISE_SEED = 0x50495341534d4f4bULL;
constexpr std::uint64_t INFERENCE_NOISE_SEED = 42ULL;
constexpr std::size_t MAX_MONOLITHIC_MODEL_PIXELS =
    static_cast<std::size_t>(1024) * 1024;
constexpr int PISA_UNET_TILE_SIZE = 96;
constexpr int PISA_UNET_TILE_OVERLAP = 32;
constexpr std::size_t PISA_VAE_ENCODER_SEGMENTS = 23;
constexpr std::size_t PISA_VAE_DECODER_SEGMENTS = 31;
constexpr int PISA_VAE_ENCODER_TILE_SIZE = 1024;
constexpr int PISA_VAE_ENCODER_PADDING = 32;
constexpr int PISA_VAE_DECODER_TILE_SIZE = 224;
constexpr int PISA_VAE_DECODER_PADDING = 11;
constexpr int SMOKE_SOURCE_WIDTH = 2;
constexpr int SMOKE_SOURCE_HEIGHT = 2;
constexpr int SMOKE_SOURCE_STRIDE = SMOKE_SOURCE_WIDTH * 4;
constexpr std::uint8_t SMOKE_SOURCE_RGBA[] = {
    16, 32, 64, 255,
    224, 48, 32, 255,
    24, 208, 80, 255,
    240, 224, 192, 255,
};

enum class NativeError : jlong {
    NONE = 0,
    LOAD_FAILED = 1,
    INVALID_ARGUMENT = 2,
    NOT_IMPLEMENTED = 3,
    INFERENCE_FAILED = 4,
    CANCELLED = 5,
    OUT_OF_MEMORY = 6,
    INTERNAL = 7,
};

jlongArray makeLoadResult(
    JNIEnv* env,
    jlong handle,
    NativeError error,
    bool gpuEnabled
) {
    jlongArray result = env->NewLongArray(3);
    if (result == nullptr) {
        return nullptr;
    }

    const jlong values[3] = {
        handle,
        static_cast<jlong>(error),
        gpuEnabled ? 1L : 0L,
    };
    env->SetLongArrayRegion(result, 0, 3, values);
    return result;
}

jlongArray makePrepareResult(
    JNIEnv* env,
    NativeError error,
    jint imageWidth = 0,
    jint imageHeight = 0,
    jint latentWidth = 0,
    jint latentHeight = 0
) {
    jlongArray result = env->NewLongArray(5);
    if (result == nullptr) {
        return nullptr;
    }

    const jlong values[5] = {
        static_cast<jlong>(error),
        static_cast<jlong>(imageWidth),
        static_cast<jlong>(imageHeight),
        static_cast<jlong>(latentWidth),
        static_cast<jlong>(latentHeight),
    };
    env->SetLongArrayRegion(result, 0, 5, values);
    return result;
}

jlongArray makeSmokeResult(
    JNIEnv* env,
    NativeError error,
    jint completedStages = 0,
    bool outputFinite = false
) {
    jlongArray result = env->NewLongArray(3);
    if (result == nullptr) {
        return nullptr;
    }

    const jlong values[3] = {
        static_cast<jlong>(error),
        static_cast<jlong>(completedStages),
        outputFinite ? 1L : 0L,
    };
    env->SetLongArrayRegion(result, 0, 3, values);
    return result;
}

jlongArray makeInferenceResult(
    JNIEnv* env,
    NativeError error,
    jint outputWidth = 0,
    jint outputHeight = 0,
    jint outputRowStrideBytes = 0,
    bool gpuUsed = false
) {
    jlongArray result = env->NewLongArray(5);
    if (result == nullptr) {
        return nullptr;
    }

    const jlong values[5] = {
        static_cast<jlong>(error),
        static_cast<jlong>(outputWidth),
        static_cast<jlong>(outputHeight),
        static_cast<jlong>(outputRowStrideBytes),
        gpuUsed ? 1L : 0L,
    };
    env->SetLongArrayRegion(result, 0, 5, values);
    return result;
}

class ScopedUtfChars {
public:
    ScopedUtfChars(JNIEnv* env, jstring value)
        : env_(env), value_(value) {
        if (value_ != nullptr) {
            chars_ = env_->GetStringUTFChars(value_, nullptr);
        }
    }

    ~ScopedUtfChars() {
        if (chars_ != nullptr) {
            env_->ReleaseStringUTFChars(value_, chars_);
        }
    }

    ScopedUtfChars(const ScopedUtfChars&) = delete;
    ScopedUtfChars& operator=(const ScopedUtfChars&) = delete;

    bool copyTo(std::string& output) const {
        if (chars_ == nullptr || chars_[0] == '\0') {
            return false;
        }
        output.assign(chars_);
        return true;
    }

private:
    JNIEnv* env_;
    jstring value_;
    const char* chars_ = nullptr;
};

bool copyRequiredPath(JNIEnv* env, jstring value, std::string& output) {
    ScopedUtfChars chars(env, value);
    return chars.copyTo(output);
}

enum class ArtifactReadResult {
    OK,
    LOAD_FAILED,
    OUT_OF_MEMORY,
};

ArtifactReadResult readFp16Artifact(
    const std::string& path,
    std::unique_ptr<std::uint8_t[]>& output,
    std::size_t& outputBytes
) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        return ArtifactReadResult::LOAD_FAILED;
    }

    const std::streamoff size = input.tellg();
    if (
        size <= 0 ||
        (size % 2) != 0 ||
        size > static_cast<std::streamoff>(
            std::numeric_limits<std::streamsize>::max()
        )
    ) {
        return ArtifactReadResult::LOAD_FAILED;
    }

    const auto byteCount = static_cast<std::size_t>(size);
    std::unique_ptr<std::uint8_t[]> buffer(
        new (std::nothrow) std::uint8_t[byteCount]
    );
    if (!buffer) {
        return ArtifactReadResult::OUT_OF_MEMORY;
    }

    input.seekg(0, std::ios::beg);
    if (!input) {
        return ArtifactReadResult::LOAD_FAILED;
    }

    if (!input.read(
        reinterpret_cast<char*>(buffer.get()),
        static_cast<std::streamsize>(byteCount)
    )) {
        return ArtifactReadResult::LOAD_FAILED;
    }

    output = std::move(buffer);
    outputBytes = byteCount;
    return ArtifactReadResult::OK;
}

#if KIRA_HAS_MNN

struct InterpreterDeleter {
    void operator()(MNN::Interpreter* interpreter) const {
        if (interpreter != nullptr) {
            MNN::Interpreter::destroy(interpreter);
        }
    }
};

using InterpreterPtr = std::unique_ptr<MNN::Interpreter, InterpreterDeleter>;

struct PisaModelBundle {
    InterpreterPtr vaeEncoder;
    InterpreterPtr unet;
    InterpreterPtr vaeDecoder;
    kira::pisa::VaeSegmentPack vaeSegmentPack;
    std::vector<InterpreterPtr> vaeEncoderSegments;
    std::vector<InterpreterPtr> vaeDecoderSegments;
    std::unique_ptr<std::uint8_t[]> emptyPrompt;
    std::size_t emptyPromptBytes = 0;
    MNN::Session* vaeEncoderSession = nullptr;
    MNN::Session* unetSession = nullptr;
    MNN::Session* vaeDecoderSession = nullptr;
    MNNForwardType backend = MNN_FORWARD_CPU;
    std::atomic<bool> cancelRequested{false};
};

InterpreterPtr loadInterpreter(const std::string& path) {
    return InterpreterPtr(MNN::Interpreter::createFromFile(path.c_str()));
}

bool loadSegmentInterpreters(
    const std::vector<std::vector<std::uint8_t>>& models,
    std::vector<InterpreterPtr>& output
) {
    output.clear();
    output.reserve(models.size());
    for (const auto& model : models) {
        if (model.empty()) {
            output.clear();
            return false;
        }
        InterpreterPtr interpreter(
            MNN::Interpreter::createFromBuffer(
                model.data(),
                model.size()
            )
        );
        if (!interpreter) {
            output.clear();
            return false;
        }
        interpreter->setSessionMode(MNN::Interpreter::Session_Release);
        output.push_back(std::move(interpreter));
    }
    return true;
}

void releaseSession(
    MNN::Interpreter* interpreter,
    MNN::Session*& session
) {
    if (interpreter != nullptr && session != nullptr) {
        interpreter->releaseSession(session);
        session = nullptr;
    }
}

void releaseSessions(PisaModelBundle& bundle) {
    releaseSession(bundle.vaeEncoder.get(), bundle.vaeEncoderSession);
    releaseSession(bundle.unet.get(), bundle.unetSession);
    releaseSession(bundle.vaeDecoder.get(), bundle.vaeDecoderSession);
}

bool sessionUsesBackend(
    MNN::Interpreter* interpreter,
    MNN::Session* session,
    MNNForwardType expectedBackend
) {
    if (interpreter == nullptr || session == nullptr) {
        return false;
    }

    int backendTypes[2] = {-1, -1};
    if (!interpreter->getSessionInfo(
        session,
        MNN::Interpreter::BACKENDS,
        backendTypes
    )) {
        return false;
    }

    return backendTypes[0] == static_cast<int>(expectedBackend);
}

bool createSessionsForBackend(
    PisaModelBundle& bundle,
    MNNForwardType requestedBackend
) {
    releaseSessions(bundle);

    MNN::BackendConfig backendConfig;
    backendConfig.precision = MNN::BackendConfig::Precision_Low;
    backendConfig.memory = MNN::BackendConfig::Memory_Low;
    backendConfig.power = MNN::BackendConfig::Power_High;

    MNN::ScheduleConfig config;
    config.type = requestedBackend;
    config.backupType = MNN_FORWARD_CPU;
    config.backendConfig = &backendConfig;

    switch (requestedBackend) {
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
            return false;
    }

    bundle.vaeEncoderSession = bundle.vaeEncoder->createSession(config);
    if (bundle.vaeEncoderSession == nullptr) {
        releaseSessions(bundle);
        return false;
    }

    bundle.unetSession = bundle.unet->createSession(config);
    if (bundle.unetSession == nullptr) {
        releaseSessions(bundle);
        return false;
    }

    bundle.vaeDecoderSession = bundle.vaeDecoder->createSession(config);
    if (bundle.vaeDecoderSession == nullptr) {
        releaseSessions(bundle);
        return false;
    }

    if (
        !sessionUsesBackend(
            bundle.vaeEncoder.get(),
            bundle.vaeEncoderSession,
            requestedBackend
        ) ||
        !sessionUsesBackend(
            bundle.unet.get(),
            bundle.unetSession,
            requestedBackend
        ) ||
        !sessionUsesBackend(
            bundle.vaeDecoder.get(),
            bundle.vaeDecoderSession,
            requestedBackend
        )
    ) {
        releaseSessions(bundle);
        return false;
    }

    bundle.backend = requestedBackend;
    return true;
}

bool createPreferredSessions(
    PisaModelBundle& bundle,
    bool preferGpu
) {
    if (preferGpu) {
        if (createSessionsForBackend(bundle, MNN_FORWARD_OPENCL)) {
            return true;
        }
        if (createSessionsForBackend(bundle, MNN_FORWARD_VULKAN)) {
            return true;
        }
    }
    return createSessionsForBackend(bundle, MNN_FORWARD_CPU);
}

bool isCancellationRequested(
    const PisaModelBundle& bundle
) {
    return bundle.cancelRequested.load(
        std::memory_order_acquire
    );
}

class ScopedCancellationReset {
public:
    explicit ScopedCancellationReset(
        PisaModelBundle& bundle
    ) : bundle_(bundle) {}

    ~ScopedCancellationReset() {
        bundle_.cancelRequested.store(
            false,
            std::memory_order_release
        );
    }

    ScopedCancellationReset(
        const ScopedCancellationReset&
    ) = delete;
    ScopedCancellationReset& operator=(
        const ScopedCancellationReset&
    ) = delete;

private:
    PisaModelBundle& bundle_;
};

bool isGpuBackend(MNNForwardType backend) {
    return backend == MNN_FORWARD_OPENCL || backend == MNN_FORWARD_VULKAN;
}

const char* sessionInfoForBackend(MNNForwardType backend) {
    switch (backend) {
        case MNN_FORWARD_OPENCL:
            return "backend=opencl;gpu=1";
        case MNN_FORWARD_VULKAN:
            return "backend=vulkan;gpu=1";
        case MNN_FORWARD_CPU:
            return "backend=cpu;gpu=0";
        default:
            return "";
    }
}

void appendHex(std::string& output, const std::string& value) {
    static constexpr char HEX[] = "0123456789abcdef";
    for (unsigned char byte : value) {
        output.push_back(HEX[(byte >> 4) & 0x0f]);
        output.push_back(HEX[byte & 0x0f]);
    }
}

bool appendTensorInfo(
    std::string& output,
    const char* graph,
    const char* role,
    const std::string& name,
    MNN::Tensor* tensor
) {
    if (tensor == nullptr || name.empty()) {
        return false;
    }

    if (!output.empty()) {
        output.push_back('\n');
    }

    output += "graph=";
    output += graph;
    output += ";role=";
    output += role;
    output += ";nameHex=";
    appendHex(output, name);
    output += ";shape=";

    const auto shape = tensor->shape();
    for (std::size_t index = 0; index < shape.size(); ++index) {
        if (index > 0) {
            output.push_back(',');
        }
        output += std::to_string(shape[index]);
    }

    const auto type = tensor->getType();
    output += ";type=";
    output += std::to_string(static_cast<unsigned int>(type.code));
    output.push_back(',');
    output += std::to_string(static_cast<unsigned int>(type.bits));
    output.push_back(',');
    output += std::to_string(static_cast<unsigned int>(type.lanes));
    output += ";dim=";
    output += std::to_string(
        static_cast<int>(tensor->getDimensionType())
    );
    return true;
}

template <typename TensorMap>
bool appendTensorMap(
    std::string& output,
    const char* graph,
    const char* role,
    const TensorMap& tensors
) {
    for (const auto& entry : tensors) {
        if (!appendTensorInfo(
            output,
            graph,
            role,
            entry.first,
            entry.second
        )) {
            return false;
        }
    }
    return true;
}

bool tensorShapeEquals(
    const MNN::Tensor* tensor,
    const std::vector<int>& expected
) {
    if (tensor == nullptr) {
        return false;
    }
    return tensor->shape() == expected;
}

bool resizeInputsAndSession(
    MNN::Interpreter* interpreter,
    MNN::Session* session,
    const std::vector<std::pair<const char*, std::vector<int>>>& inputs
) {
    if (interpreter == nullptr || session == nullptr) {
        return false;
    }

    for (const auto& input : inputs) {
        MNN::Tensor* tensor = interpreter->getSessionInput(
            session,
            input.first
        );
        if (tensor == nullptr) {
            return false;
        }
        interpreter->resizeTensor(tensor, input.second);
    }

    interpreter->resizeSession(session);

    int resizeStatus = -1;
    if (!interpreter->getSessionInfo(
        session,
        MNN::Interpreter::RESIZE_STATUS,
        &resizeStatus
    )) {
        return false;
    }
    return resizeStatus == 0;
}

bool prepareUnetGraph(
    PisaModelBundle& bundle,
    int latentWidth,
    int latentHeight
) {
    if (latentWidth <= 0 || latentHeight <= 0) {
        return false;
    }

    if (!resizeInputsAndSession(
        bundle.unet.get(),
        bundle.unetSession,
        {
            {"latent", {1, 4, latentHeight, latentWidth}},
            {"timestep", {1}},
            {"encoder_hidden_states", {1, 77, 1024}},
        }
    )) {
        return false;
    }

    return tensorShapeEquals(
        bundle.unet->getSessionOutput(
            bundle.unetSession,
            "model_pred"
        ),
        {1, 4, latentHeight, latentWidth}
    );
}

bool prepareUnetForLatentGraph(
    PisaModelBundle& bundle,
    int latentWidth,
    int latentHeight
) {
    if (latentWidth <= 0 || latentHeight <= 0) {
        return false;
    }

    int unetWidth = latentWidth;
    int unetHeight = latentHeight;
    if (
        kira::pisa::requiresTiling(
            latentWidth,
            latentHeight,
            PISA_UNET_TILE_SIZE
        )
    ) {
        kira::pisa::TilePlan tilePlan;
        if (
            !kira::pisa::buildTilePlan(
                latentWidth,
                latentHeight,
                PISA_UNET_TILE_SIZE,
                PISA_UNET_TILE_OVERLAP,
                tilePlan
            )
        ) {
            return false;
        }
        unetWidth = tilePlan.tileSize;
        unetHeight = tilePlan.tileSize;
    }

    return prepareUnetGraph(
        bundle,
        unetWidth,
        unetHeight
    );
}

bool preparePisaGraph(
    PisaModelBundle& bundle,
    int imageWidth,
    int imageHeight,
    int& latentWidth,
    int& latentHeight
) {
    if (
        imageWidth <= 0 ||
        imageHeight <= 0 ||
        imageWidth % 8 != 0 ||
        imageHeight % 8 != 0
    ) {
        return false;
    }

    latentWidth = imageWidth / 8;
    latentHeight = imageHeight / 8;

    if (!resizeInputsAndSession(
        bundle.vaeEncoder.get(),
        bundle.vaeEncoderSession,
        {
            {"image", {1, 3, imageHeight, imageWidth}},
        }
    )) {
        return false;
    }

    if (!tensorShapeEquals(
        bundle.vaeEncoder->getSessionOutput(
            bundle.vaeEncoderSession,
            "moments"
        ),
        {1, 8, latentHeight, latentWidth}
    )) {
        return false;
    }

    if (!prepareUnetForLatentGraph(
        bundle,
        latentWidth,
        latentHeight
    )) {
        return false;
    }

    if (!resizeInputsAndSession(
        bundle.vaeDecoder.get(),
        bundle.vaeDecoderSession,
        {
            {"latent", {1, 4, latentHeight, latentWidth}},
        }
    )) {
        return false;
    }

    return tensorShapeEquals(
        bundle.vaeDecoder->getSessionOutput(
            bundle.vaeDecoderSession,
            "image"
        ),
        {1, 3, imageHeight, imageWidth}
    );
}

bool checkedElementCount(
    std::initializer_list<std::size_t> dimensions,
    std::size_t& output
) {
    std::size_t count = 1;
    for (std::size_t dimension : dimensions) {
        if (
            dimension == 0 ||
            count > std::numeric_limits<std::size_t>::max() / dimension
        ) {
            return false;
        }
        count *= dimension;
    }
    output = count;
    return true;
}

NativeError mapMnnRunError(MNN::ErrorCode error) {
    switch (error) {
        case MNN::NO_ERROR:
            return NativeError::NONE;
        case MNN::OUT_OF_MEMORY:
            return NativeError::OUT_OF_MEMORY;
        default:
            return NativeError::INFERENCE_FAILED;
    }
}

bool writeUnetConditioning(PisaModelBundle& bundle) {
    MNN::Tensor* timestep =
        bundle.unet->getSessionInput(
            bundle.unetSession,
            "timestep"
        );
    MNN::Tensor* prompt =
        bundle.unet->getSessionInput(
            bundle.unetSession,
            "encoder_hidden_states"
        );
    return kira::pisa::writeIntScalarTensor(
        timestep,
        PISA_TIMESTEP
    ) && kira::pisa::writeFp16BytesNchwTensor(
        prompt,
        bundle.emptyPrompt.get(),
        bundle.emptyPromptBytes
    );
}

struct UnetTileContext {
    PisaModelBundle* bundle = nullptr;
    NativeError error = NativeError::NONE;
};

bool runUnetTileTransform(
    const float* input,
    std::size_t inputCount,
    int channels,
    int tileWidth,
    int tileHeight,
    float* output,
    std::size_t outputCount,
    void* rawContext
) {
    auto* context =
        static_cast<UnetTileContext*>(rawContext);
    if (
        context == nullptr ||
        context->bundle == nullptr ||
        input == nullptr ||
        output == nullptr ||
        channels != 4 ||
        tileWidth <= 0 ||
        tileHeight <= 0
    ) {
        if (context != nullptr) {
            context->error = NativeError::INVALID_ARGUMENT;
        }
        return false;
    }

    std::size_t expectedCount = 0;
    if (
        !checkedElementCount(
            {
                4U,
                static_cast<std::size_t>(tileHeight),
                static_cast<std::size_t>(tileWidth),
            },
            expectedCount
        ) ||
        inputCount != expectedCount ||
        outputCount != expectedCount
    ) {
        context->error = NativeError::INVALID_ARGUMENT;
        return false;
    }

    PisaModelBundle& bundle = *context->bundle;
    if (isCancellationRequested(bundle)) {
        context->error = NativeError::CANCELLED;
        return false;
    }

    MNN::Tensor* latent =
        bundle.unet->getSessionInput(
            bundle.unetSession,
            "latent"
        );
    const MNN::Tensor* prediction =
        bundle.unet->getSessionOutput(
            bundle.unetSession,
            "model_pred"
        );
    if (
        !tensorShapeEquals(
            latent,
            {1, 4, tileHeight, tileWidth}
        ) ||
        !tensorShapeEquals(
            prediction,
            {1, 4, tileHeight, tileWidth}
        ) ||
        !kira::pisa::writeFloatNchwTensor(
            latent,
            input,
            inputCount
        )
    ) {
        context->error = NativeError::INFERENCE_FAILED;
        return false;
    }

    const MNN::ErrorCode run =
        bundle.unet->runSession(bundle.unetSession);
    if (run != MNN::NO_ERROR) {
        context->error = mapMnnRunError(run);
        return false;
    }

    if (
        !kira::pisa::readFloatNchwTensor(
            prediction,
            output,
            outputCount
        )
    ) {
        context->error = NativeError::INFERENCE_FAILED;
        return false;
    }

    return true;
}

NativeError runPisaLatentPipeline(
    PisaModelBundle& bundle,
    const float* moments,
    int latentWidth,
    int latentHeight,
    std::uint64_t noiseSeed,
    float* decoderLatent,
    std::size_t latentCount,
    int& completedStages
) {
    if (
        moments == nullptr ||
        decoderLatent == nullptr ||
        latentWidth <= 0 ||
        latentHeight <= 0 ||
        latentCount == 0
    ) {
        return NativeError::INVALID_ARGUMENT;
    }

    if (isCancellationRequested(bundle)) {
        return NativeError::CANCELLED;
    }

    std::unique_ptr<float[]> noise(
        new (std::nothrow) float[latentCount]
    );
    std::unique_ptr<float[]> modelPrediction(
        new (std::nothrow) float[latentCount]
    );
    if (!noise || !modelPrediction) {
        return NativeError::OUT_OF_MEMORY;
    }

    if (
        !kira::pisa::fillGaussianNoise(
            noise.get(),
            latentCount,
            noiseSeed
        ) ||
        !kira::pisa::sampleLatentFromMoments(
            moments,
            noise.get(),
            decoderLatent,
            1,
            4,
            latentHeight,
            latentWidth,
            VAE_SCALING_FACTOR
        )
    ) {
        return NativeError::INFERENCE_FAILED;
    }

    if (!writeUnetConditioning(bundle)) {
        return NativeError::INFERENCE_FAILED;
    }
    if (isCancellationRequested(bundle)) {
        return NativeError::CANCELLED;
    }

    if (
        kira::pisa::requiresTiling(
            latentWidth,
            latentHeight,
            PISA_UNET_TILE_SIZE
        )
    ) {
        UnetTileContext tileContext;
        tileContext.bundle = &bundle;
        if (
            !kira::pisa::runTiledPlanarTransform(
                decoderLatent,
                4,
                latentWidth,
                latentHeight,
                PISA_UNET_TILE_SIZE,
                PISA_UNET_TILE_OVERLAP,
                runUnetTileTransform,
                &tileContext,
                modelPrediction.get(),
                latentCount
            )
        ) {
            return tileContext.error == NativeError::NONE
                ? NativeError::INFERENCE_FAILED
                : tileContext.error;
        }
    } else {
        MNN::Tensor* unetLatent =
            bundle.unet->getSessionInput(
                bundle.unetSession,
                "latent"
            );
        if (
            !kira::pisa::writeFloatNchwTensor(
                unetLatent,
                decoderLatent,
                latentCount
            )
        ) {
            return NativeError::INFERENCE_FAILED;
        }

        const MNN::ErrorCode unetRun =
            bundle.unet->runSession(bundle.unetSession);
        if (unetRun != MNN::NO_ERROR) {
            return mapMnnRunError(unetRun);
        }

        const MNN::Tensor* unetOutput =
            bundle.unet->getSessionOutput(
                bundle.unetSession,
                "model_pred"
            );
        if (
            !kira::pisa::readFloatNchwTensor(
                unetOutput,
                modelPrediction.get(),
                latentCount
            )
        ) {
            return NativeError::INFERENCE_FAILED;
        }
    }

    completedStages = 2;
    if (isCancellationRequested(bundle)) {
        return NativeError::CANCELLED;
    }

    if (
        !kira::pisa::buildDecoderLatent(
            decoderLatent,
            modelPrediction.get(),
            decoderLatent,
            latentCount,
            VAE_SCALING_FACTOR
        )
    ) {
        return NativeError::INFERENCE_FAILED;
    }

    return NativeError::NONE;
}

bool collectSegmentInterpreters(
    const std::vector<InterpreterPtr>& owned,
    std::vector<MNN::Interpreter*>& output
) {
    output.clear();
    output.reserve(owned.size());
    for (const InterpreterPtr& interpreter : owned) {
        if (!interpreter) {
            output.clear();
            return false;
        }
        output.push_back(interpreter.get());
    }
    return !output.empty();
}

bool segmentedVaeCancelled(void* rawContext) {
    auto* bundle = static_cast<PisaModelBundle*>(rawContext);
    return bundle != nullptr && isCancellationRequested(*bundle);
}

NativeError runPisaSegmentedGraph(
    PisaModelBundle& bundle,
    const float* encoderImage,
    int imageWidth,
    int imageHeight,
    std::uint64_t noiseSeed,
    float* decodedImage,
    std::size_t decodedImageCount,
    int& completedStages
) {
    completedStages = 0;
    if (
        encoderImage == nullptr ||
        decodedImage == nullptr ||
        imageWidth <= 0 ||
        imageHeight <= 0 ||
        imageWidth % 8 != 0 ||
        imageHeight % 8 != 0
    ) {
        return NativeError::INVALID_ARGUMENT;
    }

    const int latentWidth = imageWidth / 8;
    const int latentHeight = imageHeight / 8;
    std::size_t momentsCount = 0;
    std::size_t latentCount = 0;
    std::size_t expectedImageCount = 0;
    if (
        !checkedElementCount(
            {
                8U,
                static_cast<std::size_t>(latentHeight),
                static_cast<std::size_t>(latentWidth),
            },
            momentsCount
        ) ||
        !checkedElementCount(
            {
                4U,
                static_cast<std::size_t>(latentHeight),
                static_cast<std::size_t>(latentWidth),
            },
            latentCount
        ) ||
        !checkedElementCount(
            {
                3U,
                static_cast<std::size_t>(imageHeight),
                static_cast<std::size_t>(imageWidth),
            },
            expectedImageCount
        ) ||
        decodedImageCount < expectedImageCount
    ) {
        return NativeError::OUT_OF_MEMORY;
    }

    std::unique_ptr<float[]> moments(
        new (std::nothrow) float[momentsCount]
    );
    std::unique_ptr<float[]> decoderLatent(
        new (std::nothrow) float[latentCount]
    );
    if (!moments || !decoderLatent) {
        return NativeError::OUT_OF_MEMORY;
    }

    std::vector<MNN::Interpreter*> encoderSegments;
    std::vector<MNN::Interpreter*> decoderSegments;
    if (
        !collectSegmentInterpreters(
            bundle.vaeEncoderSegments,
            encoderSegments
        ) ||
        !collectSegmentInterpreters(
            bundle.vaeDecoderSegments,
            decoderSegments
        )
    ) {
        return NativeError::LOAD_FAILED;
    }

    if (
        !kira::pisa::runMnnSegmentedVae(
            encoderSegments,
            bundle.vaeSegmentPack.encoderAffine,
            bundle.backend,
            encoderImage,
            3,
            imageWidth,
            imageHeight,
            PISA_VAE_ENCODER_TILE_SIZE,
            PISA_VAE_ENCODER_PADDING,
            false,
            8,
            segmentedVaeCancelled,
            &bundle,
            moments.get(),
            momentsCount
        )
    ) {
        return isCancellationRequested(bundle)
            ? NativeError::CANCELLED
            : NativeError::INFERENCE_FAILED;
    }
    completedStages = 1;

    const NativeError latentResult = runPisaLatentPipeline(
        bundle,
        moments.get(),
        latentWidth,
        latentHeight,
        noiseSeed,
        decoderLatent.get(),
        latentCount,
        completedStages
    );
    if (latentResult != NativeError::NONE) {
        return latentResult;
    }

    if (
        !kira::pisa::runMnnSegmentedVae(
            decoderSegments,
            bundle.vaeSegmentPack.decoderAffine,
            bundle.backend,
            decoderLatent.get(),
            4,
            latentWidth,
            latentHeight,
            PISA_VAE_DECODER_TILE_SIZE,
            PISA_VAE_DECODER_PADDING,
            true,
            3,
            segmentedVaeCancelled,
            &bundle,
            decodedImage,
            decodedImageCount
        )
    ) {
        return isCancellationRequested(bundle)
            ? NativeError::CANCELLED
            : NativeError::INFERENCE_FAILED;
    }

    completedStages = 3;
    return NativeError::NONE;
}

NativeError runPisaPreparedGraph(
    PisaModelBundle& bundle,
    int latentWidth,
    int latentHeight,
    std::uint64_t noiseSeed,
    int& completedStages
) {
    completedStages = 0;

    if (isCancellationRequested(bundle)) {
        return NativeError::CANCELLED;
    }

    std::size_t momentsCount = 0;
    std::size_t latentCount = 0;
    if (
        !checkedElementCount(
            {
                8U,
                static_cast<std::size_t>(latentHeight),
                static_cast<std::size_t>(latentWidth),
            },
            momentsCount
        ) ||
        !checkedElementCount(
            {
                4U,
                static_cast<std::size_t>(latentHeight),
                static_cast<std::size_t>(latentWidth),
            },
            latentCount
        )
    ) {
        return NativeError::OUT_OF_MEMORY;
    }

    std::unique_ptr<float[]> moments(
        new (std::nothrow) float[momentsCount]
    );
    std::unique_ptr<float[]> decoderLatent(
        new (std::nothrow) float[latentCount]
    );
    if (!moments || !decoderLatent) {
        return NativeError::OUT_OF_MEMORY;
    }

    const MNN::ErrorCode encoderRun =
        bundle.vaeEncoder->runSession(bundle.vaeEncoderSession);
    if (encoderRun != MNN::NO_ERROR) {
        return mapMnnRunError(encoderRun);
    }
    completedStages = 1;
    if (isCancellationRequested(bundle)) {
        return NativeError::CANCELLED;
    }

    const MNN::Tensor* encoderOutput =
        bundle.vaeEncoder->getSessionOutput(
            bundle.vaeEncoderSession,
            "moments"
        );
    if (
        !kira::pisa::readFloatNchwTensor(
            encoderOutput,
            moments.get(),
            momentsCount
        )
    ) {
        return NativeError::INFERENCE_FAILED;
    }

    const NativeError latentResult = runPisaLatentPipeline(
        bundle,
        moments.get(),
        latentWidth,
        latentHeight,
        noiseSeed,
        decoderLatent.get(),
        latentCount,
        completedStages
    );
    if (latentResult != NativeError::NONE) {
        return latentResult;
    }

    MNN::Tensor* decoderInput =
        bundle.vaeDecoder->getSessionInput(
            bundle.vaeDecoderSession,
            "latent"
        );
    if (
        !kira::pisa::writeFloatNchwTensor(
            decoderInput,
            decoderLatent.get(),
            latentCount
        )
    ) {
        return NativeError::INFERENCE_FAILED;
    }

    if (isCancellationRequested(bundle)) {
        return NativeError::CANCELLED;
    }

    const MNN::ErrorCode decoderRun =
        bundle.vaeDecoder->runSession(bundle.vaeDecoderSession);
    if (decoderRun != MNN::NO_ERROR) {
        return mapMnnRunError(decoderRun);
    }
    completedStages = 3;
    if (isCancellationRequested(bundle)) {
        return NativeError::CANCELLED;
    }
    return NativeError::NONE;
}

NativeError runPisaSmoke(
    PisaModelBundle& bundle,
    int imageWidth,
    int imageHeight,
    int& completedStages,
    bool& outputFinite
) {
    completedStages = 0;
    outputFinite = false;

    int latentWidth = 0;
    int latentHeight = 0;
    if (!preparePisaGraph(
        bundle,
        imageWidth,
        imageHeight,
        latentWidth,
        latentHeight
    )) {
        return NativeError::LOAD_FAILED;
    }

    std::size_t imageCount = 0;
    if (
        !checkedElementCount(
            {
                3U,
                static_cast<std::size_t>(imageHeight),
                static_cast<std::size_t>(imageWidth),
            },
            imageCount
        )
    ) {
        return NativeError::OUT_OF_MEMORY;
    }

    std::unique_ptr<float[]> decodedImage(
        new (std::nothrow) float[imageCount]
    );
    if (!decodedImage) {
        return NativeError::OUT_OF_MEMORY;
    }

    MNN::Tensor* encoderInput =
        bundle.vaeEncoder->getSessionInput(
            bundle.vaeEncoderSession,
            "image"
        );
    if (isCancellationRequested(bundle)) {
        return NativeError::CANCELLED;
    }

    if (
        !kira::pisa::writeRgba8888BicubicNormalizedTensor(
            encoderInput,
            SMOKE_SOURCE_RGBA,
            SMOKE_SOURCE_WIDTH,
            SMOKE_SOURCE_HEIGHT,
            SMOKE_SOURCE_STRIDE
        )
    ) {
        return NativeError::INFERENCE_FAILED;
    }

    const NativeError graphResult = runPisaPreparedGraph(
        bundle,
        latentWidth,
        latentHeight,
        SMOKE_NOISE_SEED,
        completedStages
    );
    if (graphResult != NativeError::NONE) {
        return graphResult;
    }

    const MNN::Tensor* decoderOutput =
        bundle.vaeDecoder->getSessionOutput(
            bundle.vaeDecoderSession,
            "image"
        );
    if (
        !kira::pisa::readFloatNchwTensor(
            decoderOutput,
            decodedImage.get(),
            imageCount
        )
    ) {
        return NativeError::INFERENCE_FAILED;
    }

    for (std::size_t index = 0; index < imageCount; ++index) {
        if (!std::isfinite(decodedImage[index])) {
            return NativeError::INFERENCE_FAILED;
        }
    }
    outputFinite = true;
    return NativeError::NONE;
}

bool appendGraphInfo(
    std::string& output,
    const char* graph,
    MNN::Interpreter* interpreter,
    MNN::Session* session
) {
    if (interpreter == nullptr || session == nullptr) {
        return false;
    }

    return appendTensorMap(
        output,
        graph,
        "input",
        interpreter->getSessionInputAll(session)
    ) && appendTensorMap(
        output,
        graph,
        "output",
        interpreter->getSessionOutputAll(session)
    );
}

#endif

}  // namespace

extern "C" JNIEXPORT jlongArray JNICALL
Java_com_ikegami99_kiraenhance_inference_mnn_MnnPisaNativeBridge_nativeLoadModel(
    JNIEnv* env,
    jobject /* thiz */,
    jstring vaeEncoderPath,
    jstring unetPath,
    jstring vaeDecoderPath,
    jstring emptyPromptPath,
    jstring vaeSegmentPackPath,
    jboolean preferGpu
) {
    std::string vaeEncoder;
    std::string unet;
    std::string vaeDecoder;
    std::string emptyPrompt;
    std::string vaeSegmentPack;
    if (
        !copyRequiredPath(env, vaeEncoderPath, vaeEncoder) ||
        !copyRequiredPath(env, unetPath, unet) ||
        !copyRequiredPath(env, vaeDecoderPath, vaeDecoder) ||
        !copyRequiredPath(env, emptyPromptPath, emptyPrompt) ||
        !copyRequiredPath(env, vaeSegmentPackPath, vaeSegmentPack)
    ) {
        return makeLoadResult(env, 0L, NativeError::INVALID_ARGUMENT, false);
    }

#if KIRA_HAS_MNN
    std::unique_ptr<PisaModelBundle> bundle(
        new (std::nothrow) PisaModelBundle()
    );
    if (!bundle) {
        return makeLoadResult(env, 0L, NativeError::OUT_OF_MEMORY, false);
    }

    if (
        !kira::pisa::loadVaeSegmentPackFile(
            vaeSegmentPack,
            bundle->vaeSegmentPack
        ) ||
        bundle->vaeSegmentPack.encoderModels.size() !=
            PISA_VAE_ENCODER_SEGMENTS ||
        bundle->vaeSegmentPack.decoderModels.size() !=
            PISA_VAE_DECODER_SEGMENTS ||
        bundle->vaeSegmentPack.encoderAffine.size() + 1 !=
            PISA_VAE_ENCODER_SEGMENTS ||
        bundle->vaeSegmentPack.decoderAffine.size() + 1 !=
            PISA_VAE_DECODER_SEGMENTS
    ) {
        return makeLoadResult(env, 0L, NativeError::LOAD_FAILED, false);
    }

    if (
        !loadSegmentInterpreters(
            bundle->vaeSegmentPack.encoderModels,
            bundle->vaeEncoderSegments
        ) ||
        !loadSegmentInterpreters(
            bundle->vaeSegmentPack.decoderModels,
            bundle->vaeDecoderSegments
        )
    ) {
        return makeLoadResult(env, 0L, NativeError::LOAD_FAILED, false);
    }

    bundle->vaeEncoder = loadInterpreter(vaeEncoder);
    if (!bundle->vaeEncoder) {
        return makeLoadResult(env, 0L, NativeError::LOAD_FAILED, false);
    }

    bundle->unet = loadInterpreter(unet);
    if (!bundle->unet) {
        return makeLoadResult(env, 0L, NativeError::LOAD_FAILED, false);
    }

    bundle->vaeDecoder = loadInterpreter(vaeDecoder);
    if (!bundle->vaeDecoder) {
        return makeLoadResult(env, 0L, NativeError::LOAD_FAILED, false);
    }

    const ArtifactReadResult promptResult = readFp16Artifact(
        emptyPrompt,
        bundle->emptyPrompt,
        bundle->emptyPromptBytes
    );
    if (promptResult == ArtifactReadResult::OUT_OF_MEMORY) {
        return makeLoadResult(env, 0L, NativeError::OUT_OF_MEMORY, false);
    }
    if (promptResult != ArtifactReadResult::OK) {
        return makeLoadResult(env, 0L, NativeError::LOAD_FAILED, false);
    }
    if (bundle->emptyPromptBytes != EMPTY_PROMPT_FP16_BYTES) {
        return makeLoadResult(env, 0L, NativeError::LOAD_FAILED, false);
    }

    bundle->vaeEncoder->setSessionMode(MNN::Interpreter::Session_Release);
    bundle->unet->setSessionMode(MNN::Interpreter::Session_Release);
    bundle->vaeDecoder->setSessionMode(MNN::Interpreter::Session_Release);

    if (!createPreferredSessions(*bundle, preferGpu == JNI_TRUE)) {
        return makeLoadResult(env, 0L, NativeError::LOAD_FAILED, false);
    }

    const bool gpuEnabled = isGpuBackend(bundle->backend);
    static_assert(sizeof(std::intptr_t) <= sizeof(jlong));
    const auto handle = static_cast<jlong>(
        reinterpret_cast<std::intptr_t>(bundle.release())
    );
    return makeLoadResult(env, handle, NativeError::NONE, gpuEnabled);
#else
    (void)preferGpu;
    return makeLoadResult(env, 0L, NativeError::LOAD_FAILED, false);
#endif
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_ikegami99_kiraenhance_inference_mnn_MnnPisaNativeBridge_nativeGraphInfo(
    JNIEnv* env,
    jobject /* thiz */,
    jlong handle
) {
#if KIRA_HAS_MNN
    if (handle == 0L) {
        return env->NewStringUTF("");
    }

    auto* bundle = reinterpret_cast<PisaModelBundle*>(
        static_cast<std::intptr_t>(handle)
    );

    std::string output;
    if (
        !appendGraphInfo(
            output,
            "vae_encoder",
            bundle->vaeEncoder.get(),
            bundle->vaeEncoderSession
        ) ||
        !appendGraphInfo(
            output,
            "unet",
            bundle->unet.get(),
            bundle->unetSession
        ) ||
        !appendGraphInfo(
            output,
            "vae_decoder",
            bundle->vaeDecoder.get(),
            bundle->vaeDecoderSession
        )
    ) {
        return env->NewStringUTF("");
    }

    return env->NewStringUTF(output.c_str());
#else
    (void)handle;
    return env->NewStringUTF("");
#endif
}

extern "C" JNIEXPORT jlongArray JNICALL
Java_com_ikegami99_kiraenhance_inference_mnn_MnnPisaNativeBridge_nativePrepareGraph(
    JNIEnv* env,
    jobject /* thiz */,
    jlong handle,
    jint imageWidth,
    jint imageHeight
) {
#if KIRA_HAS_MNN
    if (
        handle == 0L ||
        imageWidth <= 0 ||
        imageHeight <= 0 ||
        imageWidth % 8 != 0 ||
        imageHeight % 8 != 0
    ) {
        return makePrepareResult(env, NativeError::INVALID_ARGUMENT);
    }

    auto* bundle = reinterpret_cast<PisaModelBundle*>(
        static_cast<std::intptr_t>(handle)
    );
    if (bundle == nullptr) {
        return makePrepareResult(env, NativeError::INVALID_ARGUMENT);
    }

    int latentWidth = 0;
    int latentHeight = 0;
    if (!preparePisaGraph(
        *bundle,
        imageWidth,
        imageHeight,
        latentWidth,
        latentHeight
    )) {
        return makePrepareResult(env, NativeError::LOAD_FAILED);
    }

    return makePrepareResult(
        env,
        NativeError::NONE,
        imageWidth,
        imageHeight,
        latentWidth,
        latentHeight
    );
#else
    (void)handle;
    (void)imageWidth;
    (void)imageHeight;
    return makePrepareResult(env, NativeError::NOT_IMPLEMENTED);
#endif
}

extern "C" JNIEXPORT jlongArray JNICALL
Java_com_ikegami99_kiraenhance_inference_mnn_MnnPisaNativeBridge_nativeSmokeGraph(
    JNIEnv* env,
    jobject /* thiz */,
    jlong handle,
    jint imageWidth,
    jint imageHeight
) {
#if KIRA_HAS_MNN
    if (
        handle == 0L ||
        imageWidth <= 0 ||
        imageHeight <= 0 ||
        imageWidth % 8 != 0 ||
        imageHeight % 8 != 0
    ) {
        return makeSmokeResult(env, NativeError::INVALID_ARGUMENT);
    }

    auto* bundle = reinterpret_cast<PisaModelBundle*>(
        static_cast<std::intptr_t>(handle)
    );
    if (bundle == nullptr) {
        return makeSmokeResult(env, NativeError::INVALID_ARGUMENT);
    }

    int completedStages = 0;
    bool outputFinite = false;
    const NativeError result = runPisaSmoke(
        *bundle,
        imageWidth,
        imageHeight,
        completedStages,
        outputFinite
    );
    return makeSmokeResult(
        env,
        result,
        completedStages,
        outputFinite
    );
#else
    (void)handle;
    (void)imageWidth;
    (void)imageHeight;
    return makeSmokeResult(env, NativeError::NOT_IMPLEMENTED);
#endif
}

extern "C" JNIEXPORT jlongArray JNICALL
Java_com_ikegami99_kiraenhance_inference_mnn_MnnPisaNativeBridge_nativeInfer(
    JNIEnv* env,
    jobject /* thiz */,
    jlong handle,
    jobject inputPixels,
    jint width,
    jint height,
    jint inputRowStrideBytes,
    jobject outputPixels,
    jlong outputCapacityBytes
) {
#if KIRA_HAS_MNN
    if (
        handle == 0L ||
        inputPixels == nullptr ||
        outputPixels == nullptr ||
        width <= 1 ||
        height <= 1 ||
        width > std::numeric_limits<int>::max() / 4 ||
        inputRowStrideBytes < width * 4 ||
        outputCapacityBytes <= 0
    ) {
        return makeInferenceResult(env, NativeError::INVALID_ARGUMENT);
    }

    auto* inputData = static_cast<std::uint8_t*>(
        env->GetDirectBufferAddress(inputPixels)
    );
    auto* outputData = static_cast<std::uint8_t*>(
        env->GetDirectBufferAddress(outputPixels)
    );
    const jlong inputCapacity = env->GetDirectBufferCapacity(inputPixels);
    const jlong outputCapacity = env->GetDirectBufferCapacity(outputPixels);
    if (
        inputData == nullptr ||
        outputData == nullptr ||
        inputCapacity < 0 ||
        outputCapacity < 0
    ) {
        return makeInferenceResult(env, NativeError::INVALID_ARGUMENT);
    }

    const jlong requiredInputBytes =
        static_cast<jlong>(inputRowStrideBytes) * static_cast<jlong>(height);
    if (
        requiredInputBytes <= 0 ||
        requiredInputBytes > inputCapacity ||
        outputCapacityBytes > outputCapacity
    ) {
        return makeInferenceResult(env, NativeError::INVALID_ARGUMENT);
    }

    auto* bundle = reinterpret_cast<PisaModelBundle*>(
        static_cast<std::intptr_t>(handle)
    );
    if (bundle == nullptr) {
        return makeInferenceResult(env, NativeError::INVALID_ARGUMENT);
    }
    ScopedCancellationReset cancellationReset(*bundle);

    kira::pisa::ResizePlan resizePlan;
    if (!kira::pisa::buildResizePlan(width, height, resizePlan)) {
        return makeInferenceResult(env, NativeError::INVALID_ARGUMENT);
    }

    const std::int64_t outputWidth64 =
        static_cast<std::int64_t>(resizePlan.outputWidth);
    const std::int64_t outputHeight64 =
        static_cast<std::int64_t>(resizePlan.outputHeight);
    if (
        outputWidth64 <= 0 ||
        outputHeight64 <= 0 ||
        outputWidth64 >
            std::numeric_limits<std::int64_t>::max() /
                outputHeight64 /
                4
    ) {
        return makeInferenceResult(env, NativeError::OUT_OF_MEMORY);
    }

    const jlong requiredOutputBytes = static_cast<jlong>(
        outputWidth64 * outputHeight64 * 4
    );
    if (outputCapacityBytes < requiredOutputBytes) {
        return makeInferenceResult(env, NativeError::INVALID_ARGUMENT);
    }

    std::size_t modelPixels = 0;
    if (
        !checkedElementCount(
            {
                static_cast<std::size_t>(resizePlan.modelHeight),
                static_cast<std::size_t>(resizePlan.modelWidth),
            },
            modelPixels
        )
    ) {
        return makeInferenceResult(
            env,
            NativeError::OUT_OF_MEMORY,
            0,
            0,
            0,
            isGpuBackend(bundle->backend)
        );
    }

    const bool useSegmentedVae =
        modelPixels > MAX_MONOLITHIC_MODEL_PIXELS ||
        kira::pisa::requiresVaeTiling(
            resizePlan.modelWidth,
            resizePlan.modelHeight,
            PISA_VAE_ENCODER_TILE_SIZE,
            PISA_VAE_ENCODER_PADDING
        );

    int latentWidth = 0;
    int latentHeight = 0;
    bool graphPrepared = false;
    if (useSegmentedVae) {
        if (
            resizePlan.modelWidth % 8 == 0 &&
            resizePlan.modelHeight % 8 == 0
        ) {
            latentWidth = resizePlan.modelWidth / 8;
            latentHeight = resizePlan.modelHeight / 8;
            graphPrepared = prepareUnetForLatentGraph(
                *bundle,
                latentWidth,
                latentHeight
            );
        }
    } else {
        graphPrepared = preparePisaGraph(
            *bundle,
            resizePlan.modelWidth,
            resizePlan.modelHeight,
            latentWidth,
            latentHeight
        );
    }
    if (!graphPrepared) {
        return makeInferenceResult(
            env,
            NativeError::LOAD_FAILED,
            0,
            0,
            0,
            isGpuBackend(bundle->backend)
        );
    }

    std::size_t imageCount = 0;
    if (
        !checkedElementCount(
            {3U, modelPixels},
            imageCount
        )
    ) {
        return makeInferenceResult(
            env,
            NativeError::OUT_OF_MEMORY,
            0,
            0,
            0,
            isGpuBackend(bundle->backend)
        );
    }

    if (
        resizePlan.modelWidth >
        std::numeric_limits<int>::max() / 4
    ) {
        return makeInferenceResult(
            env,
            NativeError::OUT_OF_MEMORY,
            0,
            0,
            0,
            isGpuBackend(bundle->backend)
        );
    }

    std::size_t modelRgbaBytes = 0;
    if (
        !checkedElementCount(
            {modelPixels, 4U},
            modelRgbaBytes
        )
    ) {
        return makeInferenceResult(
            env,
            NativeError::OUT_OF_MEMORY,
            0,
            0,
            0,
            isGpuBackend(bundle->backend)
        );
    }

    std::unique_ptr<std::uint8_t[]> modelRgba(
        new (std::nothrow) std::uint8_t[modelRgbaBytes]
    );
    std::unique_ptr<float[]> sourceImage(
        new (std::nothrow) float[imageCount]
    );
    std::unique_ptr<float[]> decodedImage(
        new (std::nothrow) float[imageCount]
    );
    if (!modelRgba || !sourceImage || !decodedImage) {
        return makeInferenceResult(
            env,
            NativeError::OUT_OF_MEMORY,
            0,
            0,
            0,
            isGpuBackend(bundle->backend)
        );
    }

    if (isCancellationRequested(*bundle)) {
        return makeInferenceResult(
            env,
            NativeError::CANCELLED,
            0,
            0,
            0,
            isGpuBackend(bundle->backend)
        );
    }

    const int modelRowStrideBytes =
        resizePlan.modelWidth * 4;
    if (
        !kira::pisa::preparePisaModelRgba(
            inputData,
            inputRowStrideBytes,
            resizePlan,
            modelRgba.get(),
            modelRowStrideBytes,
            modelRgbaBytes
        ) ||
        !kira::pisa::rgba8888ToNormalizedNchw(
            modelRgba.get(),
            resizePlan.modelWidth,
            resizePlan.modelHeight,
            modelRowStrideBytes,
            sourceImage.get(),
            imageCount
        )
    ) {
        return makeInferenceResult(
            env,
            NativeError::INFERENCE_FAILED,
            0,
            0,
            0,
            isGpuBackend(bundle->backend)
        );
    }

    int completedStages = 0;
    NativeError graphResult = NativeError::NONE;
    if (useSegmentedVae) {
        graphResult = runPisaSegmentedGraph(
            *bundle,
            sourceImage.get(),
            resizePlan.modelWidth,
            resizePlan.modelHeight,
            INFERENCE_NOISE_SEED,
            decodedImage.get(),
            imageCount,
            completedStages
        );
    } else {
        MNN::Tensor* encoderInput =
            bundle->vaeEncoder->getSessionInput(
                bundle->vaeEncoderSession,
                "image"
            );
        if (
            !kira::pisa::writeFloatNchwTensor(
                encoderInput,
                sourceImage.get(),
                imageCount
            )
        ) {
            return makeInferenceResult(
                env,
                NativeError::INFERENCE_FAILED,
                0,
                0,
                0,
                isGpuBackend(bundle->backend)
            );
        }

        graphResult = runPisaPreparedGraph(
            *bundle,
            latentWidth,
            latentHeight,
            INFERENCE_NOISE_SEED,
            completedStages
        );
        if (graphResult == NativeError::NONE && completedStages == 3) {
            const MNN::Tensor* decoderOutput =
                bundle->vaeDecoder->getSessionOutput(
                    bundle->vaeDecoderSession,
                    "image"
                );
            if (
                !kira::pisa::readFloatNchwTensor(
                    decoderOutput,
                    decodedImage.get(),
                    imageCount
                )
            ) {
                graphResult = NativeError::INFERENCE_FAILED;
            }
        }
    }

    if (graphResult != NativeError::NONE || completedStages != 3) {
        return makeInferenceResult(
            env,
            graphResult == NativeError::NONE
                ? NativeError::INFERENCE_FAILED
                : graphResult,
            0,
            0,
            0,
            isGpuBackend(bundle->backend)
        );
    }

    // The VAE consumes [-1, 1], while upstream AdaIN uses
    // ToTensor(input_image), i.e. the exact uint8 RGB / 255 grid.
    if (
        !kira::pisa::rgba8888ToUnitNchw(
            modelRgba.get(),
            resizePlan.modelWidth,
            resizePlan.modelHeight,
            modelRowStrideBytes,
            sourceImage.get(),
            imageCount
        )
    ) {
        return makeInferenceResult(
            env,
            NativeError::INFERENCE_FAILED,
            0,
            0,
            0,
            isGpuBackend(bundle->backend)
        );
    }

    for (std::size_t index = 0; index < imageCount; ++index) {
        const float decodedValue = decodedImage[index];
        if (!std::isfinite(decodedValue)) {
            return makeInferenceResult(
                env,
                NativeError::INFERENCE_FAILED,
                0,
                0,
                0,
                isGpuBackend(bundle->backend)
            );
        }

        decodedImage[index] = std::clamp(
            decodedValue * 0.5f + 0.5f,
            0.0f,
            1.0f
        );
    }

    // Upstream converts the decoder tensor to PIL before AdaIN,
    // which truncates float RGB values onto the uint8 grid.
    if (
        !kira::pisa::quantizeUnitPlanarLikeTorchvision(
            decodedImage.get(),
            imageCount
        )
    ) {
        return makeInferenceResult(
            env,
            NativeError::INFERENCE_FAILED,
            0,
            0,
            0,
            isGpuBackend(bundle->backend)
        );
    }

    if (isCancellationRequested(*bundle)) {
        return makeInferenceResult(
            env,
            NativeError::CANCELLED,
            0,
            0,
            0,
            isGpuBackend(bundle->backend)
        );
    }

    if (
        !kira::pisa::adainColorFixRgbPlanar(
            decodedImage.get(),
            modelPixels,
            sourceImage.get(),
            modelPixels,
            decodedImage.get()
        )
    ) {
        return makeInferenceResult(
            env,
            NativeError::INFERENCE_FAILED,
            0,
            0,
            0,
            isGpuBackend(bundle->backend)
        );
    }

    if (isCancellationRequested(*bundle)) {
        return makeInferenceResult(
            env,
            NativeError::CANCELLED,
            0,
            0,
            0,
            isGpuBackend(bundle->backend)
        );
    }

    if (
        resizePlan.outputWidth >
        std::numeric_limits<int>::max() / 4
    ) {
        return makeInferenceResult(
            env,
            NativeError::OUT_OF_MEMORY,
            0,
            0,
            0,
            isGpuBackend(bundle->backend)
        );
    }
    const int outputRowStrideBytes =
        resizePlan.outputWidth * 4;

    const bool needsOutputResize =
        resizePlan.modelWidth != resizePlan.outputWidth ||
        resizePlan.modelHeight != resizePlan.outputHeight;
    if (needsOutputResize) {
        if (
            !kira::pisa::unitNchwToRgba8888LikeTorchvision(
                decodedImage.get(),
                resizePlan.modelWidth,
                resizePlan.modelHeight,
                modelRgba.get(),
                modelRowStrideBytes,
                modelRgbaBytes
            )
        ) {
            return makeInferenceResult(
                env,
                NativeError::INFERENCE_FAILED,
                0,
                0,
                0,
                isGpuBackend(bundle->backend)
            );
        }

        if (isCancellationRequested(*bundle)) {
            return makeInferenceResult(
                env,
                NativeError::CANCELLED,
                0,
                0,
                0,
                isGpuBackend(bundle->backend)
            );
        }

        if (
            !kira::pisa::resizeRgba8888PillowRgb(
                modelRgba.get(),
                resizePlan.modelWidth,
                resizePlan.modelHeight,
                modelRowStrideBytes,
                outputData,
                resizePlan.outputWidth,
                resizePlan.outputHeight,
                outputRowStrideBytes,
                static_cast<std::size_t>(
                    requiredOutputBytes
                ),
                kira::pisa::PillowResizeFilter::BICUBIC
            )
        ) {
            return makeInferenceResult(
                env,
                NativeError::INFERENCE_FAILED,
                0,
                0,
                0,
                isGpuBackend(bundle->backend)
            );
        }
    } else if (
        !kira::pisa::unitNchwToRgba8888LikeTorchvision(
            decodedImage.get(),
            resizePlan.modelWidth,
            resizePlan.modelHeight,
            outputData,
            outputRowStrideBytes,
            static_cast<std::size_t>(requiredOutputBytes)
        )
    ) {
        return makeInferenceResult(
            env,
            NativeError::INFERENCE_FAILED,
            0,
            0,
            0,
            isGpuBackend(bundle->backend)
        );
    }

    return makeInferenceResult(
        env,
        NativeError::NONE,
        resizePlan.outputWidth,
        resizePlan.outputHeight,
        outputRowStrideBytes,
        isGpuBackend(bundle->backend)
    );
#else
    (void)handle;
    (void)inputPixels;
    (void)width;
    (void)height;
    (void)inputRowStrideBytes;
    (void)outputPixels;
    (void)outputCapacityBytes;
    return makeInferenceResult(env, NativeError::NOT_IMPLEMENTED);
#endif
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_ikegami99_kiraenhance_inference_mnn_MnnPisaNativeBridge_nativeSessionInfo(
    JNIEnv* env,
    jobject /* thiz */,
    jlong handle
) {
#if KIRA_HAS_MNN
    if (handle == 0L) {
        return env->NewStringUTF("");
    }

    auto* bundle = reinterpret_cast<PisaModelBundle*>(
        static_cast<std::intptr_t>(handle)
    );
    return env->NewStringUTF(sessionInfoForBackend(bundle->backend));
#else
    (void)handle;
    return env->NewStringUTF("");
#endif
}

extern "C" JNIEXPORT void JNICALL
Java_com_ikegami99_kiraenhance_inference_mnn_MnnPisaNativeBridge_nativeCancel(
    JNIEnv* /* env */,
    jobject /* thiz */,
    jlong handle
) {
#if KIRA_HAS_MNN
    if (handle == 0L) {
        return;
    }

    auto* bundle = reinterpret_cast<PisaModelBundle*>(
        static_cast<std::intptr_t>(handle)
    );
    if (bundle != nullptr) {
        bundle->cancelRequested.store(
            true,
            std::memory_order_release
        );
    }
#else
    (void)handle;
#endif
}

extern "C" JNIEXPORT void JNICALL
Java_com_ikegami99_kiraenhance_inference_mnn_MnnPisaNativeBridge_nativeUnloadModel(
    JNIEnv* /* env */,
    jobject /* thiz */,
    jlong handle
) {
#if KIRA_HAS_MNN
    if (handle == 0L) {
        return;
    }

    auto* bundle = reinterpret_cast<PisaModelBundle*>(
        static_cast<std::intptr_t>(handle)
    );
    delete bundle;
#else
    (void)handle;
#endif
}
