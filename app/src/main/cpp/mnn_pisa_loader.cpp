#include <jni.h>

#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <new>
#include <string>

#if KIRA_HAS_MNN
#include <MNN/Interpreter.hpp>
#endif

namespace {

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
    std::unique_ptr<std::uint8_t[]> emptyPrompt;
    std::size_t emptyPromptBytes = 0;
    MNN::Session* vaeEncoderSession = nullptr;
    MNN::Session* unetSession = nullptr;
    MNN::Session* vaeDecoderSession = nullptr;
    MNNForwardType backend = MNN_FORWARD_CPU;
};

InterpreterPtr loadInterpreter(const std::string& path) {
    return InterpreterPtr(MNN::Interpreter::createFromFile(path.c_str()));
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
    jboolean preferGpu
) {
    std::string vaeEncoder;
    std::string unet;
    std::string vaeDecoder;
    std::string emptyPrompt;
    if (
        !copyRequiredPath(env, vaeEncoderPath, vaeEncoder) ||
        !copyRequiredPath(env, unetPath, unet) ||
        !copyRequiredPath(env, vaeDecoderPath, vaeDecoder) ||
        !copyRequiredPath(env, emptyPromptPath, emptyPrompt)
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
