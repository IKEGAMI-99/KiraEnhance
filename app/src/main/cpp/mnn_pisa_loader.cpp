#include <jni.h>

#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

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

bool readFp16Artifact(const std::string& path, std::vector<std::uint8_t>& output) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        return false;
    }

    const std::streamoff size = input.tellg();
    if (size <= 0 || (size % 2) != 0) {
        return false;
    }

    output.resize(static_cast<std::size_t>(size));
    input.seekg(0, std::ios::beg);
    if (!input) {
        return false;
    }

    return static_cast<bool>(
        input.read(
            reinterpret_cast<char*>(output.data()),
            static_cast<std::streamsize>(output.size())
        )
    );
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
    std::vector<std::uint8_t> emptyPrompt;
};

InterpreterPtr loadInterpreter(const std::string& path) {
    return InterpreterPtr(MNN::Interpreter::createFromFile(path.c_str()));
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
    (void)preferGpu;

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
    try {
        auto bundle = std::make_unique<PisaModelBundle>();

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

        if (!readFp16Artifact(emptyPrompt, bundle->emptyPrompt)) {
            return makeLoadResult(env, 0L, NativeError::LOAD_FAILED, false);
        }

        static_assert(sizeof(std::intptr_t) <= sizeof(jlong));
        const auto handle = static_cast<jlong>(
            reinterpret_cast<std::intptr_t>(bundle.release())
        );
        return makeLoadResult(env, handle, NativeError::NONE, false);
    } catch (const std::bad_alloc&) {
        return makeLoadResult(env, 0L, NativeError::OUT_OF_MEMORY, false);
    } catch (...) {
        return makeLoadResult(env, 0L, NativeError::INTERNAL, false);
    }
#else
    return makeLoadResult(env, 0L, NativeError::LOAD_FAILED, false);
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
