#include <jni.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <new>
#include <string>

#include <net.h>
#include <platform.h>
#if NCNN_VULKAN
#include <gpu.h>
#endif

namespace {

enum NativeError : jlong {
    ERROR_NONE = 0,
    ERROR_LOAD_PARAM_FAILED = 1,
    ERROR_LOAD_MODEL_FAILED = 2,
    ERROR_INVALID_ARGUMENT = 3,
    ERROR_INTERNAL = 4,
};

struct ModelContext {
    ncnn::Net net;
    bool gpu_enabled = false;
    std::atomic<bool> cancelled{false};
    std::string input_blob_name;
    std::string output_blob_name;
    int native_scale = 1;
    int pre_padding = 0;
};

#if NCNN_VULKAN
std::mutex g_gpu_mutex;
bool g_gpu_instance_active = false;
int g_gpu_users = 0;

bool acquire_gpu() {
    std::lock_guard<std::mutex> guard(g_gpu_mutex);

    if (!g_gpu_instance_active) {
        if (ncnn::create_gpu_instance() != 0) {
            return false;
        }
        g_gpu_instance_active = true;
    }

    if (ncnn::get_gpu_count() <= 0) {
        if (g_gpu_users == 0) {
            ncnn::destroy_gpu_instance();
            g_gpu_instance_active = false;
        }
        return false;
    }

    ++g_gpu_users;
    return true;
}

void release_gpu() {
    std::lock_guard<std::mutex> guard(g_gpu_mutex);
    if (g_gpu_users <= 0) {
        return;
    }

    --g_gpu_users;
    if (g_gpu_users == 0 && g_gpu_instance_active) {
        ncnn::destroy_gpu_instance();
        g_gpu_instance_active = false;
    }
}

int query_gpu_count() {
    std::lock_guard<std::mutex> guard(g_gpu_mutex);

    if (g_gpu_instance_active) {
        return ncnn::get_gpu_count();
    }

    if (ncnn::create_gpu_instance() != 0) {
        return 0;
    }
    const int count = ncnn::get_gpu_count();
    ncnn::destroy_gpu_instance();
    return count;
}
#else
bool acquire_gpu() { return false; }
void release_gpu() {}
int query_gpu_count() { return 0; }
#endif

std::string to_string(JNIEnv* env, jstring value) {
    if (value == nullptr) {
        return {};
    }

    const char* chars = env->GetStringUTFChars(value, nullptr);
    if (chars == nullptr) {
        return {};
    }

    std::string result(chars);
    env->ReleaseStringUTFChars(value, chars);
    return result;
}

jlongArray make_load_result(
    JNIEnv* env,
    jlong handle,
    NativeError error,
    bool gpu_enabled
) {
    jlong values[3] = {
        handle,
        static_cast<jlong>(error),
        gpu_enabled ? 1L : 0L,
    };

    jlongArray array = env->NewLongArray(3);
    if (array != nullptr) {
        env->SetLongArrayRegion(array, 0, 3, values);
    }
    return array;
}

ModelContext* from_handle(jlong handle) {
    return reinterpret_cast<ModelContext*>(static_cast<intptr_t>(handle));
}

jlong to_handle(ModelContext* context) {
    return static_cast<jlong>(reinterpret_cast<intptr_t>(context));
}

}  // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_com_ikegami99_kiraenhance_inference_ncnn_NcnnNativeBridge_nativeRuntimeInfo(
    JNIEnv* env,
    jobject /* thiz */
) {
    const int gpu_count = query_gpu_count();

    std::string info = "ncnn=" + std::to_string(NCNN_VERSION_NUMBER)
        + ";vulkan=" + std::string(NCNN_VULKAN ? "1" : "0")
        + ";gpuCount=" + std::to_string(gpu_count);

    return env->NewStringUTF(info.c_str());
}

extern "C" JNIEXPORT jlongArray JNICALL
Java_com_ikegami99_kiraenhance_inference_ncnn_NcnnNativeBridge_nativeLoadModel(
    JNIEnv* env,
    jobject /* thiz */,
    jstring param_path_value,
    jstring bin_path_value,
    jstring input_blob_value,
    jstring output_blob_value,
    jint native_scale,
    jint pre_padding,
    jboolean prefer_gpu
) {
    const std::string param_path = to_string(env, param_path_value);
    const std::string bin_path = to_string(env, bin_path_value);
    const std::string input_blob_name = to_string(env, input_blob_value);
    const std::string output_blob_name = to_string(env, output_blob_value);

    if (param_path.empty() || bin_path.empty() || input_blob_name.empty()
        || output_blob_name.empty() || native_scale <= 0 || pre_padding < 0) {
        return make_load_result(env, 0L, ERROR_INVALID_ARGUMENT, false);
    }

    bool gpu_enabled = false;
#if NCNN_VULKAN
    if (prefer_gpu == JNI_TRUE) {
        gpu_enabled = acquire_gpu();
    }
#endif

    std::unique_ptr<ModelContext> context(new (std::nothrow) ModelContext());
    if (!context) {
        if (gpu_enabled) {
            release_gpu();
        }
        return make_load_result(env, 0L, ERROR_INTERNAL, false);
    }

    context->gpu_enabled = gpu_enabled;
    context->input_blob_name = input_blob_name;
    context->output_blob_name = output_blob_name;
    context->native_scale = native_scale;
    context->pre_padding = pre_padding;
    context->net.opt.use_vulkan_compute = gpu_enabled;
    context->net.opt.use_fp16_packed = true;
    context->net.opt.use_fp16_storage = true;
    context->net.opt.use_fp16_arithmetic = false;

#if NCNN_VULKAN
    if (gpu_enabled) {
        context->net.set_vulkan_device(0);
    }
#endif

    if (context->net.load_param(param_path.c_str()) != 0) {
        context.reset();
        if (gpu_enabled) {
            release_gpu();
        }
        return make_load_result(env, 0L, ERROR_LOAD_PARAM_FAILED, false);
    }

    if (context->net.load_model(bin_path.c_str()) != 0) {
        context.reset();
        if (gpu_enabled) {
            release_gpu();
        }
        return make_load_result(env, 0L, ERROR_LOAD_MODEL_FAILED, false);
    }

    ModelContext* raw_context = context.release();
    return make_load_result(
        env,
        to_handle(raw_context),
        ERROR_NONE,
        gpu_enabled
    );
}

extern "C" JNIEXPORT void JNICALL
Java_com_ikegami99_kiraenhance_inference_ncnn_NcnnNativeBridge_nativeUnload(
    JNIEnv* /* env */,
    jobject /* thiz */,
    jlong handle
) {
    ModelContext* context = from_handle(handle);
    if (context == nullptr) {
        return;
    }

    const bool gpu_enabled = context->gpu_enabled;
    delete context;
    if (gpu_enabled) {
        release_gpu();
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_ikegami99_kiraenhance_inference_ncnn_NcnnNativeBridge_nativeCancel(
    JNIEnv* /* env */,
    jobject /* thiz */,
    jlong handle
) {
    ModelContext* context = from_handle(handle);
    if (context != nullptr) {
        context->cancelled.store(true, std::memory_order_relaxed);
    }
}
