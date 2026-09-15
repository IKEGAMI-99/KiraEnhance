#include <jni.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
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
    ERROR_INFERENCE_FAILED = 5,
    ERROR_CANCELLED = 6,
    ERROR_BUFFER_TOO_SMALL = 7,
    ERROR_OUT_OF_MEMORY = 8,
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
        if (ncnn::create_gpu_instance() != 0) return false;
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
    if (g_gpu_users <= 0) return;

    --g_gpu_users;
    if (g_gpu_users == 0 && g_gpu_instance_active) {
        ncnn::destroy_gpu_instance();
        g_gpu_instance_active = false;
    }
}

int query_gpu_count() {
    std::lock_guard<std::mutex> guard(g_gpu_mutex);
    if (g_gpu_instance_active) return ncnn::get_gpu_count();
    if (ncnn::create_gpu_instance() != 0) return 0;

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
    if (value == nullptr) return {};
    const char* chars = env->GetStringUTFChars(value, nullptr);
    if (chars == nullptr) return {};
    std::string result(chars);
    env->ReleaseStringUTFChars(value, chars);
    return result;
}

jlongArray make_load_result(JNIEnv* env, jlong handle, NativeError error, bool gpu_enabled) {
    const jlong values[3] = {
        handle,
        static_cast<jlong>(error),
        gpu_enabled ? 1L : 0L,
    };
    jlongArray array = env->NewLongArray(3);
    if (array != nullptr) env->SetLongArrayRegion(array, 0, 3, values);
    return array;
}

jlongArray make_inference_result(
    JNIEnv* env,
    NativeError error,
    int output_width,
    int output_height,
    int output_row_stride,
    bool gpu_used
) {
    const jlong values[5] = {
        static_cast<jlong>(error),
        static_cast<jlong>(output_width),
        static_cast<jlong>(output_height),
        static_cast<jlong>(output_row_stride),
        gpu_used ? 1L : 0L,
    };
    jlongArray array = env->NewLongArray(5);
    if (array != nullptr) env->SetLongArrayRegion(array, 0, 5, values);
    return array;
}

ModelContext* from_handle(jlong handle) {
    return reinterpret_cast<ModelContext*>(static_cast<intptr_t>(handle));
}

jlong to_handle(ModelContext* context) {
    return static_cast<jlong>(reinterpret_cast<intptr_t>(context));
}

bool checked_multiply(jlong a, jlong b, jlong* result) {
    if (a < 0 || b < 0 || result == nullptr) return false;
    if (a != 0 && b > std::numeric_limits<jlong>::max() / a) return false;
    *result = a * b;
    return true;
}

unsigned char to_u8(float value) {
    const float scaled = value * 255.f + 0.5f;
    const float clipped = std::max(0.f, std::min(255.f, scaled));
    return static_cast<unsigned char>(clipped);
}

unsigned char sample_alpha_bilinear(
    const unsigned char* rgba,
    int width,
    int height,
    int row_stride,
    int output_x,
    int output_y,
    int scale
) {
    const float source_x = (static_cast<float>(output_x) + 0.5f) / scale - 0.5f;
    const float source_y = (static_cast<float>(output_y) + 0.5f) / scale - 0.5f;
    const float clamped_x = std::max(0.f, std::min(static_cast<float>(width - 1), source_x));
    const float clamped_y = std::max(0.f, std::min(static_cast<float>(height - 1), source_y));

    const int x0 = static_cast<int>(std::floor(clamped_x));
    const int y0 = static_cast<int>(std::floor(clamped_y));
    const int x1 = std::min(x0 + 1, width - 1);
    const int y1 = std::min(y0 + 1, height - 1);
    const float fx = clamped_x - x0;
    const float fy = clamped_y - y0;

    const auto alpha_at = [&](int x, int y) -> float {
        return static_cast<float>(rgba[static_cast<size_t>(y) * row_stride + x * 4 + 3]);
    };

    const float top = alpha_at(x0, y0) * (1.f - fx) + alpha_at(x1, y0) * fx;
    const float bottom = alpha_at(x0, y1) * (1.f - fx) + alpha_at(x1, y1) * fx;
    const float alpha = top * (1.f - fy) + bottom * fy;
    return static_cast<unsigned char>(std::max(0.f, std::min(255.f, alpha + 0.5f)));
}

NativeError infer_rgba(
    ModelContext* context,
    const unsigned char* input_ptr,
    int width,
    int height,
    int input_row_stride,
    unsigned char* output_ptr,
    int output_width,
    int output_height,
    int output_row_stride
) {
    ncnn::Mat input = ncnn::Mat::from_pixels(
        input_ptr,
        ncnn::Mat::PIXEL_RGBA2RGB,
        width,
        height,
        input_row_stride
    );
    if (input.empty()) return ERROR_OUT_OF_MEMORY;

    const float norm_values[3] = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
    input.substract_mean_normalize(nullptr, norm_values);

    if (context->cancelled.load(std::memory_order_relaxed)) return ERROR_CANCELLED;

    ncnn::Extractor extractor = context->net.create_extractor();
    if (extractor.input(context->input_blob_name.c_str(), input) != 0) {
        return ERROR_INFERENCE_FAILED;
    }

    ncnn::Mat output;
    if (extractor.extract(context->output_blob_name.c_str(), output) != 0 || output.empty()) {
        return ERROR_INFERENCE_FAILED;
    }

    if (context->cancelled.load(std::memory_order_relaxed)) return ERROR_CANCELLED;

    if (output.w != output_width || output.h != output_height || output.c < 3
        || output.elempack != 1 || output.elemsize != sizeof(float)) {
        return ERROR_INFERENCE_FAILED;
    }

    const float* red = output.channel(0);
    const float* green = output.channel(1);
    const float* blue = output.channel(2);
    if (red == nullptr || green == nullptr || blue == nullptr) return ERROR_INFERENCE_FAILED;

    for (int y = 0; y < output_height; ++y) {
        if (context->cancelled.load(std::memory_order_relaxed)) return ERROR_CANCELLED;

        unsigned char* destination_row = output_ptr + static_cast<size_t>(y) * output_row_stride;
        const size_t tensor_row = static_cast<size_t>(y) * output_width;
        for (int x = 0; x < output_width; ++x) {
            const size_t tensor_index = tensor_row + x;
            unsigned char* destination = destination_row + static_cast<size_t>(x) * 4;
            destination[0] = to_u8(red[tensor_index]);
            destination[1] = to_u8(green[tensor_index]);
            destination[2] = to_u8(blue[tensor_index]);
            destination[3] = sample_alpha_bilinear(
                input_ptr,
                width,
                height,
                input_row_stride,
                x,
                y,
                context->native_scale
            );
        }
    }

    return ERROR_NONE;
}

}  // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_com_ikegami99_kiraenhance_inference_ncnn_NcnnNativeBridge_nativeRuntimeInfo(
    JNIEnv* env,
    jobject /* thiz */
) {
    const int gpu_count = query_gpu_count();
    const std::string info = "ncnn=" + std::to_string(NCNN_VERSION_NUMBER)
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
    if (prefer_gpu == JNI_TRUE) gpu_enabled = acquire_gpu();
#endif

    std::unique_ptr<ModelContext> context(new (std::nothrow) ModelContext());
    if (!context) {
        if (gpu_enabled) release_gpu();
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
    if (gpu_enabled) context->net.set_vulkan_device(0);
#endif

    if (context->net.load_param(param_path.c_str()) != 0) {
        context.reset();
        if (gpu_enabled) release_gpu();
        return make_load_result(env, 0L, ERROR_LOAD_PARAM_FAILED, false);
    }

    if (context->net.load_model(bin_path.c_str()) != 0) {
        context.reset();
        if (gpu_enabled) release_gpu();
        return make_load_result(env, 0L, ERROR_LOAD_MODEL_FAILED, false);
    }

    ModelContext* raw_context = context.release();
    return make_load_result(env, to_handle(raw_context), ERROR_NONE, gpu_enabled);
}

extern "C" JNIEXPORT jlongArray JNICALL
Java_com_ikegami99_kiraenhance_inference_ncnn_NcnnNativeBridge_nativeInfer(
    JNIEnv* env,
    jobject /* thiz */,
    jlong handle,
    jobject input_pixels,
    jint width,
    jint height,
    jint input_row_stride,
    jobject output_pixels,
    jlong output_capacity_bytes
) {
    ModelContext* context = from_handle(handle);
    if (context == nullptr || input_pixels == nullptr || output_pixels == nullptr
        || width <= 0 || height <= 0 || input_row_stride <= 0
        || context->native_scale <= 0 || output_capacity_bytes <= 0) {
        return make_inference_result(env, ERROR_INVALID_ARGUMENT, 0, 0, 0, false);
    }

    const jlong minimum_input_stride = static_cast<jlong>(width) * 4;
    if (minimum_input_stride > std::numeric_limits<jint>::max()
        || static_cast<jlong>(input_row_stride) < minimum_input_stride) {
        return make_inference_result(env, ERROR_INVALID_ARGUMENT, 0, 0, 0, context->gpu_enabled);
    }

    auto* input_ptr = static_cast<unsigned char*>(env->GetDirectBufferAddress(input_pixels));
    auto* output_ptr = static_cast<unsigned char*>(env->GetDirectBufferAddress(output_pixels));
    const jlong input_capacity = env->GetDirectBufferCapacity(input_pixels);
    const jlong actual_output_capacity = env->GetDirectBufferCapacity(output_pixels);
    if (input_ptr == nullptr || output_ptr == nullptr || input_capacity < 0 || actual_output_capacity < 0) {
        return make_inference_result(env, ERROR_INVALID_ARGUMENT, 0, 0, 0, context->gpu_enabled);
    }

    jlong required_input_bytes = 0;
    if (!checked_multiply(static_cast<jlong>(input_row_stride), static_cast<jlong>(height), &required_input_bytes)
        || required_input_bytes > input_capacity) {
        return make_inference_result(env, ERROR_BUFFER_TOO_SMALL, 0, 0, 0, context->gpu_enabled);
    }

    jlong output_width_long = 0;
    jlong output_height_long = 0;
    jlong output_row_stride_long = 0;
    jlong required_output_bytes = 0;
    if (!checked_multiply(static_cast<jlong>(width), context->native_scale, &output_width_long)
        || !checked_multiply(static_cast<jlong>(height), context->native_scale, &output_height_long)
        || !checked_multiply(output_width_long, 4, &output_row_stride_long)
        || !checked_multiply(output_row_stride_long, output_height_long, &required_output_bytes)
        || output_width_long > std::numeric_limits<int>::max()
        || output_height_long > std::numeric_limits<int>::max()
        || output_row_stride_long > std::numeric_limits<int>::max()) {
        return make_inference_result(env, ERROR_OUT_OF_MEMORY, 0, 0, 0, context->gpu_enabled);
    }

    if (required_output_bytes > output_capacity_bytes || required_output_bytes > actual_output_capacity) {
        return make_inference_result(env, ERROR_BUFFER_TOO_SMALL, 0, 0, 0, context->gpu_enabled);
    }

    const int output_width = static_cast<int>(output_width_long);
    const int output_height = static_cast<int>(output_height_long);
    const int output_row_stride = static_cast<int>(output_row_stride_long);

    context->cancelled.store(false, std::memory_order_relaxed);
    const NativeError result = infer_rgba(
        context,
        input_ptr,
        width,
        height,
        input_row_stride,
        output_ptr,
        output_width,
        output_height,
        output_row_stride
    );

    if (result != ERROR_NONE) {
        return make_inference_result(env, result, 0, 0, 0, context->gpu_enabled);
    }

    return make_inference_result(
        env,
        ERROR_NONE,
        output_width,
        output_height,
        output_row_stride,
        context->gpu_enabled
    );
}

extern "C" JNIEXPORT void JNICALL
Java_com_ikegami99_kiraenhance_inference_ncnn_NcnnNativeBridge_nativeUnload(
    JNIEnv* /* env */,
    jobject /* thiz */,
    jlong handle
) {
    ModelContext* context = from_handle(handle);
    if (context == nullptr) return;

    const bool gpu_enabled = context->gpu_enabled;
    delete context;
    if (gpu_enabled) release_gpu();
}

extern "C" JNIEXPORT void JNICALL
Java_com_ikegami99_kiraenhance_inference_ncnn_NcnnNativeBridge_nativeCancel(
    JNIEnv* /* env */,
    jobject /* thiz */,
    jlong handle
) {
    ModelContext* context = from_handle(handle);
    if (context != nullptr) context->cancelled.store(true, std::memory_order_relaxed);
}
