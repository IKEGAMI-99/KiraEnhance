#include <jni.h>

#include <string>

#include <net.h>
#include <platform.h>
#if NCNN_VULKAN
#include <gpu.h>
#endif

extern "C" JNIEXPORT jstring JNICALL
Java_com_ikegami99_kiraenhance_inference_ncnn_NcnnNativeBridge_nativeRuntimeInfo(
    JNIEnv* env,
    jobject /* thiz */
) {
    int gpu_count = 0;

#if NCNN_VULKAN
    const int create_result = ncnn::create_gpu_instance();
    if (create_result == 0) {
        gpu_count = ncnn::get_gpu_count();
    }
    ncnn::destroy_gpu_instance();
#endif

    std::string info = "ncnn=" + std::to_string(NCNN_VERSION_NUMBER)
        + ";vulkan=" + std::string(NCNN_VULKAN ? "1" : "0")
        + ";gpuCount=" + std::to_string(gpu_count);

    return env->NewStringUTF(info.c_str());
}
