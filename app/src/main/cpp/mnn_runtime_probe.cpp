#include <jni.h>

#include <string>

#if KIRA_HAS_MNN
#include <MNN/Interpreter.hpp>
#endif

extern "C" JNIEXPORT jstring JNICALL
Java_com_ikegami99_kiraenhance_inference_mnn_MnnPisaNativeBridge_nativeRuntimeInfo(
    JNIEnv* env,
    jobject /* thiz */
) {
#if KIRA_HAS_MNN
    const char* version = MNN::getVersion();
    const std::string info = "mnn="
        + std::string(version != nullptr ? version : "unknown")
        + ";linked=1";
#else
    const std::string info = "mnn=unavailable;linked=0";
#endif
    return env->NewStringUTF(info.c_str());
}
