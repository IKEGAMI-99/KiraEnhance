package com.ikegami99.kiraenhance.inference.ncnn

enum class NcnnNativeError(val code: Int) {
    NONE(0),
    LOAD_PARAM_FAILED(1),
    LOAD_MODEL_FAILED(2),
    INVALID_ARGUMENT(3),
    INTERNAL(4),
    ;

    companion object {
        fun fromCode(code: Int): NcnnNativeError =
            entries.firstOrNull { it.code == code } ?: INTERNAL
    }
}

data class NcnnNativeLoadResult(
    val handle: Long,
    val errorCode: NcnnNativeError,
    val gpuEnabled: Boolean,
)

interface NcnnNativeApi {
    fun runtimeInfo(): NcnnRuntimeInfo

    fun loadModel(
        paramPath: String,
        binPath: String,
        inputBlobName: String,
        outputBlobName: String,
        nativeScale: Int,
        prePadding: Int,
        preferGpu: Boolean,
    ): NcnnNativeLoadResult

    fun unload(handle: Long)

    fun cancel(handle: Long)
}
