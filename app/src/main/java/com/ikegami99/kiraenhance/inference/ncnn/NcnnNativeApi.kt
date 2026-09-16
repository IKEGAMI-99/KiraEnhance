package com.ikegami99.kiraenhance.inference.ncnn

import java.nio.ByteBuffer

enum class NcnnNativeError(val code: Int) {
    NONE(0),
    LOAD_PARAM_FAILED(1),
    LOAD_MODEL_FAILED(2),
    INVALID_ARGUMENT(3),
    INTERNAL(4),
    INFERENCE_FAILED(5),
    CANCELLED(6),
    BUFFER_TOO_SMALL(7),
    OUT_OF_MEMORY(8),
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

data class NcnnNativeInferenceResult(
    val errorCode: NcnnNativeError,
    val outputWidth: Int,
    val outputHeight: Int,
    val outputRowStrideBytes: Int,
    val gpuUsed: Boolean,
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

    fun infer(
        handle: Long,
        inputPixels: ByteBuffer,
        width: Int,
        height: Int,
        inputRowStrideBytes: Int,
        outputPixels: ByteBuffer,
        outputCapacityBytes: Long,
    ): NcnnNativeInferenceResult = NcnnNativeInferenceResult(
        errorCode = NcnnNativeError.INTERNAL,
        outputWidth = 0,
        outputHeight = 0,
        outputRowStrideBytes = 0,
        gpuUsed = false,
    )

    fun unload(handle: Long)

    fun cancel(handle: Long)
}
