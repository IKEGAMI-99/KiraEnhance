package com.ikegami99.kiraenhance.inference.mnn

import java.nio.ByteBuffer

enum class MnnPisaNativeError {
    NONE,
    LOAD_FAILED,
    INVALID_ARGUMENT,
    NOT_IMPLEMENTED,
    INFERENCE_FAILED,
    CANCELLED,
    OUT_OF_MEMORY,
    INTERNAL,
}

data class MnnPisaNativeLoadResult(
    val handle: Long,
    val errorCode: MnnPisaNativeError,
    val gpuEnabled: Boolean,
)

data class MnnPisaNativeInferenceResult(
    val errorCode: MnnPisaNativeError,
    val outputWidth: Int,
    val outputHeight: Int,
    val outputRowStrideBytes: Int,
    val gpuUsed: Boolean,
)

data class MnnPisaDiagnosticsSnapshot(
    val modelVersion: String,
    val sessionInfo: MnnPisaSessionInfo?,
    val tensorInfo: List<MnnPisaTensorInfo>?,
)

interface MnnPisaNativeApi {
    fun loadModel(
        vaeEncoderPath: String,
        unetPath: String,
        vaeDecoderPath: String,
        emptyPromptPath: String,
        preferGpu: Boolean,
    ): MnnPisaNativeLoadResult

    fun sessionInfo(handle: Long): MnnPisaSessionInfo? = null

    fun graphInfo(handle: Long): List<MnnPisaTensorInfo>? = null

    fun infer(
        handle: Long,
        inputPixels: ByteBuffer,
        width: Int,
        height: Int,
        inputRowStrideBytes: Int,
        outputPixels: ByteBuffer,
        outputCapacityBytes: Long,
    ): MnnPisaNativeInferenceResult

    fun unload(handle: Long)

    fun cancel(handle: Long)
}
