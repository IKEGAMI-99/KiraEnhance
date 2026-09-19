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

data class MnnPisaNativePrepareResult(
    val errorCode: MnnPisaNativeError,
    val imageWidth: Int,
    val imageHeight: Int,
    val latentWidth: Int,
    val latentHeight: Int,
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
        vaeSegmentPackPath: String,
        preferGpu: Boolean,
    ): MnnPisaNativeLoadResult

    fun sessionInfo(handle: Long): MnnPisaSessionInfo? = null

    fun graphInfo(handle: Long): List<MnnPisaTensorInfo>? = null

    fun prepareGraph(
        handle: Long,
        imageWidth: Int,
        imageHeight: Int,
    ): MnnPisaNativePrepareResult = MnnPisaNativePrepareResult(
        errorCode = MnnPisaNativeError.NOT_IMPLEMENTED,
        imageWidth = 0,
        imageHeight = 0,
        latentWidth = 0,
        latentHeight = 0,
    )

    fun smokeGraph(
        handle: Long,
        imageWidth: Int,
        imageHeight: Int,
    ): MnnPisaNativeSmokeResult = MnnPisaNativeSmokeResult(
        errorCode = MnnPisaNativeError.NOT_IMPLEMENTED,
        completedStages = 0,
        outputFinite = false,
    )

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
