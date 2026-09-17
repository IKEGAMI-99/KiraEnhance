package com.ikegami99.kiraenhance.inference.mnn

import java.nio.ByteBuffer

object MnnPisaNativeBridge : MnnPisaNativeApi {
    override fun loadModel(
        vaeEncoderPath: String,
        unetPath: String,
        vaeDecoderPath: String,
        emptyPromptPath: String,
        preferGpu: Boolean,
    ): MnnPisaNativeLoadResult = MnnPisaNativeLoadResult(
        handle = 0L,
        errorCode = MnnPisaNativeError.NOT_IMPLEMENTED,
        gpuEnabled = false,
    )

    override fun infer(
        handle: Long,
        inputPixels: ByteBuffer,
        width: Int,
        height: Int,
        inputRowStrideBytes: Int,
        outputPixels: ByteBuffer,
        outputCapacityBytes: Long,
    ): MnnPisaNativeInferenceResult = MnnPisaNativeInferenceResult(
        errorCode = MnnPisaNativeError.NOT_IMPLEMENTED,
        outputWidth = 0,
        outputHeight = 0,
        outputRowStrideBytes = 0,
        gpuUsed = false,
    )

    override fun unload(handle: Long) = Unit

    override fun cancel(handle: Long) = Unit
}
