package com.ikegami99.kiraenhance.inference.mnn

import java.nio.ByteBuffer

object MnnPisaNativeBridge : MnnPisaNativeApi {
    private val nativeLibraryLoaded by lazy {
        System.loadLibrary("kiraenhance")
        true
    }

    private external fun nativeRuntimeInfo(): String

    private external fun nativeLoadModel(
        vaeEncoderPath: String,
        unetPath: String,
        vaeDecoderPath: String,
        emptyPromptPath: String,
        preferGpu: Boolean,
    ): LongArray

    private external fun nativeUnloadModel(handle: Long)

    private fun ensureNativeLibraryLoaded() {
        nativeLibraryLoaded
    }

    fun runtimeInfo(): MnnRuntimeInfo {
        ensureNativeLibraryLoaded()
        return MnnRuntimeInfo.parse(nativeRuntimeInfo())
    }

    override fun loadModel(
        vaeEncoderPath: String,
        unetPath: String,
        vaeDecoderPath: String,
        emptyPromptPath: String,
        preferGpu: Boolean,
    ): MnnPisaNativeLoadResult = runCatching {
        ensureNativeLibraryLoaded()
        MnnPisaNativeLoadResultCodec.decode(
            nativeLoadModel(
                vaeEncoderPath = vaeEncoderPath,
                unetPath = unetPath,
                vaeDecoderPath = vaeDecoderPath,
                emptyPromptPath = emptyPromptPath,
                preferGpu = preferGpu,
            ),
        )
    }.getOrElse {
        nativeLoadFailure()
    }

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

    override fun unload(handle: Long) {
        if (handle == 0L) {
            return
        }
        runCatching {
            ensureNativeLibraryLoaded()
            nativeUnloadModel(handle)
        }
    }

    override fun cancel(handle: Long) = Unit

    private fun nativeLoadFailure() = MnnPisaNativeLoadResult(
        handle = 0L,
        errorCode = MnnPisaNativeError.LOAD_FAILED,
        gpuEnabled = false,
    )
}
