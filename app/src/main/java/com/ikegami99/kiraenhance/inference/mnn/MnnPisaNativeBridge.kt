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

    private external fun nativeSessionInfo(handle: Long): String

    private external fun nativeGraphInfo(handle: Long): String

    private external fun nativePrepareGraph(
        handle: Long,
        imageWidth: Int,
        imageHeight: Int,
    ): LongArray

    private external fun nativeInfer(
        handle: Long,
        inputPixels: ByteBuffer,
        width: Int,
        height: Int,
        inputRowStrideBytes: Int,
        outputPixels: ByteBuffer,
        outputCapacityBytes: Long,
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

    override fun sessionInfo(handle: Long): MnnPisaSessionInfo? {
        if (handle == 0L) {
            return null
        }
        return runCatching {
            ensureNativeLibraryLoaded()
            MnnPisaSessionInfo.parse(nativeSessionInfo(handle))
        }.getOrNull()
    }

    override fun graphInfo(handle: Long): List<MnnPisaTensorInfo>? {
        if (handle == 0L) {
            return null
        }
        return runCatching {
            ensureNativeLibraryLoaded()
            val raw = nativeGraphInfo(handle)
            if (raw.isBlank()) {
                null
            } else {
                MnnPisaTensorInfoCodec.decode(raw)
            }
        }.getOrNull()
    }

    override fun prepareGraph(
        handle: Long,
        imageWidth: Int,
        imageHeight: Int,
    ): MnnPisaNativePrepareResult = runCatching {
        ensureNativeLibraryLoaded()
        MnnPisaNativePrepareResultCodec.decode(
            nativePrepareGraph(
                handle = handle,
                imageWidth = imageWidth,
                imageHeight = imageHeight,
            ),
        )
    }.getOrElse {
        MnnPisaNativePrepareResult(
            errorCode = MnnPisaNativeError.INTERNAL,
            imageWidth = 0,
            imageHeight = 0,
            latentWidth = 0,
            latentHeight = 0,
        )
    }

    override fun infer(
        handle: Long,
        inputPixels: ByteBuffer,
        width: Int,
        height: Int,
        inputRowStrideBytes: Int,
        outputPixels: ByteBuffer,
        outputCapacityBytes: Long,
    ): MnnPisaNativeInferenceResult = runCatching {
        ensureNativeLibraryLoaded()
        MnnPisaNativeInferenceResultCodec.decode(
            nativeInfer(
                handle = handle,
                inputPixels = inputPixels,
                width = width,
                height = height,
                inputRowStrideBytes = inputRowStrideBytes,
                outputPixels = outputPixels,
                outputCapacityBytes = outputCapacityBytes,
            ),
        )
    }.getOrElse {
        MnnPisaNativeInferenceResult(
            errorCode = MnnPisaNativeError.INTERNAL,
            outputWidth = 0,
            outputHeight = 0,
            outputRowStrideBytes = 0,
            gpuUsed = false,
        )
    }

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
