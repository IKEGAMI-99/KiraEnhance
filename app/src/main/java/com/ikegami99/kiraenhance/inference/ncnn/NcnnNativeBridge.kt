package com.ikegami99.kiraenhance.inference.ncnn

import java.nio.ByteBuffer

object NcnnNativeBridge : NcnnNativeApi {
    init {
        System.loadLibrary("kiraenhance")
    }

    private external fun nativeRuntimeInfo(): String

    private external fun nativeLoadModel(
        paramPath: String,
        binPath: String,
        inputBlobName: String,
        outputBlobName: String,
        nativeScale: Int,
        prePadding: Int,
        preferGpu: Boolean,
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

    private external fun nativeUnload(handle: Long)

    private external fun nativeCancel(handle: Long)

    override fun runtimeInfo(): NcnnRuntimeInfo =
        NcnnRuntimeInfo.parse(nativeRuntimeInfo())

    override fun loadModel(
        paramPath: String,
        binPath: String,
        inputBlobName: String,
        outputBlobName: String,
        nativeScale: Int,
        prePadding: Int,
        preferGpu: Boolean,
    ): NcnnNativeLoadResult {
        val raw = nativeLoadModel(
            paramPath,
            binPath,
            inputBlobName,
            outputBlobName,
            nativeScale,
            prePadding,
            preferGpu,
        )
        require(raw.size == NATIVE_LOAD_RESULT_SIZE) {
            "Malformed native ncnn load result"
        }

        return NcnnNativeLoadResult(
            handle = raw[0],
            errorCode = NcnnNativeError.fromCode(raw[1].toInt()),
            gpuEnabled = raw[2] == 1L,
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
    ): NcnnNativeInferenceResult {
        val raw = nativeInfer(
            handle,
            inputPixels,
            width,
            height,
            inputRowStrideBytes,
            outputPixels,
            outputCapacityBytes,
        )
        require(raw.size == NATIVE_INFERENCE_RESULT_SIZE) {
            "Malformed native ncnn inference result"
        }

        return NcnnNativeInferenceResult(
            errorCode = NcnnNativeError.fromCode(raw[0].toInt()),
            outputWidth = raw[1].toInt(),
            outputHeight = raw[2].toInt(),
            outputRowStrideBytes = raw[3].toInt(),
            gpuUsed = raw[4] == 1L,
        )
    }

    override fun unload(handle: Long) {
        if (handle != 0L) {
            nativeUnload(handle)
        }
    }

    override fun cancel(handle: Long) {
        if (handle != 0L) {
            nativeCancel(handle)
        }
    }

    private const val NATIVE_LOAD_RESULT_SIZE = 3
    private const val NATIVE_INFERENCE_RESULT_SIZE = 5
}
