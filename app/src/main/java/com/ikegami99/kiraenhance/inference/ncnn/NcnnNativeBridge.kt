package com.ikegami99.kiraenhance.inference.ncnn

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
}
