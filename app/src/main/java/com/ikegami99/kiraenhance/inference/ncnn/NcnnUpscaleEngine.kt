package com.ikegami99.kiraenhance.inference.ncnn

import com.ikegami99.kiraenhance.inference.EngineError
import com.ikegami99.kiraenhance.inference.EngineErrorCode
import com.ikegami99.kiraenhance.inference.ModelLoadRequest
import com.ikegami99.kiraenhance.inference.ModelLoadResult
import com.ikegami99.kiraenhance.inference.UpscaleEngine
import com.ikegami99.kiraenhance.inference.UpscaleInput
import com.ikegami99.kiraenhance.inference.UpscaleResult
import com.ikegami99.kiraenhance.inference.UpscaleSettings

class NcnnUpscaleEngine(
    private val nativeApi: NcnnNativeApi = NcnnNativeBridge,
) : UpscaleEngine {
    private val lock = Any()
    private var nativeHandle: Long = 0L
    private var loadedRequest: ModelLoadRequest? = null
    private var gpuEnabled: Boolean = false

    override fun load(request: ModelLoadRequest): ModelLoadResult = synchronized(lock) {
        unloadLocked()

        val bundle = try {
            NcnnModelBundle.resolve(request)
        } catch (error: IllegalArgumentException) {
            return@synchronized ModelLoadResult.Failed(
                EngineError(
                    code = EngineErrorCode.MODEL_ARTIFACT_MISSING,
                    message = error.message ?: "Required ncnn model artifacts are missing",
                ),
            )
        }

        if (!bundle.paramFile.isFile || !bundle.binFile.isFile) {
            return@synchronized ModelLoadResult.Failed(
                EngineError(
                    code = EngineErrorCode.MODEL_ARTIFACT_MISSING,
                    message = "ncnn model requires readable .param and .bin files",
                ),
            )
        }

        val runtime = nativeApi.runtimeInfo()
        val preferGpu = request.capabilities.supportsGpu && runtime.hasUsableVulkanGpu
        val nativeResult = nativeApi.loadModel(
            paramPath = bundle.paramFile.absolutePath,
            binPath = bundle.binFile.absolutePath,
            inputBlobName = request.capabilities.inputBlobName,
            outputBlobName = request.capabilities.outputBlobName,
            nativeScale = request.capabilities.nativeScale,
            prePadding = request.capabilities.prePadding,
            preferGpu = preferGpu,
        )

        if (nativeResult.errorCode != NcnnNativeError.NONE || nativeResult.handle == 0L) {
            if (nativeResult.handle != 0L) {
                nativeApi.unload(nativeResult.handle)
            }
            return@synchronized ModelLoadResult.Failed(
                EngineError(
                    code = EngineErrorCode.MODEL_LOAD_FAILED,
                    message = nativeErrorMessage(nativeResult.errorCode),
                ),
            )
        }

        nativeHandle = nativeResult.handle
        loadedRequest = request
        gpuEnabled = nativeResult.gpuEnabled
        ModelLoadResult.Loaded(gpuEnabled = gpuEnabled)
    }

    override fun isLoaded(): Boolean = synchronized(lock) {
        nativeHandle != 0L && loadedRequest != null
    }

    override fun upscale(
        input: UpscaleInput,
        settings: UpscaleSettings,
    ): UpscaleResult = synchronized(lock) {
        if (nativeHandle == 0L || loadedRequest == null) {
            return@synchronized UpscaleResult.Failed(
                EngineError(
                    code = EngineErrorCode.MODEL_LOAD_FAILED,
                    message = "No ncnn model is loaded",
                ),
            )
        }

        UpscaleResult.Failed(
            EngineError(
                code = EngineErrorCode.INFERENCE_FAILED,
                message = "ncnn image inference is not implemented yet",
            ),
        )
    }

    override fun cancel() = synchronized(lock) {
        if (nativeHandle != 0L) {
            nativeApi.cancel(nativeHandle)
        }
    }

    override fun unload() = synchronized(lock) {
        unloadLocked()
    }

    private fun unloadLocked() {
        if (nativeHandle != 0L) {
            nativeApi.unload(nativeHandle)
        }
        nativeHandle = 0L
        loadedRequest = null
        gpuEnabled = false
    }

    private fun nativeErrorMessage(error: NcnnNativeError): String = when (error) {
        NcnnNativeError.NONE -> "Unknown ncnn model load failure"
        NcnnNativeError.LOAD_PARAM_FAILED -> "ncnn failed to load the model .param file"
        NcnnNativeError.LOAD_MODEL_FAILED -> "ncnn failed to load the model .bin file"
        NcnnNativeError.INVALID_ARGUMENT -> "Invalid ncnn model load arguments"
        NcnnNativeError.INTERNAL -> "Internal ncnn model load failure"
    }
}
