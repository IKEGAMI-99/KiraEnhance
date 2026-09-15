package com.ikegami99.kiraenhance.inference.ncnn

import com.ikegami99.kiraenhance.inference.EngineError
import com.ikegami99.kiraenhance.inference.EngineErrorCode
import com.ikegami99.kiraenhance.inference.ModelLoadRequest
import com.ikegami99.kiraenhance.inference.ModelLoadResult
import com.ikegami99.kiraenhance.inference.PixelFormat
import com.ikegami99.kiraenhance.inference.UpscaleEngine
import com.ikegami99.kiraenhance.inference.UpscaleInput
import com.ikegami99.kiraenhance.inference.UpscaleOutput
import com.ikegami99.kiraenhance.inference.UpscaleResult
import com.ikegami99.kiraenhance.inference.UpscaleSettings
import java.nio.ByteBuffer

class NcnnUpscaleEngine(
    private val nativeApi: NcnnNativeApi = NcnnNativeBridge,
) : UpscaleEngine {
    private val stateLock = Any()
    private var nativeHandle: Long = 0L
    private var loadedRequest: ModelLoadRequest? = null
    private var gpuEnabled: Boolean = false
    private var inferenceInProgress: Boolean = false
    private var pendingUnload: Boolean = false

    override fun load(request: ModelLoadRequest): ModelLoadResult {
        synchronized(stateLock) {
            if (inferenceInProgress) {
                return ModelLoadResult.Failed(
                    EngineError(
                        code = EngineErrorCode.INTERNAL,
                        message = "Cannot replace an ncnn model while inference is running",
                    ),
                )
            }
            unloadLocked()
        }

        val bundle = try {
            NcnnModelBundle.resolve(request)
        } catch (error: IllegalArgumentException) {
            return ModelLoadResult.Failed(
                EngineError(
                    code = EngineErrorCode.MODEL_ARTIFACT_MISSING,
                    message = error.message ?: "Required ncnn model artifacts are missing",
                ),
            )
        }

        if (!bundle.paramFile.isFile || !bundle.binFile.isFile) {
            return ModelLoadResult.Failed(
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
            return ModelLoadResult.Failed(
                EngineError(
                    code = EngineErrorCode.MODEL_LOAD_FAILED,
                    message = nativeErrorMessage(nativeResult.errorCode),
                ),
            )
        }

        synchronized(stateLock) {
            nativeHandle = nativeResult.handle
            loadedRequest = request
            gpuEnabled = nativeResult.gpuEnabled
            pendingUnload = false
        }
        return ModelLoadResult.Loaded(gpuEnabled = nativeResult.gpuEnabled)
    }

    override fun isLoaded(): Boolean = synchronized(stateLock) {
        nativeHandle != 0L && loadedRequest != null
    }

    override fun upscale(
        input: UpscaleInput,
        settings: UpscaleSettings,
    ): UpscaleResult {
        val handle: Long
        val request: ModelLoadRequest
        synchronized(stateLock) {
            handle = nativeHandle
            request = loadedRequest ?: return UpscaleResult.Failed(
                EngineError(
                    code = EngineErrorCode.MODEL_LOAD_FAILED,
                    message = "No ncnn model is loaded",
                ),
            )
            if (handle == 0L) {
                return UpscaleResult.Failed(
                    EngineError(
                        code = EngineErrorCode.MODEL_LOAD_FAILED,
                        message = "No ncnn model is loaded",
                    ),
                )
            }
            if (inferenceInProgress) {
                return UpscaleResult.Failed(
                    EngineError(
                        code = EngineErrorCode.INFERENCE_FAILED,
                        message = "An ncnn inference is already running",
                    ),
                )
            }
            inferenceInProgress = true
        }

        try {
            val validationError = validateInference(input, settings, request)
            if (validationError != null) {
                return UpscaleResult.Failed(validationError)
            }

            val outputWidthLong = input.width.toLong() * settings.outputScale.toLong()
            val outputHeightLong = input.height.toLong() * settings.outputScale.toLong()
            val outputRowStrideLong = outputWidthLong * BYTES_PER_RGBA_PIXEL
            val outputBytes = outputRowStrideLong * outputHeightLong

            if (
                outputWidthLong > Int.MAX_VALUE ||
                outputHeightLong > Int.MAX_VALUE ||
                outputRowStrideLong > Int.MAX_VALUE ||
                outputBytes <= 0L
            ) {
                return UpscaleResult.Failed(
                    EngineError(
                        code = EngineErrorCode.OUT_OF_MEMORY,
                        message = "Requested output dimensions are too large",
                    ),
                )
            }

            if (outputBytes > MAX_SMOKE_OUTPUT_BYTES) {
                return UpscaleResult.Failed(
                    EngineError(
                        code = EngineErrorCode.OUT_OF_MEMORY,
                        message = "Small-image inference is limited to 64 MiB output; use tiled inference",
                    ),
                )
            }

            val directInput = input.pixels.asDirectSlice()
            val output = try {
                ByteBuffer.allocateDirect(outputBytes.toInt())
            } catch (_: OutOfMemoryError) {
                return UpscaleResult.Failed(
                    EngineError(
                        code = EngineErrorCode.OUT_OF_MEMORY,
                        message = "Unable to allocate output pixel buffer",
                    ),
                )
            }

            val nativeResult = nativeApi.infer(
                handle = handle,
                inputPixels = directInput,
                width = input.width,
                height = input.height,
                inputRowStrideBytes = input.rowStrideBytes,
                outputPixels = output,
                outputCapacityBytes = outputBytes,
            )

            if (nativeResult.errorCode != NcnnNativeError.NONE) {
                return UpscaleResult.Failed(nativeInferenceError(nativeResult.errorCode))
            }

            val outputWidth = outputWidthLong.toInt()
            val outputHeight = outputHeightLong.toInt()
            val outputRowStride = outputRowStrideLong.toInt()
            if (
                nativeResult.outputWidth != outputWidth ||
                nativeResult.outputHeight != outputHeight ||
                nativeResult.outputRowStrideBytes != outputRowStride
            ) {
                return UpscaleResult.Failed(
                    EngineError(
                        code = EngineErrorCode.INTERNAL,
                        message = "ncnn returned unexpected output dimensions",
                    ),
                )
            }

            output.position(0)
            output.limit(outputBytes.toInt())
            return UpscaleResult.Success(
                output = UpscaleOutput(
                    width = outputWidth,
                    height = outputHeight,
                    rowStrideBytes = outputRowStride,
                    pixelFormat = PixelFormat.RGBA_8888,
                    pixels = output,
                ),
                usedGpu = nativeResult.gpuUsed,
            )
        } finally {
            synchronized(stateLock) {
                inferenceInProgress = false
                if (pendingUnload) {
                    unloadLocked()
                    pendingUnload = false
                }
            }
        }
    }

    override fun cancel() {
        val handle = synchronized(stateLock) { nativeHandle }
        if (handle != 0L) {
            nativeApi.cancel(handle)
        }
    }

    override fun unload() {
        val handleToCancel: Long
        synchronized(stateLock) {
            if (inferenceInProgress) {
                pendingUnload = true
                handleToCancel = nativeHandle
            } else {
                unloadLocked()
                return
            }
        }
        if (handleToCancel != 0L) {
            nativeApi.cancel(handleToCancel)
        }
    }

    private fun validateInference(
        input: UpscaleInput,
        settings: UpscaleSettings,
        request: ModelLoadRequest,
    ): EngineError? {
        if (input.pixelFormat != request.capabilities.pixelFormat || input.pixelFormat != PixelFormat.RGBA_8888) {
            return EngineError(
                code = EngineErrorCode.INVALID_INPUT,
                message = "Unsupported ncnn input pixel format",
            )
        }
        if (settings.outputScale != request.capabilities.nativeScale) {
            return EngineError(
                code = EngineErrorCode.INVALID_INPUT,
                message = "This ncnn smoke path only supports the model's native scale",
            )
        }
        return null
    }

    private fun ByteBuffer.asDirectSlice(): ByteBuffer {
        val source = duplicate()
        if (source.isDirect) {
            return source.slice()
        }

        return ByteBuffer.allocateDirect(source.remaining()).apply {
            put(source)
            flip()
        }
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
        NcnnNativeError.INFERENCE_FAILED -> "ncnn inference failed"
        NcnnNativeError.CANCELLED -> "ncnn operation was cancelled"
        NcnnNativeError.BUFFER_TOO_SMALL -> "ncnn output buffer is too small"
        NcnnNativeError.OUT_OF_MEMORY -> "ncnn ran out of memory"
    }

    private fun nativeInferenceError(error: NcnnNativeError): EngineError = when (error) {
        NcnnNativeError.CANCELLED -> EngineError(
            code = EngineErrorCode.CANCELLED,
            message = "ncnn inference was cancelled",
        )
        NcnnNativeError.OUT_OF_MEMORY -> EngineError(
            code = EngineErrorCode.OUT_OF_MEMORY,
            message = "ncnn could not allocate inference memory",
        )
        NcnnNativeError.INVALID_ARGUMENT,
        NcnnNativeError.BUFFER_TOO_SMALL,
        -> EngineError(
            code = EngineErrorCode.INVALID_INPUT,
            message = nativeErrorMessage(error),
        )
        NcnnNativeError.NONE,
        NcnnNativeError.LOAD_PARAM_FAILED,
        NcnnNativeError.LOAD_MODEL_FAILED,
        NcnnNativeError.INTERNAL,
        NcnnNativeError.INFERENCE_FAILED,
        -> EngineError(
            code = EngineErrorCode.INFERENCE_FAILED,
            message = nativeErrorMessage(error),
        )
    }

    private companion object {
        const val BYTES_PER_RGBA_PIXEL = 4L
        const val MAX_SMOKE_OUTPUT_BYTES = 64L * 1024L * 1024L
    }
}
