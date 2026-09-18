package com.ikegami99.kiraenhance.inference.mnn

import com.ikegami99.kiraenhance.inference.EngineError
import com.ikegami99.kiraenhance.inference.EngineErrorCode
import com.ikegami99.kiraenhance.inference.ModelLoadRequest
import com.ikegami99.kiraenhance.inference.ModelLoadResult
import com.ikegami99.kiraenhance.inference.PixelFormat
import com.ikegami99.kiraenhance.inference.UpscaleEngine
import com.ikegami99.kiraenhance.inference.UpscaleInput
import com.ikegami99.kiraenhance.inference.UpscaleOutput
import com.ikegami99.kiraenhance.inference.UpscaleProgress
import com.ikegami99.kiraenhance.inference.UpscaleResult
import com.ikegami99.kiraenhance.inference.UpscaleSettings
import java.io.File
import java.nio.ByteBuffer

class MnnPisaUpscaleEngine(
    private val nativeApi: MnnPisaNativeApi = MnnPisaNativeBridge,
    private val onDiagnostics: (MnnPisaDiagnosticsSnapshot) -> Unit = {},
) : UpscaleEngine {
    private var nativeHandle: Long = 0L
    private var loadedRequest: ModelLoadRequest? = null
    private var diagnosticsSnapshot: MnnPisaDiagnosticsSnapshot? = null
    private var currentProgress = UpscaleProgress()

    override fun load(request: ModelLoadRequest): ModelLoadResult {
        unload()

        if (request.modelId != MODEL_ID) {
            return loadFailure("PiSA-SR engine cannot load model ${request.modelId}")
        }
        if (request.capabilities.pixelFormat != PixelFormat.RGBA_8888) {
            return loadFailure("PiSA-SR requires RGBA_8888 input")
        }
        if (request.capabilities.nativeScale != NATIVE_SCALE) {
            return loadFailure("PiSA-SR requires native 4x scale")
        }

        val files = LinkedHashMap<String, File>()
        for (fileName in PisaModelRequestFactory.REQUIRED_ARTIFACTS) {
            val artifact = request.artifact(fileName)
                ?: return missingArtifact(fileName)
            val file = File(artifact.absolutePath)
            if (!file.isFile) {
                return missingArtifact(fileName)
            }
            files[fileName] = file
        }

        val nativeResult = runCatching {
            nativeApi.loadModel(
                vaeEncoderPath = files.getValue(PisaModelRequestFactory.VAE_ENCODER_FILE).absolutePath,
                unetPath = files.getValue(PisaModelRequestFactory.UNET_FILE).absolutePath,
                vaeDecoderPath = files.getValue(PisaModelRequestFactory.VAE_DECODER_FILE).absolutePath,
                emptyPromptPath = files.getValue(PisaModelRequestFactory.EMPTY_PROMPT_FILE).absolutePath,
                preferGpu = request.capabilities.supportsGpu,
            )
        }.getOrElse { error ->
            return loadFailure("MNN PiSA-SR native load failed: ${error.message ?: error.javaClass.simpleName}")
        }

        if (nativeResult.errorCode != MnnPisaNativeError.NONE || nativeResult.handle == 0L) {
            return loadFailure("MNN PiSA-SR native load failed: ${nativeResult.errorCode}")
        }

        nativeHandle = nativeResult.handle
        loadedRequest = request
        diagnosticsSnapshot = MnnPisaDiagnosticsSnapshot(
            modelVersion = request.version,
            sessionInfo = runCatching {
                nativeApi.sessionInfo(nativeResult.handle)
            }.getOrNull(),
            tensorInfo = runCatching {
                nativeApi.graphInfo(nativeResult.handle)?.toList()
            }.getOrNull(),
        )
        diagnosticsSnapshot?.let { snapshot ->
            runCatching {
                onDiagnostics(snapshot)
            }
        }
        currentProgress = UpscaleProgress()
        return ModelLoadResult.Loaded(gpuEnabled = nativeResult.gpuEnabled)
    }

    override fun isLoaded(): Boolean = nativeHandle != 0L && loadedRequest != null

    fun diagnostics(): MnnPisaDiagnosticsSnapshot? = diagnosticsSnapshot

    fun prepareGraph(
        imageWidth: Int,
        imageHeight: Int,
    ): MnnPisaNativePrepareResult {
        val handle = nativeHandle
        if (handle == 0L || loadedRequest == null) {
            return prepareFailure(MnnPisaNativeError.LOAD_FAILED)
        }
        if (imageWidth <= 0 || imageHeight <= 0) {
            return prepareFailure(MnnPisaNativeError.INVALID_ARGUMENT)
        }

        return runCatching {
            nativeApi.prepareGraph(
                handle = handle,
                imageWidth = imageWidth,
                imageHeight = imageHeight,
            )
        }.getOrElse {
            prepareFailure(MnnPisaNativeError.INTERNAL)
        }
    }

    fun smokeGraph(
        imageWidth: Int,
        imageHeight: Int,
    ): MnnPisaNativeSmokeResult {
        val handle = nativeHandle
        if (handle == 0L || loadedRequest == null) {
            return smokeFailure(MnnPisaNativeError.LOAD_FAILED)
        }
        if (imageWidth <= 0 || imageHeight <= 0) {
            return smokeFailure(MnnPisaNativeError.INVALID_ARGUMENT)
        }

        return runCatching {
            nativeApi.smokeGraph(
                handle = handle,
                imageWidth = imageWidth,
                imageHeight = imageHeight,
            )
        }.getOrElse {
            smokeFailure(MnnPisaNativeError.INTERNAL)
        }
    }

    override fun progress(): UpscaleProgress = currentProgress

    override fun upscale(
        input: UpscaleInput,
        settings: UpscaleSettings,
    ): UpscaleResult {
        val request = loadedRequest
            ?: return failed(EngineErrorCode.MODEL_LOAD_FAILED, "PiSA-SR model is not loaded")
        val handle = nativeHandle
        if (handle == 0L) {
            return failed(EngineErrorCode.MODEL_LOAD_FAILED, "PiSA-SR model is not loaded")
        }
        if (input.pixelFormat != request.capabilities.pixelFormat) {
            return failed(EngineErrorCode.INVALID_INPUT, "PiSA-SR requires RGBA_8888 input")
        }
        if (settings.outputScale != request.capabilities.nativeScale) {
            return failed(
                EngineErrorCode.INVALID_INPUT,
                "PiSA-SR supports ${request.capabilities.nativeScale}x output in this engine",
            )
        }

        val outputWidthLong = input.width.toLong() * settings.outputScale.toLong()
        val outputHeightLong = input.height.toLong() * settings.outputScale.toLong()
        if (
            outputWidthLong <= 0L ||
            outputHeightLong <= 0L ||
            outputWidthLong > Int.MAX_VALUE ||
            outputHeightLong > Int.MAX_VALUE
        ) {
            return failed(EngineErrorCode.OUT_OF_MEMORY, "PiSA-SR output dimensions are too large")
        }

        if (
            input.width < MIN_MONOLITHIC_SOURCE_SIDE ||
            input.height < MIN_MONOLITHIC_SOURCE_SIDE
        ) {
            return failed(
                EngineErrorCode.INFERENCE_FAILED,
                "PiSA-SR minimum-size preprocessing is not implemented yet",
            )
        }

        if (outputWidthLong > Long.MAX_VALUE / outputHeightLong) {
            return failed(EngineErrorCode.OUT_OF_MEMORY, "PiSA-SR output dimensions are too large")
        }
        val outputPixelsLong = outputWidthLong * outputHeightLong
        if (outputPixelsLong > MAX_MONOLITHIC_OUTPUT_PIXELS) {
            return failed(
                EngineErrorCode.INFERENCE_FAILED,
                "PiSA-SR large-image VAE tiled inference is not implemented yet",
            )
        }

        val outputWidth = outputWidthLong.toInt()
        val outputHeight = outputHeightLong.toInt()
        val rowStrideLong = outputWidthLong * BYTES_PER_PIXEL
        if (
            rowStrideLong > Int.MAX_VALUE ||
            rowStrideLong <= 0L ||
            outputHeightLong > Int.MAX_VALUE.toLong() / rowStrideLong
        ) {
            return failed(EngineErrorCode.OUT_OF_MEMORY, "PiSA-SR output buffer is too large")
        }
        val capacityLong = rowStrideLong * outputHeightLong

        val outputBuffer = try {
            ByteBuffer.allocateDirect(capacityLong.toInt())
        } catch (_: OutOfMemoryError) {
            return failed(EngineErrorCode.OUT_OF_MEMORY, "Unable to allocate PiSA-SR output buffer")
        }

        currentProgress = UpscaleProgress(completedTiles = 0, totalTiles = 1)
        val nativeResult = runCatching {
            nativeApi.infer(
                handle = handle,
                inputPixels = input.pixels,
                width = input.width,
                height = input.height,
                inputRowStrideBytes = input.rowStrideBytes,
                outputPixels = outputBuffer,
                outputCapacityBytes = capacityLong,
            )
        }.getOrElse { error ->
            return failed(
                EngineErrorCode.INFERENCE_FAILED,
                "MNN PiSA-SR inference failed: ${error.message ?: error.javaClass.simpleName}",
            )
        }

        if (nativeResult.errorCode != MnnPisaNativeError.NONE) {
            return failed(mapInferenceError(nativeResult.errorCode), nativeErrorMessage(nativeResult.errorCode))
        }

        val nativeWidth = nativeResult.outputWidth
        val nativeHeight = nativeResult.outputHeight
        val nativeStride = nativeResult.outputRowStrideBytes
        if (
            nativeWidth <= 0 || nativeHeight <= 0 ||
            nativeStride < nativeWidth * BYTES_PER_PIXEL ||
            nativeWidth != outputWidth || nativeHeight != outputHeight
        ) {
            return failed(EngineErrorCode.INFERENCE_FAILED, "MNN PiSA-SR returned invalid output dimensions")
        }

        val requiredBytes = nativeStride.toLong() * nativeHeight.toLong()
        if (requiredBytes > outputBuffer.capacity().toLong()) {
            return failed(EngineErrorCode.INFERENCE_FAILED, "MNN PiSA-SR returned an oversized output buffer")
        }

        outputBuffer.position(0)
        outputBuffer.limit(requiredBytes.toInt())
        currentProgress = UpscaleProgress(completedTiles = 1, totalTiles = 1)
        return UpscaleResult.Success(
            output = UpscaleOutput(
                width = nativeWidth,
                height = nativeHeight,
                rowStrideBytes = nativeStride,
                pixelFormat = PixelFormat.RGBA_8888,
                pixels = outputBuffer,
            ),
            usedGpu = nativeResult.gpuUsed,
        )
    }

    override fun cancel() {
        if (nativeHandle != 0L) {
            nativeApi.cancel(nativeHandle)
        }
    }

    override fun unload() {
        val handle = nativeHandle
        nativeHandle = 0L
        loadedRequest = null
        diagnosticsSnapshot = null
        currentProgress = UpscaleProgress()
        if (handle != 0L) {
            nativeApi.unload(handle)
        }
    }

    private fun missingArtifact(fileName: String): ModelLoadResult.Failed = ModelLoadResult.Failed(
        EngineError(
            code = EngineErrorCode.MODEL_ARTIFACT_MISSING,
            message = "Missing PiSA-SR artifact: $fileName",
        ),
    )

    private fun loadFailure(message: String): ModelLoadResult.Failed = ModelLoadResult.Failed(
        EngineError(
            code = EngineErrorCode.MODEL_LOAD_FAILED,
            message = message,
        ),
    )

    private fun failed(code: EngineErrorCode, message: String): UpscaleResult.Failed = UpscaleResult.Failed(
        EngineError(code = code, message = message),
    )

    private fun prepareFailure(error: MnnPisaNativeError) = MnnPisaNativePrepareResult(
        errorCode = error,
        imageWidth = 0,
        imageHeight = 0,
        latentWidth = 0,
        latentHeight = 0,
    )

    private fun smokeFailure(error: MnnPisaNativeError) = MnnPisaNativeSmokeResult(
        errorCode = error,
        completedStages = 0,
        outputFinite = false,
    )

    private fun mapInferenceError(error: MnnPisaNativeError): EngineErrorCode = when (error) {
        MnnPisaNativeError.CANCELLED -> EngineErrorCode.CANCELLED
        MnnPisaNativeError.OUT_OF_MEMORY -> EngineErrorCode.OUT_OF_MEMORY
        MnnPisaNativeError.INVALID_ARGUMENT -> EngineErrorCode.INVALID_INPUT
        MnnPisaNativeError.NOT_IMPLEMENTED,
        MnnPisaNativeError.INFERENCE_FAILED,
        -> EngineErrorCode.INFERENCE_FAILED
        MnnPisaNativeError.LOAD_FAILED -> EngineErrorCode.MODEL_LOAD_FAILED
        MnnPisaNativeError.INTERNAL,
        MnnPisaNativeError.NONE,
        -> EngineErrorCode.INTERNAL
    }

    private fun nativeErrorMessage(error: MnnPisaNativeError): String = when (error) {
        MnnPisaNativeError.NOT_IMPLEMENTED -> "PiSA-SR MNN image path does not support this input yet"
        else -> "MNN PiSA-SR inference failed: $error"
    }

    private companion object {
        const val MODEL_ID = "pisa-sr"
        const val NATIVE_SCALE = 4
        const val BYTES_PER_PIXEL = 4
        const val MIN_MONOLITHIC_SOURCE_SIDE = 128
        const val MAX_MONOLITHIC_OUTPUT_PIXELS = 1024L * 1024L
    }
}
