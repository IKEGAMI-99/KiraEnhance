package com.ikegami99.kiraenhance.inference.mnn

import com.ikegami99.kiraenhance.inference.EngineCapabilities
import com.ikegami99.kiraenhance.inference.EngineErrorCode
import com.ikegami99.kiraenhance.inference.ModelArtifactFile
import com.ikegami99.kiraenhance.inference.ModelLoadRequest
import com.ikegami99.kiraenhance.inference.ModelLoadResult
import com.ikegami99.kiraenhance.inference.PixelFormat
import com.ikegami99.kiraenhance.inference.UpscaleInput
import com.ikegami99.kiraenhance.inference.UpscaleProgress
import com.ikegami99.kiraenhance.inference.UpscaleResult
import com.ikegami99.kiraenhance.inference.UpscaleSettings
import java.io.File
import java.nio.ByteBuffer
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class MnnPisaUpscaleEngineTest {
    @get:Rule
    val temporaryFolder = TemporaryFolder()

    @Test
    fun `loads and unloads complete PiSA package through native boundary`() {
        val native = FakeMnnPisaNativeApi()
        val engine = MnnPisaUpscaleEngine(native)

        val result = engine.load(request())

        assertEquals(ModelLoadResult.Loaded(gpuEnabled = false), result)
        assertTrue(engine.isLoaded())
        assertEquals(1, native.loadCalls)

        engine.cancel()
        assertEquals(1, native.cancelCalls)

        engine.unload()
        assertEquals(1, native.unloadCalls)
        assertFalse(engine.isLoaded())
    }

    @Test
    fun `captures native session and graph diagnostics after load`() {
        val sessionInfo = MnnPisaSessionInfo(
            backend = MnnPisaBackend.OPENCL,
            gpuEnabled = true,
        )
        val tensorInfo = listOf(
            MnnPisaTensorInfo(
                graph = MnnPisaGraph.UNET,
                role = MnnTensorRole.INPUT,
                name = "latent",
                shape = listOf(1, 4, 64, 64),
                typeCode = 2,
                typeBits = 16,
                typeLanes = 1,
                dimensionType = 1,
            ),
        )
        val native = FakeMnnPisaNativeApi(
            loadResult = MnnPisaNativeLoadResult(
                handle = 7L,
                errorCode = MnnPisaNativeError.NONE,
                gpuEnabled = true,
            ),
            sessionInfo = sessionInfo,
            graphInfo = tensorInfo,
        )
        val engine = MnnPisaUpscaleEngine(native)

        assertTrue(engine.load(request()) is ModelLoadResult.Loaded)
        assertEquals(
            MnnPisaDiagnosticsSnapshot(
                modelVersion = "converted-test",
                sessionInfo = sessionInfo,
                tensorInfo = tensorInfo,
            ),
            engine.diagnostics(),
        )
        assertEquals(1, native.sessionInfoCalls)
        assertEquals(1, native.graphInfoCalls)

        engine.unload()
        assertNull(engine.diagnostics())
    }

    @Test
    fun `diagnostic probe failure does not fail model load`() {
        val native = FakeMnnPisaNativeApi(
            throwOnDiagnostics = true,
        )
        val engine = MnnPisaUpscaleEngine(native)

        assertTrue(engine.load(request()) is ModelLoadResult.Loaded)
        assertEquals(
            MnnPisaDiagnosticsSnapshot(
                modelVersion = "converted-test",
                sessionInfo = null,
                tensorInfo = null,
            ),
            engine.diagnostics(),
        )
    }

    @Test
    fun `publishes diagnostics snapshot once after successful load`() {
        val sessionInfo = MnnPisaSessionInfo(
            backend = MnnPisaBackend.CPU,
            gpuEnabled = false,
        )
        val tensorInfo = listOf(
            MnnPisaTensorInfo(
                graph = MnnPisaGraph.VAE_DECODER,
                role = MnnTensorRole.OUTPUT,
                name = "image",
                shape = listOf(1, 3, 512, 512),
                typeCode = 2,
                typeBits = 16,
                typeLanes = 1,
                dimensionType = 1,
            ),
        )
        val native = FakeMnnPisaNativeApi(
            sessionInfo = sessionInfo,
            graphInfo = tensorInfo,
        )
        val published = mutableListOf<MnnPisaDiagnosticsSnapshot>()
        val engine = MnnPisaUpscaleEngine(
            nativeApi = native,
            onDiagnostics = published::add,
        )

        assertTrue(engine.load(request()) is ModelLoadResult.Loaded)

        assertEquals(
            listOf(
                MnnPisaDiagnosticsSnapshot(
                    modelVersion = "converted-test",
                    sessionInfo = sessionInfo,
                    tensorInfo = tensorInfo,
                ),
            ),
            published,
        )
    }

    @Test
    fun `diagnostics callback failure does not fail model load`() {
        val engine = MnnPisaUpscaleEngine(
            nativeApi = FakeMnnPisaNativeApi(),
            onDiagnostics = { error("logger unavailable") },
        )

        assertTrue(engine.load(request()) is ModelLoadResult.Loaded)
        assertTrue(engine.isLoaded())
    }

    @Test
    fun `prepares loaded graph through native boundary`() {
        val expected = MnnPisaNativePrepareResult(
            errorCode = MnnPisaNativeError.NONE,
            imageWidth = 768,
            imageHeight = 512,
            latentWidth = 96,
            latentHeight = 64,
        )
        val native = FakeMnnPisaNativeApi(
            prepareResult = expected,
        )
        val engine = MnnPisaUpscaleEngine(native)
        assertTrue(engine.load(request()) is ModelLoadResult.Loaded)

        val result = engine.prepareGraph(
            imageWidth = 768,
            imageHeight = 512,
        )

        assertEquals(expected, result)
        assertEquals(1, native.prepareCalls)
    }

    @Test
    fun `graph preparation rejects unloaded engine`() {
        val native = FakeMnnPisaNativeApi()
        val engine = MnnPisaUpscaleEngine(native)

        val result = engine.prepareGraph(
            imageWidth = 512,
            imageHeight = 512,
        )

        assertEquals(MnnPisaNativeError.LOAD_FAILED, result.errorCode)
        assertEquals(0, native.prepareCalls)
    }

    @Test
    fun `runs full graph smoke through native boundary`() {
        val expected = MnnPisaNativeSmokeResult(
            errorCode = MnnPisaNativeError.NONE,
            completedStages = 3,
            outputFinite = true,
        )
        val native = FakeMnnPisaNativeApi(
            smokeResult = expected,
        )
        val engine = MnnPisaUpscaleEngine(native)
        assertTrue(engine.load(request()) is ModelLoadResult.Loaded)

        val result = engine.smokeGraph(
            imageWidth = 512,
            imageHeight = 512,
        )

        assertEquals(expected, result)
        assertEquals(1, native.smokeCalls)
    }

    @Test
    fun `smoke rejects unloaded engine`() {
        val native = FakeMnnPisaNativeApi()
        val engine = MnnPisaUpscaleEngine(native)

        val result = engine.smokeGraph(
            imageWidth = 512,
            imageHeight = 512,
        )

        assertEquals(MnnPisaNativeError.LOAD_FAILED, result.errorCode)
        assertEquals(0, native.smokeCalls)
    }

    @Test
    fun `bounded real inference returns native output metadata`() {
        val native = FakeMnnPisaNativeApi(
            inferenceResult = MnnPisaNativeInferenceResult(
                errorCode = MnnPisaNativeError.NONE,
                outputWidth = 512,
                outputHeight = 512,
                outputRowStrideBytes = 2048,
                gpuUsed = true,
            ),
        )
        val engine = MnnPisaUpscaleEngine(native)
        assertTrue(engine.load(request()) is ModelLoadResult.Loaded)

        val result = engine.upscale(
            input = UpscaleInput(
                width = 128,
                height = 128,
                rowStrideBytes = 512,
                pixelFormat = PixelFormat.RGBA_8888,
                pixels = ByteBuffer.allocateDirect(128 * 128 * 4),
            ),
            settings = UpscaleSettings(outputScale = 4),
        )

        assertTrue(result is UpscaleResult.Success)
        val success = result as UpscaleResult.Success
        assertEquals(512, success.output.width)
        assertEquals(512, success.output.height)
        assertEquals(2048, success.output.rowStrideBytes)
        assertTrue(success.usedGpu)
        assertEquals(1, native.inferCalls)
        assertEquals(
            UpscaleProgress(completedTiles = 1, totalTiles = 1),
            engine.progress(),
        )
    }

    @Test
    fun `odd sized PiSA input keeps exact four-x app output`() {
        val native = FakeMnnPisaNativeApi(
            inferenceResult = MnnPisaNativeInferenceResult(
                errorCode = MnnPisaNativeError.NONE,
                outputWidth = 516,
                outputHeight = 512,
                outputRowStrideBytes = 2064,
                gpuUsed = true,
            ),
        )
        val engine = MnnPisaUpscaleEngine(native)
        assertTrue(engine.load(request()) is ModelLoadResult.Loaded)

        val result = engine.upscale(
            input = UpscaleInput(
                width = 129,
                height = 128,
                rowStrideBytes = 516,
                pixelFormat = PixelFormat.RGBA_8888,
                pixels = ByteBuffer.allocateDirect(129 * 128 * 4),
            ),
            settings = UpscaleSettings(outputScale = 4),
        )

        assertTrue(result is UpscaleResult.Success)
        val success = result as UpscaleResult.Success
        assertEquals(516, success.output.width)
        assertEquals(512, success.output.height)
        assertEquals(2064, success.output.rowStrideBytes)
        assertEquals(1, native.inferCalls)
    }

    @Test
    fun `large PiSA input is rejected before native allocation path`() {
        val native = FakeMnnPisaNativeApi()
        val engine = MnnPisaUpscaleEngine(native)
        assertTrue(engine.load(request()) is ModelLoadResult.Loaded)

        val result = engine.upscale(
            input = UpscaleInput(
                width = 512,
                height = 512,
                rowStrideBytes = 2048,
                pixelFormat = PixelFormat.RGBA_8888,
                pixels = ByteBuffer.allocateDirect(512 * 512 * 4),
            ),
            settings = UpscaleSettings(outputScale = 4),
        )

        assertTrue(result is UpscaleResult.Failed)
        assertEquals(
            EngineErrorCode.INFERENCE_FAILED,
            (result as UpscaleResult.Failed).error.code,
        )
        assertTrue(result.error.message.contains("tiled inference"))
        assertEquals(0, native.inferCalls)
    }

    @Test
    fun `not implemented inference becomes typed failure instead of fake output`() {
        val native = FakeMnnPisaNativeApi(
            inferenceResult = MnnPisaNativeInferenceResult(
                errorCode = MnnPisaNativeError.NOT_IMPLEMENTED,
                outputWidth = 0,
                outputHeight = 0,
                outputRowStrideBytes = 0,
                gpuUsed = false,
            ),
        )
        val engine = MnnPisaUpscaleEngine(native)
        assertTrue(engine.load(request()) is ModelLoadResult.Loaded)

        val result = engine.upscale(
            input = UpscaleInput(
                width = 128,
                height = 128,
                rowStrideBytes = 512,
                pixelFormat = PixelFormat.RGBA_8888,
                pixels = ByteBuffer.allocateDirect(128 * 128 * 4),
            ),
            settings = UpscaleSettings(outputScale = 4),
        )

        assertTrue(result is UpscaleResult.Failed)
        assertEquals(
            EngineErrorCode.INFERENCE_FAILED,
            (result as UpscaleResult.Failed).error.code,
        )
    }

    @Test
    fun `default native bridge maps unavailable JNI to internal failure on host JVM`() {
        val result = MnnPisaNativeBridge.infer(
            handle = 1L,
            inputPixels = ByteBuffer.allocateDirect(4),
            width = 1,
            height = 1,
            inputRowStrideBytes = 4,
            outputPixels = ByteBuffer.allocateDirect(64),
            outputCapacityBytes = 64,
        )

        assertEquals(MnnPisaNativeError.INTERNAL, result.errorCode)
        assertEquals(0, result.outputWidth)
        assertEquals(0, result.outputHeight)
    }

    private fun request(): ModelLoadRequest {
        val directory = temporaryFolder.newFolder("pisa")
        val artifacts = REQUIRED_FILES.map { fileName ->
            val file = File(directory, fileName).apply { writeBytes(byteArrayOf(1)) }
            ModelArtifactFile(fileName, file.absolutePath)
        }
        return ModelLoadRequest(
            modelId = "pisa-sr",
            version = "converted-test",
            artifacts = artifacts,
            capabilities = EngineCapabilities(
                nativeScale = 4,
                pixelFormat = PixelFormat.RGBA_8888,
                inputBlobName = "image",
                outputBlobName = "output",
                prePadding = 0,
                supportsGpu = true,
            ),
        )
    }

    private class FakeMnnPisaNativeApi(
        private val loadResult: MnnPisaNativeLoadResult = MnnPisaNativeLoadResult(
            handle = 7L,
            errorCode = MnnPisaNativeError.NONE,
            gpuEnabled = false,
        ),
        private val inferenceResult: MnnPisaNativeInferenceResult = MnnPisaNativeInferenceResult(
            errorCode = MnnPisaNativeError.NOT_IMPLEMENTED,
            outputWidth = 0,
            outputHeight = 0,
            outputRowStrideBytes = 0,
            gpuUsed = false,
        ),
        private val prepareResult: MnnPisaNativePrepareResult = MnnPisaNativePrepareResult(
            errorCode = MnnPisaNativeError.NOT_IMPLEMENTED,
            imageWidth = 0,
            imageHeight = 0,
            latentWidth = 0,
            latentHeight = 0,
        ),
        private val smokeResult: MnnPisaNativeSmokeResult = MnnPisaNativeSmokeResult(
            errorCode = MnnPisaNativeError.NOT_IMPLEMENTED,
            completedStages = 0,
            outputFinite = false,
        ),
        private val sessionInfo: MnnPisaSessionInfo? = null,
        private val graphInfo: List<MnnPisaTensorInfo>? = null,
        private val throwOnDiagnostics: Boolean = false,
    ) : MnnPisaNativeApi {
        var loadCalls = 0
        var unloadCalls = 0
        var cancelCalls = 0
        var sessionInfoCalls = 0
        var graphInfoCalls = 0
        var prepareCalls = 0
        var smokeCalls = 0
        var inferCalls = 0

        override fun loadModel(
            vaeEncoderPath: String,
            unetPath: String,
            vaeDecoderPath: String,
            emptyPromptPath: String,
            preferGpu: Boolean,
        ): MnnPisaNativeLoadResult {
            loadCalls += 1
            return loadResult
        }

        override fun sessionInfo(handle: Long): MnnPisaSessionInfo? {
            sessionInfoCalls += 1
            if (throwOnDiagnostics) {
                error("session diagnostics unavailable")
            }
            return sessionInfo
        }

        override fun graphInfo(handle: Long): List<MnnPisaTensorInfo>? {
            graphInfoCalls += 1
            if (throwOnDiagnostics) {
                error("graph diagnostics unavailable")
            }
            return graphInfo
        }

        override fun prepareGraph(
            handle: Long,
            imageWidth: Int,
            imageHeight: Int,
        ): MnnPisaNativePrepareResult {
            prepareCalls += 1
            return prepareResult
        }

        override fun smokeGraph(
            handle: Long,
            imageWidth: Int,
            imageHeight: Int,
        ): MnnPisaNativeSmokeResult {
            smokeCalls += 1
            return smokeResult
        }

        override fun infer(
            handle: Long,
            inputPixels: ByteBuffer,
            width: Int,
            height: Int,
            inputRowStrideBytes: Int,
            outputPixels: ByteBuffer,
            outputCapacityBytes: Long,
        ): MnnPisaNativeInferenceResult {
            inferCalls += 1
            return inferenceResult
        }

        override fun unload(handle: Long) {
            unloadCalls += 1
        }

        override fun cancel(handle: Long) {
            cancelCalls += 1
        }
    }

    private companion object {
        val REQUIRED_FILES = listOf(
            "vae_encoder.mnn",
            "unet_default.mnn",
            "vae_decoder.mnn",
            "empty_prompt.fp16",
        )
    }
}
