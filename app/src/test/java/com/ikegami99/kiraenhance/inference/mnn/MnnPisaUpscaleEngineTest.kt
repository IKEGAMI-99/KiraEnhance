package com.ikegami99.kiraenhance.inference.mnn

import com.ikegami99.kiraenhance.inference.EngineCapabilities
import com.ikegami99.kiraenhance.inference.EngineErrorCode
import com.ikegami99.kiraenhance.inference.ModelArtifactFile
import com.ikegami99.kiraenhance.inference.ModelLoadRequest
import com.ikegami99.kiraenhance.inference.ModelLoadResult
import com.ikegami99.kiraenhance.inference.PixelFormat
import com.ikegami99.kiraenhance.inference.UpscaleInput
import com.ikegami99.kiraenhance.inference.UpscaleResult
import com.ikegami99.kiraenhance.inference.UpscaleSettings
import java.io.File
import java.nio.ByteBuffer
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
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
                width = 1,
                height = 1,
                rowStrideBytes = 4,
                pixelFormat = PixelFormat.RGBA_8888,
                pixels = ByteBuffer.allocateDirect(4),
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
    fun `default native bridge explicitly reports inference not implemented`() {
        val result = MnnPisaNativeBridge.infer(
            handle = 1L,
            inputPixels = ByteBuffer.allocateDirect(4),
            width = 1,
            height = 1,
            inputRowStrideBytes = 4,
            outputPixels = ByteBuffer.allocateDirect(64),
            outputCapacityBytes = 64,
        )

        assertEquals(MnnPisaNativeError.NOT_IMPLEMENTED, result.errorCode)
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
    ) : MnnPisaNativeApi {
        var loadCalls = 0
        var unloadCalls = 0
        var cancelCalls = 0

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

        override fun infer(
            handle: Long,
            inputPixels: ByteBuffer,
            width: Int,
            height: Int,
            inputRowStrideBytes: Int,
            outputPixels: ByteBuffer,
            outputCapacityBytes: Long,
        ): MnnPisaNativeInferenceResult = inferenceResult

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
