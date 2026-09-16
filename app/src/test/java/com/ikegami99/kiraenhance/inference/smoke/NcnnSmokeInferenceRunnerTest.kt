package com.ikegami99.kiraenhance.inference.smoke

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
import com.ikegami99.kiraenhance.model.EnhancementMode
import com.ikegami99.kiraenhance.model.ModelArtifactDescriptor
import com.ikegami99.kiraenhance.model.ModelBackend
import com.ikegami99.kiraenhance.model.ModelDescriptor
import java.nio.ByteBuffer
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class NcnnSmokeInferenceRunnerTest {
    @Test
    fun runLoadsUltraSharpRunsDeterministic4xTileAndDecodesPixels() {
        val engine = FakeEngine(
            loadResult = ModelLoadResult.Loaded(gpuEnabled = true),
            upscaleResult = UpscaleResult.Success(
                output = rgbaOutput(
                    width = 4,
                    height = 4,
                    rgba = intArrayOf(0x11, 0x22, 0x33, 0x44),
                ),
                usedGpu = true,
            ),
        )
        val runner = NcnnSmokeInferenceRunner { engine }

        val result = runner.run(
            model = ultraSharpModel(),
            artifactPath = { fileName -> "/models/ultrasharp/v1/$fileName" },
            image = NcnnSmokeImage(
                width = 1,
                height = 1,
                argbPixels = intArrayOf(0x44112233),
            ),
        )

        assertTrue(result is NcnnSmokeRunResult.Success)
        result as NcnnSmokeRunResult.Success
        assertEquals(4, result.width)
        assertEquals(4, result.height)
        assertEquals(16, result.argbPixels.size)
        assertTrue(result.argbPixels.all { it == 0x44112233 })
        assertTrue(result.usedGpu)

        assertEquals("ultrasharp", engine.loadedRequest?.modelId)
        assertEquals("/models/ultrasharp/v1/model.param", engine.loadedRequest?.artifact("model.param")?.absolutePath)
        assertEquals("/models/ultrasharp/v1/model.bin", engine.loadedRequest?.artifact("model.bin")?.absolutePath)
        assertEquals(4, engine.lastSettings?.outputScale)
        assertTrue(engine.lastSettings?.useGpu == true)
        assertEquals(128, engine.lastSettings?.tileSize)
        assertTrue(engine.closed)
    }

    @Test
    fun runReturnsLoadFailureWithoutStartingInferenceAndStillClosesEngine() {
        val expected = EngineError(
            code = EngineErrorCode.MODEL_LOAD_FAILED,
            message = "model rejected",
        )
        val engine = FakeEngine(
            loadResult = ModelLoadResult.Failed(expected),
            upscaleResult = UpscaleResult.Failed(
                EngineError(
                    code = EngineErrorCode.INFERENCE_FAILED,
                    message = "unused",
                ),
            ),
        )
        val runner = NcnnSmokeInferenceRunner { engine }

        val result = runner.run(
            model = ultraSharpModel(),
            artifactPath = { fileName -> "/models/$fileName" },
            image = NcnnSmokeImage(
                width = 1,
                height = 1,
                argbPixels = intArrayOf(0xFF000000.toInt()),
            ),
        )

        assertTrue(result is NcnnSmokeRunResult.Failed)
        result as NcnnSmokeRunResult.Failed
        assertEquals(expected, result.error)
        assertFalse(engine.upscaleCalled)
        assertTrue(engine.closed)
    }

    private class FakeEngine(
        private val loadResult: ModelLoadResult,
        private val upscaleResult: UpscaleResult,
    ) : UpscaleEngine {
        var loadedRequest: ModelLoadRequest? = null
        var lastSettings: UpscaleSettings? = null
        var upscaleCalled: Boolean = false
        var closed: Boolean = false

        override fun load(request: ModelLoadRequest): ModelLoadResult {
            loadedRequest = request
            return loadResult
        }

        override fun isLoaded(): Boolean = loadResult is ModelLoadResult.Loaded

        override fun progress(): UpscaleProgress = UpscaleProgress()

        override fun upscale(input: UpscaleInput, settings: UpscaleSettings): UpscaleResult {
            upscaleCalled = true
            lastSettings = settings
            return upscaleResult
        }

        override fun cancel() = Unit

        override fun unload() {
            closed = true
        }
    }

    private fun rgbaOutput(
        width: Int,
        height: Int,
        rgba: IntArray,
    ): UpscaleOutput {
        val rowStride = width * 4
        val pixels = ByteBuffer.allocateDirect(rowStride * height)
        repeat(width * height) {
            rgba.forEach { channel -> pixels.put(channel.toByte()) }
        }
        pixels.flip()
        return UpscaleOutput(
            width = width,
            height = height,
            rowStrideBytes = rowStride,
            pixelFormat = PixelFormat.RGBA_8888,
            pixels = pixels,
        )
    }

    private fun ultraSharpModel(): ModelDescriptor = ModelDescriptor(
        id = "ultrasharp",
        displayName = "4x-UltraSharp",
        mode = EnhancementMode.ULTRASHARP,
        version = "v1",
        backend = ModelBackend.NCNN,
        artifacts = listOf(
            ModelArtifactDescriptor(
                fileName = "model.param",
                downloadUrl = "https://example.com/model.param",
                fileSizeBytes = 1L,
                sha256 = "a".repeat(64),
            ),
            ModelArtifactDescriptor(
                fileName = "model.bin",
                downloadUrl = "https://example.com/model.bin",
                fileSizeBytes = 1L,
                sha256 = "b".repeat(64),
            ),
        ),
        supportedScales = listOf(4),
        minAppVersion = "0.1.0-alpha01",
        estimatedRamMb = 2048,
        licenseName = "CC BY-NC-SA 4.0",
        licenseUrl = "https://creativecommons.org/licenses/by-nc-sa/4.0/",
        description = "test",
        community = true,
    )
}
