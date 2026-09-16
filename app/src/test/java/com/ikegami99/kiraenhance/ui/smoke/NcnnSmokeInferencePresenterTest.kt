package com.ikegami99.kiraenhance.ui.smoke

import com.ikegami99.kiraenhance.inference.ModelLoadRequest
import com.ikegami99.kiraenhance.inference.ModelLoadResult
import com.ikegami99.kiraenhance.inference.PixelFormat
import com.ikegami99.kiraenhance.inference.UpscaleEngine
import com.ikegami99.kiraenhance.inference.UpscaleInput
import com.ikegami99.kiraenhance.inference.UpscaleOutput
import com.ikegami99.kiraenhance.inference.UpscaleProgress
import com.ikegami99.kiraenhance.inference.UpscaleResult
import com.ikegami99.kiraenhance.inference.UpscaleSettings
import com.ikegami99.kiraenhance.inference.smoke.NcnnSmokeImage
import com.ikegami99.kiraenhance.inference.smoke.NcnnSmokeInferenceRunner
import com.ikegami99.kiraenhance.model.EnhancementMode
import com.ikegami99.kiraenhance.model.ModelArtifactDescriptor
import com.ikegami99.kiraenhance.model.ModelBackend
import com.ikegami99.kiraenhance.model.ModelDescriptor
import java.nio.ByteBuffer
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class NcnnSmokeInferencePresenterTest {
    @Test
    fun successfulInferenceBuildsCompletedPresentation() {
        val engine = FakeEngine()
        val times = ArrayDeque(listOf(1_000L, 1_250L))
        val presenter = NcnnSmokeInferencePresenter(
            runner = NcnnSmokeInferenceRunner { engine },
            nowMs = { times.removeFirst() },
        )

        val result = presenter.run(
            model = ultraSharpModel(),
            artifactPath = { fileName -> "/models/$fileName" },
            image = NcnnSmokeImage(
                width = 1,
                height = 1,
                argbPixels = intArrayOf(0xFF102030.toInt()),
            ),
        )

        assertEquals("4x処理が完了しました", result.state.status)
        assertTrue(result.state.modelInstalled)
        assertTrue(result.state.imageSelected)
        assertEquals("1 × 1", result.state.inputDescription)
        assertEquals("4 × 4", result.state.outputDescription)
        assertEquals("Vulkan GPU", result.state.runtimeDetails)
        assertEquals(250L, result.state.elapsedMs)
        assertNotNull(result.output)
        assertEquals(4, result.output?.width)
        assertEquals(4, result.output?.height)
    }

    private class FakeEngine : UpscaleEngine {
        override fun load(request: ModelLoadRequest): ModelLoadResult =
            ModelLoadResult.Loaded(gpuEnabled = true)

        override fun isLoaded(): Boolean = true

        override fun progress(): UpscaleProgress = UpscaleProgress()

        override fun upscale(input: UpscaleInput, settings: UpscaleSettings): UpscaleResult {
            val width = input.width * 4
            val height = input.height * 4
            val rowStride = width * 4
            val pixels = ByteBuffer.allocateDirect(rowStride * height)
            repeat(width * height) {
                pixels.put(0x10)
                pixels.put(0x20)
                pixels.put(0x30)
                pixels.put(0xFF.toByte())
            }
            pixels.flip()
            return UpscaleResult.Success(
                output = UpscaleOutput(
                    width = width,
                    height = height,
                    rowStrideBytes = rowStride,
                    pixelFormat = PixelFormat.RGBA_8888,
                    pixels = pixels,
                ),
                usedGpu = true,
            )
        }

        override fun cancel() = Unit

        override fun unload() = Unit
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
