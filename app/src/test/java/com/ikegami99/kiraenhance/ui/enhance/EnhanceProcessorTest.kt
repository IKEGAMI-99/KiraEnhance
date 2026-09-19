package com.ikegami99.kiraenhance.ui.enhance

import com.ikegami99.kiraenhance.inference.EngineCapabilities
import com.ikegami99.kiraenhance.inference.ModelArtifactFile
import com.ikegami99.kiraenhance.inference.ModelLoadRequest
import com.ikegami99.kiraenhance.inference.ModelLoadResult
import com.ikegami99.kiraenhance.inference.ModelRequestFactory
import com.ikegami99.kiraenhance.inference.PixelFormat
import com.ikegami99.kiraenhance.inference.UpscaleEngine
import com.ikegami99.kiraenhance.inference.UpscaleInput
import com.ikegami99.kiraenhance.inference.UpscaleProgress
import com.ikegami99.kiraenhance.inference.UpscaleResult
import com.ikegami99.kiraenhance.inference.UpscaleSettings
import com.ikegami99.kiraenhance.model.EnhancementMode
import com.ikegami99.kiraenhance.model.ModelArtifactDescriptor
import com.ikegami99.kiraenhance.model.ModelBackend
import com.ikegami99.kiraenhance.model.ModelDescriptor
import org.junit.Assert.assertEquals
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test

class EnhanceProcessorTest {
    @Test
    fun `prepares execution from injected binding without backend assumptions`() {
        val engine = FakeUpscaleEngine()
        var factoryCalled = false
        val factory = ModelRequestFactory { model, artifactPath ->
            factoryCalled = true
            ModelLoadRequest(
                modelId = model.id,
                version = model.version,
                artifacts = listOf(ModelArtifactFile("weights.bin", artifactPath("weights.bin"))),
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
        val binding = EnhanceEngineBinding(
            engine = engine,
            requestFactory = factory,
            outputScale = 4,
            saveModeName = "Test",
        )

        val execution = EnhanceProcessor().prepareExecution(
            model = model(),
            binding = binding,
            artifactPath = { "/models/$it" },
        )

        assertTrue(factoryCalled)
        assertSame(engine, execution.engine)
        assertEquals("test-model", execution.request.modelId)
        assertEquals("/models/weights.bin", execution.request.artifact("weights.bin")?.absolutePath)
        assertEquals(4, execution.settings.outputScale)
    }

    private fun model() = ModelDescriptor(
        id = "test-model",
        displayName = "Test Model",
        mode = EnhancementMode.BALANCED,
        version = "1",
        backend = ModelBackend.MNN,
        artifacts = listOf(
            ModelArtifactDescriptor(
                fileName = "weights.bin",
                downloadUrl = "https://example.test/weights.bin",
                fileSizeBytes = 1,
                sha256 = "a".repeat(64),
            ),
        ),
        supportedScales = listOf(4),
        minAppVersion = "0.1.0",
        estimatedRamMb = 1024,
        licenseName = "test",
        licenseUrl = "https://example.test/license",
        description = "test",
        community = false,
    )

    private class FakeUpscaleEngine : UpscaleEngine {
        override fun load(request: ModelLoadRequest): ModelLoadResult = error("not used")
        override fun isLoaded(): Boolean = false
        override fun progress(): UpscaleProgress = UpscaleProgress()
        override fun upscale(input: UpscaleInput, settings: UpscaleSettings): UpscaleResult = error("not used")
        override fun cancel() = Unit
        override fun unload() = Unit
    }
}
