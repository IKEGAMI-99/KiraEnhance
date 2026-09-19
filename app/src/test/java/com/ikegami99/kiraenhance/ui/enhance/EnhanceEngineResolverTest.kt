package com.ikegami99.kiraenhance.ui.enhance

import com.ikegami99.kiraenhance.inference.ModelLoadRequest
import com.ikegami99.kiraenhance.inference.ModelLoadResult
import com.ikegami99.kiraenhance.inference.ModelRequestFactory
import com.ikegami99.kiraenhance.inference.UpscaleEngine
import com.ikegami99.kiraenhance.inference.UpscaleInput
import com.ikegami99.kiraenhance.inference.UpscaleProgress
import com.ikegami99.kiraenhance.inference.UpscaleResult
import com.ikegami99.kiraenhance.inference.UpscaleSettings
import com.ikegami99.kiraenhance.inference.ncnn.UltraSharpModelRequestFactory
import com.ikegami99.kiraenhance.model.EnhancementMode
import com.ikegami99.kiraenhance.model.ModelArtifactDescriptor
import com.ikegami99.kiraenhance.model.ModelBackend
import com.ikegami99.kiraenhance.model.ModelDescriptor
import org.junit.Assert.assertEquals
import org.junit.Assert.assertSame
import org.junit.Test

class EnhanceEngineResolverTest {
    private val ncnnEngine = FakeUpscaleEngine()
    private val mnnEngine = FakeUpscaleEngine()
    private val pisaFactory = ModelRequestFactory { _, _ -> error("not invoked by resolver") }
    private val resolver = EnhanceEngineResolver(
        ncnnProvider = { ncnnEngine },
        mnnPisaProvider = { mnnEngine },
        pisaRequestFactory = pisaFactory,
    )

    @Test
    fun `maps UltraSharp ncnn to 4x binding`() {
        val binding = resolver.resolve(
            model(mode = EnhancementMode.ULTRASHARP, backend = ModelBackend.NCNN),
        )

        assertSame(ncnnEngine, binding.engine)
        assertSame(UltraSharpModelRequestFactory, binding.requestFactory)
        assertEquals(4, binding.outputScale)
        assertEquals("UltraSharp", binding.saveModeName)
        assertEquals(
            EnhanceOutputSize(width = 516, height = 512),
            binding.plannedOutputSize(inputWidth = 129, inputHeight = 128),
        )
    }

    @Test
    fun `maps balanced MNN to PiSA binding`() {
        val binding = resolver.resolve(
            model(mode = EnhancementMode.BALANCED, backend = ModelBackend.MNN),
        )

        assertSame(mnnEngine, binding.engine)
        assertSame(pisaFactory, binding.requestFactory)
        assertEquals(4, binding.outputScale)
        assertEquals("PiSA-SR", binding.saveModeName)
        assertEquals(
            EnhanceOutputSize(width = 512, height = 512),
            binding.plannedOutputSize(inputWidth = 129, inputHeight = 128),
        )
        assertEquals(
            EnhanceOutputSize(width = 256, height = 384),
            binding.plannedOutputSize(inputWidth = 64, inputHeight = 96),
        )
    }

    @Test(expected = IllegalArgumentException::class)
    fun `rejects unsupported backend and mode pairing`() {
        resolver.resolve(
            model(mode = EnhancementMode.ULTRASHARP, backend = ModelBackend.MNN),
        )
    }

    private fun model(
        mode: EnhancementMode,
        backend: ModelBackend,
    ) = ModelDescriptor(
        id = if (mode == EnhancementMode.ULTRASHARP) "ultrasharp" else "pisa-sr",
        displayName = if (mode == EnhancementMode.ULTRASHARP) "4x-UltraSharp" else "PiSA-SR",
        mode = mode,
        version = "test",
        backend = backend,
        artifacts = listOf(
            ModelArtifactDescriptor(
                fileName = "model.bin",
                downloadUrl = "https://example.test/model.bin",
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
