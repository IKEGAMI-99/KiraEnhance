package com.ikegami99.kiraenhance.inference.ncnn

import com.ikegami99.kiraenhance.model.EnhancementMode
import com.ikegami99.kiraenhance.model.ModelArtifactDescriptor
import com.ikegami99.kiraenhance.model.ModelBackend
import com.ikegami99.kiraenhance.model.ModelDescriptor
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class UltraSharpModelRequestFactoryTest {
    @Test
    fun `creates native 4x ncnn request from installed UltraSharp descriptor`() {
        val request = UltraSharpModelRequestFactory.create(
            model = model(),
            artifactPath = { "/models/ultrasharp/$it" },
        )

        assertEquals("ultrasharp", request.modelId)
        assertEquals(4, request.capabilities.nativeScale)
        assertEquals("data", request.capabilities.inputBlobName)
        assertEquals("output", request.capabilities.outputBlobName)
        assertEquals(10, request.capabilities.prePadding)
        assertTrue(request.capabilities.supportsGpu)
        assertEquals("/models/ultrasharp/model.param", request.artifact("model.param")?.absolutePath)
        assertEquals("/models/ultrasharp/model.bin", request.artifact("model.bin")?.absolutePath)
    }

    @Test(expected = IllegalArgumentException::class)
    fun `rejects non ncnn UltraSharp descriptor`() {
        UltraSharpModelRequestFactory.create(
            model = model(backend = ModelBackend.MNN),
            artifactPath = { "/models/$it" },
        )
    }

    private fun model(backend: ModelBackend = ModelBackend.NCNN) = ModelDescriptor(
        id = "ultrasharp",
        displayName = "4x-UltraSharp",
        mode = EnhancementMode.ULTRASHARP,
        version = "4x-fp16-test",
        backend = backend,
        artifacts = listOf(
            ModelArtifactDescriptor("model.param", "https://example.test/model.param", 1, "a".repeat(64)),
            ModelArtifactDescriptor("model.bin", "https://example.test/model.bin", 2, "b".repeat(64)),
        ),
        supportedScales = listOf(4),
        minAppVersion = "0.1.0",
        estimatedRamMb = 1024,
        licenseName = "test",
        licenseUrl = "https://example.test/license",
        description = "test",
        community = true,
    )
}
