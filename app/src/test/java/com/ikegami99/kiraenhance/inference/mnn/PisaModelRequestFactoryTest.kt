package com.ikegami99.kiraenhance.inference.mnn

import com.ikegami99.kiraenhance.model.EnhancementMode
import com.ikegami99.kiraenhance.model.ModelArtifactDescriptor
import com.ikegami99.kiraenhance.model.ModelBackend
import com.ikegami99.kiraenhance.model.ModelDescriptor
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class PisaModelRequestFactoryTest {
    @Test
    fun `creates native 4x MNN request from PiSA package`() {
        val request = PisaModelRequestFactory.create(
            model = model(),
            artifactPath = { "/models/pisa/$it" },
        )

        assertEquals("pisa-sr", request.modelId)
        assertEquals(4, request.capabilities.nativeScale)
        assertEquals("image", request.capabilities.inputBlobName)
        assertEquals("output", request.capabilities.outputBlobName)
        assertEquals(0, request.capabilities.prePadding)
        assertTrue(request.capabilities.supportsGpu)
        assertEquals("/models/pisa/vae_encoder.mnn", request.artifact("vae_encoder.mnn")?.absolutePath)
        assertEquals("/models/pisa/unet_default.mnn", request.artifact("unet_default.mnn")?.absolutePath)
        assertEquals("/models/pisa/vae_decoder.mnn", request.artifact("vae_decoder.mnn")?.absolutePath)
        assertEquals("/models/pisa/empty_prompt.fp16", request.artifact("empty_prompt.fp16")?.absolutePath)
        assertEquals("/models/pisa/vae_segments.pack", request.artifact("vae_segments.pack")?.absolutePath)
    }

    @Test(expected = IllegalArgumentException::class)
    fun `rejects incomplete PiSA artifact package`() {
        PisaModelRequestFactory.create(
            model = model().copy(artifacts = model().artifacts.dropLast(1)),
            artifactPath = { "/models/pisa/$it" },
        )
    }

    private fun model() = ModelDescriptor(
        id = "pisa-sr",
        displayName = "PiSA-SR",
        mode = EnhancementMode.BALANCED,
        version = "converted-test",
        backend = ModelBackend.MNN,
        artifacts = listOf(
            artifact("vae_encoder.mnn", "a"),
            artifact("unet_default.mnn", "b"),
            artifact("vae_decoder.mnn", "c"),
            artifact("empty_prompt.fp16", "d"),
            artifact("vae_segments.pack", "e"),
        ),
        supportedScales = listOf(4),
        minAppVersion = "0.1.0",
        estimatedRamMb = 8000,
        licenseName = "test",
        licenseUrl = "https://example.test/license",
        description = "test",
        community = false,
    )

    private fun artifact(fileName: String, hashChar: String) = ModelArtifactDescriptor(
        fileName = fileName,
        downloadUrl = "https://example.test/$fileName",
        fileSizeBytes = 1,
        sha256 = hashChar.repeat(64),
    )
}
