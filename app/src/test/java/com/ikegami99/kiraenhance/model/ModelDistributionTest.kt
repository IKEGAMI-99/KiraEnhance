package com.ikegami99.kiraenhance.model

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ModelDistributionTest {
    @Test
    fun `rejects example invalid placeholder artifacts`() {
        assertFalse(
            model(
                url = "https://example.invalid/pisa/model.mnn",
                sha256 = "ab".repeat(32),
            ).isDownloadableProductionArtifactSet(),
        )
    }

    @Test
    fun `rejects non https artifacts`() {
        assertFalse(
            model(
                url = "http://models.example.com/pisa/model.mnn",
                sha256 = "ab".repeat(32),
            ).isDownloadableProductionArtifactSet(),
        )
    }

    @Test
    fun `rejects zero size artifacts`() {
        assertFalse(
            model(
                url = "https://models.example.com/pisa/model.mnn",
                size = 0,
                sha256 = "ab".repeat(32),
            ).isDownloadableProductionArtifactSet(),
        )
    }

    @Test
    fun `rejects placeholder sha made from one repeated character`() {
        assertFalse(
            model(
                url = "https://models.example.com/pisa/model.mnn",
                sha256 = "a".repeat(64),
            ).isDownloadableProductionArtifactSet(),
        )
    }

    @Test
    fun `accepts https artifact with size and non placeholder sha`() {
        assertTrue(
            model(
                url = "https://models.example.com/pisa/model.mnn",
                sha256 = "0123456789abcdef".repeat(4),
            ).isDownloadableProductionArtifactSet(),
        )
    }

    private fun model(
        url: String,
        size: Long = 1024,
        sha256: String,
    ) = ModelDescriptor(
        id = "pisa-sr",
        displayName = "PiSA-SR",
        mode = EnhancementMode.BALANCED,
        version = "test",
        backend = ModelBackend.MNN,
        artifacts = listOf(
            ModelArtifactDescriptor(
                fileName = "model.mnn",
                downloadUrl = url,
                fileSizeBytes = size,
                sha256 = sha256,
            ),
        ),
        supportedScales = listOf(4),
        minAppVersion = "0.1.0-alpha01",
        estimatedRamMb = 1024,
        licenseName = "test",
        licenseUrl = "https://models.example.com/license",
        description = "test",
    )
}
