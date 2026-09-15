package com.ikegami99.kiraenhance.download

import androidx.work.ExistingWorkPolicy
import androidx.work.NetworkType
import com.ikegami99.kiraenhance.model.ControlType
import com.ikegami99.kiraenhance.model.EnhancementMode
import com.ikegami99.kiraenhance.model.ModelArtifactDescriptor
import com.ikegami99.kiraenhance.model.ModelBackend
import com.ikegami99.kiraenhance.model.ModelControlDescriptor
import com.ikegami99.kiraenhance.model.ModelDescriptor
import org.junit.Assert.assertEquals
import org.junit.Test

class ModelDownloadManagerTest {
    @Test
    fun wifiOnlyRequiresUnmeteredNetwork() {
        assertEquals(NetworkType.UNMETERED, ModelDownloadManager.networkTypeFor(wifiOnly = true))
        assertEquals(NetworkType.CONNECTED, ModelDownloadManager.networkTypeFor(wifiOnly = false))
    }

    @Test
    fun createsStableUniqueWorkName() {
        assertEquals(
            "model-download-pisa-sr-candidate-1",
            ModelDownloadManager.workName(model()),
        )
    }

    @Test
    fun reinstallReplacesExistingUniqueWork() {
        assertEquals(
            ExistingWorkPolicy.KEEP,
            ModelDownloadManager.workPolicyFor(replaceExisting = false),
        )
        assertEquals(
            ExistingWorkPolicy.REPLACE,
            ModelDownloadManager.workPolicyFor(replaceExisting = true),
        )
    }

    private fun model() = ModelDescriptor(
        id = "pisa-sr",
        displayName = "Kira Balance",
        mode = EnhancementMode.BALANCED,
        version = "candidate-1",
        backend = ModelBackend.MNN,
        artifacts = listOf(
            ModelArtifactDescriptor(
                fileName = "model.mnn",
                downloadUrl = "https://example.invalid/pisa-sr.mnn",
                fileSizeBytes = 2_000_000,
                sha256 = "b".repeat(64),
            ),
        ),
        supportedScales = listOf(2, 4),
        minAppVersion = "0.1.0-alpha01",
        estimatedRamMb = 8_000,
        licenseName = "Apache-2.0",
        licenseUrl = "https://example.invalid/license",
        description = "test model",
        community = false,
        controls = listOf(
            ModelControlDescriptor(
                id = "fidelity",
                label = "忠実度",
                type = ControlType.SLIDER,
                min = 0.0,
                max = 1.0,
                defaultValue = 0.65,
            ),
        ),
    )
}
