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
    fun everyUserEnqueueReplacesStaleUniqueWork() {
        assertEquals(
            ExistingWorkPolicy.REPLACE,
            ModelDownloadManager.workPolicyFor(replaceExisting = false),
        )
        assertEquals(
            ExistingWorkPolicy.REPLACE,
            ModelDownloadManager.workPolicyFor(replaceExisting = true),
        )
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsPlaceholderArtifactsBeforeEnqueue() {
        ModelDownloadManager.validateForDownload(model())
    }

    @Test
    fun serializesEveryArtifactIntoWorkerData() {
        val model = model(
            artifacts = listOf(
                artifact("model.param", 10_000L, "a"),
                artifact("model.bin", 33_400_000L, "b"),
            ),
        )

        val data = ModelDownloadManager.inputDataFor(model)

        assertEquals(model.id, data.getString(ModelDownloadWorker.KEY_MODEL_ID))
        assertEquals(model.version, data.getString(ModelDownloadWorker.KEY_VERSION))
        assertEquals(2, data.getInt(ModelDownloadWorker.KEY_ARTIFACT_COUNT, -1))
        assertEquals("model.param", data.getString(ModelDownloadWorker.artifactFileNameKey(0)))
        assertEquals("https://example.invalid/model.param", data.getString(ModelDownloadWorker.artifactUrlKey(0)))
        assertEquals(10_000L, data.getLong(ModelDownloadWorker.artifactSizeKey(0), -1L))
        assertEquals("a".repeat(64), data.getString(ModelDownloadWorker.artifactSha256Key(0)))
        assertEquals("model.bin", data.getString(ModelDownloadWorker.artifactFileNameKey(1)))
        assertEquals(33_400_000L, data.getLong(ModelDownloadWorker.artifactSizeKey(1), -1L))
    }

    private fun model(
        artifacts: List<ModelArtifactDescriptor> = listOf(artifact("model.mnn", 2_000_000L, "b")),
    ) = ModelDescriptor(
        id = "pisa-sr",
        displayName = "Kira Balance",
        mode = EnhancementMode.BALANCED,
        version = "candidate-1",
        backend = ModelBackend.MNN,
        artifacts = artifacts,
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

    private fun artifact(fileName: String, size: Long, shaCharacter: String) = ModelArtifactDescriptor(
        fileName = fileName,
        downloadUrl = "https://example.invalid/$fileName",
        fileSizeBytes = size,
        sha256 = shaCharacter.repeat(64),
    )
}
