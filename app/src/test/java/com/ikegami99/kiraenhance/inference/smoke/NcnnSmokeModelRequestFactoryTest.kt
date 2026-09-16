package com.ikegami99.kiraenhance.inference.smoke

import com.ikegami99.kiraenhance.inference.PixelFormat
import com.ikegami99.kiraenhance.model.EnhancementMode
import com.ikegami99.kiraenhance.model.ModelArtifactDescriptor
import com.ikegami99.kiraenhance.model.ModelBackend
import com.ikegami99.kiraenhance.model.ModelDescriptor
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class NcnnSmokeModelRequestFactoryTest {
    @Test
    fun buildsUltraSharpRequestFromInstalledArtifactPaths() {
        val model = ModelDescriptor(
            id = "ultrasharp",
            displayName = "4x-UltraSharp",
            mode = EnhancementMode.ULTRASHARP,
            version = "4x-fp16-b32be1b",
            backend = ModelBackend.NCNN,
            artifacts = listOf(
                artifact("model.param"),
                artifact("model.bin"),
            ),
            supportedScales = listOf(4),
            minAppVersion = "0.1.0-alpha01",
            estimatedRamMb = 2048,
            licenseName = "CC BY-NC-SA 4.0",
            licenseUrl = "https://creativecommons.org/licenses/by-nc-sa/4.0/",
            description = "test",
            community = true,
        )

        val request = NcnnSmokeModelRequestFactory.create(model) { fileName ->
            "/data/user/0/com.ikegami99.kiraenhance/files/models/ultrasharp/${model.version}/$fileName"
        }

        assertEquals("ultrasharp", request.modelId)
        assertEquals("4x-fp16-b32be1b", request.version)
        assertEquals(
            listOf("model.param", "model.bin"),
            request.artifacts.map { it.fileName },
        )
        assertTrue(request.artifacts.all { it.absolutePath.startsWith("/data/user/0/") })
        assertEquals(4, request.capabilities.nativeScale)
        assertEquals(PixelFormat.RGBA_8888, request.capabilities.pixelFormat)
        assertEquals("data", request.capabilities.inputBlobName)
        assertEquals("output", request.capabilities.outputBlobName)
        assertEquals(10, request.capabilities.prePadding)
        assertTrue(request.capabilities.supportsGpu)
    }

    private fun artifact(fileName: String) = ModelArtifactDescriptor(
        fileName = fileName,
        downloadUrl = "https://example.com/$fileName",
        fileSizeBytes = 1L,
        sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    )
}
