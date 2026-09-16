package com.ikegami99.kiraenhance.model

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class BundledModelManifestTest {
    private val parser = ModelManifestParser()

    @Test
    fun ultraSharpUsesVerifiedPinnedNcnnArtifacts() {
        val manifestFile = File("src/main/assets/model-manifest.json")
        assertTrue("Bundled manifest must exist", manifestFile.isFile)

        val manifest = parser.parse(manifestFile.readText())
        val ultraSharp = manifest.models.single { it.id == "ultrasharp" }

        assertEquals("4x-fp16-b32be1b", ultraSharp.version)
        assertEquals(ModelBackend.NCNN, ultraSharp.backend)
        assertEquals(EnhancementMode.ULTRASHARP, ultraSharp.mode)
        assertEquals(listOf(4), ultraSharp.supportedScales)
        assertTrue(ultraSharp.community)
        assertEquals("CC BY-NC-SA 4.0", ultraSharp.licenseName)
        assertEquals(
            "https://creativecommons.org/licenses/by-nc-sa/4.0/",
            ultraSharp.licenseUrl,
        )

        val param = ultraSharp.artifacts.single { it.fileName == "model.param" }
        assertEquals(165_579L, param.fileSizeBytes)
        assertEquals(
            "6b34402c8c156b5b9f2b14347d65ee03391492b07f5237bf3173596b56be4e46",
            param.sha256,
        )
        assertEquals(
            "https://huggingface.co/Kim2091/UltraSharp/resolve/b32be1b8c6f81d1f6ef8ce11f016aad3a139d1e0/NCNN/4x-UltraSharp-fp16.param",
            param.downloadUrl,
        )

        val bin = ultraSharp.artifacts.single { it.fileName == "model.bin" }
        assertEquals(33_424_520L, bin.fileSizeBytes)
        assertEquals(
            "713ce69a8642b1907cc24da7573560dcc933faf0c74e48ab0b3b379245618701",
            bin.sha256,
        )
        assertEquals(
            "https://huggingface.co/Kim2091/UltraSharp/resolve/b32be1b8c6f81d1f6ef8ce11f016aad3a139d1e0/NCNN/4x-UltraSharp-fp16.bin",
            bin.downloadUrl,
        )

        assertFalse(ultraSharp.artifacts.any { it.downloadUrl.contains("example.invalid") })
        assertEquals(33_590_099L, ultraSharp.totalFileSizeBytes)
    }
}
