package com.ikegami99.kiraenhance.model

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ModelManifestParserTest {
    private val parser = ModelManifestParser()

    @Test
    fun `parses four modes including community UltraSharp and PiSA control`() {
        val manifest = parser.parse(validManifestJson())

        assertEquals(1, manifest.schemaVersion)
        assertEquals(4, manifest.models.size)
        assertEquals(
            listOf(
                EnhancementMode.FIDELITY,
                EnhancementMode.BALANCED,
                EnhancementMode.DETAIL,
                EnhancementMode.ULTRASHARP,
            ),
            manifest.models.map { it.mode },
        )

        val pisa = manifest.models.first { it.id == "pisa-sr" }
        assertEquals(ModelBackend.MNN, pisa.backend)
        assertEquals(listOf(2, 4), pisa.supportedScales)
        assertEquals(ControlType.SLIDER, pisa.controls.single().type)

        val ultraSharp = manifest.models.first { it.id == "4x-ultrasharp" }
        assertTrue(ultraSharp.community)
        assertEquals("Community model license", ultraSharp.licenseName)
        assertFalse(ultraSharp.licenseUrl.isBlank())
    }

    @Test(expected = IllegalArgumentException::class)
    fun `rejects unknown backend`() {
        parser.parse(validManifestJson().replace("\"MNN\"", "\"TENSORRT\""))
    }

    @Test(expected = IllegalArgumentException::class)
    fun `rejects malformed sha256`() {
        parser.parse(validManifestJson().replace("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "xyz"))
    }

    @Test(expected = IllegalArgumentException::class)
    fun `rejects duplicate model ids`() {
        parser.parse(validManifestJson().replace("\"hat-s\"", "\"pisa-sr\""))
    }

    @Test(expected = IllegalArgumentException::class)
    fun `rejects invalid slider default`() {
        parser.parse(validManifestJson().replace("\"defaultValue\": 0.65", "\"defaultValue\": 1.25"))
    }

    private fun validManifestJson() = """
        {
          "schemaVersion": 1,
          "models": [
            {
              "id": "realesrgan-anime",
              "displayName": "RealESRGAN Anime",
              "mode": "FIDELITY",
              "version": "1.0.0",
              "backend": "NCNN",
              "downloadUrl": "https://example.com/realesrgan.bin",
              "fileSizeBytes": 1000000,
              "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
              "supportedScales": [4],
              "minAppVersion": "0.1.0",
              "estimatedRamMb": 3500,
              "licenseName": "Example license",
              "licenseUrl": "https://example.com/license",
              "description": "Fast fidelity-oriented model",
              "community": false,
              "controls": []
            },
            {
              "id": "pisa-sr",
              "displayName": "PiSA-SR",
              "mode": "BALANCED",
              "version": "1.0.0",
              "backend": "MNN",
              "downloadUrl": "https://example.com/pisa.bin",
              "fileSizeBytes": 2000000,
              "sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
              "supportedScales": [2, 4],
              "minAppVersion": "0.1.0",
              "estimatedRamMb": 6000,
              "licenseName": "Example license",
              "licenseUrl": "https://example.com/license",
              "description": "Balanced PiSA-SR target",
              "community": false,
              "controls": [
                {
                  "id": "fidelityStrength",
                  "displayName": "忠実度",
                  "type": "SLIDER",
                  "min": 0.0,
                  "max": 1.0,
                  "defaultValue": 0.65,
                  "choices": []
                }
              ]
            },
            {
              "id": "hat-s",
              "displayName": "HAT-S",
              "mode": "DETAIL",
              "version": "1.0.0",
              "backend": "MNN",
              "downloadUrl": "https://example.com/hat.bin",
              "fileSizeBytes": 3000000,
              "sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
              "supportedScales": [4],
              "minAppVersion": "0.1.0",
              "estimatedRamMb": 7000,
              "licenseName": "Example license",
              "licenseUrl": "https://example.com/license",
              "description": "Detail-oriented model",
              "community": false,
              "controls": []
            },
            {
              "id": "4x-ultrasharp",
              "displayName": "4x-UltraSharp",
              "mode": "ULTRASHARP",
              "version": "1.0.0",
              "backend": "NCNN",
              "downloadUrl": "https://example.com/ultrasharp.bin",
              "fileSizeBytes": 4000000,
              "sha256": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
              "supportedScales": [4],
              "minAppVersion": "0.1.0",
              "estimatedRamMb": 4500,
              "licenseName": "Community model license",
              "licenseUrl": "https://example.com/community-license",
              "description": "Community UltraSharp model",
              "community": true,
              "controls": []
            }
          ]
        }
    """.trimIndent()
}
