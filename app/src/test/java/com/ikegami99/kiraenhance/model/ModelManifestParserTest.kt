package com.ikegami99.kiraenhance.model

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class ModelManifestParserTest {
    private val parser = ModelManifestParser()

    @Test
    fun parsesFourUserFacingModesAndControls() {
        val manifest = parser.parse(validManifestJson())

        assertEquals(1, manifest.schemaVersion)
        assertEquals(4, manifest.models.size)

        val fidelity = manifest.models.first { it.id == "realesrgan-anime" }
        assertEquals(EnhancementMode.FIDELITY, fidelity.mode)
        assertEquals(ModelBackend.NCNN, fidelity.backend)
        assertEquals(listOf(2, 4), fidelity.supportedScales)
        assertFalse(fidelity.community)

        val pisa = manifest.models.first { it.id == "pisa-sr" }
        assertEquals(EnhancementMode.BALANCED, pisa.mode)
        assertEquals(ModelBackend.MNN, pisa.backend)
        assertEquals(1, pisa.controls.size)
        assertEquals(ControlType.SLIDER, pisa.controls.single().type)
        assertEquals(0.65, pisa.controls.single().defaultValue as Double, 0.0)

        val ultraSharp = manifest.models.first { it.id == "ultrasharp" }
        assertEquals(EnhancementMode.ULTRASHARP, ultraSharp.mode)
        assertTrue(ultraSharp.community)
        assertEquals("CC BY-NC-SA 4.0", ultraSharp.licenseName)
    }

    @Test
    fun rejectsUnknownBackend() {
        val json = validManifestJson().replace("\"backend\": \"MNN\"", "\"backend\": \"MAGIC\"")

        assertThrows(IllegalArgumentException::class.java) {
            parser.parse(json)
        }
    }

    @Test
    fun rejectsMalformedSha256() {
        val json = validManifestJson().replace(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "not-a-sha",
        )

        assertThrows(IllegalArgumentException::class.java) {
            parser.parse(json)
        }
    }

    @Test
    fun rejectsDuplicateModelIds() {
        val json = validManifestJson().replace("\"id\": \"pisa-sr\"", "\"id\": \"realesrgan-anime\"")

        assertThrows(IllegalArgumentException::class.java) {
            parser.parse(json)
        }
    }

    @Test
    fun rejectsUnsupportedScale() {
        val json = validManifestJson().replace("\"supportedScales\": [2, 4]", "\"supportedScales\": [3, 4]")

        assertThrows(IllegalArgumentException::class.java) {
            parser.parse(json)
        }
    }

    @Test
    fun rejectsSliderDefaultOutsideRange() {
        val json = validManifestJson().replace("\"defaultValue\": 0.65", "\"defaultValue\": 1.5")

        assertThrows(IllegalArgumentException::class.java) {
            parser.parse(json)
        }
    }

    private fun validManifestJson(): String =
        """
        {
          "schemaVersion": 1,
          "models": [
            {
              "id": "realesrgan-anime",
              "displayName": "Kira Fidelity",
              "mode": "FIDELITY",
              "version": "candidate-1",
              "backend": "NCNN",
              "downloadUrl": "https://example.invalid/realesrgan-anime.bin",
              "fileSizeBytes": 1000000,
              "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
              "supportedScales": [2, 4],
              "minAppVersion": "0.1.0-alpha01",
              "estimatedRamMb": 2048,
              "licenseName": "candidate-license",
              "licenseUrl": "https://example.invalid/license/realesrgan",
              "description": "Fast faithful anime upscaler candidate.",
              "community": false,
              "controls": []
            },
            {
              "id": "pisa-sr",
              "displayName": "Kira Balance",
              "mode": "BALANCED",
              "version": "candidate-1",
              "backend": "MNN",
              "downloadUrl": "https://example.invalid/pisa-sr.bin",
              "fileSizeBytes": 2000000,
              "sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
              "supportedScales": [2, 4],
              "minAppVersion": "0.1.0-alpha01",
              "estimatedRamMb": 8000,
              "licenseName": "Apache-2.0",
              "licenseUrl": "https://example.invalid/license/pisa",
              "description": "Required balanced PiSA-SR target.",
              "community": false,
              "controls": [
                {
                  "id": "fidelity",
                  "label": "忠実度",
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
              "displayName": "Kira Detail",
              "mode": "DETAIL",
              "version": "candidate-1",
              "backend": "MNN",
              "downloadUrl": "https://example.invalid/hat-s.bin",
              "fileSizeBytes": 3000000,
              "sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
              "supportedScales": [4],
              "minAppVersion": "0.1.0-alpha01",
              "estimatedRamMb": 4096,
              "licenseName": "candidate-license",
              "licenseUrl": "https://example.invalid/license/hat",
              "description": "Detail-oriented HAT-S candidate.",
              "community": false,
              "controls": []
            },
            {
              "id": "ultrasharp",
              "displayName": "UltraSharp",
              "mode": "ULTRASHARP",
              "version": "4x-candidate",
              "backend": "NCNN",
              "downloadUrl": "https://example.invalid/ultrasharp.bin",
              "fileSizeBytes": 4000000,
              "sha256": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
              "supportedScales": [4],
              "minAppVersion": "0.1.0-alpha01",
              "estimatedRamMb": 2048,
              "licenseName": "CC BY-NC-SA 4.0",
              "licenseUrl": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
              "description": "Optional community model.",
              "community": true,
              "controls": []
            }
          ]
        }
        """.trimIndent()
}
