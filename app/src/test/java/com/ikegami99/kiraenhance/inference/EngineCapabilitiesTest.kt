package com.ikegami99.kiraenhance.inference

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class EngineCapabilitiesTest {
    @Test
    fun acceptsEsrganStyleCapabilities() {
        val capabilities = EngineCapabilities(
            nativeScale = 4,
            pixelFormat = PixelFormat.RGBA_8888,
            inputBlobName = "data",
            outputBlobName = "output",
            prePadding = 10,
            supportsGpu = true,
        )

        assertEquals(4, capabilities.nativeScale)
        assertEquals("data", capabilities.inputBlobName)
        assertEquals("output", capabilities.outputBlobName)
        assertTrue(capabilities.supportsGpu)
    }

    @Test
    fun rejectsInvalidNativeScale() {
        assertThrows(IllegalArgumentException::class.java) {
            EngineCapabilities(
                nativeScale = 0,
                pixelFormat = PixelFormat.RGBA_8888,
                inputBlobName = "data",
                outputBlobName = "output",
                prePadding = 10,
                supportsGpu = true,
            )
        }
    }

    @Test
    fun rejectsBlankBlobNames() {
        assertThrows(IllegalArgumentException::class.java) {
            EngineCapabilities(
                nativeScale = 4,
                pixelFormat = PixelFormat.RGBA_8888,
                inputBlobName = " ",
                outputBlobName = "output",
                prePadding = 10,
                supportsGpu = true,
            )
        }
    }

    @Test
    fun rejectsNegativePrePadding() {
        assertThrows(IllegalArgumentException::class.java) {
            EngineCapabilities(
                nativeScale = 4,
                pixelFormat = PixelFormat.RGBA_8888,
                inputBlobName = "data",
                outputBlobName = "output",
                prePadding = -1,
                supportsGpu = true,
            )
        }
    }

    @Test
    fun modelLoadRequestRejectsDuplicateArtifactNames() {
        assertThrows(IllegalArgumentException::class.java) {
            ModelLoadRequest(
                modelId = "ultrasharp",
                version = "4x-candidate",
                artifacts = listOf(
                    ModelArtifactFile("model.bin", "/tmp/a.bin"),
                    ModelArtifactFile("model.bin", "/tmp/b.bin"),
                ),
                capabilities = EngineCapabilities(
                    nativeScale = 4,
                    pixelFormat = PixelFormat.RGBA_8888,
                    inputBlobName = "data",
                    outputBlobName = "output",
                    prePadding = 10,
                    supportsGpu = true,
                ),
            )
        }
    }
}
