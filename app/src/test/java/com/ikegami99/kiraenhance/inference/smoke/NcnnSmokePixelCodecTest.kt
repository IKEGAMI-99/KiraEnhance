package com.ikegami99.kiraenhance.inference.smoke

import com.ikegami99.kiraenhance.inference.PixelFormat
import com.ikegami99.kiraenhance.inference.UpscaleOutput
import java.nio.ByteBuffer
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class NcnnSmokePixelCodecTest {
    @Test
    fun preservesArgbChannelsAcrossRgbaBuffers() {
        val argb = intArrayOf(
            0xFF112233.toInt(),
            0x80445566.toInt(),
        )

        val input = NcnnSmokePixelCodec.encodeArgb(
            width = 2,
            height = 1,
            argb = argb,
        )

        assertEquals(2, input.width)
        assertEquals(1, input.height)
        assertEquals(8, input.rowStrideBytes)
        assertEquals(PixelFormat.RGBA_8888, input.pixelFormat)
        assertTrue(input.pixels.isDirect)

        val encoded = ByteArray(8)
        input.pixels.duplicate().get(encoded)
        assertArrayEquals(
            byteArrayOf(
                0x11, 0x22, 0x33, 0xFF.toByte(),
                0x44, 0x55, 0x66, 0x80.toByte(),
            ),
            encoded,
        )

        val outputPixels = ByteBuffer.allocateDirect(8).apply {
            put(encoded)
            flip()
        }
        val decoded = NcnnSmokePixelCodec.decodeToArgb(
            UpscaleOutput(
                width = 2,
                height = 1,
                rowStrideBytes = 8,
                pixelFormat = PixelFormat.RGBA_8888,
                pixels = outputPixels,
            ),
        )

        assertArrayEquals(argb, decoded)
    }
}
