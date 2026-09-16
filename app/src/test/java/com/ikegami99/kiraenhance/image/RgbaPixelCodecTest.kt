package com.ikegami99.kiraenhance.image

import com.ikegami99.kiraenhance.inference.PixelFormat
import com.ikegami99.kiraenhance.inference.UpscaleOutput
import java.nio.ByteBuffer
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

class RgbaPixelCodecTest {
    @Test
    fun `encodeArgb writes rgba byte order`() {
        val input = RgbaPixelCodec.encodeArgb(
            width = 1,
            height = 1,
            argb = intArrayOf(0x7F123456),
        )
        val pixels = input.pixels.duplicate()

        assertEquals(0x12, pixels.get(0).toInt() and 0xFF)
        assertEquals(0x34, pixels.get(1).toInt() and 0xFF)
        assertEquals(0x56, pixels.get(2).toInt() and 0xFF)
        assertEquals(0x7F, pixels.get(3).toInt() and 0xFF)
    }

    @Test
    fun `decodeRow respects output row stride`() {
        val buffer = ByteBuffer.allocateDirect(16).apply {
            put(byteArrayOf(1, 2, 3, 4, 5, 6, 7, 8))
            put(byteArrayOf(99, 99, 99, 99))
            put(byteArrayOf(9, 10, 11, 12))
            flip()
        }
        val output = UpscaleOutput(
            width = 2,
            height = 1,
            rowStrideBytes = 12,
            pixelFormat = PixelFormat.RGBA_8888,
            pixels = buffer,
        )
        val destination = IntArray(2)

        RgbaPixelCodec.decodeRow(output, y = 0, destination = destination)

        assertArrayEquals(
            intArrayOf(0x04010203, 0x08050607),
            destination,
        )
    }
}
