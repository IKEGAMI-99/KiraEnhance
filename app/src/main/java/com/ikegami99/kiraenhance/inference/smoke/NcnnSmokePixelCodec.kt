package com.ikegami99.kiraenhance.inference.smoke

import com.ikegami99.kiraenhance.inference.PixelFormat
import com.ikegami99.kiraenhance.inference.UpscaleInput
import com.ikegami99.kiraenhance.inference.UpscaleOutput
import java.nio.ByteBuffer

object NcnnSmokePixelCodec {
    fun encodeArgb(
        width: Int,
        height: Int,
        argb: IntArray,
    ): UpscaleInput {
        require(width > 0 && height > 0) { "Image dimensions must be positive" }
        require(argb.size == width * height) { "ARGB pixel count does not match image dimensions" }

        val rowStrideBytes = width * BYTES_PER_PIXEL
        val buffer = ByteBuffer.allocateDirect(rowStrideBytes * height)
        argb.forEach { pixel ->
            buffer.put(((pixel ushr 16) and 0xFF).toByte())
            buffer.put(((pixel ushr 8) and 0xFF).toByte())
            buffer.put((pixel and 0xFF).toByte())
            buffer.put(((pixel ushr 24) and 0xFF).toByte())
        }
        buffer.flip()

        return UpscaleInput(
            width = width,
            height = height,
            rowStrideBytes = rowStrideBytes,
            pixelFormat = PixelFormat.RGBA_8888,
            pixels = buffer,
        )
    }

    fun decodeToArgb(output: UpscaleOutput): IntArray {
        require(output.pixelFormat == PixelFormat.RGBA_8888) { "Smoke output must be RGBA_8888" }
        require(output.rowStrideBytes >= output.width * BYTES_PER_PIXEL) {
            "Output row stride is smaller than one RGBA row"
        }

        val source = output.pixels.duplicate()
        val base = source.position()
        return IntArray(output.width * output.height) { index ->
            val x = index % output.width
            val y = index / output.width
            val offset = base + y * output.rowStrideBytes + x * BYTES_PER_PIXEL
            val red = source.get(offset).toInt() and 0xFF
            val green = source.get(offset + 1).toInt() and 0xFF
            val blue = source.get(offset + 2).toInt() and 0xFF
            val alpha = source.get(offset + 3).toInt() and 0xFF
            (alpha shl 24) or (red shl 16) or (green shl 8) or blue
        }
    }

    private const val BYTES_PER_PIXEL = 4
}
