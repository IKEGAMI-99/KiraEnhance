package com.ikegami99.kiraenhance.inference.mnn

import org.junit.Assert.assertEquals
import org.junit.Test

class MnnPisaNativeInferenceResultCodecTest {
    @Test
    fun `decodes successful native inference payload`() {
        val result = MnnPisaNativeInferenceResultCodec.decode(
            longArrayOf(
                MnnPisaNativeError.NONE.ordinal.toLong(),
                2048L,
                1024L,
                8192L,
                1L,
                123_456_789L,
            ),
        )

        assertEquals(MnnPisaNativeError.NONE, result.errorCode)
        assertEquals(2048, result.outputWidth)
        assertEquals(1024, result.outputHeight)
        assertEquals(8192, result.outputRowStrideBytes)
        assertEquals(true, result.gpuUsed)
        assertEquals(123_456_789L, result.segmentedVaePeakTrackedBytes)
    }

    @Test
    fun `preserves native failure with empty dimensions`() {
        val result = MnnPisaNativeInferenceResultCodec.decode(
            longArrayOf(
                MnnPisaNativeError.NOT_IMPLEMENTED.ordinal.toLong(),
                0L,
                0L,
                0L,
                0L,
                0L,
            ),
        )

        assertEquals(MnnPisaNativeError.NOT_IMPLEMENTED, result.errorCode)
        assertEquals(0, result.outputWidth)
        assertEquals(0, result.outputHeight)
    }

    @Test
    fun `rejects malformed successful payload`() {
        val result = MnnPisaNativeInferenceResultCodec.decode(
            longArrayOf(
                MnnPisaNativeError.NONE.ordinal.toLong(),
                512L,
                512L,
                1024L,
                0L,
                0L,
            ),
        )

        assertEquals(MnnPisaNativeError.INTERNAL, result.errorCode)
    }
    @Test
    fun `rejects negative segmented VAE peak bytes`() {
        val result = MnnPisaNativeInferenceResultCodec.decode(
            longArrayOf(
                MnnPisaNativeError.NONE.ordinal.toLong(),
                512L,
                512L,
                2048L,
                0L,
                -1L,
            ),
        )

        assertEquals(MnnPisaNativeError.INTERNAL, result.errorCode)
    }

}
