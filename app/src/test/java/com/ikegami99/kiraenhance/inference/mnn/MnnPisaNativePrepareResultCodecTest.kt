package com.ikegami99.kiraenhance.inference.mnn

import org.junit.Assert.assertEquals
import org.junit.Test

class MnnPisaNativePrepareResultCodecTest {
    @Test
    fun `decodes successful graph preparation`() {
        val result = MnnPisaNativePrepareResultCodec.decode(
            longArrayOf(
                MnnPisaNativeError.NONE.ordinal.toLong(),
                768L,
                512L,
                96L,
                64L,
            ),
        )

        assertEquals(MnnPisaNativeError.NONE, result.errorCode)
        assertEquals(768, result.imageWidth)
        assertEquals(512, result.imageHeight)
        assertEquals(96, result.latentWidth)
        assertEquals(64, result.latentHeight)
    }

    @Test
    fun `preserves native preparation failure`() {
        val result = MnnPisaNativePrepareResultCodec.decode(
            longArrayOf(
                MnnPisaNativeError.INVALID_ARGUMENT.ordinal.toLong(),
                0L,
                0L,
                0L,
                0L,
            ),
        )

        assertEquals(MnnPisaNativeError.INVALID_ARGUMENT, result.errorCode)
    }

    @Test
    fun `rejects malformed successful payload`() {
        val result = MnnPisaNativePrepareResultCodec.decode(
            longArrayOf(
                MnnPisaNativeError.NONE.ordinal.toLong(),
                512L,
                512L,
                0L,
                64L,
            ),
        )

        assertEquals(MnnPisaNativeError.INTERNAL, result.errorCode)
    }
}
