package com.ikegami99.kiraenhance.inference.mnn

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class MnnPisaNativeSmokeResultCodecTest {
    @Test
    fun `decodes successful full graph smoke`() {
        val result = MnnPisaNativeSmokeResultCodec.decode(
            longArrayOf(
                MnnPisaNativeError.NONE.ordinal.toLong(),
                3L,
                1L,
            ),
        )

        assertEquals(MnnPisaNativeError.NONE, result.errorCode)
        assertEquals(3, result.completedStages)
        assertTrue(result.outputFinite)
    }

    @Test
    fun `preserves partial native failure stage`() {
        val result = MnnPisaNativeSmokeResultCodec.decode(
            longArrayOf(
                MnnPisaNativeError.INFERENCE_FAILED.ordinal.toLong(),
                1L,
                0L,
            ),
        )

        assertEquals(MnnPisaNativeError.INFERENCE_FAILED, result.errorCode)
        assertEquals(1, result.completedStages)
        assertFalse(result.outputFinite)
    }

    @Test
    fun `rejects impossible successful payload`() {
        val result = MnnPisaNativeSmokeResultCodec.decode(
            longArrayOf(
                MnnPisaNativeError.NONE.ordinal.toLong(),
                2L,
                1L,
            ),
        )

        assertEquals(MnnPisaNativeError.INTERNAL, result.errorCode)
    }

    @Test
    fun `rejects malformed finite flag`() {
        val result = MnnPisaNativeSmokeResultCodec.decode(
            longArrayOf(
                MnnPisaNativeError.INFERENCE_FAILED.ordinal.toLong(),
                2L,
                7L,
            ),
        )

        assertEquals(MnnPisaNativeError.INTERNAL, result.errorCode)
    }
}
