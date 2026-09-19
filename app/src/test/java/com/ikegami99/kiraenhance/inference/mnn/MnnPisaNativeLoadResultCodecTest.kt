package com.ikegami99.kiraenhance.inference.mnn

import org.junit.Assert.assertEquals
import org.junit.Test

class MnnPisaNativeLoadResultCodecTest {
    @Test
    fun `decodes successful native load payload`() {
        assertEquals(
            MnnPisaNativeLoadResult(
                handle = 41L,
                errorCode = MnnPisaNativeError.NONE,
                gpuEnabled = false,
            ),
            MnnPisaNativeLoadResultCodec.decode(
                longArrayOf(41L, MnnPisaNativeError.NONE.ordinal.toLong(), 0L),
            ),
        )
    }

    @Test
    fun `decodes explicit native failure payload`() {
        assertEquals(
            MnnPisaNativeLoadResult(
                handle = 0L,
                errorCode = MnnPisaNativeError.LOAD_FAILED,
                gpuEnabled = false,
            ),
            MnnPisaNativeLoadResultCodec.decode(
                longArrayOf(0L, MnnPisaNativeError.LOAD_FAILED.ordinal.toLong(), 0L),
            ),
        )
    }

    @Test
    fun `rejects malformed native load payload length`() {
        assertInternalFailure(
            MnnPisaNativeLoadResultCodec.decode(longArrayOf(41L)),
        )
    }

    @Test
    fun `rejects unknown native error ordinal`() {
        assertInternalFailure(
            MnnPisaNativeLoadResultCodec.decode(longArrayOf(41L, 999L, 0L)),
        )
    }

    @Test
    fun `rejects invalid gpu flag`() {
        assertInternalFailure(
            MnnPisaNativeLoadResultCodec.decode(
                longArrayOf(41L, MnnPisaNativeError.NONE.ordinal.toLong(), 2L),
            ),
        )
    }

    @Test
    fun `rejects success without a native handle`() {
        assertInternalFailure(
            MnnPisaNativeLoadResultCodec.decode(
                longArrayOf(0L, MnnPisaNativeError.NONE.ordinal.toLong(), 0L),
            ),
        )
    }

    private fun assertInternalFailure(result: MnnPisaNativeLoadResult) {
        assertEquals(0L, result.handle)
        assertEquals(MnnPisaNativeError.INTERNAL, result.errorCode)
        assertEquals(false, result.gpuEnabled)
    }
}
