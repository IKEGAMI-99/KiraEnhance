package com.ikegami99.kiraenhance.inference.mnn

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class MnnRuntimeInfoTest {
    @Test
    fun parsesLinkedNativeRuntimeInfo() {
        val info = MnnRuntimeInfo.parse("mnn=3.6.1;linked=1")

        assertEquals("3.6.1", info.mnnVersion)
        assertTrue(info.linked)
    }

    @Test
    fun parsesUnavailableRuntimeInfo() {
        val info = MnnRuntimeInfo.parse("mnn=unavailable;linked=0")

        assertEquals("unavailable", info.mnnVersion)
        assertFalse(info.linked)
    }

    @Test
    fun rejectsMalformedRuntimeInfo() {
        assertThrows(IllegalArgumentException::class.java) {
            MnnRuntimeInfo.parse("mnn=3.6.1;linked=maybe")
        }
    }
}
