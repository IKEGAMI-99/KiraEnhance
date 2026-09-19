package com.ikegami99.kiraenhance.inference.mnn

import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class MnnRuntimeProbeInstrumentedTest {
    @Test
    fun reportsLinkedMnn361Runtime() {
        val info = MnnPisaNativeBridge.runtimeInfo()

        assertEquals("3.6.1", info.mnnVersion)
        assertTrue(info.linked)
    }
}
