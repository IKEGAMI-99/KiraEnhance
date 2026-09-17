package com.ikegami99.kiraenhance.diagnostics

import com.ikegami99.kiraenhance.inference.mnn.MnnRuntimeInfo
import org.junit.Assert.assertEquals
import org.junit.Test

class MnnRuntimeDiagnosticsTest {
    @Test
    fun `reports linked runtime version`() {
        val message = MnnRuntimeDiagnostics.capture {
            MnnRuntimeInfo(mnnVersion = "3.6.1", linked = true)
        }

        assertEquals("version=3.6.1 linked=true", message)
    }

    @Test
    fun `reports native probe failure without throwing`() {
        val message = MnnRuntimeDiagnostics.capture {
            throw UnsatisfiedLinkError("missing libMNN.so")
        }

        assertEquals(
            "probe_failed type=UnsatisfiedLinkError message=missing libMNN.so",
            message,
        )
    }
}
