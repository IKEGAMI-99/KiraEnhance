package com.ikegami99.kiraenhance.inference.mnn

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class MnnPisaSessionInfoTest {
    @Test
    fun `parses OpenCL session info`() {
        assertEquals(
            MnnPisaSessionInfo(
                backend = MnnPisaBackend.OPENCL,
                gpuEnabled = true,
            ),
            MnnPisaSessionInfo.parse("backend=opencl;gpu=1"),
        )
    }

    @Test
    fun `parses Vulkan session info`() {
        assertEquals(
            MnnPisaSessionInfo(
                backend = MnnPisaBackend.VULKAN,
                gpuEnabled = true,
            ),
            MnnPisaSessionInfo.parse("backend=vulkan;gpu=1"),
        )
    }

    @Test
    fun `parses CPU session info`() {
        assertEquals(
            MnnPisaSessionInfo(
                backend = MnnPisaBackend.CPU,
                gpuEnabled = false,
            ),
            MnnPisaSessionInfo.parse("backend=cpu;gpu=0"),
        )
    }

    @Test
    fun `rejects inconsistent or malformed session info`() {
        assertNull(MnnPisaSessionInfo.parse("backend=opencl;gpu=0"))
        assertNull(MnnPisaSessionInfo.parse("backend=cpu;gpu=1"))
        assertNull(MnnPisaSessionInfo.parse("backend=metal;gpu=1"))
        assertNull(MnnPisaSessionInfo.parse("gpu=1"))
        assertNull(MnnPisaSessionInfo.parse(""))
    }
}
