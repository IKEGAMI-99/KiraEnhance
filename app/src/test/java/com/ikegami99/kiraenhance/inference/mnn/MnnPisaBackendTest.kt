package com.ikegami99.kiraenhance.inference.mnn

import org.junit.Assert.assertEquals
import org.junit.Test

class MnnPisaBackendTest {
    @Test
    fun `gpu preference tries OpenCL then Vulkan then CPU`() {
        assertEquals(
            listOf(
                MnnPisaBackend.OPENCL,
                MnnPisaBackend.VULKAN,
                MnnPisaBackend.CPU,
            ),
            MnnPisaBackend.preference(preferGpu = true),
        )
    }

    @Test
    fun `cpu preference skips gpu backends`() {
        assertEquals(
            listOf(MnnPisaBackend.CPU),
            MnnPisaBackend.preference(preferGpu = false),
        )
    }
}
