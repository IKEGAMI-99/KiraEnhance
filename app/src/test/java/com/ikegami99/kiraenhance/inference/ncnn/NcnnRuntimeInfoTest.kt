package com.ikegami99.kiraenhance.inference.ncnn

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class NcnnRuntimeInfoTest {
    @Test
    fun parsesNativeRuntimeInfo() {
        val info = NcnnRuntimeInfo.parse("ncnn=20260526;vulkan=1;gpuCount=1")

        assertEquals("20260526", info.ncnnVersion)
        assertTrue(info.vulkanCompiled)
        assertEquals(1, info.gpuCount)
        assertTrue(info.hasUsableVulkanGpu)
    }

    @Test
    fun reportsNoUsableGpuWhenVulkanIsUnavailable() {
        val info = NcnnRuntimeInfo.parse("ncnn=20260526;vulkan=0;gpuCount=0")

        assertFalse(info.vulkanCompiled)
        assertFalse(info.hasUsableVulkanGpu)
    }

    @Test
    fun rejectsMalformedRuntimeInfo() {
        assertThrows(IllegalArgumentException::class.java) {
            NcnnRuntimeInfo.parse("ncnn=20260526;vulkan=maybe")
        }
    }
}
