package com.ikegami99.kiraenhance.device

import org.junit.Assert.assertEquals
import org.junit.Test

class DeviceCapabilityDetectorTest {
    @Test
    fun `non 64 bit devices are unsupported`() {
        assertEquals(
            SupportTier.UNSUPPORTED,
            classifyDevice(totalRamMb = 16_000, hasVulkan12 = true, is64Bit = false),
        )
    }

    @Test
    fun `devices without Vulkan 1_2 are unsupported`() {
        assertEquals(
            SupportTier.UNSUPPORTED,
            classifyDevice(totalRamMb = 16_000, hasVulkan12 = false, is64Bit = true),
        )
    }

    @Test
    fun `12 GB class devices are recommended`() {
        assertEquals(
            SupportTier.RECOMMENDED,
            classifyDevice(totalRamMb = 12_000, hasVulkan12 = true, is64Bit = true),
        )
    }

    @Test
    fun `8 GB class devices are supported`() {
        assertEquals(
            SupportTier.SUPPORTED,
            classifyDevice(totalRamMb = 8_000, hasVulkan12 = true, is64Bit = true),
        )
    }

    @Test
    fun `lower memory capable devices are not recommended`() {
        assertEquals(
            SupportTier.NOT_RECOMMENDED,
            classifyDevice(totalRamMb = 7_999, hasVulkan12 = true, is64Bit = true),
        )
    }
}
