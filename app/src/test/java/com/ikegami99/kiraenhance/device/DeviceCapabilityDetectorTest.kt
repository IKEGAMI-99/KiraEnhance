package com.ikegami99.kiraenhance.device

import org.junit.Assert.assertEquals
import org.junit.Test

class DeviceCapabilityDetectorTest {
    @Test
    fun rejectsDevicesWithout64BitSupport() {
        assertEquals(
            SupportTier.UNSUPPORTED,
            classifyDevice(totalRamMb = 16_000, hasVulkan12 = true, is64Bit = false),
        )
    }

    @Test
    fun rejectsDevicesWithoutVulkan12() {
        assertEquals(
            SupportTier.UNSUPPORTED,
            classifyDevice(totalRamMb = 16_000, hasVulkan12 = false, is64Bit = true),
        )
    }

    @Test
    fun marksTwelveGigabytesOrMoreAsRecommended() {
        assertEquals(
            SupportTier.RECOMMENDED,
            classifyDevice(totalRamMb = 12_000, hasVulkan12 = true, is64Bit = true),
        )
    }

    @Test
    fun marksEightToTwelveGigabytesAsSupported() {
        assertEquals(
            SupportTier.SUPPORTED,
            classifyDevice(totalRamMb = 8_000, hasVulkan12 = true, is64Bit = true),
        )
    }

    @Test
    fun marksLowMemoryDevicesAsNotRecommended() {
        assertEquals(
            SupportTier.NOT_RECOMMENDED,
            classifyDevice(totalRamMb = 7_999, hasVulkan12 = true, is64Bit = true),
        )
    }
}
