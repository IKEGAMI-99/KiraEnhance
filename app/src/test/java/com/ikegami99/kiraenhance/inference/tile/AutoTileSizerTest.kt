package com.ikegami99.kiraenhance.inference.tile

import org.junit.Assert.assertEquals
import org.junit.Test

class AutoTileSizerTest {
    @Test
    fun missingDeviceInfoFallsBackTo128() {
        assertEquals(128, AutoTileSizer.initial(totalRamMb = null, hasUsableGpu = true))
    }

    @Test
    fun missingGpuFallsBackTo128EvenWithLargeRam() {
        assertEquals(128, AutoTileSizer.initial(totalRamMb = 16_384, hasUsableGpu = false))
    }

    @Test
    fun gpuDevicesScaleConservativelyWithRam() {
        assertEquals(128, AutoTileSizer.initial(totalRamMb = 4_096, hasUsableGpu = true))
        assertEquals(192, AutoTileSizer.initial(totalRamMb = 6_144, hasUsableGpu = true))
        assertEquals(256, AutoTileSizer.initial(totalRamMb = 8_192, hasUsableGpu = true))
        assertEquals(384, AutoTileSizer.initial(totalRamMb = 12_288, hasUsableGpu = true))
        assertEquals(384, AutoTileSizer.initial(totalRamMb = 16_384, hasUsableGpu = true))
    }
}
