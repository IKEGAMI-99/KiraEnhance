package com.ikegami99.kiraenhance.image

import java.time.Instant
import java.time.ZoneId
import org.junit.Assert.assertEquals
import org.junit.Test

class EnhanceOutputNamerTest {
    @Test
    fun `png filename is deterministic and filesystem safe`() {
        val name = EnhanceOutputNamer.png(
            instant = Instant.parse("2026-09-16T14:35:42Z"),
            zoneId = ZoneId.of("Asia/Tokyo"),
            mode = "UltraSharp",
            scale = 4,
        )

        assertEquals("KiraEnhance_20260916_233542_UltraSharp_4x.png", name)
    }
}
