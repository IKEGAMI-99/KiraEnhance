package com.ikegami99.kiraenhance.inference.smoke

import org.junit.Assert.assertEquals
import org.junit.Test

class NcnnSmokeCropPlannerTest {
    @Test
    fun centerSquareCapsLandscapeImageAt256Pixels() {
        val crop = NcnnSmokeCropPlanner.centerSquare(
            imageWidth = 1920,
            imageHeight = 1080,
        )

        assertEquals(NcnnSmokeCropRect(left = 832, top = 412, width = 256, height = 256), crop)
    }

    @Test
    fun centerSquareCapsPortraitImageAt256Pixels() {
        val crop = NcnnSmokeCropPlanner.centerSquare(
            imageWidth = 1080,
            imageHeight = 1920,
        )

        assertEquals(NcnnSmokeCropRect(left = 412, top = 832, width = 256, height = 256), crop)
    }

    @Test
    fun centerSquareUsesShorterEdgeWhenImageIsSmallerThan256() {
        val crop = NcnnSmokeCropPlanner.centerSquare(
            imageWidth = 100,
            imageHeight = 80,
        )

        assertEquals(NcnnSmokeCropRect(left = 10, top = 0, width = 80, height = 80), crop)
    }
}
