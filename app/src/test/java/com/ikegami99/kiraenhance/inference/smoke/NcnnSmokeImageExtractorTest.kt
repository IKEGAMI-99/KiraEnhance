package com.ikegami99.kiraenhance.inference.smoke

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

class NcnnSmokeImageExtractorTest {
    @Test
    fun cropCopiesRequestedPixelsRowByRow() {
        val image = NcnnSmokeImage(
            width = 4,
            height = 3,
            argbPixels = intArrayOf(
                0, 1, 2, 3,
                4, 5, 6, 7,
                8, 9, 10, 11,
            ),
        )

        val cropped = NcnnSmokeImageExtractor.crop(
            image = image,
            rect = NcnnSmokeCropRect(
                left = 1,
                top = 1,
                width = 2,
                height = 2,
            ),
        )

        assertEquals(2, cropped.width)
        assertEquals(2, cropped.height)
        assertArrayEquals(intArrayOf(5, 6, 9, 10), cropped.argbPixels)
    }

    @Test(expected = IllegalArgumentException::class)
    fun cropRejectsRectangleOutsideImage() {
        NcnnSmokeImageExtractor.crop(
            image = NcnnSmokeImage(
                width = 2,
                height = 2,
                argbPixels = intArrayOf(0, 1, 2, 3),
            ),
            rect = NcnnSmokeCropRect(
                left = 1,
                top = 1,
                width = 2,
                height = 2,
            ),
        )
    }
}
