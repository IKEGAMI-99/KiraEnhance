package com.ikegami99.kiraenhance.ui.smoke

import com.ikegami99.kiraenhance.inference.smoke.NcnnSmokeImage
import org.junit.Assert.assertEquals
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test

class NcnnSmokeSelectionPresenterTest {
    @Test
    fun selectedImageIsReadyForSmokeInference() {
        val image = NcnnSmokeImage(
            width = 128,
            height = 96,
            argbPixels = IntArray(128 * 96),
        )

        val result = NcnnSmokeSelectionPresenter.present(image)

        assertTrue(result.state.modelInstalled)
        assertTrue(result.state.imageSelected)
        assertEquals("画像を選択しました", result.state.status)
        assertEquals("128 × 96", result.state.inputDescription)
        assertSame(image, result.image)
    }
}
