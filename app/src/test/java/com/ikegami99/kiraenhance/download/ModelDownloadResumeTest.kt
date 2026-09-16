package com.ikegami99.kiraenhance.download

import org.junit.Assert.assertTrue
import org.junit.Test

class ModelDownloadResumeTest {
    @Test
    fun http416WhileResumingRestartsFromZero() {
        assertTrue(
            RangeResumePolicy.shouldRestartFromZero(
                responseCode = 416,
                resumeOffset = 128L,
            ),
        )
    }
}
