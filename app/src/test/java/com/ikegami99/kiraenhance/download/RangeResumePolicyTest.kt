package com.ikegami99.kiraenhance.download

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class RangeResumePolicyTest {
    @Test
    fun http416RestartsOnlyWhenResuming() {
        assertTrue(RangeResumePolicy.shouldRestartFromZero(416, 128L))
        assertFalse(RangeResumePolicy.shouldRestartFromZero(416, 0L))
        assertFalse(RangeResumePolicy.shouldRestartFromZero(206, 128L))
    }
}
