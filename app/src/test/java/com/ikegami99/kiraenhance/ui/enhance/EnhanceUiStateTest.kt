package com.ikegami99.kiraenhance.ui.enhance

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class EnhanceUiStateTest {
    @Test
    fun `ready image becomes processing with 4x target size`() {
        val ready = EnhanceUiReducer.reduce(
            EnhanceUiState(),
            EnhanceEvent.ImageReady(width = 1080, height = 1920),
        )
        val processing = EnhanceUiReducer.reduce(ready, EnhanceEvent.ProcessingStarted)

        assertEquals(EnhanceStage.PROCESSING, processing.stage)
        assertEquals(4320, processing.targetWidth)
        assertEquals(7680, processing.targetHeight)
        assertEquals(0f, processing.progressFraction)
    }

    @Test
    fun `tile progress is clamped and elapsed time is retained`() {
        val processing = EnhanceUiReducer.reduce(
            EnhanceUiReducer.reduce(
                EnhanceUiState(),
                EnhanceEvent.ImageReady(width = 100, height = 200),
            ),
            EnhanceEvent.ProcessingStarted,
        )

        val progressed = EnhanceUiReducer.reduce(
            processing,
            EnhanceEvent.Progress(completedTiles = 7, totalTiles = 5, elapsedMs = 1234),
        )

        assertEquals(1f, progressed.progressFraction)
        assertEquals(1234L, progressed.elapsedMs)
        assertEquals("5 / 5 tiles", progressed.progressText)
    }

    @Test
    fun `completed session can move to saved state`() {
        val completed = EnhanceUiReducer.reduce(
            EnhanceUiState(stage = EnhanceStage.PROCESSING),
            EnhanceEvent.ProcessingCompleted(elapsedMs = 2500),
        )
        val saved = EnhanceUiReducer.reduce(
            completed,
            EnhanceEvent.Saved("Pictures/KiraEnhance/result.png"),
        )

        assertEquals(EnhanceStage.SAVED, saved.stage)
        assertTrue(saved.status.contains("保存"))
        assertEquals("Pictures/KiraEnhance/result.png", saved.savedLocation)
    }

    @Test
    fun `failure keeps actionable message`() {
        val failed = EnhanceUiReducer.reduce(
            EnhanceUiState(stage = EnhanceStage.PROCESSING),
            EnhanceEvent.Failed("モデルを読み込めません"),
        )

        assertEquals(EnhanceStage.ERROR, failed.stage)
        assertEquals("モデルを読み込めません", failed.status)
    }
}
