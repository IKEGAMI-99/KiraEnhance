package com.ikegami99.kiraenhance.ui.models

import androidx.work.WorkInfo
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class ModelDownloadUiReducerTest {
    @Test
    fun queuedAndBlockedRemainDistinct() {
        assertEquals(
            ModelDownloadState.QUEUED,
            ModelDownloadUiReducer.stateFor(WorkInfo.State.ENQUEUED),
        )
        assertEquals(
            ModelDownloadState.BLOCKED,
            ModelDownloadUiReducer.stateFor(WorkInfo.State.BLOCKED),
        )
    }

    @Test
    fun waitingMessagesExplainWhyWorkHasNotStarted() {
        assertEquals(
            "Wi‑Fiのみ設定のため、非従量制ネットワークを待っています",
            ModelDownloadUiReducer.waitingMessage(ModelDownloadState.QUEUED, wifiOnly = true),
        )
        assertEquals(
            "ネットワーク接続または実行開始を待っています",
            ModelDownloadUiReducer.waitingMessage(ModelDownloadState.QUEUED, wifiOnly = false),
        )
        assertEquals(
            "前提となる処理の完了を待っています",
            ModelDownloadUiReducer.waitingMessage(ModelDownloadState.BLOCKED, wifiOnly = false),
        )
        assertNull(
            ModelDownloadUiReducer.waitingMessage(ModelDownloadState.RUNNING, wifiOnly = false),
        )
    }
}
