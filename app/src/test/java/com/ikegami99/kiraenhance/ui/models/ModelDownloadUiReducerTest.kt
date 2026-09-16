package com.ikegami99.kiraenhance.ui.models

import androidx.work.WorkInfo
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class ModelDownloadUiReducerTest {
    @Test
    fun queuedBlockedAndRetryWaitRemainDistinct() {
        assertEquals(
            ModelDownloadState.QUEUED,
            ModelDownloadUiReducer.stateFor(WorkInfo.State.ENQUEUED, runAttemptCount = 0),
        )
        assertEquals(
            ModelDownloadState.RETRY_WAIT,
            ModelDownloadUiReducer.stateFor(WorkInfo.State.ENQUEUED, runAttemptCount = 1),
        )
        assertEquals(
            ModelDownloadState.BLOCKED,
            ModelDownloadUiReducer.stateFor(WorkInfo.State.BLOCKED, runAttemptCount = 0),
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
            "ダウンロード再試行を待っています",
            ModelDownloadUiReducer.waitingMessage(ModelDownloadState.RETRY_WAIT, wifiOnly = false),
        )
        assertEquals(
            "前提となる処理の完了を待っています",
            ModelDownloadUiReducer.waitingMessage(ModelDownloadState.BLOCKED, wifiOnly = false),
        )
        assertNull(
            ModelDownloadUiReducer.waitingMessage(ModelDownloadState.RUNNING, wifiOnly = false),
        )
    }

    @Test
    fun disablingWifiOnlyReplacesWaitingWorkAndPreservesPartialFiles() {
        val queued = ModelDownloadUiReducer.restartDecision(
            previousWifiOnly = true,
            newWifiOnly = false,
            state = ModelDownloadState.QUEUED,
        )
        assertEquals(
            ModelDownloadRestartDecision(
                replaceExisting = true,
                deleteLocalFiles = false,
            ),
            queued,
        )

        val blocked = ModelDownloadUiReducer.restartDecision(
            previousWifiOnly = true,
            newWifiOnly = false,
            state = ModelDownloadState.BLOCKED,
        )
        assertEquals(
            ModelDownloadRestartDecision(
                replaceExisting = true,
                deleteLocalFiles = false,
            ),
            blocked,
        )

        val retryWait = ModelDownloadUiReducer.restartDecision(
            previousWifiOnly = true,
            newWifiOnly = false,
            state = ModelDownloadState.RETRY_WAIT,
        )
        assertEquals(
            ModelDownloadRestartDecision(
                replaceExisting = true,
                deleteLocalFiles = false,
            ),
            retryWait,
        )

        assertNull(
            ModelDownloadUiReducer.restartDecision(
                previousWifiOnly = true,
                newWifiOnly = false,
                state = ModelDownloadState.RUNNING,
            ),
        )
    }

    @Test
    fun modelManagerDefaultsWifiOnlyOff() {
        assertEquals(false, ModelManagerUiState().wifiOnly)
    }
}
