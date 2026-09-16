package com.ikegami99.kiraenhance.ui.update

import androidx.work.WorkInfo

enum class AppUpdateStage {
    IDLE,
    CHECKING,
    UP_TO_DATE,
    AVAILABLE,
    DOWNLOADING,
    READY_TO_INSTALL,
    ERROR,
}

internal object AppUpdateUiReducer {
    fun stageFor(workState: WorkInfo.State): AppUpdateStage = when (workState) {
        WorkInfo.State.ENQUEUED,
        WorkInfo.State.BLOCKED,
        WorkInfo.State.RUNNING,
        -> AppUpdateStage.DOWNLOADING

        WorkInfo.State.SUCCEEDED -> AppUpdateStage.READY_TO_INSTALL
        WorkInfo.State.FAILED -> AppUpdateStage.ERROR
        WorkInfo.State.CANCELLED -> AppUpdateStage.AVAILABLE
    }

    fun progressPercent(downloadedBytes: Long, totalBytes: Long): Int {
        if (totalBytes <= 0L) return 0
        return ((downloadedBytes.coerceAtLeast(0L) * 100L) / totalBytes)
            .coerceIn(0L, 100L)
            .toInt()
    }
}
