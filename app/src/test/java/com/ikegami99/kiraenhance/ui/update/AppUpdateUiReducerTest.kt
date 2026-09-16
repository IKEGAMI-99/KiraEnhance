package com.ikegami99.kiraenhance.ui.update

import androidx.work.WorkInfo
import org.junit.Assert.assertEquals
import org.junit.Test

class AppUpdateUiReducerTest {
    @Test
    fun `work states map to update stages`() {
        assertEquals(AppUpdateStage.DOWNLOADING, AppUpdateUiReducer.stageFor(WorkInfo.State.ENQUEUED))
        assertEquals(AppUpdateStage.DOWNLOADING, AppUpdateUiReducer.stageFor(WorkInfo.State.BLOCKED))
        assertEquals(AppUpdateStage.DOWNLOADING, AppUpdateUiReducer.stageFor(WorkInfo.State.RUNNING))
        assertEquals(AppUpdateStage.READY_TO_INSTALL, AppUpdateUiReducer.stageFor(WorkInfo.State.SUCCEEDED))
        assertEquals(AppUpdateStage.ERROR, AppUpdateUiReducer.stageFor(WorkInfo.State.FAILED))
        assertEquals(AppUpdateStage.AVAILABLE, AppUpdateUiReducer.stageFor(WorkInfo.State.CANCELLED))
    }

    @Test
    fun `progress percent clamps to zero through one hundred`() {
        assertEquals(50, AppUpdateUiReducer.progressPercent(50L, 100L))
        assertEquals(100, AppUpdateUiReducer.progressPercent(150L, 100L))
        assertEquals(0, AppUpdateUiReducer.progressPercent(-10L, 100L))
        assertEquals(0, AppUpdateUiReducer.progressPercent(10L, 0L))
    }

    @Test
    fun `primary action follows update stage`() {
        assertEquals(AppUpdatePrimaryAction.CHECK, AppUpdateUiReducer.primaryActionFor(AppUpdateStage.IDLE))
        assertEquals(AppUpdatePrimaryAction.NONE, AppUpdateUiReducer.primaryActionFor(AppUpdateStage.CHECKING))
        assertEquals(AppUpdatePrimaryAction.CHECK, AppUpdateUiReducer.primaryActionFor(AppUpdateStage.UP_TO_DATE))
        assertEquals(AppUpdatePrimaryAction.DOWNLOAD, AppUpdateUiReducer.primaryActionFor(AppUpdateStage.AVAILABLE))
        assertEquals(AppUpdatePrimaryAction.NONE, AppUpdateUiReducer.primaryActionFor(AppUpdateStage.DOWNLOADING))
        assertEquals(AppUpdatePrimaryAction.INSTALL, AppUpdateUiReducer.primaryActionFor(AppUpdateStage.READY_TO_INSTALL))
        assertEquals(AppUpdatePrimaryAction.CHECK, AppUpdateUiReducer.primaryActionFor(AppUpdateStage.ERROR))
    }
}
