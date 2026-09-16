package com.ikegami99.kiraenhance.ui.models

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class ModelCardStatusTextTest {
    @Test
    fun queuedWifiOnlyUsesSpecificWaitReason() {
        assertEquals(
            "Wi‑Fiのみ設定のため、非従量制ネットワークを待っています",
            ModelCardStatusText.textFor(
                downloadState = ModelDownloadState.QUEUED,
                wifiOnly = true,
                progressPercent = 0,
                downloadedText = "0 B",
                totalText = "31.9 MB",
            ),
        )
    }

    @Test
    fun blockedUsesPrerequisiteWaitReason() {
        assertEquals(
            "前提となる処理の完了を待っています",
            ModelCardStatusText.textFor(
                downloadState = ModelDownloadState.BLOCKED,
                wifiOnly = false,
                progressPercent = 0,
                downloadedText = "0 B",
                totalText = "31.9 MB",
            ),
        )
    }

    @Test
    fun runningUsesProgressText() {
        assertEquals(
            "42% • 13.4 MB / 31.9 MB",
            ModelCardStatusText.textFor(
                downloadState = ModelDownloadState.RUNNING,
                wifiOnly = false,
                progressPercent = 42,
                downloadedText = "13.4 MB",
                totalText = "31.9 MB",
            ),
        )
    }

    @Test
    fun idleHasNoStatusText() {
        assertNull(
            ModelCardStatusText.textFor(
                downloadState = ModelDownloadState.IDLE,
                wifiOnly = false,
                progressPercent = 0,
                downloadedText = "0 B",
                totalText = "31.9 MB",
            ),
        )
    }
}
