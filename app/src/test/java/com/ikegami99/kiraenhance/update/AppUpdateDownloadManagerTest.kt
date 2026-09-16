package com.ikegami99.kiraenhance.update

import androidx.work.ExistingWorkPolicy
import androidx.work.NetworkType
import org.junit.Assert.assertEquals
import org.junit.Test

class AppUpdateDownloadManagerTest {
    private val info = AppUpdateInfo(
        versionName = "0.1.0-alpha02",
        versionCode = 2,
        releaseNotes = "test",
        apkUrl = "https://example.com/KiraEnhance.apk",
        apkSha256 = "a".repeat(64),
    )

    @Test
    fun updateDownloadUsesConnectedNetworkAndReplacement() {
        assertEquals(NetworkType.CONNECTED, AppUpdateDownloadManager.requiredNetworkType())
        assertEquals(ExistingWorkPolicy.REPLACE, AppUpdateDownloadManager.workPolicy())
        assertEquals("app-update-download-2", AppUpdateDownloadManager.workName(versionCode = 2))
    }

    @Test
    fun serializesVerifiedReleaseMetadataForWorker() {
        val data = AppUpdateDownloadManager.inputDataFor(info)

        assertEquals(2, data.getInt(AppUpdateDownloadWorker.KEY_VERSION_CODE, -1))
        assertEquals("0.1.0-alpha02", data.getString(AppUpdateDownloadWorker.KEY_VERSION_NAME))
        assertEquals(info.apkUrl, data.getString(AppUpdateDownloadWorker.KEY_APK_URL))
        assertEquals(info.apkSha256, data.getString(AppUpdateDownloadWorker.KEY_APK_SHA256))
    }
}
