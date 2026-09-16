package com.ikegami99.kiraenhance.update

import org.junit.Assert.assertEquals
import org.junit.Test

class AppUpdateCheckerTest {
    private val newerJson = """
        {
          "versionName": "0.1.0-alpha02",
          "versionCode": 2,
          "releaseNotes": "test",
          "apkUrl": "https://example.com/app.apk",
          "apkSha256": "${"b".repeat(64)}"
        }
    """.trimIndent()

    @Test
    fun reportsAvailableWhenRemoteVersionCodeIsHigher() {
        val checker = AppUpdateChecker(UpdateManifestSource { newerJson })
        val result = checker.check(currentVersionCode = 1)
        assertEquals(2, (result as AppUpdateCheckResult.Available).info.versionCode)
    }

    @Test
    fun reportsUpToDateWhenVersionCodeMatches() {
        val checker = AppUpdateChecker(UpdateManifestSource { newerJson })
        val result = checker.check(currentVersionCode = 2)
        assertEquals("0.1.0-alpha02", (result as AppUpdateCheckResult.UpToDate).latestVersionName)
    }

    @Test
    fun reportsUpToDateWhenInstalledVersionIsNewer() {
        val checker = AppUpdateChecker(UpdateManifestSource { newerJson })
        val result = checker.check(currentVersionCode = 3)
        assertEquals("0.1.0-alpha02", (result as AppUpdateCheckResult.UpToDate).latestVersionName)
    }
}
