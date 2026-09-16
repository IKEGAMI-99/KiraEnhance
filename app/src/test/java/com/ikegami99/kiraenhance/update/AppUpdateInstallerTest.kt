package com.ikegami99.kiraenhance.update

import java.io.File
import java.nio.file.Files
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class AppUpdateInstallerTest {
    @Test
    fun `opens unknown sources settings when install permission is missing`() {
        val filesDir = Files.createTempDirectory("kiraenhance-files").toFile()
        val apk = trustedApk(filesDir)
        val platform = FakeInstallPlatform(filesDir, canInstall = false)
        val installer = AppUpdateInstaller(platform)

        val step = installer.begin(apk)

        assertEquals(AppUpdateInstallStep.REQUEST_UNKNOWN_SOURCES, step)
        assertEquals(1, platform.settingsLaunches)
        assertTrue(platform.installerLaunches.isEmpty())
    }

    @Test
    fun `opens package installer with app scoped file provider when permission exists`() {
        val filesDir = Files.createTempDirectory("kiraenhance-files").toFile()
        val apk = trustedApk(filesDir)
        val platform = FakeInstallPlatform(filesDir, canInstall = true)
        val installer = AppUpdateInstaller(platform)

        val step = installer.begin(apk)

        assertEquals(AppUpdateInstallStep.OPEN_PACKAGE_INSTALLER, step)
        assertEquals(0, platform.settingsLaunches)
        assertEquals(
            listOf(apk.canonicalFile to "com.ikegami99.kiraenhance.fileprovider"),
            platform.installerLaunches,
        )
    }

    @Test
    fun `rejects apk outside private update directory`() {
        val filesDir = Files.createTempDirectory("kiraenhance-files").toFile()
        val outside = File(filesDir, "outside.apk").apply { writeBytes(byteArrayOf(1)) }
        val platform = FakeInstallPlatform(filesDir, canInstall = true)
        val installer = AppUpdateInstaller(platform)

        assertThrows(IllegalArgumentException::class.java) {
            installer.begin(outside)
        }
        assertTrue(platform.installerLaunches.isEmpty())
    }

    private fun trustedApk(filesDir: File): File {
        val updates = File(filesDir, "updates").apply { mkdirs() }
        return File(updates, "KiraEnhance-2.apk").apply { writeBytes(byteArrayOf(1)) }
    }

    private class FakeInstallPlatform(
        override val filesDir: File,
        private val canInstall: Boolean,
    ) : AppUpdateInstallPlatform {
        override val applicationId: String = "com.ikegami99.kiraenhance"
        var settingsLaunches: Int = 0
        val installerLaunches = mutableListOf<Pair<File, String>>()

        override fun canRequestPackageInstalls(): Boolean = canInstall

        override fun openUnknownSourcesSettings() {
            settingsLaunches += 1
        }

        override fun openPackageInstaller(apkFile: File, authority: String) {
            installerLaunches += apkFile.canonicalFile to authority
        }
    }
}
