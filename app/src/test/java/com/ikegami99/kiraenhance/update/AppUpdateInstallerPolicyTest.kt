package com.ikegami99.kiraenhance.update

import java.io.File
import java.nio.file.Files
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class AppUpdateInstallerPolicyTest {
    @Test
    fun `requests unknown source permission before launching installer`() {
        assertEquals(
            AppUpdateInstallStep.REQUEST_UNKNOWN_SOURCES,
            AppUpdateInstallerPolicy.nextStep(canRequestPackageInstalls = false),
        )
    }

    @Test
    fun `launches package installer when permission is already granted`() {
        assertEquals(
            AppUpdateInstallStep.OPEN_PACKAGE_INSTALLER,
            AppUpdateInstallerPolicy.nextStep(canRequestPackageInstalls = true),
        )
    }

    @Test
    fun `only accepts apk inside private updates directory`() {
        val filesDir = Files.createTempDirectory("kiraenhance-files").toFile()
        val updatesDir = File(filesDir, "updates").apply { mkdirs() }
        val trusted = File(updatesDir, "KiraEnhance-2.apk").apply { writeBytes(byteArrayOf(1)) }
        val sibling = File(filesDir, "KiraEnhance-2.apk").apply { writeBytes(byteArrayOf(1)) }
        val wrongExtension = File(updatesDir, "KiraEnhance-2.tmp").apply { writeBytes(byteArrayOf(1)) }

        assertTrue(AppUpdateInstallerPolicy.isTrustedApk(filesDir, trusted))
        assertFalse(AppUpdateInstallerPolicy.isTrustedApk(filesDir, sibling))
        assertFalse(AppUpdateInstallerPolicy.isTrustedApk(filesDir, wrongExtension))
    }

    @Test
    fun `file provider authority is scoped to application id`() {
        assertEquals(
            "com.ikegami99.kiraenhance.fileprovider",
            AppUpdateInstallerPolicy.fileProviderAuthority("com.ikegami99.kiraenhance"),
        )
    }
}
