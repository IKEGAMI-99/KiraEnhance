package com.ikegami99.kiraenhance.update

import java.io.File

enum class AppUpdateInstallStep {
    REQUEST_UNKNOWN_SOURCES,
    OPEN_PACKAGE_INSTALLER,
}

object AppUpdateInstallerPolicy {
    fun nextStep(canRequestPackageInstalls: Boolean): AppUpdateInstallStep =
        if (canRequestPackageInstalls) {
            AppUpdateInstallStep.OPEN_PACKAGE_INSTALLER
        } else {
            AppUpdateInstallStep.REQUEST_UNKNOWN_SOURCES
        }

    fun fileProviderAuthority(applicationId: String): String = "$applicationId.fileprovider"

    fun isTrustedApk(filesDir: File, apkFile: File): Boolean {
        if (!apkFile.isFile || !apkFile.name.endsWith(".apk", ignoreCase = true)) return false

        val updatesDir = File(filesDir, "updates").canonicalFile
        val candidate = apkFile.canonicalFile
        return candidate.parentFile == updatesDir
    }
}
