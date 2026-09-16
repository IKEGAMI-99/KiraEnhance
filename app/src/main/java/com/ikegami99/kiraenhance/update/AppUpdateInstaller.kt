package com.ikegami99.kiraenhance.update

import java.io.File

interface AppUpdateInstallPlatform {
    val filesDir: File
    val applicationId: String

    fun canRequestPackageInstalls(): Boolean

    fun openUnknownSourcesSettings()

    fun openPackageInstaller(apkFile: File, authority: String)
}

class AppUpdateInstaller(
    private val platform: AppUpdateInstallPlatform,
) {
    fun begin(apkFile: File): AppUpdateInstallStep {
        require(AppUpdateInstallerPolicy.isTrustedApk(platform.filesDir, apkFile)) {
            "Untrusted update APK path"
        }

        val canonicalApk = apkFile.canonicalFile
        val step = AppUpdateInstallerPolicy.nextStep(platform.canRequestPackageInstalls())
        when (step) {
            AppUpdateInstallStep.REQUEST_UNKNOWN_SOURCES -> platform.openUnknownSourcesSettings()
            AppUpdateInstallStep.OPEN_PACKAGE_INSTALLER -> platform.openPackageInstaller(
                apkFile = canonicalApk,
                authority = AppUpdateInstallerPolicy.fileProviderAuthority(platform.applicationId),
            )
        }
        return step
    }
}
