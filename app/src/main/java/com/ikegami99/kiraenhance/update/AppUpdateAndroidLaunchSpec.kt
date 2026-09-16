package com.ikegami99.kiraenhance.update

data class AndroidLaunchSpec(
    val action: String,
    val data: String,
    val mimeType: String? = null,
    val newTask: Boolean = true,
    val grantReadPermission: Boolean = false,
)

object AppUpdateAndroidLaunchSpec {
    fun unknownSources(applicationId: String): AndroidLaunchSpec = AndroidLaunchSpec(
        action = "android.settings.MANAGE_UNKNOWN_APP_SOURCES",
        data = "package:$applicationId",
    )

    fun packageInstaller(contentUri: String): AndroidLaunchSpec = AndroidLaunchSpec(
        action = "android.intent.action.VIEW",
        data = contentUri,
        mimeType = "application/vnd.android.package-archive",
        grantReadPermission = true,
    )
}
