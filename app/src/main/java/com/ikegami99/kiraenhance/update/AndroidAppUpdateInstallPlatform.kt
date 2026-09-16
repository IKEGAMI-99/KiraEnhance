package com.ikegami99.kiraenhance.update

import android.content.ClipData
import android.content.Context
import android.content.Intent
import android.net.Uri
import androidx.core.content.FileProvider
import java.io.File

class AndroidAppUpdateInstallPlatform(
    context: Context,
) : AppUpdateInstallPlatform {
    private val appContext = context.applicationContext

    override val filesDir: File
        get() = appContext.filesDir

    override val applicationId: String
        get() = appContext.packageName

    override fun canRequestPackageInstalls(): Boolean =
        appContext.packageManager.canRequestPackageInstalls()

    override fun openUnknownSourcesSettings() {
        launch(AppUpdateAndroidLaunchSpec.unknownSources(applicationId))
    }

    override fun openPackageInstaller(apkFile: File, authority: String) {
        val contentUri = FileProvider.getUriForFile(
            appContext,
            authority,
            apkFile,
        )
        launch(
            spec = AppUpdateAndroidLaunchSpec.packageInstaller(contentUri.toString()),
            clipUri = contentUri,
        )
    }

    private fun launch(
        spec: AndroidLaunchSpec,
        clipUri: Uri? = null,
    ) {
        val dataUri = Uri.parse(spec.data)
        val intent = Intent(spec.action).apply {
            if (spec.mimeType != null) {
                setDataAndType(dataUri, spec.mimeType)
            } else {
                data = dataUri
            }
            if (spec.newTask) addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            if (spec.grantReadPermission) addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            if (clipUri != null) {
                clipData = ClipData.newRawUri("KiraEnhance update", clipUri)
            }
        }
        appContext.startActivity(intent)
    }
}
