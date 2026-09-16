package com.ikegami99.kiraenhance.update

import android.content.Context
import androidx.work.Constraints
import androidx.work.Data
import androidx.work.ExistingWorkPolicy
import androidx.work.NetworkType
import androidx.work.OneTimeWorkRequest
import androidx.work.WorkManager
import java.util.UUID

class AppUpdateDownloadManager(
    context: Context,
    private val workManager: WorkManager = WorkManager.getInstance(context.applicationContext),
) {
    fun enqueue(info: AppUpdateInfo): UUID {
        val request = OneTimeWorkRequest.Builder(AppUpdateDownloadWorker::class.java)
            .setConstraints(
                Constraints.Builder()
                    .setRequiredNetworkType(requiredNetworkType())
                    .build(),
            )
            .setInputData(inputDataFor(info))
            .addTag("app-update-download")
            .addTag("app-update-download-${info.versionCode}")
            .build()

        workManager.enqueueUniqueWork(
            workName(info.versionCode),
            workPolicy(),
            request,
        )
        return request.id
    }

    companion object {
        internal fun requiredNetworkType(): NetworkType = NetworkType.CONNECTED

        internal fun workPolicy(): ExistingWorkPolicy = ExistingWorkPolicy.REPLACE

        internal fun workName(versionCode: Int): String = "app-update-download-$versionCode"

        internal fun inputDataFor(info: AppUpdateInfo): Data = Data.Builder()
            .putInt(AppUpdateDownloadWorker.KEY_VERSION_CODE, info.versionCode)
            .putString(AppUpdateDownloadWorker.KEY_VERSION_NAME, info.versionName)
            .putString(AppUpdateDownloadWorker.KEY_APK_URL, info.apkUrl)
            .putString(AppUpdateDownloadWorker.KEY_APK_SHA256, info.apkSha256)
            .build()
    }
}
