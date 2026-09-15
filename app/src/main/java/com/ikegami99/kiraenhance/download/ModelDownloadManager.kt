package com.ikegami99.kiraenhance.download

import android.content.Context
import androidx.work.Constraints
import androidx.work.Data
import androidx.work.ExistingWorkPolicy
import androidx.work.NetworkType
import androidx.work.OneTimeWorkRequest
import androidx.work.WorkManager
import com.ikegami99.kiraenhance.model.ModelDescriptor
import java.util.UUID

class ModelDownloadManager(
    context: Context,
    private val workManager: WorkManager = WorkManager.getInstance(context.applicationContext),
) {
    fun enqueue(model: ModelDescriptor, wifiOnly: Boolean): UUID {
        val constraints = Constraints.Builder()
            .setRequiredNetworkType(networkTypeFor(wifiOnly))
            .build()

        val input = Data.Builder()
            .putString(ModelDownloadWorker.KEY_MODEL_ID, model.id)
            .putString(ModelDownloadWorker.KEY_VERSION, model.version)
            .putString(ModelDownloadWorker.KEY_URL, model.downloadUrl)
            .putString(ModelDownloadWorker.KEY_SHA256, model.sha256)
            .putLong(ModelDownloadWorker.KEY_FILE_SIZE_BYTES, model.fileSizeBytes)
            .build()

        val request = OneTimeWorkRequest.Builder(ModelDownloadWorker::class.java)
            .setConstraints(constraints)
            .setInputData(input)
            .addTag("model-download")
            .addTag("model-download-${model.id}")
            .build()

        workManager.enqueueUniqueWork(
            workName(model),
            ExistingWorkPolicy.KEEP,
            request,
        )
        return request.id
    }

    companion object {
        internal fun networkTypeFor(wifiOnly: Boolean): NetworkType =
            if (wifiOnly) NetworkType.UNMETERED else NetworkType.CONNECTED

        internal fun workName(model: ModelDescriptor): String =
            "model-download-${model.id}-${model.version}"
    }
}
