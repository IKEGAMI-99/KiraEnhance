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
    fun enqueue(
        model: ModelDescriptor,
        wifiOnly: Boolean,
        replaceExisting: Boolean = false,
    ): UUID {
        val constraints = Constraints.Builder()
            .setRequiredNetworkType(networkTypeFor(wifiOnly))
            .build()

        val request = OneTimeWorkRequest.Builder(ModelDownloadWorker::class.java)
            .setConstraints(constraints)
            .setInputData(inputDataFor(model))
            .addTag("model-download")
            .addTag("model-download-${model.id}")
            .build()

        workManager.enqueueUniqueWork(
            workName(model),
            workPolicyFor(replaceExisting),
            request,
        )
        return request.id
    }

    companion object {
        internal fun networkTypeFor(wifiOnly: Boolean): NetworkType =
            if (wifiOnly) NetworkType.UNMETERED else NetworkType.CONNECTED

        internal fun workName(model: ModelDescriptor): String =
            "model-download-${model.id}-${model.version}"

        internal fun workPolicyFor(replaceExisting: Boolean): ExistingWorkPolicy =
            if (replaceExisting) ExistingWorkPolicy.REPLACE else ExistingWorkPolicy.KEEP

        internal fun inputDataFor(model: ModelDescriptor): Data {
            val builder = Data.Builder()
                .putString(ModelDownloadWorker.KEY_MODEL_ID, model.id)
                .putString(ModelDownloadWorker.KEY_VERSION, model.version)
                .putInt(ModelDownloadWorker.KEY_ARTIFACT_COUNT, model.artifacts.size)

            model.artifacts.forEachIndexed { index, artifact ->
                builder
                    .putString(ModelDownloadWorker.artifactFileNameKey(index), artifact.fileName)
                    .putString(ModelDownloadWorker.artifactUrlKey(index), artifact.downloadUrl)
                    .putString(ModelDownloadWorker.artifactSha256Key(index), artifact.sha256)
                    .putLong(ModelDownloadWorker.artifactSizeKey(index), artifact.fileSizeBytes)
            }

            return builder.build()
        }
    }
}
