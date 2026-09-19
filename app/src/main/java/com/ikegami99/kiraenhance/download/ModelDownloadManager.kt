package com.ikegami99.kiraenhance.download

import android.content.Context
import androidx.work.Constraints
import androidx.work.Data
import androidx.work.ExistingWorkPolicy
import androidx.work.NetworkType
import androidx.work.OneTimeWorkRequest
import androidx.work.WorkManager
import com.ikegami99.kiraenhance.diagnostics.AppDiagnosticLogger
import com.ikegami99.kiraenhance.model.ModelDescriptor
import com.ikegami99.kiraenhance.model.requireDownloadable
import java.util.UUID

class ModelDownloadManager(
    context: Context,
    private val workManager: WorkManager = WorkManager.getInstance(context.applicationContext),
) {
    private val logger = AppDiagnosticLogger.get(context)

    fun enqueue(
        model: ModelDescriptor,
        wifiOnly: Boolean,
        replaceExisting: Boolean = false,
    ): UUID {
        validateForDownload(model)

        val networkType = networkTypeFor(wifiOnly)
        val constraints = Constraints.Builder()
            .setRequiredNetworkType(networkType)
            .build()

        val request = OneTimeWorkRequest.Builder(ModelDownloadWorker::class.java)
            .setConstraints(constraints)
            .setInputData(inputDataFor(model))
            .addTag("model-download")
            .addTag("model-download-${model.id}")
            .build()
        val policy = workPolicyFor(replaceExisting)

        logger.log(
            "ModelDownloadManager",
            "enqueue model=${model.id} version=${model.version} request=${request.id} " +
                "wifiOnly=$wifiOnly network=$networkType policy=$policy artifacts=${model.artifacts.size}",
        )

        workManager.enqueueUniqueWork(
            workName(model),
            policy,
            request,
        )
        return request.id
    }

    companion object {
        internal fun validateForDownload(model: ModelDescriptor) {
            model.requireDownloadable()
        }

        internal fun networkTypeFor(wifiOnly: Boolean): NetworkType =
            if (wifiOnly) NetworkType.UNMETERED else NetworkType.CONNECTED

        internal fun workName(model: ModelDescriptor): String =
            "model-download-${model.id}-${model.version}"

        internal fun workPolicyFor(replaceExisting: Boolean): ExistingWorkPolicy =
            ExistingWorkPolicy.REPLACE

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
