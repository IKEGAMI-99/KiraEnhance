package com.ikegami99.kiraenhance.download

import android.content.Context
import androidx.work.Data
import androidx.work.Worker
import androidx.work.WorkerParameters
import com.ikegami99.kiraenhance.model.InstalledModelStore
import com.ikegami99.kiraenhance.util.Sha256
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.lang.Math.addExact
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import okhttp3.OkHttpClient
import okhttp3.Request

class ModelDownloadWorker(
    appContext: Context,
    workerParams: WorkerParameters,
) : Worker(appContext, workerParams) {
    private val client = OkHttpClient()
    private val store = InstalledModelStore(appContext)

    override fun doWork(): Result {
        val modelId = inputData.getString(KEY_MODEL_ID) ?: return failure("Missing model id")
        val version = inputData.getString(KEY_VERSION) ?: return failure("Missing model version")
        val artifacts = readArtifacts() ?: return failure("Invalid model artifact metadata")
        val totalExpected = try {
            artifacts.fold(0L) { total, artifact -> addExact(total, artifact.expectedSize) }
        } catch (_: ArithmeticException) {
            return failure("Model bundle is too large")
        }

        return try {
            var completedBytes = 0L
            artifacts.forEach { artifact ->
                val finalFile = try {
                    store.artifactFile(modelId, version, artifact.fileName)
                } catch (_: IllegalArgumentException) {
                    return failure("Unsafe model path")
                }
                val partFile = try {
                    store.partialFile(modelId, version, artifact.fileName)
                } catch (_: IllegalArgumentException) {
                    return failure("Unsafe model path")
                }

                if (isValidCompletedFile(finalFile, artifact.expectedSize, artifact.expectedSha)) {
                    completedBytes = addExact(completedBytes, artifact.expectedSize)
                    setProgressAsync(progressData(modelId, completedBytes, totalExpected))
                    return@forEach
                }

                if (preparePartialFile(partFile, artifact, finalFile)) {
                    completedBytes = addExact(completedBytes, artifact.expectedSize)
                    setProgressAsync(progressData(modelId, completedBytes, totalExpected))
                    return@forEach
                }

                downloadArtifact(
                    modelId = modelId,
                    artifact = artifact,
                    partFile = partFile,
                    baseCompletedBytes = completedBytes,
                    totalBundleBytes = totalExpected,
                )?.let { return it }

                if (partFile.length() != artifact.expectedSize) {
                    return if (partFile.length() < artifact.expectedSize) {
                        Result.retry()
                    } else {
                        partFile.delete()
                        failure("Downloaded artifact is larger than expected: ${artifact.fileName}")
                    }
                }

                if (!Sha256.matches(partFile, artifact.expectedSha)) {
                    partFile.delete()
                    return failure("SHA-256 verification failed: ${artifact.fileName}")
                }

                activate(partFile, finalFile)
                completedBytes = addExact(completedBytes, artifact.expectedSize)
                setProgressAsync(progressData(modelId, completedBytes, totalExpected))
            }

            Result.success(successData(modelId, totalExpected))
        } catch (_: IOException) {
            Result.retry()
        } catch (error: RuntimeException) {
            failure(error.message ?: "Model download failed")
        }
    }

    private fun readArtifacts(): List<DownloadArtifact>? {
        val count = inputData.getInt(KEY_ARTIFACT_COUNT, -1)
        if (count !in 1..MAX_ARTIFACTS) return null

        val artifacts = ArrayList<DownloadArtifact>(count)
        for (index in 0 until count) {
            val fileName = inputData.getString(artifactFileNameKey(index)) ?: return null
            val url = inputData.getString(artifactUrlKey(index)) ?: return null
            val sha256 = inputData.getString(artifactSha256Key(index)) ?: return null
            val size = inputData.getLong(artifactSizeKey(index), -1L)
            if (size <= 0L || !url.startsWith("https://", ignoreCase = true)) return null
            if (!SHA_256.matches(sha256)) return null

            artifacts += DownloadArtifact(
                fileName = fileName,
                url = url,
                expectedSize = size,
                expectedSha = sha256,
            )
        }
        if (artifacts.map { it.fileName }.distinct().size != artifacts.size) return null
        return artifacts
    }

    private fun preparePartialFile(
        partFile: File,
        artifact: DownloadArtifact,
        finalFile: File,
    ): Boolean {
        partFile.parentFile?.mkdirs()
        if (!partFile.exists()) return false

        if (partFile.length() > artifact.expectedSize) {
            partFile.delete()
            return false
        }

        if (partFile.length() == artifact.expectedSize) {
            if (Sha256.matches(partFile, artifact.expectedSha)) {
                activate(partFile, finalFile)
                return true
            }
            partFile.delete()
        }
        return false
    }

    private fun downloadArtifact(
        modelId: String,
        artifact: DownloadArtifact,
        partFile: File,
        baseCompletedBytes: Long,
        totalBundleBytes: Long,
    ): Result? {
        var resumeOffset = if (partFile.exists()) partFile.length() else 0L

        while (true) {
            val requestBuilder = Request.Builder().url(artifact.url)
            if (resumeOffset > 0L) {
                requestBuilder.header("Range", "bytes=$resumeOffset-")
            }

            client.newCall(requestBuilder.build()).execute().use { response ->
                if (RangeResumePolicy.shouldRestartFromZero(response.code, resumeOffset)) {
                    partFile.delete()
                    resumeOffset = 0L
                    return@use
                }

                if (response.code == 408 || response.code == 429 || response.code in 500..599) {
                    return Result.retry()
                }
                if (response.code !in listOf(200, 206)) {
                    return failure("Download failed with HTTP ${response.code}: ${artifact.fileName}")
                }

                var append = resumeOffset > 0L && response.code == 206
                if (resumeOffset > 0L && response.code == 200) {
                    resumeOffset = 0L
                    append = false
                }

                if (append) {
                    val contentRange = response.header("Content-Range")
                    if (contentRange == null || !contentRange.startsWith("bytes $resumeOffset-")) {
                        partFile.delete()
                        return Result.retry()
                    }
                }

                val body = response.body
                FileOutputStream(partFile, append).use { output ->
                    body.byteStream().use { input ->
                        val buffer = ByteArray(BUFFER_SIZE)
                        var downloaded = resumeOffset
                        var lastReportedBytes = resumeOffset
                        var lastReportedAt = System.nanoTime()

                        while (true) {
                            if (isStopped) throw IOException("Download stopped")
                            val read = input.read(buffer)
                            if (read < 0) break
                            if (read == 0) continue

                            output.write(buffer, 0, read)
                            downloaded += read
                            if (downloaded > artifact.expectedSize) {
                                output.flush()
                                partFile.delete()
                                return failure("Downloaded artifact is larger than expected: ${artifact.fileName}")
                            }

                            val now = System.nanoTime()
                            val enoughBytes = downloaded - lastReportedBytes >= PROGRESS_BYTES
                            val enoughTime = now - lastReportedAt >= PROGRESS_NANOS
                            if (enoughBytes && enoughTime) {
                                setProgressAsync(
                                    progressData(
                                        modelId = modelId,
                                        downloaded = baseCompletedBytes + downloaded,
                                        total = totalBundleBytes,
                                    ),
                                )
                                lastReportedBytes = downloaded
                                lastReportedAt = now
                            }
                        }
                        output.fd.sync()
                    }
                }
                return null
            }
        }
    }

    private fun isValidCompletedFile(file: File, expectedSize: Long, expectedSha: String): Boolean =
        file.isFile && file.length() == expectedSize && Sha256.matches(file, expectedSha)

    private fun activate(partFile: File, finalFile: File) {
        finalFile.parentFile?.mkdirs()
        val source = partFile.toPath()
        val target = finalFile.toPath()
        try {
            Files.move(
                source,
                target,
                StandardCopyOption.ATOMIC_MOVE,
                StandardCopyOption.REPLACE_EXISTING,
            )
        } catch (_: AtomicMoveNotSupportedException) {
            Files.move(source, target, StandardCopyOption.REPLACE_EXISTING)
        }
    }

    private fun progressData(modelId: String, downloaded: Long, total: Long): Data =
        Data.Builder()
            .putString(KEY_PROGRESS_MODEL_ID, modelId)
            .putLong(KEY_BYTES_DOWNLOADED, downloaded)
            .putLong(KEY_TOTAL_BYTES, total)
            .build()

    private fun successData(modelId: String, total: Long): Data =
        Data.Builder()
            .putString(KEY_PROGRESS_MODEL_ID, modelId)
            .putLong(KEY_BYTES_DOWNLOADED, total)
            .putLong(KEY_TOTAL_BYTES, total)
            .build()

    private fun failure(message: String): Result =
        Result.failure(Data.Builder().putString(KEY_ERROR, message).build())

    private data class DownloadArtifact(
        val fileName: String,
        val url: String,
        val expectedSize: Long,
        val expectedSha: String,
    )

    companion object {
        const val KEY_MODEL_ID = "modelId"
        const val KEY_VERSION = "version"
        const val KEY_ARTIFACT_COUNT = "artifactCount"

        const val KEY_PROGRESS_MODEL_ID = "modelId"
        const val KEY_BYTES_DOWNLOADED = "bytesDownloaded"
        const val KEY_TOTAL_BYTES = "totalBytes"
        const val KEY_ERROR = "error"

        internal fun artifactFileNameKey(index: Int) = "artifactFileName_$index"
        internal fun artifactUrlKey(index: Int) = "artifactUrl_$index"
        internal fun artifactSha256Key(index: Int) = "artifactSha256_$index"
        internal fun artifactSizeKey(index: Int) = "artifactSize_$index"

        private const val MAX_ARTIFACTS = 16
        private const val BUFFER_SIZE = 128 * 1024
        private const val PROGRESS_BYTES = 1024L * 1024L
        private const val PROGRESS_NANOS = 250_000_000L
        private val SHA_256 = Regex("^[A-Fa-f0-9]{64}$")
    }
}
