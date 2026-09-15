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
        val url = inputData.getString(KEY_URL) ?: return failure("Missing download URL")
        val expectedSha = inputData.getString(KEY_SHA256) ?: return failure("Missing SHA-256")
        val expectedSize = inputData.getLong(KEY_FILE_SIZE_BYTES, -1L)
        if (expectedSize <= 0L) return failure("Invalid model file size")
        if (!url.startsWith("https://", ignoreCase = true)) return failure("Model URL must use HTTPS")

        val finalFile = try {
            store.modelFile(modelId, version)
        } catch (error: IllegalArgumentException) {
            return failure("Unsafe model path")
        }
        val partFile = try {
            store.partialFile(modelId, version)
        } catch (error: IllegalArgumentException) {
            return failure("Unsafe model path")
        }

        return try {
            if (isValidCompletedFile(finalFile, expectedSize, expectedSha)) {
                return Result.success(successData(modelId, expectedSize))
            }

            preparePartialFile(partFile, expectedSize, expectedSha, finalFile, modelId)?.let {
                return it
            }

            download(
                modelId = modelId,
                url = url,
                expectedSize = expectedSize,
                partFile = partFile,
            )?.let { return it }

            if (partFile.length() != expectedSize) {
                return if (partFile.length() < expectedSize) {
                    Result.retry()
                } else {
                    partFile.delete()
                    failure("Downloaded file is larger than expected")
                }
            }

            if (!Sha256.matches(partFile, expectedSha)) {
                partFile.delete()
                return failure("SHA-256 verification failed")
            }

            activate(partFile, finalFile)
            setProgressAsync(progressData(modelId, expectedSize, expectedSize))
            Result.success(successData(modelId, expectedSize))
        } catch (error: IOException) {
            Result.retry()
        } catch (error: RuntimeException) {
            failure(error.message ?: "Model download failed")
        }
    }

    private fun preparePartialFile(
        partFile: File,
        expectedSize: Long,
        expectedSha: String,
        finalFile: File,
        modelId: String,
    ): Result? {
        partFile.parentFile?.mkdirs()
        if (!partFile.exists()) return null

        if (partFile.length() > expectedSize) {
            partFile.delete()
            return null
        }

        if (partFile.length() == expectedSize) {
            if (Sha256.matches(partFile, expectedSha)) {
                activate(partFile, finalFile)
                return Result.success(successData(modelId, expectedSize))
            }
            partFile.delete()
        }
        return null
    }

    private fun download(
        modelId: String,
        url: String,
        expectedSize: Long,
        partFile: File,
    ): Result? {
        var resumeOffset = if (partFile.exists()) partFile.length() else 0L
        val requestBuilder = Request.Builder().url(url)
        if (resumeOffset > 0L) {
            requestBuilder.header("Range", "bytes=$resumeOffset-")
        }

        client.newCall(requestBuilder.build()).execute().use { response ->
            if (response.code == 408 || response.code == 429 || response.code in 500..599) {
                return Result.retry()
            }
            if (response.code !in listOf(200, 206)) {
                return failure("Download failed with HTTP ${response.code}")
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

                        val now = System.nanoTime()
                        val enoughBytes = downloaded - lastReportedBytes >= PROGRESS_BYTES
                        val enoughTime = now - lastReportedAt >= PROGRESS_NANOS
                        if (enoughBytes && enoughTime) {
                            setProgressAsync(progressData(modelId, downloaded, expectedSize))
                            lastReportedBytes = downloaded
                            lastReportedAt = now
                        }
                    }
                    output.fd.sync()
                }
            }
        }
        return null
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

    companion object {
        const val KEY_MODEL_ID = "modelId"
        const val KEY_VERSION = "version"
        const val KEY_URL = "url"
        const val KEY_SHA256 = "sha256"
        const val KEY_FILE_SIZE_BYTES = "fileSizeBytes"

        const val KEY_PROGRESS_MODEL_ID = "modelId"
        const val KEY_BYTES_DOWNLOADED = "bytesDownloaded"
        const val KEY_TOTAL_BYTES = "totalBytes"
        const val KEY_ERROR = "error"

        private const val BUFFER_SIZE = 128 * 1024
        private const val PROGRESS_BYTES = 1024L * 1024L
        private const val PROGRESS_NANOS = 250_000_000L
    }
}
