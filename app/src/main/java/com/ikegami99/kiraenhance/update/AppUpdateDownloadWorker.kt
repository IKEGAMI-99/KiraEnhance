package com.ikegami99.kiraenhance.update

import android.content.Context
import androidx.work.Data
import androidx.work.Worker
import androidx.work.WorkerParameters
import com.ikegami99.kiraenhance.download.RangeResumePolicy
import com.ikegami99.kiraenhance.util.Sha256
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import okhttp3.OkHttpClient
import okhttp3.Request

class AppUpdateDownloadWorker(
    appContext: Context,
    workerParams: WorkerParameters,
) : Worker(appContext, workerParams) {
    private val client = OkHttpClient()

    override fun doWork(): Result {
        val versionCode = inputData.getInt(KEY_VERSION_CODE, -1)
        val versionName = inputData.getString(KEY_VERSION_NAME).orEmpty()
        val apkUrl = inputData.getString(KEY_APK_URL).orEmpty()
        val expectedSha = inputData.getString(KEY_APK_SHA256).orEmpty()

        if (versionCode <= 0 || versionName.isBlank()) return failure("Invalid update version")
        if (!apkUrl.startsWith("https://", ignoreCase = true)) return failure("Invalid update APK URL")
        if (!SHA_256.matches(expectedSha)) return failure("Invalid update APK SHA-256")

        val updatesDir = File(applicationContext.filesDir, UPDATES_DIRECTORY)
        val finalFile = File(updatesDir, "KiraEnhance-$versionCode.apk")
        val partFile = File(updatesDir, "KiraEnhance-$versionCode.apk.part")

        return try {
            updatesDir.mkdirs()

            if (finalFile.isFile && Sha256.matches(finalFile, expectedSha)) {
                return Result.success(successData(finalFile))
            }
            if (finalFile.exists()) finalFile.delete()

            download(apkUrl, expectedSha, partFile, finalFile)
        } catch (_: IOException) {
            Result.retry()
        } catch (error: RuntimeException) {
            failure(error.message ?: "App update download failed")
        }
    }

    private fun download(
        apkUrl: String,
        expectedSha: String,
        partFile: File,
        finalFile: File,
    ): Result {
        var resumeOffset = if (partFile.isFile) partFile.length() else 0L

        while (true) {
            val requestBuilder = Request.Builder().url(apkUrl)
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
                    return failure("Update download failed with HTTP ${response.code}")
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
                val responseLength = body.contentLength()
                val totalBytes = when {
                    responseLength < 0L -> 0L
                    append -> resumeOffset + responseLength
                    else -> responseLength
                }

                FileOutputStream(partFile, append).use { output ->
                    body.byteStream().use { input ->
                        val buffer = ByteArray(BUFFER_SIZE)
                        var downloaded = resumeOffset
                        var lastReportedBytes = downloaded
                        var lastReportedAt = System.nanoTime()

                        setProgressAsync(progressData(downloaded, totalBytes))

                        while (true) {
                            if (isStopped) throw IOException("Update download stopped")
                            val read = input.read(buffer)
                            if (read < 0) break
                            if (read == 0) continue

                            output.write(buffer, 0, read)
                            downloaded += read

                            val now = System.nanoTime()
                            val enoughBytes = downloaded - lastReportedBytes >= PROGRESS_BYTES
                            val enoughTime = now - lastReportedAt >= PROGRESS_NANOS
                            if (enoughBytes && enoughTime) {
                                setProgressAsync(progressData(downloaded, totalBytes))
                                lastReportedBytes = downloaded
                                lastReportedAt = now
                            }
                        }
                        output.fd.sync()
                    }
                }

                if (!Sha256.matches(partFile, expectedSha)) {
                    partFile.delete()
                    return failure("Update APK SHA-256 verification failed")
                }

                activate(partFile, finalFile)
                return Result.success(successData(finalFile))
            }
        }
    }

    private fun activate(partFile: File, finalFile: File) {
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

    private fun progressData(downloaded: Long, total: Long): Data = Data.Builder()
        .putLong(KEY_BYTES_DOWNLOADED, downloaded)
        .putLong(KEY_TOTAL_BYTES, total)
        .build()

    private fun successData(file: File): Data = Data.Builder()
        .putLong(KEY_BYTES_DOWNLOADED, file.length())
        .putLong(KEY_TOTAL_BYTES, file.length())
        .putString(KEY_APK_PATH, file.absolutePath)
        .build()

    private fun failure(message: String): Result =
        Result.failure(Data.Builder().putString(KEY_ERROR, message).build())

    companion object {
        const val KEY_VERSION_CODE = "versionCode"
        const val KEY_VERSION_NAME = "versionName"
        const val KEY_APK_URL = "apkUrl"
        const val KEY_APK_SHA256 = "apkSha256"
        const val KEY_BYTES_DOWNLOADED = "bytesDownloaded"
        const val KEY_TOTAL_BYTES = "totalBytes"
        const val KEY_APK_PATH = "apkPath"
        const val KEY_ERROR = "error"

        private const val UPDATES_DIRECTORY = "updates"
        private const val BUFFER_SIZE = 128 * 1024
        private const val PROGRESS_BYTES = 1024L * 1024L
        private const val PROGRESS_NANOS = 250_000_000L
        private val SHA_256 = Regex("^[A-Fa-f0-9]{64}$")
    }
}
