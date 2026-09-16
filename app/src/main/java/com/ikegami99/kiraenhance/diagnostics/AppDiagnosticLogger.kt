package com.ikegami99.kiraenhance.diagnostics

import android.content.Context
import android.net.Uri
import android.os.Build
import android.util.Log
import java.io.File

class AppDiagnosticLogger private constructor(
    private val appContext: Context,
) {
    private val store = DiagnosticLogStore(
        File(appContext.filesDir, "diagnostics/kiraenhance.log"),
    )

    fun log(tag: String, message: String) {
        runCatching {
            store.append(tag, message)
        }.onFailure { error ->
            Log.e("KiraEnhanceLog", "Failed to persist diagnostic event", error)
        }
    }

    fun exportTo(uri: Uri): Result<Long> = runCatching {
        val resolver = appContext.contentResolver
        val writtenBytes = resolver.openOutputStream(uri, "w")
            ?.use { output ->
                val count = store.copyTo(output)
                output.flush()
                count
            }
            ?: error("保存先を開けませんでした")

        check(writtenBytes > 0L) { "保存するログが0バイトです" }

        val verifiedBytes = resolver.openInputStream(uri)
            ?.use { input ->
                val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
                var total = 0L
                while (true) {
                    val read = input.read(buffer)
                    if (read < 0) break
                    total += read
                }
                total
            }
            ?: error("保存したログを再読込できませんでした")

        check(verifiedBytes == writtenBytes) {
            "保存後のログ検証に失敗しました: wrote=$writtenBytes read=$verifiedBytes"
        }
        check(verifiedBytes > 0L) { "保存したログが0バイトです" }
        verifiedBytes
    }

    fun exportFileName(): String = DiagnosticLogStore.exportFileName()

    fun logAppSession() {
        val packageInfo = runCatching {
            appContext.packageManager.getPackageInfo(appContext.packageName, 0)
        }.getOrNull()
        log(
            "App",
            "session version=${packageInfo?.versionName ?: "unknown"} " +
                "android=${Build.VERSION.SDK_INT} device=${Build.MANUFACTURER}/${Build.MODEL}",
        )
    }

    companion object {
        @Volatile
        private var instance: AppDiagnosticLogger? = null

        fun get(context: Context): AppDiagnosticLogger = instance ?: synchronized(this) {
            instance ?: AppDiagnosticLogger(context.applicationContext).also { instance = it }
        }
    }
}
