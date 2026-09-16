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

    fun exportTo(uri: Uri): Result<Unit> = runCatching {
        val output = appContext.contentResolver.openOutputStream(uri, "w")
            ?: error("保存先を開けませんでした")
        output.use { store.copyTo(it) }
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
