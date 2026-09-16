package com.ikegami99.kiraenhance.update

import okhttp3.OkHttpClient
import okhttp3.Request

fun interface UpdateManifestSource {
    fun load(): String
}

class HttpUpdateManifestSource(
    private val client: OkHttpClient = OkHttpClient(),
    private val manifestUrl: String = DEFAULT_UPDATE_MANIFEST_URL,
) : UpdateManifestSource {
    override fun load(): String {
        val request = Request.Builder()
            .url(manifestUrl)
            .build()

        client.newCall(request).execute().use { response ->
            require(response.isSuccessful) {
                "Update check failed with HTTP ${response.code}"
            }
            return response.body.string()
        }
    }
}

sealed interface AppUpdateCheckResult {
    data class Available(val info: AppUpdateInfo) : AppUpdateCheckResult
    data class UpToDate(val latestVersionName: String) : AppUpdateCheckResult
}

class AppUpdateChecker(
    private val source: UpdateManifestSource,
    private val parser: AppUpdateManifestParser = AppUpdateManifestParser(),
) {
    fun check(currentVersionCode: Int): AppUpdateCheckResult {
        require(currentVersionCode > 0) { "currentVersionCode must be positive" }
        val info = parser.parse(source.load())
        return if (info.versionCode > currentVersionCode) {
            AppUpdateCheckResult.Available(info)
        } else {
            AppUpdateCheckResult.UpToDate(info.versionName)
        }
    }
}

const val DEFAULT_UPDATE_MANIFEST_URL =
    "https://github.com/IKEGAMI-99/KiraEnhance/releases/latest/download/update-manifest.json"
