package com.ikegami99.kiraenhance.update

import com.squareup.moshi.JsonDataException
import com.squareup.moshi.Moshi
import com.squareup.moshi.kotlin.reflect.KotlinJsonAdapterFactory
import java.net.URI

class AppUpdateManifestParser(
    moshi: Moshi = Moshi.Builder()
        .addLast(KotlinJsonAdapterFactory())
        .build(),
) {
    private val adapter = moshi.adapter(AppUpdateInfo::class.java)

    fun parse(json: String): AppUpdateInfo {
        val info = try {
            adapter.fromJson(json)
                ?: throw IllegalArgumentException("Update manifest is empty")
        } catch (error: IllegalArgumentException) {
            throw error
        } catch (error: JsonDataException) {
            throw IllegalArgumentException("Invalid update manifest", error)
        } catch (error: RuntimeException) {
            throw IllegalArgumentException("Invalid update manifest", error)
        }

        validate(info)
        return info
    }

    private fun validate(info: AppUpdateInfo) {
        require(info.versionName.isNotBlank()) { "versionName must not be blank" }
        require(info.versionCode > 0) { "versionCode must be positive" }
        require(info.releaseNotes.isNotBlank()) { "releaseNotes must not be blank" }
        require(isHttpsUrl(info.apkUrl)) { "apkUrl must use HTTPS" }
        require(SHA_256.matches(info.apkSha256)) { "apkSha256 must be a 64-character hexadecimal SHA-256" }
    }

    private fun isHttpsUrl(value: String): Boolean = runCatching {
        val uri = URI(value)
        uri.scheme.equals("https", ignoreCase = true) && !uri.host.isNullOrBlank()
    }.getOrDefault(false)

    private companion object {
        val SHA_256 = Regex("^[A-Fa-f0-9]{64}$")
    }
}
