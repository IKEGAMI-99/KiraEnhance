# In-App Update Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let KiraEnhance manually check GitHub Releases, download and verify a stable-signed APK, and hand it to Android's system package installer from inside the app.

**Architecture:** GitHub Releases hosts `update-manifest.json`, the signed APK, and its checksum. The app uses a small manifest parser/checker, a WorkManager APK downloader with SHA-256 verification, a narrow FileProvider, and a dedicated Compose update screen. Release signing exists only in GitHub Actions secrets; debug CI remains unsigned with the normal debug key.

**Tech Stack:** Kotlin, OkHttp 5.3.2, Moshi 1.15.2, WorkManager 2.11.2, Android FileProvider/package installer intents, Compose Material 3, GitHub Actions, JUnit 4.

**Spec:** `docs/superpowers/specs/2026-09-16-model-download-and-app-update-design.md`

## Global Constraints

- Android `minSdk = 28`, `targetSdk = 36`.
- Update checks are manual only.
- Update source of truth is GitHub Releases.
- Manifest URL is `https://github.com/IKEGAMI-99/KiraEnhance/releases/latest/download/update-manifest.json`.
- Manifest fields are `versionName`, `versionCode`, `releaseNotes`, `apkUrl`, and `apkSha256`.
- APKs keep application ID `com.ikegami99.kiraenhance` and one stable release signing certificate.
- Android's system installer confirmation remains mandatory.
- A checksum mismatch deletes the APK and never enables Install.
- Required GitHub Secrets are `KIRA_RELEASE_KEYSTORE_B64`, `KIRA_RELEASE_STORE_PASSWORD`, `KIRA_RELEASE_KEY_ALIAS`, and `KIRA_RELEASE_KEY_PASSWORD`.

---

### Task 1: Define, parse, and compare update metadata

**Files:**
- Create: `app/src/main/java/com/ikegami99/kiraenhance/update/AppUpdateInfo.kt`
- Create: `app/src/main/java/com/ikegami99/kiraenhance/update/AppUpdateManifestParser.kt`
- Create: `app/src/main/java/com/ikegami99/kiraenhance/update/AppUpdateChecker.kt`
- Create: `app/src/test/java/com/ikegami99/kiraenhance/update/AppUpdateManifestParserTest.kt`
- Create: `app/src/test/java/com/ikegami99/kiraenhance/update/AppUpdateCheckerTest.kt`

**Interfaces:**
- Produces: `AppUpdateInfo(versionName: String, versionCode: Int, releaseNotes: String, apkUrl: String, apkSha256: String)`.
- Produces: `UpdateManifestSource.load(): String`.
- Produces: `AppUpdateCheckResult.Available(info: AppUpdateInfo)` and `AppUpdateCheckResult.UpToDate(latestVersionName: String)`.
- Produces: `AppUpdateChecker.check(currentVersionCode: Int): AppUpdateCheckResult`.

- [ ] **Step 1: Write failing parser and comparison tests**

```kotlin
package com.ikegami99.kiraenhance.update

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class AppUpdateManifestParserTest {
    private val parser = AppUpdateManifestParser()

    @Test
    fun parsesValidManifest() {
        val info = parser.parse(
            """
            {
              "versionName": "0.1.0-alpha02",
              "versionCode": 2,
              "releaseNotes": "Updater test build",
              "apkUrl": "https://github.com/IKEGAMI-99/KiraEnhance/releases/download/v0.1.0-alpha02/KiraEnhance-0.1.0-alpha02.apk",
              "apkSha256": "${"a".repeat(64)}"
            }
            """.trimIndent(),
        )
        assertEquals("0.1.0-alpha02", info.versionName)
        assertEquals(2, info.versionCode)
        assertEquals("a".repeat(64), info.apkSha256)
    }

    @Test
    fun rejectsNonHttpsApkUrl() {
        assertThrows(IllegalArgumentException::class.java) {
            parser.parse(
                """
                {
                  "versionName": "0.1.0-alpha02",
                  "versionCode": 2,
                  "releaseNotes": "test",
                  "apkUrl": "http://example.invalid/app.apk",
                  "apkSha256": "${"a".repeat(64)}"
                }
                """.trimIndent(),
            )
        }
    }
}
```

```kotlin
package com.ikegami99.kiraenhance.update

import org.junit.Assert.assertEquals
import org.junit.Test

class AppUpdateCheckerTest {
    private val newerJson = """
        {
          "versionName": "0.1.0-alpha02",
          "versionCode": 2,
          "releaseNotes": "test",
          "apkUrl": "https://example.com/app.apk",
          "apkSha256": "${"b".repeat(64)}"
        }
    """.trimIndent()

    @Test
    fun higherVersionCodeIsAvailable() {
        val checker = AppUpdateChecker(UpdateManifestSource { newerJson })
        val result = checker.check(currentVersionCode = 1)
        assertEquals(2, (result as AppUpdateCheckResult.Available).info.versionCode)
    }

    @Test
    fun sameVersionCodeIsUpToDate() {
        val checker = AppUpdateChecker(UpdateManifestSource { newerJson })
        val result = checker.check(currentVersionCode = 2)
        assertEquals("0.1.0-alpha02", (result as AppUpdateCheckResult.UpToDate).latestVersionName)
    }
}
```

- [ ] **Step 2: Run tests and verify RED**

```bash
gradle :app:testDebugUnitTest --tests 'com.ikegami99.kiraenhance.update.AppUpdateManifestParserTest' --tests 'com.ikegami99.kiraenhance.update.AppUpdateCheckerTest' --stacktrace
```

Expected: compilation failure because the update classes do not exist.

- [ ] **Step 3: Implement the metadata model and parser**

`AppUpdateInfo.kt`:

```kotlin
package com.ikegami99.kiraenhance.update

data class AppUpdateInfo(
    val versionName: String,
    val versionCode: Int,
    val releaseNotes: String,
    val apkUrl: String,
    val apkSha256: String,
)
```

`AppUpdateManifestParser.kt`:

```kotlin
package com.ikegami99.kiraenhance.update

import com.squareup.moshi.JsonDataException
import com.squareup.moshi.Moshi
import com.squareup.moshi.kotlin.reflect.KotlinJsonAdapterFactory
import java.net.URI

class AppUpdateManifestParser(
    moshi: Moshi = Moshi.Builder().addLast(KotlinJsonAdapterFactory()).build(),
) {
    private val adapter = moshi.adapter(AppUpdateInfo::class.java)

    fun parse(json: String): AppUpdateInfo {
        val info = try {
            adapter.fromJson(json) ?: throw IllegalArgumentException("Update manifest is empty")
        } catch (error: IllegalArgumentException) {
            throw error
        } catch (error: JsonDataException) {
            throw IllegalArgumentException("Invalid update manifest", error)
        } catch (error: RuntimeException) {
            throw IllegalArgumentException("Invalid update manifest", error)
        }
        require(info.versionName.isNotBlank()) { "versionName must not be blank" }
        require(info.versionCode > 0) { "versionCode must be positive" }
        require(info.apkSha256.matches(Regex("^[A-Fa-f0-9]{64}$"))) { "Invalid APK SHA-256" }
        val uri = URI(info.apkUrl)
        require(uri.scheme.equals("https", ignoreCase = true) && !uri.host.isNullOrBlank()) {
            "APK URL must use HTTPS"
        }
        return info
    }
}
```

- [ ] **Step 4: Implement HTTP source and version-code comparison**

`AppUpdateChecker.kt`:

```kotlin
package com.ikegami99.kiraenhance.update

import okhttp3.OkHttpClient
import okhttp3.Request

const val DEFAULT_UPDATE_MANIFEST_URL =
    "https://github.com/IKEGAMI-99/KiraEnhance/releases/latest/download/update-manifest.json"

fun interface UpdateManifestSource {
    fun load(): String
}

class HttpUpdateManifestSource(
    private val client: OkHttpClient = OkHttpClient(),
    private val manifestUrl: String = DEFAULT_UPDATE_MANIFEST_URL,
) : UpdateManifestSource {
    override fun load(): String {
        val request = Request.Builder().url(manifestUrl).build()
        client.newCall(request).execute().use { response ->
            require(response.isSuccessful) { "Update check failed with HTTP ${response.code}" }
            return response.body.string()
        }
    }
}

sealed interface AppUpdateCheckResult {
    data class Available(val info: AppUpdateInfo) : AppUpdateCheckResult
    data class UpToDate(val latestVersionName: String) : AppUpdateCheckResult
}

fun interface AppUpdateCheckUseCase {
    fun check(currentVersionCode: Int): AppUpdateCheckResult
}

class AppUpdateChecker(
    private val source: UpdateManifestSource,
    private val parser: AppUpdateManifestParser = AppUpdateManifestParser(),
) : AppUpdateCheckUseCase {
    override fun check(currentVersionCode: Int): AppUpdateCheckResult {
        val info = parser.parse(source.load())
        return if (info.versionCode > currentVersionCode) {
            AppUpdateCheckResult.Available(info)
        } else {
            AppUpdateCheckResult.UpToDate(info.versionName)
        }
    }
}
```

- [ ] **Step 5: Run tests and verify GREEN**

```bash
gradle :app:testDebugUnitTest --tests 'com.ikegami99.kiraenhance.update.AppUpdateManifestParserTest' --tests 'com.ikegami99.kiraenhance.update.AppUpdateCheckerTest' --stacktrace
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add app/src/main/java/com/ikegami99/kiraenhance/update app/src/test/java/com/ikegami99/kiraenhance/update
git commit -m "feat: add app update manifest checker"
```

---

### Task 2: Download, resume, and verify the update APK

**Files:**
- Create: `app/src/main/java/com/ikegami99/kiraenhance/download/RangeResumePolicy.kt`
- Modify: `app/src/main/java/com/ikegami99/kiraenhance/download/ModelDownloadWorker.kt`
- Modify: `app/src/test/java/com/ikegami99/kiraenhance/download/ModelDownloadResumeTest.kt`
- Create: `app/src/test/java/com/ikegami99/kiraenhance/download/RangeResumePolicyTest.kt`
- Create: `app/src/main/java/com/ikegami99/kiraenhance/update/AppUpdateDownloadManager.kt`
- Create: `app/src/main/java/com/ikegami99/kiraenhance/update/AppUpdateDownloadWorker.kt`
- Create: `app/src/test/java/com/ikegami99/kiraenhance/update/AppUpdateDownloadManagerTest.kt`

**Interfaces:**
- Produces: `RangeResumePolicy.shouldRestartFromZero(responseCode: Int, resumeOffset: Long): Boolean`.
- Produces: `AppUpdateDownloadManager.enqueue(info: AppUpdateInfo): UUID`.
- Produces worker output/progress keys `KEY_BYTES_DOWNLOADED`, `KEY_TOTAL_BYTES`, `KEY_APK_PATH`, and `KEY_ERROR`.

- [ ] **Step 1: Write failing Range policy and manager tests**

```kotlin
package com.ikegami99.kiraenhance.download

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class RangeResumePolicyTest {
    @Test
    fun http416RestartsOnlyWhenResuming() {
        assertTrue(RangeResumePolicy.shouldRestartFromZero(416, 128L))
        assertFalse(RangeResumePolicy.shouldRestartFromZero(416, 0L))
        assertFalse(RangeResumePolicy.shouldRestartFromZero(206, 128L))
    }
}
```

```kotlin
package com.ikegami99.kiraenhance.update

import androidx.work.NetworkType
import org.junit.Assert.assertEquals
import org.junit.Test

class AppUpdateDownloadManagerTest {
    private val info = AppUpdateInfo(
        versionName = "0.1.0-alpha02",
        versionCode = 2,
        releaseNotes = "test",
        apkUrl = "https://example.com/app.apk",
        apkSha256 = "c".repeat(64),
    )

    @Test
    fun updateWorkUsesConnectedNetworkAndStableName() {
        assertEquals(NetworkType.CONNECTED, AppUpdateDownloadManager.requiredNetworkType())
        assertEquals("app-update-download-2", AppUpdateDownloadManager.workName(2))
        val data = AppUpdateDownloadManager.inputDataFor(info)
        assertEquals(info.apkUrl, data.getString(AppUpdateDownloadWorker.KEY_APK_URL))
        assertEquals(info.apkSha256, data.getString(AppUpdateDownloadWorker.KEY_APK_SHA256))
        assertEquals(2, data.getInt(AppUpdateDownloadWorker.KEY_VERSION_CODE, -1))
    }
}
```

- [ ] **Step 2: Run tests and verify RED**

```bash
gradle :app:testDebugUnitTest --tests 'com.ikegami99.kiraenhance.download.RangeResumePolicyTest' --tests 'com.ikegami99.kiraenhance.update.AppUpdateDownloadManagerTest' --stacktrace
```

Expected: compilation failure because the new policy and update manager do not exist.

- [ ] **Step 3: Extract the shared Range policy and keep the model regression green**

`RangeResumePolicy.kt`:

```kotlin
package com.ikegami99.kiraenhance.download

internal object RangeResumePolicy {
    fun shouldRestartFromZero(responseCode: Int, resumeOffset: Long): Boolean =
        responseCode == 416 && resumeOffset > 0L
}
```

Change `ModelDownloadWorker` to call `RangeResumePolicy.shouldRestartFromZero(response.code, resumeOffset)`. Change `ModelDownloadResumeTest` to assert the shared policy instead of the former companion helper.

- [ ] **Step 4: Implement the update download manager**

`AppUpdateDownloadManager.kt`:

```kotlin
package com.ikegami99.kiraenhance.update

import android.content.Context
import androidx.work.Constraints
import androidx.work.Data
import androidx.work.ExistingWorkPolicy
import androidx.work.NetworkType
import androidx.work.OneTimeWorkRequest
import androidx.work.WorkManager
import java.util.UUID

class AppUpdateDownloadManager(
    context: Context,
    private val workManager: WorkManager = WorkManager.getInstance(context.applicationContext),
) {
    fun enqueue(info: AppUpdateInfo): UUID {
        val request = OneTimeWorkRequest.Builder(AppUpdateDownloadWorker::class.java)
            .setConstraints(
                Constraints.Builder()
                    .setRequiredNetworkType(requiredNetworkType())
                    .build(),
            )
            .setInputData(inputDataFor(info))
            .addTag("app-update-download")
            .build()
        workManager.enqueueUniqueWork(
            workName(info.versionCode),
            ExistingWorkPolicy.REPLACE,
            request,
        )
        return request.id
    }

    companion object {
        internal fun requiredNetworkType(): NetworkType = NetworkType.CONNECTED
        internal fun workName(versionCode: Int): String = "app-update-download-$versionCode"
        internal fun inputDataFor(info: AppUpdateInfo): Data = Data.Builder()
            .putString(AppUpdateDownloadWorker.KEY_APK_URL, info.apkUrl)
            .putString(AppUpdateDownloadWorker.KEY_APK_SHA256, info.apkSha256)
            .putInt(AppUpdateDownloadWorker.KEY_VERSION_CODE, info.versionCode)
            .build()
    }
}
```

- [ ] **Step 5: Implement the update worker**

`AppUpdateDownloadWorker.kt`:

```kotlin
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
        val url = inputData.getString(KEY_APK_URL) ?: return failure("Missing APK URL")
        val sha = inputData.getString(KEY_APK_SHA256) ?: return failure("Missing APK SHA-256")
        val versionCode = inputData.getInt(KEY_VERSION_CODE, -1)
        if (versionCode <= 0 || !url.startsWith("https://", ignoreCase = true)) {
            return failure("Invalid update metadata")
        }
        val updateDir = File(applicationContext.filesDir, "updates").apply { mkdirs() }
        val finalFile = File(updateDir, "KiraEnhance-$versionCode.apk")
        val partFile = File(updateDir, "KiraEnhance-$versionCode.apk.part")

        return try {
            download(url, partFile)
            if (!Sha256.matches(partFile, sha)) {
                partFile.delete()
                return failure("APK SHA-256 verification failed")
            }
            activate(partFile, finalFile)
            Result.success(
                Data.Builder()
                    .putString(KEY_APK_PATH, finalFile.absolutePath)
                    .build(),
            )
        } catch (_: IOException) {
            Result.retry()
        }
    }

    private fun download(url: String, partFile: File) {
        var resumeOffset = if (partFile.isFile) partFile.length() else 0L
        while (true) {
            val builder = Request.Builder().url(url)
            if (resumeOffset > 0L) builder.header("Range", "bytes=$resumeOffset-")
            client.newCall(builder.build()).execute().use { response ->
                if (RangeResumePolicy.shouldRestartFromZero(response.code, resumeOffset)) {
                    partFile.delete()
                    resumeOffset = 0L
                    return@use
                }
                if (response.code == 408 || response.code == 429 || response.code in 500..599) {
                    throw IOException("Transient HTTP ${response.code}")
                }
                require(response.code == 200 || response.code == 206) {
                    "Update download failed with HTTP ${response.code}"
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
                        resumeOffset = 0L
                        return@use
                    }
                }
                val bodyLength = response.body.contentLength()
                val total = if (bodyLength >= 0L) resumeOffset + bodyLength else 0L
                FileOutputStream(partFile, append).use { output ->
                    response.body.byteStream().use { input ->
                        val buffer = ByteArray(128 * 1024)
                        var downloaded = resumeOffset
                        while (true) {
                            if (isStopped) throw IOException("Update download stopped")
                            val read = input.read(buffer)
                            if (read < 0) break
                            if (read == 0) continue
                            output.write(buffer, 0, read)
                            downloaded += read
                            setProgressAsync(
                                Data.Builder()
                                    .putLong(KEY_BYTES_DOWNLOADED, downloaded)
                                    .putLong(KEY_TOTAL_BYTES, total)
                                    .build(),
                            )
                        }
                        output.fd.sync()
                    }
                }
                return
            }
        }
    }

    private fun activate(partFile: File, finalFile: File) {
        try {
            Files.move(
                partFile.toPath(),
                finalFile.toPath(),
                StandardCopyOption.ATOMIC_MOVE,
                StandardCopyOption.REPLACE_EXISTING,
            )
        } catch (_: AtomicMoveNotSupportedException) {
            Files.move(partFile.toPath(), finalFile.toPath(), StandardCopyOption.REPLACE_EXISTING)
        }
    }

    private fun failure(message: String): Result =
        Result.failure(Data.Builder().putString(KEY_ERROR, message).build())

    companion object {
        const val KEY_APK_URL = "apkUrl"
        const val KEY_APK_SHA256 = "apkSha256"
        const val KEY_VERSION_CODE = "versionCode"
        const val KEY_BYTES_DOWNLOADED = "bytesDownloaded"
        const val KEY_TOTAL_BYTES = "totalBytes"
        const val KEY_APK_PATH = "apkPath"
        const val KEY_ERROR = "error"
    }
}
```

During implementation, convert `IllegalArgumentException` from non-200/206 responses into `Result.failure` in `doWork()` by adding:

```kotlin
} catch (error: IllegalArgumentException) {
    failure(error.message ?: "Update download failed")
} catch (_: IOException) {
    Result.retry()
}
```

- [ ] **Step 6: Run all unit tests and CI-equivalent compile**

```bash
gradle :app:assembleDebug :app:testDebugUnitTest :app:assembleDebugAndroidTest --stacktrace
```

Expected: PASS, including the existing model HTTP 416 regression.

- [ ] **Step 7: Commit**

```bash
git add app/src/main/java/com/ikegami99/kiraenhance/download app/src/main/java/com/ikegami99/kiraenhance/update app/src/test/java/com/ikegami99/kiraenhance/download app/src/test/java/com/ikegami99/kiraenhance/update
git commit -m "feat: download and verify app updates"
```

---

### Task 3: Add safe package-installer handoff

**Files:**
- Create: `app/src/main/java/com/ikegami99/kiraenhance/update/AppUpdateInstaller.kt`
- Create: `app/src/main/res/xml/update_file_paths.xml`
- Modify: `app/src/main/AndroidManifest.xml`
- Create: `app/src/test/java/com/ikegami99/kiraenhance/update/AppUpdateInstallPolicyTest.kt`

**Interfaces:**
- Produces: `AppUpdateInstallPolicy.action(canRequestPackageInstalls: Boolean): AppUpdateInstallAction`.
- Produces: `AppUpdatePathPolicy.isManagedFile(updateDir: File, candidate: File): Boolean`.
- Produces: `AppUpdateInstaller.installOrRequestPermission(apkFile: File)`.

- [ ] **Step 1: Write failing install-policy tests**

```kotlin
package com.ikegami99.kiraenhance.update

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class AppUpdateInstallPolicyTest {
    @Test
    fun permissionPolicyUsesSystemSettingsBeforeInstaller() {
        assertEquals(
            AppUpdateInstallAction.REQUEST_PERMISSION,
            AppUpdateInstallPolicy.action(canRequestPackageInstalls = false),
        )
        assertEquals(
            AppUpdateInstallAction.INSTALL,
            AppUpdateInstallPolicy.action(canRequestPackageInstalls = true),
        )
    }

    @Test
    fun installerAcceptsOnlyFilesInsideUpdateDirectory() {
        val root = File("build/test-updates").absoluteFile
        assertTrue(AppUpdatePathPolicy.isManagedFile(root, File(root, "KiraEnhance-2.apk")))
        assertFalse(AppUpdatePathPolicy.isManagedFile(root, File(root.parentFile, "outside.apk")))
    }
}
```

- [ ] **Step 2: Run test and verify RED**

```bash
gradle :app:testDebugUnitTest --tests 'com.ikegami99.kiraenhance.update.AppUpdateInstallPolicyTest' --stacktrace
```

Expected: install policy classes unresolved.

- [ ] **Step 3: Implement policy and installer**

`AppUpdateInstaller.kt`:

```kotlin
package com.ikegami99.kiraenhance.update

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.provider.Settings
import androidx.core.content.FileProvider
import java.io.File

enum class AppUpdateInstallAction { REQUEST_PERMISSION, INSTALL }

internal object AppUpdateInstallPolicy {
    fun action(canRequestPackageInstalls: Boolean): AppUpdateInstallAction =
        if (canRequestPackageInstalls) AppUpdateInstallAction.INSTALL
        else AppUpdateInstallAction.REQUEST_PERMISSION
}

internal object AppUpdatePathPolicy {
    fun isManagedFile(updateDir: File, candidate: File): Boolean = runCatching {
        val root = updateDir.canonicalFile
        val file = candidate.canonicalFile
        file.parentFile == root && file.name.endsWith(".apk", ignoreCase = true)
    }.getOrDefault(false)
}

class AppUpdateInstaller(private val context: Context) {
    fun installOrRequestPermission(apkFile: File) {
        val updateDir = File(context.filesDir, "updates")
        require(AppUpdatePathPolicy.isManagedFile(updateDir, apkFile)) { "Unsafe update APK path" }
        require(apkFile.isFile) { "Update APK does not exist" }

        when (AppUpdateInstallPolicy.action(context.packageManager.canRequestPackageInstalls())) {
            AppUpdateInstallAction.REQUEST_PERMISSION -> {
                context.startActivity(
                    Intent(
                        Settings.ACTION_MANAGE_UNKNOWN_APP_SOURCES,
                        Uri.parse("package:${context.packageName}"),
                    ).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK),
                )
            }
            AppUpdateInstallAction.INSTALL -> {
                val uri = FileProvider.getUriForFile(
                    context,
                    "${context.packageName}.updates",
                    apkFile,
                )
                context.startActivity(
                    Intent(Intent.ACTION_VIEW)
                        .setDataAndType(uri, "application/vnd.android.package-archive")
                        .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_ACTIVITY_NEW_TASK),
                )
            }
        }
    }
}
```

Add to `AndroidManifest.xml`:

```xml
<uses-permission android:name="android.permission.REQUEST_INSTALL_PACKAGES" />
```

Inside `<application>` add:

```xml
<provider
    android:name="androidx.core.content.FileProvider"
    android:authorities="${applicationId}.updates"
    android:exported="false"
    android:grantUriPermissions="true">
    <meta-data
        android:name="android.support.FILE_PROVIDER_PATHS"
        android:resource="@xml/update_file_paths" />
</provider>
```

`app/src/main/res/xml/update_file_paths.xml`:

```xml
<?xml version="1.0" encoding="utf-8"?>
<paths xmlns:android="http://schemas.android.com/apk/res/android">
    <files-path name="updates" path="updates/" />
</paths>
```

- [ ] **Step 4: Run tests and manifest compile**

```bash
gradle :app:testDebugUnitTest :app:assembleDebug --stacktrace
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/src/main/java/com/ikegami99/kiraenhance/update/AppUpdateInstaller.kt app/src/main/res/xml/update_file_paths.xml app/src/main/AndroidManifest.xml app/src/test/java/com/ikegami99/kiraenhance/update/AppUpdateInstallPolicyTest.kt
git commit -m "feat: hand verified updates to Android installer"
```

---

### Task 4: Add the update screen and navigation

**Files:**
- Create: `app/src/main/java/com/ikegami99/kiraenhance/ui/update/AppUpdateViewModel.kt`
- Create: `app/src/main/java/com/ikegami99/kiraenhance/ui/update/AppUpdateScreen.kt`
- Create: `app/src/main/java/com/ikegami99/kiraenhance/ui/update/AppUpdateRoute.kt`
- Modify: `app/src/main/java/com/ikegami99/kiraenhance/KiraEnhanceApp.kt`
- Modify: `app/src/main/java/com/ikegami99/kiraenhance/ui/home/HomeScreen.kt`
- Modify: `app/build.gradle.kts`
- Create: `app/src/androidTest/java/com/ikegami99/kiraenhance/ui/update/AppUpdateScreenTest.kt`

**Interfaces:**
- Produces: `AppUpdateUiState`.
- Produces: `AppUpdateViewModel.checkForUpdates()` and `AppUpdateViewModel.downloadUpdate()`.
- Produces: `AppUpdateRoute(onBack: () -> Unit)`.

- [ ] **Step 1: Enable BuildConfig and ViewModel coroutines**

In `app/build.gradle.kts`, add:

```kotlin
buildFeatures {
    compose = true
    buildConfig = true
}
```

and:

```kotlin
implementation("androidx.lifecycle:lifecycle-viewmodel-ktx:2.10.0")
```

- [ ] **Step 2: Create the screen state and ViewModel**

`AppUpdateViewModel.kt` uses this state:

```kotlin
data class AppUpdateUiState(
    val currentVersionName: String,
    val currentVersionCode: Int,
    val checking: Boolean = false,
    val availableUpdate: AppUpdateInfo? = null,
    val upToDate: Boolean = false,
    val downloading: Boolean = false,
    val bytesDownloaded: Long = 0L,
    val totalBytes: Long = 0L,
    val verifiedApkPath: String? = null,
    val errorMessage: String? = null,
)
```

The ViewModel constructor is:

```kotlin
class AppUpdateViewModel(
    private val currentVersionName: String,
    private val currentVersionCode: Int,
    private val checker: AppUpdateCheckUseCase,
    private val downloadManager: AppUpdateDownloadManager,
    private val workManager: WorkManager,
) : ViewModel()
```

`checkForUpdates()` sets `checking = true`, calls `checker.check(currentVersionCode)` inside `viewModelScope.launch(Dispatchers.IO)`, then updates state on the ViewModel scope. Available sets `availableUpdate`; UpToDate sets `upToDate = true`; exceptions set `errorMessage`.

`downloadUpdate()` requires `availableUpdate`, calls `downloadManager.enqueue(info)`, observes that WorkInfo ID, maps progress keys to state, and on `SUCCEEDED` reads `KEY_APK_PATH`. It never creates `verifiedApkPath` from an ENQUEUED/RUNNING/FAILED work item.

- [ ] **Step 3: Write the failing Compose UI test**

`AppUpdateScreenTest.kt` must render three states and assert:

```kotlin
composeRule.onNodeWithText("アプリ更新").assertIsDisplayed()
composeRule.onNodeWithText("現在 0.1.0-alpha01").assertIsDisplayed()
composeRule.onNodeWithText("アップデートを確認").assertIsDisplayed()
```

For an available update:

```kotlin
composeRule.onNodeWithText("0.1.0-alpha02").assertIsDisplayed()
composeRule.onNodeWithText("Updater test build").assertIsDisplayed()
composeRule.onNodeWithText("ダウンロード").assertIsDisplayed()
```

For a verified APK state:

```kotlin
composeRule.onNodeWithText("インストール").assertIsEnabled()
```

- [ ] **Step 4: Implement screen and route**

`AppUpdateScreen` signature:

```kotlin
@Composable
fun AppUpdateScreen(
    state: AppUpdateUiState,
    onBack: () -> Unit,
    onCheck: () -> Unit,
    onDownload: () -> Unit,
    onInstall: () -> Unit,
    modifier: Modifier = Modifier,
)
```

Render current version at top, a check button disabled while `checking`, latest version/release notes when `availableUpdate != null`, progress when `downloading`, `errorMessage` in error color, and `インストール` only when `verifiedApkPath != null`.

`AppUpdateRoute` constructs its factory using:

```kotlin
AppUpdateViewModel(
    currentVersionName = BuildConfig.VERSION_NAME,
    currentVersionCode = BuildConfig.VERSION_CODE,
    checker = AppUpdateChecker(HttpUpdateManifestSource()),
    downloadManager = AppUpdateDownloadManager(context.applicationContext),
    workManager = WorkManager.getInstance(context.applicationContext),
)
```

On Install, resolve `state.verifiedApkPath` to a `File` and call:

```kotlin
AppUpdateInstaller(context.applicationContext).installOrRequestPermission(File(path))
```

- [ ] **Step 5: Wire Home and navigation**

Add to `KiraEnhanceApp.kt`:

```kotlin
private const val UPDATE_ROUTE = "update"
```

Pass `onOpenAppUpdate = { navController.navigate(UPDATE_ROUTE) }` to Home and add:

```kotlin
composable(UPDATE_ROUTE) {
    AppUpdateRoute(onBack = { navController.popBackStack() })
}
```

Add `onOpenAppUpdate: () -> Unit` to `HomeScreen` and an outlined button:

```kotlin
OutlinedButton(
    onClick = onOpenAppUpdate,
    modifier = Modifier.fillMaxWidth(),
) {
    Text("アプリ更新")
}
```

- [ ] **Step 6: Run CI-equivalent verification**

```bash
gradle :app:assembleDebug :app:testDebugUnitTest :app:assembleDebugAndroidTest --stacktrace
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add app/build.gradle.kts app/src/main/java/com/ikegami99/kiraenhance/KiraEnhanceApp.kt app/src/main/java/com/ikegami99/kiraenhance/ui/home/HomeScreen.kt app/src/main/java/com/ikegami99/kiraenhance/ui/update app/src/androidTest/java/com/ikegami99/kiraenhance/ui/update
git commit -m "feat: add in-app update screen"
```

---

### Task 5: Add stable release signing and GitHub Release publishing

**Files:**
- Modify: `app/build.gradle.kts`
- Create: `.github/workflows/release.yml`

**Interfaces:**
- Consumes: `KIRA_VERSION_NAME`, `KIRA_VERSION_CODE`, `KIRA_KEYSTORE_PATH`, `KIRA_RELEASE_STORE_PASSWORD`, `KIRA_RELEASE_KEY_ALIAS`, `KIRA_RELEASE_KEY_PASSWORD`.
- Produces: `KiraEnhance-<version>.apk`, `KiraEnhance-<version>.apk.sha256`, and `update-manifest.json` release assets.

- [ ] **Step 1: Make version and signing configuration environment-driven**

At the top of `app/build.gradle.kts` add:

```kotlin
val kiraVersionName = providers.environmentVariable("KIRA_VERSION_NAME").orNull ?: "0.1.0-alpha01"
val kiraVersionCode = providers.environmentVariable("KIRA_VERSION_CODE").orNull?.toIntOrNull() ?: 1
val releaseStorePath = providers.environmentVariable("KIRA_KEYSTORE_PATH").orNull
val releaseStorePassword = providers.environmentVariable("KIRA_RELEASE_STORE_PASSWORD").orNull
val releaseKeyAlias = providers.environmentVariable("KIRA_RELEASE_KEY_ALIAS").orNull
val releaseKeyPassword = providers.environmentVariable("KIRA_RELEASE_KEY_PASSWORD").orNull
val releaseSigningReady = listOf(
    releaseStorePath,
    releaseStorePassword,
    releaseKeyAlias,
    releaseKeyPassword,
).all { !it.isNullOrBlank() }
```

Set default config to:

```kotlin
versionCode = kiraVersionCode
versionName = kiraVersionName
```

Inside `android` add:

```kotlin
signingConfigs {
    if (releaseSigningReady) {
        create("release") {
            storeFile = file(requireNotNull(releaseStorePath))
            storePassword = requireNotNull(releaseStorePassword)
            keyAlias = requireNotNull(releaseKeyAlias)
            keyPassword = requireNotNull(releaseKeyPassword)
        }
    }
}

buildTypes {
    getByName("release") {
        signingConfigs.findByName("release")?.let { signingConfig = it }
    }
}
```

- [ ] **Step 2: Create the release workflow**

`.github/workflows/release.yml` must use `workflow_dispatch` inputs `version_name`, `version_code`, and `release_notes`, grant `contents: write`, install the same JDK/Android/ncnn/Gradle versions as `android.yml`, then run these release-specific steps.

Validate and materialize secrets:

```yaml
      - name: Validate release inputs and secrets
        env:
          KEYSTORE_B64: ${{ secrets.KIRA_RELEASE_KEYSTORE_B64 }}
          STORE_PASSWORD: ${{ secrets.KIRA_RELEASE_STORE_PASSWORD }}
          KEY_ALIAS: ${{ secrets.KIRA_RELEASE_KEY_ALIAS }}
          KEY_PASSWORD: ${{ secrets.KIRA_RELEASE_KEY_PASSWORD }}
        run: |
          set -euo pipefail
          [[ '${{ inputs.version_name }}' =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-][A-Za-z0-9.-]+)?$ ]]
          [[ '${{ inputs.version_code }}' =~ ^[1-9][0-9]*$ ]]
          test -n "$KEYSTORE_B64"
          test -n "$STORE_PASSWORD"
          test -n "$KEY_ALIAS"
          test -n "$KEY_PASSWORD"
          printf '%s' "$KEYSTORE_B64" | base64 --decode > "$RUNNER_TEMP/kiraenhance-release.jks"
          echo "KIRA_KEYSTORE_PATH=$RUNNER_TEMP/kiraenhance-release.jks" >> "$GITHUB_ENV"
          echo "KIRA_RELEASE_STORE_PASSWORD=$STORE_PASSWORD" >> "$GITHUB_ENV"
          echo "KIRA_RELEASE_KEY_ALIAS=$KEY_ALIAS" >> "$GITHUB_ENV"
          echo "KIRA_RELEASE_KEY_PASSWORD=$KEY_PASSWORD" >> "$GITHUB_ENV"
          echo "KIRA_VERSION_NAME=${{ inputs.version_name }}" >> "$GITHUB_ENV"
          echo "KIRA_VERSION_CODE=${{ inputs.version_code }}" >> "$GITHUB_ENV"
```

Build:

```yaml
      - name: Build signed release APK
        run: gradle :app:assembleRelease :app:testDebugUnitTest --stacktrace
```

Prepare release assets safely, including notes as a file:

```yaml
      - name: Prepare release assets
        env:
          RELEASE_NOTES: ${{ inputs.release_notes }}
        run: |
          set -euo pipefail
          VERSION='${{ inputs.version_name }}'
          VERSION_CODE='${{ inputs.version_code }}'
          TAG="v$VERSION"
          APK_NAME="KiraEnhance-$VERSION.apk"
          cp app/build/outputs/apk/release/app-release.apk "$RUNNER_TEMP/$APK_NAME"
          SHA="$(sha256sum "$RUNNER_TEMP/$APK_NAME" | awk '{print $1}')"
          printf '%s  %s\n' "$SHA" "$APK_NAME" > "$RUNNER_TEMP/$APK_NAME.sha256"
          printf '%s\n' "$RELEASE_NOTES" > "$RUNNER_TEMP/release-notes.md"
          APK_URL="https://github.com/${GITHUB_REPOSITORY}/releases/download/$TAG/$APK_NAME"
          jq -n \
            --arg versionName "$VERSION" \
            --argjson versionCode "$VERSION_CODE" \
            --arg releaseNotes "$RELEASE_NOTES" \
            --arg apkUrl "$APK_URL" \
            --arg apkSha256 "$SHA" \
            '{versionName:$versionName,versionCode:$versionCode,releaseNotes:$releaseNotes,apkUrl:$apkUrl,apkSha256:$apkSha256}' \
            > "$RUNNER_TEMP/update-manifest.json"
          echo "TAG=$TAG" >> "$GITHUB_ENV"
          echo "APK_NAME=$APK_NAME" >> "$GITHUB_ENV"
```

Publish:

```yaml
      - name: Publish GitHub Release
        env:
          GH_TOKEN: ${{ github.token }}
        run: |
          gh release create "$TAG" \
            --title "KiraEnhance ${{ inputs.version_name }}" \
            --notes-file "$RUNNER_TEMP/release-notes.md" \
            "$RUNNER_TEMP/$APK_NAME" \
            "$RUNNER_TEMP/$APK_NAME.sha256" \
            "$RUNNER_TEMP/update-manifest.json"
```

- [ ] **Step 3: Verify debug CI still builds without release secrets**

```bash
gradle :app:assembleDebug :app:testDebugUnitTest :app:assembleDebugAndroidTest --stacktrace
```

Expected: PASS with no release-signing variables present.

- [ ] **Step 4: Commit**

```bash
git add app/build.gradle.kts .github/workflows/release.yml
git commit -m "ci: add signed KiraEnhance release workflow"
```

---

### Task 6: Release and real-device acceptance

**Files:**
- No source files.

**Interfaces:**
- Validates GitHub Release -> manifest check -> APK download -> SHA-256 verification -> Android installer.

- [ ] **Step 1: Configure the four GitHub Actions secrets**

Store the base64-encoded stable keystore and credentials under the exact secret names from Global Constraints. Never commit the keystore or credentials.

- [ ] **Step 2: Publish the first stable-signed baseline**

Run the release workflow with `version_name = 0.1.0-alpha02`, `version_code = 2`, and release notes `First stable-signed updater baseline`.

- [ ] **Step 3: Install the baseline manually once**

If Android rejects replacing the current debug APK because its signature differs, uninstall the debug build and install the stable-signed `0.1.0-alpha02` APK. Keep this release keystore permanently.

- [ ] **Step 4: Publish a higher-version test update**

Run the release workflow with `version_name = 0.1.0-alpha03`, `version_code = 3`, and release notes `In-app updater acceptance build`.

- [ ] **Step 5: Verify in-app replacement**

From `アプリ更新`, confirm `0.1.0-alpha03` and its notes appear, download it, wait for checksum verification, tap Install, grant install-from-this-source permission if prompted, and confirm Android offers an in-place replacement rather than a package/signature mismatch.

- [ ] **Step 6: Require final branch CI evidence**

```bash
gradle :app:assembleDebug :app:testDebugUnitTest :app:assembleDebugAndroidTest --stacktrace
```

Expected: PASS with zero failing unit tests and successful debug/androidTest APK assembly.
