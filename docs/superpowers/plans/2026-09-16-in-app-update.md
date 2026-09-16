# In-App Update Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let KiraEnhance manually check GitHub Releases, download and verify a stable-signed APK, and hand it to Android's system package installer from inside the app.

**Architecture:** GitHub Releases hosts a stable `update-manifest.json` plus the signed APK and checksum. The app uses a small manifest parser/checker, a WorkManager download worker, a narrow `FileProvider`, and a dedicated Compose update screen. Release signing is handled only by GitHub Actions secrets; development debug APKs remain unchanged.

**Tech Stack:** Kotlin, OkHttp 5.3.2, Moshi 1.15.2, WorkManager 2.11.2, Android FileProvider/package installer intents, Compose Material 3, GitHub Actions, JUnit 4.

**Spec:** `docs/superpowers/specs/2026-09-16-model-download-and-app-update-design.md`

## Global Constraints

- Android `minSdk = 28`, `targetSdk = 36`.
- Update checks are manual only.
- Update source of truth is GitHub Releases.
- The stable manifest URL is `https://github.com/IKEGAMI-99/KiraEnhance/releases/latest/download/update-manifest.json`.
- Manifest fields are `versionName`, `versionCode`, `releaseNotes`, `apkUrl`, and `apkSha256`.
- APKs must keep application ID `com.ikegami99.kiraenhance` and one stable release signing certificate.
- Never silently install; Android's system installer confirmation is mandatory.
- A checksum mismatch deletes the APK and never enables Install.
- Required GitHub Secrets: `KIRA_RELEASE_KEYSTORE_B64`, `KIRA_RELEASE_STORE_PASSWORD`, `KIRA_RELEASE_KEY_ALIAS`, `KIRA_RELEASE_KEY_PASSWORD`.

---

### Task 1: Define and validate update metadata

**Files:**
- Create: `app/src/main/java/com/ikegami99/kiraenhance/update/AppUpdateInfo.kt`
- Create: `app/src/main/java/com/ikegami99/kiraenhance/update/AppUpdateManifestParser.kt`
- Create: `app/src/main/java/com/ikegami99/kiraenhance/update/AppUpdateChecker.kt`
- Create: `app/src/test/java/com/ikegami99/kiraenhance/update/AppUpdateManifestParserTest.kt`
- Create: `app/src/test/java/com/ikegami99/kiraenhance/update/AppUpdateCheckerTest.kt`

**Interfaces:**
- Produces: `data class AppUpdateInfo(versionName: String, versionCode: Int, releaseNotes: String, apkUrl: String, apkSha256: String)`
- Produces: `fun interface UpdateManifestSource { fun load(): String }`
- Produces: `class HttpUpdateManifestSource(client: OkHttpClient, manifestUrl: String = DEFAULT_UPDATE_MANIFEST_URL)`
- Produces: `sealed interface AppUpdateCheckResult { data class Available(...); data class UpToDate(...) }`
- Produces: `AppUpdateChecker.check(currentVersionCode: Int): AppUpdateCheckResult`

- [ ] **Step 1: Write failing parser tests**

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

- [ ] **Step 2: Write failing version comparison tests**

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
    fun reportsAvailableWhenRemoteVersionCodeIsHigher() {
        val checker = AppUpdateChecker(UpdateManifestSource { newerJson })
        val result = checker.check(currentVersionCode = 1)
        assertEquals(2, (result as AppUpdateCheckResult.Available).info.versionCode)
    }

    @Test
    fun reportsUpToDateWhenVersionCodeMatches() {
        val checker = AppUpdateChecker(UpdateManifestSource { newerJson })
        val result = checker.check(currentVersionCode = 2)
        assertEquals("0.1.0-alpha02", (result as AppUpdateCheckResult.UpToDate).latestVersionName)
    }
}
```

- [ ] **Step 3: Run tests and verify RED**

```bash
gradle :app:testDebugUnitTest --tests 'com.ikegami99.kiraenhance.update.*' --stacktrace
```

Expected: compilation failure because the update classes do not exist.

- [ ] **Step 4: Implement metadata, parser, HTTP source, and checker**

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

`AppUpdateManifestParser.kt` validates nonblank version name, positive version code, HTTPS APK URL, and a 64-hex SHA-256 using the same Moshi/KotlinJsonAdapterFactory pattern as `ModelManifestParser`.

`AppUpdateChecker.kt`:

```kotlin
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

class AppUpdateChecker(
    private val source: UpdateManifestSource,
    private val parser: AppUpdateManifestParser = AppUpdateManifestParser(),
) {
    fun check(currentVersionCode: Int): AppUpdateCheckResult {
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
```

- [ ] **Step 5: Run tests and verify GREEN**

```bash
gradle :app:testDebugUnitTest --tests 'com.ikegami99.kiraenhance.update.*' --stacktrace
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add app/src/main/java/com/ikegami99/kiraenhance/update app/src/test/java/com/ikegami99/kiraenhance/update
git commit -m "feat: add app update manifest checker"
```

---

### Task 2: Download and verify update APKs

**Files:**
- Create: `app/src/main/java/com/ikegami99/kiraenhance/update/AppUpdateDownloadManager.kt`
- Create: `app/src/main/java/com/ikegami99/kiraenhance/update/AppUpdateDownloadWorker.kt`
- Create: `app/src/main/java/com/ikegami99/kiraenhance/download/RangeResumePolicy.kt`
- Modify: `app/src/main/java/com/ikegami99/kiraenhance/download/ModelDownloadWorker.kt`
- Modify: `app/src/test/java/com/ikegami99/kiraenhance/download/ModelDownloadResumeTest.kt`
- Create: `app/src/test/java/com/ikegami99/kiraenhance/download/RangeResumePolicyTest.kt`
- Create: `app/src/test/java/com/ikegami99/kiraenhance/update/AppUpdateDownloadManagerTest.kt`

**Interfaces:**
- Produces: `RangeResumePolicy.shouldRestartFromZero(responseCode: Int, resumeOffset: Long): Boolean`
- Produces: `AppUpdateDownloadManager.enqueue(info: AppUpdateInfo): UUID`
- Produces worker progress keys `KEY_BYTES_DOWNLOADED`, `KEY_TOTAL_BYTES`, `KEY_APK_PATH`, `KEY_ERROR`.

- [ ] **Step 1: Write failing shared Range policy tests**

```kotlin
class RangeResumePolicyTest {
    @Test
    fun http416RestartsOnlyWhenResuming() {
        assertTrue(RangeResumePolicy.shouldRestartFromZero(416, 128L))
        assertFalse(RangeResumePolicy.shouldRestartFromZero(416, 0L))
        assertFalse(RangeResumePolicy.shouldRestartFromZero(206, 128L))
    }
}
```

- [ ] **Step 2: Verify RED**

```bash
gradle :app:testDebugUnitTest --tests 'com.ikegami99.kiraenhance.download.RangeResumePolicyTest' --stacktrace
```

Expected: `RangeResumePolicy` unresolved.

- [ ] **Step 3: Extract the shared Range policy without changing behavior**

```kotlin
package com.ikegami99.kiraenhance.download

internal object RangeResumePolicy {
    fun shouldRestartFromZero(responseCode: Int, resumeOffset: Long): Boolean =
        responseCode == 416 && resumeOffset > 0L
}
```

Change `ModelDownloadWorker` to call `RangeResumePolicy.shouldRestartFromZero(...)`, and update the existing resume regression test to reference the shared policy.

- [ ] **Step 4: Write failing AppUpdateDownloadManager serialization tests**

Verify WorkManager input data contains APK URL, SHA-256, version code, and a stable unique work name `app-update-download-2` for version code 2.

```kotlin
@Test
fun createsStableUpdateWorkName() {
    assertEquals("app-update-download-2", AppUpdateDownloadManager.workName(versionCode = 2))
}
```

- [ ] **Step 5: Implement update manager and worker**

The manager must use `NetworkType.CONNECTED` and `ExistingWorkPolicy.REPLACE`.

The worker must:

```text
1. Store partial data at filesDir/updates/KiraEnhance-<versionCode>.apk.part.
2. Resume with Range when the partial file is non-empty.
3. On HTTP 416 while resuming, delete the partial file and retry once from byte zero in the same worker loop.
4. Accept only HTTP 200 or 206 for body transfer.
5. If a resumed request receives 200, overwrite from zero rather than append.
6. SHA-256 verify against apkSha256.
7. Delete the bad file on checksum mismatch.
8. Atomically move the verified file to filesDir/updates/KiraEnhance-<versionCode>.apk.
9. Return the absolute verified APK path in KEY_APK_PATH.
```

Use the existing `Sha256.matches(file, expectedSha)` utility rather than a second checksum implementation.

- [ ] **Step 6: Run unit tests and CI-equivalent compile**

```bash
gradle :app:assembleDebug :app:testDebugUnitTest :app:assembleDebugAndroidTest --stacktrace
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add app/src/main/java/com/ikegami99/kiraenhance/download app/src/main/java/com/ikegami99/kiraenhance/update app/src/test/java/com/ikegami99/kiraenhance/download app/src/test/java/com/ikegami99/kiraenhance/update
git commit -m "feat: download and verify app updates"
```

---

### Task 3: Add safe Android package-installer handoff

**Files:**
- Create: `app/src/main/java/com/ikegami99/kiraenhance/update/AppUpdateInstaller.kt`
- Create: `app/src/main/res/xml/update_file_paths.xml`
- Modify: `app/src/main/AndroidManifest.xml`
- Create: `app/src/test/java/com/ikegami99/kiraenhance/update/AppUpdateInstallPolicyTest.kt`

**Interfaces:**
- Produces: `AppUpdateInstallPolicy.action(canRequestPackageInstalls: Boolean): AppUpdateInstallAction`
- Produces: `AppUpdateInstaller.installOrRequestPermission(apkFile: File)`

- [ ] **Step 1: Write failing pure install-policy test**

```kotlin
package com.ikegami99.kiraenhance.update

import org.junit.Assert.assertEquals
import org.junit.Test

class AppUpdateInstallPolicyTest {
    @Test
    fun requestsPermissionBeforeInstallerWhenNeeded() {
        assertEquals(
            AppUpdateInstallAction.REQUEST_PERMISSION,
            AppUpdateInstallPolicy.action(canRequestPackageInstalls = false),
        )
        assertEquals(
            AppUpdateInstallAction.INSTALL,
            AppUpdateInstallPolicy.action(canRequestPackageInstalls = true),
        )
    }
}
```

- [ ] **Step 2: Verify RED**

```bash
gradle :app:testDebugUnitTest --tests 'com.ikegami99.kiraenhance.update.AppUpdateInstallPolicyTest' --stacktrace
```

Expected: update install policy unresolved.

- [ ] **Step 3: Implement installer policy and Android intents**

Policy:

```kotlin
enum class AppUpdateInstallAction { REQUEST_PERMISSION, INSTALL }

internal object AppUpdateInstallPolicy {
    fun action(canRequestPackageInstalls: Boolean): AppUpdateInstallAction =
        if (canRequestPackageInstalls) AppUpdateInstallAction.INSTALL
        else AppUpdateInstallAction.REQUEST_PERMISSION
}
```

Installer behavior:

```kotlin
val canInstall = context.packageManager.canRequestPackageInstalls()
if (!canInstall) {
    context.startActivity(
        Intent(
            Settings.ACTION_MANAGE_UNKNOWN_APP_SOURCES,
            Uri.parse("package:${context.packageName}"),
        ).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK),
    )
    return
}

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
```

Manifest additions:

```xml
<uses-permission android:name="android.permission.REQUEST_INSTALL_PACKAGES" />
```

and inside `<application>`:

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

`update_file_paths.xml` must expose only the update folder:

```xml
<?xml version="1.0" encoding="utf-8"?>
<paths xmlns:android="http://schemas.android.com/apk/res/android">
    <files-path name="updates" path="updates/" />
</paths>
```

- [ ] **Step 4: Run unit tests and Android manifest compile**

```bash
gradle :app:testDebugUnitTest :app:assembleDebug --stacktrace
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/src/main/java/com/ikegami99/kiraenhance/update app/src/main/res/xml/update_file_paths.xml app/src/main/AndroidManifest.xml app/src/test/java/com/ikegami99/kiraenhance/update
git commit -m "feat: hand verified updates to Android installer"
```

---

### Task 4: Add the App Update screen and navigation

**Files:**
- Create: `app/src/main/java/com/ikegami99/kiraenhance/ui/update/AppUpdateViewModel.kt`
- Create: `app/src/main/java/com/ikegami99/kiraenhance/ui/update/AppUpdateScreen.kt`
- Create: `app/src/main/java/com/ikegami99/kiraenhance/ui/update/AppUpdateRoute.kt`
- Modify: `app/src/main/java/com/ikegami99/kiraenhance/KiraEnhanceApp.kt`
- Modify: `app/src/main/java/com/ikegami99/kiraenhance/ui/home/HomeScreen.kt`
- Modify: `app/build.gradle.kts`
- Create: `app/src/androidTest/java/com/ikegami99/kiraenhance/ui/update/AppUpdateScreenTest.kt`

**Interfaces:**
- Produces: `AppUpdateUiState`
- Produces: `AppUpdateViewModel.checkForUpdates()` and `downloadUpdate()`
- Consumes: `AppUpdateInstaller.installOrRequestPermission(file)` from the route.

- [ ] **Step 1: Add required ViewModel coroutine/build config support**

In dependencies add:

```kotlin
implementation("androidx.lifecycle:lifecycle-viewmodel-ktx:2.10.0")
```

In `buildFeatures` add:

```kotlin
buildConfig = true
```

- [ ] **Step 2: Write failing Compose screen test**

Render an idle state and assert `アプリ更新`, current version, and `アップデートを確認` are visible. Render an available state and assert latest version, release notes, and `ダウンロード` are visible. Render a verified-download state and assert `インストール` is enabled.

- [ ] **Step 3: Implement ViewModel state and manual check flow**

Use this state shape:

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

`checkForUpdates()` runs `AppUpdateChecker.check(...)` on `Dispatchers.IO`, updates `availableUpdate` for Available, and sets `upToDate = true` for UpToDate. `downloadUpdate()` enqueues `AppUpdateDownloadManager`, observes that work ID, updates progress, and only sets `verifiedApkPath` from a successful worker output.

- [ ] **Step 4: Implement screen and route**

`AppUpdateScreen` displays current version, check button, latest version/release notes, linear download progress, errors, and the install button only when `verifiedApkPath != null`.

`AppUpdateRoute` creates the ViewModel factory with `BuildConfig.VERSION_NAME` and `BuildConfig.VERSION_CODE`, and on Install verifies the path still points inside `filesDir/updates` before calling `AppUpdateInstaller`.

- [ ] **Step 5: Wire navigation from Home**

Add:

```kotlin
private const val UPDATE_ROUTE = "update"
```

Add `onOpenAppUpdate` to `HomeScreen`, render an outlined `アプリ更新` button, and add a `composable(UPDATE_ROUTE)` destination in `KiraEnhanceApp`.

- [ ] **Step 6: Run unit and UI compile verification**

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

### Task 5: Make release versioning and signing reproducible

**Files:**
- Modify: `app/build.gradle.kts`
- Create: `.github/workflows/release.yml`

**Interfaces:**
- Consumes environment: `KIRA_VERSION_NAME`, `KIRA_VERSION_CODE`, `KIRA_KEYSTORE_PATH`, `KIRA_RELEASE_STORE_PASSWORD`, `KIRA_RELEASE_KEY_ALIAS`, `KIRA_RELEASE_KEY_PASSWORD`.
- Produces release assets: `KiraEnhance-<version>.apk`, matching `.sha256`, and `update-manifest.json`.

- [ ] **Step 1: Make Gradle version values environment-overridable**

At top level:

```kotlin
val kiraVersionName = providers.environmentVariable("KIRA_VERSION_NAME").orNull ?: "0.1.0-alpha01"
val kiraVersionCode = providers.environmentVariable("KIRA_VERSION_CODE").orNull?.toIntOrNull() ?: 1
```

Use these in `defaultConfig`.

Configure a release signing config only when all four signing environment variables are present. The release workflow must fail before Gradle if any secret is absent; debug CI remains unaffected.

- [ ] **Step 2: Add release workflow with explicit manual inputs**

Create `.github/workflows/release.yml` with `workflow_dispatch` inputs `version_name`, `version_code`, and `release_notes`, `permissions: contents: write`, the same Android/ncnn setup as `android.yml`, and these release-specific steps:

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

Generate assets:

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
            --notes '${{ inputs.release_notes }}' \
            "$RUNNER_TEMP/$APK_NAME" \
            "$RUNNER_TEMP/$APK_NAME.sha256" \
            "$RUNNER_TEMP/update-manifest.json"
```

- [ ] **Step 3: Verify development CI still succeeds without release secrets**

```bash
gradle :app:assembleDebug :app:testDebugUnitTest :app:assembleDebugAndroidTest --stacktrace
```

Expected: PASS with no release-signing environment variables.

- [ ] **Step 4: Commit**

```bash
git add app/build.gradle.kts .github/workflows/release.yml
git commit -m "ci: add signed KiraEnhance release workflow"
```

---

### Task 6: Release and real-device acceptance

**Files:**
- No source change expected.

**Interfaces:**
- Validates the complete GitHub Release -> app check -> download -> checksum -> Android installer flow.

- [ ] **Step 1: Configure the four GitHub Actions secrets**

Store the base64-encoded stable keystore and its store password, key alias, and key password under the exact secret names in Global Constraints. Do not commit the keystore or passwords.

- [ ] **Step 2: Publish the first stable-signed build**

Run the release workflow with `version_name = 0.1.0-alpha02`, `version_code = 2`, and release notes describing the updater test build.

- [ ] **Step 3: Install the stable-signed baseline manually once**

Because current debug APK signing may differ, uninstall the debug build if Android rejects replacement, then install the new stable-signed APK. Preserve this keystore permanently for all future updates.

- [ ] **Step 4: Publish the next stable-signed build**

Run the workflow with a higher version code, for example `version_name = 0.1.0-alpha03` and `version_code = 3`.

- [ ] **Step 5: Verify the in-app flow on device**

From `アプリ更新`, check for updates, confirm the new version and notes appear, download it, verify progress completes, tap Install, grant unknown-app-source permission if prompted, and confirm Android offers an in-place update without package/signature mismatch.

- [ ] **Step 6: Final CI evidence**

Before publishing any test APK from the implementation branch, require:

```bash
gradle :app:assembleDebug :app:testDebugUnitTest :app:assembleDebugAndroidTest --stacktrace
```

Expected: PASS with zero failing unit tests and successful debug/androidTest APK assembly.
