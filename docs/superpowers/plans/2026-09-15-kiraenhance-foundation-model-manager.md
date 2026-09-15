# KiraEnhance Foundation & Model Manager Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a launchable Android KiraEnhance app shell with the final beginner-first visual direction, a typed model registry, device capability detection, verified/resumable model downloads, and a functional model-management screen, without adding AI inference yet.

**Architecture:** Use a single Android app module with Kotlin/Jetpack Compose for UI and domain logic. Model metadata is parsed into immutable descriptors, persisted model state lives under app-private storage, and WorkManager performs resumable downloads into `.part` files before SHA-256 verification and atomic activation. Keep inference behind interfaces that are introduced later; this milestone must not bake MNN or ncnn assumptions into UI code.

**Tech Stack:** Android, Kotlin, Jetpack Compose, Material 3, WorkManager, Moshi, Android NDK-ready Gradle configuration, JUnit, Compose UI tests.

**Spec:** `docs/superpowers/specs/2026-09-15-kiraenhance-design.md`

## Global Constraints

- Android only for v1.
- Package name: `com.ikegami99.kiraenhance`.
- UI: Kotlin + Jetpack Compose.
- Native inference core will use C++/NDK/JNI in later plans; this milestone configures NDK but does not add model inference.
- Primary future runtime: MNN; optional secondary runtime: ncnn/Vulkan.
- Reference device class: POCO F7 Ultra / Snapdragon 8 Elite.
- AI inference must remain fully local; network access is only for app/model metadata and downloads.
- App distribution target is GitHub Releases; model metadata lives in GitHub and large model binaries on Hugging Face.
- Use stable toolchain versions current on 2026-09-15: AGP `9.4.0`, Gradle `9.6.0`, compileSdk `37`, targetSdk `36`, minSdk `28`, NDK `28.2.13676358`, Compose BOM `2026.08.00`, WorkManager `2.11.2`, JDK `17`.
- Initial app version: `0.1.0-alpha01`, versionCode `1`.
- Do not bundle AI model binaries in the APK.
- No image data or full source file paths may be uploaded or included in diagnostic logs.
- UI labels for primary modes are `忠実`, `おすすめ`, `高精細`, `UltraSharp`.

---

## File map for this milestone

```text
KiraEnhance/
├── settings.gradle.kts
├── build.gradle.kts
├── gradle.properties
├── gradle/wrapper/gradle-wrapper.properties
├── .gitignore
├── .github/workflows/android.yml
└── app/
    ├── build.gradle.kts
    ├── proguard-rules.pro
    └── src/
        ├── main/
        │   ├── AndroidManifest.xml
        │   ├── assets/model_manifest.json
        │   ├── java/com/ikegami99/kiraenhance/
        │   │   ├── MainActivity.kt
        │   │   ├── KiraEnhanceApp.kt
        │   │   ├── data/model/ModelManifestParser.kt
        │   │   ├── data/model/ModelRepository.kt
        │   │   ├── data/model/ModelStorage.kt
        │   │   ├── data/model/ModelDownloadWorker.kt
        │   │   ├── data/model/Sha256Verifier.kt
        │   │   ├── domain/model/EnhanceMode.kt
        │   │   ├── domain/model/ModelDescriptor.kt
        │   │   ├── domain/model/ModelInstallState.kt
        │   │   ├── domain/device/DeviceCapabilityDetector.kt
        │   │   ├── domain/device/DeviceSupportEvaluator.kt
        │   │   ├── ui/home/HomeScreen.kt
        │   │   ├── ui/home/HomeViewModel.kt
        │   │   ├── ui/models/ModelManagerScreen.kt
        │   │   ├── ui/models/ModelManagerViewModel.kt
        │   │   ├── ui/theme/Color.kt
        │   │   ├── ui/theme/Theme.kt
        │   │   └── ui/theme/Type.kt
        │   └── res/values/strings.xml
        ├── test/java/com/ikegami99/kiraenhance/
        │   ├── data/model/ModelManifestParserTest.kt
        │   ├── data/model/Sha256VerifierTest.kt
        │   ├── data/model/ModelStorageTest.kt
        │   └── domain/device/DeviceSupportEvaluatorTest.kt
        └── androidTest/java/com/ikegami99/kiraenhance/
            └── ui/ModelManagerScreenTest.kt
```

---

### Task 1: Bootstrap the Android project and CI

**Files:**
- Create: `settings.gradle.kts`
- Create: `build.gradle.kts`
- Create: `gradle.properties`
- Create: `gradle/wrapper/gradle-wrapper.properties`
- Create: `.gitignore`
- Create: `app/build.gradle.kts`
- Create: `app/proguard-rules.pro`
- Create: `app/src/main/AndroidManifest.xml`
- Create: `app/src/main/res/values/strings.xml`
- Create: `app/src/main/java/com/ikegami99/kiraenhance/MainActivity.kt`
- Create: `app/src/main/java/com/ikegami99/kiraenhance/KiraEnhanceApp.kt`
- Create: `.github/workflows/android.yml`

**Interfaces:**
- Produces a buildable `:app` module and `KiraEnhanceApp()` Compose entry point used by later tasks.

- [ ] **Step 1: Create a minimal build configuration**

Use AGP 9.4.0 and Gradle 9.6.0. `settings.gradle.kts`:

```kotlin
pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}
rootProject.name = "KiraEnhance"
include(":app")
```

Root `build.gradle.kts`:

```kotlin
plugins {
    id("com.android.application") version "9.4.0" apply false
}
```

`gradle/wrapper/gradle-wrapper.properties`:

```properties
distributionBase=GRADLE_USER_HOME
distributionPath=wrapper/dists
distributionUrl=https\://services.gradle.org/distributions/gradle-9.6.0-bin.zip
networkTimeout=10000
validateDistributionUrl=true
zipStoreBase=GRADLE_USER_HOME
zipStorePath=wrapper/dists
```

- [ ] **Step 2: Configure the app module**

`app/build.gradle.kts` must include:

```kotlin
plugins {
    id("com.android.application")
}

android {
    namespace = "com.ikegami99.kiraenhance"
    compileSdk = 37
    ndkVersion = "28.2.13676358"

    defaultConfig {
        applicationId = "com.ikegami99.kiraenhance"
        minSdk = 28
        targetSdk = 36
        versionCode = 1
        versionName = "0.1.0-alpha01"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildFeatures {
        compose = true
        buildConfig = true
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

dependencies {
    val composeBom = platform("androidx.compose:compose-bom:2026.08.00")
    implementation(composeBom)
    androidTestImplementation(composeBom)

    implementation("androidx.activity:activity-compose:1.13.0")
    implementation("androidx.compose.foundation:foundation")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-tooling-preview")
    debugImplementation("androidx.compose.ui:ui-tooling")

    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.11.0")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.11.0")
    implementation("androidx.work:work-runtime:2.11.2")
    implementation("com.squareup.moshi:moshi:1.15.2")
    implementation("com.squareup.moshi:moshi-kotlin:1.15.2")

    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.3.0")
    androidTestImplementation("androidx.compose.ui:ui-test-junit4")
    debugImplementation("androidx.compose.ui:ui-test-manifest")
}
```

If a listed AndroidX stable version has moved between planning and execution, keep the exact version above for reproducibility unless Gradle proves it is unavailable.

- [ ] **Step 3: Add the minimal launcher**

`MainActivity.kt`:

```kotlin
package com.ikegami99.kiraenhance

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent { KiraEnhanceApp() }
    }
}
```

`KiraEnhanceApp.kt` initially renders `Text("KiraEnhance")` inside `MaterialTheme`.

- [ ] **Step 4: Build the project**

Run:

```bash
./gradlew :app:assembleDebug :app:testDebugUnitTest
```

Expected: both tasks complete with `BUILD SUCCESSFUL`.

- [ ] **Step 5: Add CI**

`.github/workflows/android.yml` checks out the repository, sets up JDK 17, runs `./gradlew :app:testDebugUnitTest :app:assembleDebug`, and uploads `app-debug.apk` as an artifact.

- [ ] **Step 6: Commit**

```bash
git add .
git commit -m "build: bootstrap KiraEnhance Android app"
```

---

### Task 2: Define model metadata and parse the manifest

**Files:**
- Create: `app/src/main/java/com/ikegami99/kiraenhance/domain/model/EnhanceMode.kt`
- Create: `app/src/main/java/com/ikegami99/kiraenhance/domain/model/ModelDescriptor.kt`
- Create: `app/src/main/java/com/ikegami99/kiraenhance/domain/model/ModelInstallState.kt`
- Create: `app/src/main/java/com/ikegami99/kiraenhance/data/model/ModelManifestParser.kt`
- Create: `app/src/main/assets/model_manifest.json`
- Test: `app/src/test/java/com/ikegami99/kiraenhance/data/model/ModelManifestParserTest.kt`

**Interfaces:**
- Produces `EnhanceMode`, `ModelDescriptor`, `ModelParameterDescriptor`, `ModelManifestParser.parse(json: String): List<ModelDescriptor>`.
- Later tasks consume `ModelDescriptor.id`, `download`, `sha256`, `backend`, and capability fields.

- [ ] **Step 1: Write the failing parser test**

```kotlin
@Test
fun parsesUltraSharpAsCommunityModel() {
    val models = ModelManifestParser().parse(sampleManifest)
    val ultra = models.single { it.id == "ultrasharp-4x" }
    assertEquals(EnhanceMode.ULTRASHARP, ultra.mode)
    assertTrue(ultra.communityModel)
    assertEquals(listOf(2, 4), ultra.outputScales)
}
```

Also assert that `pisa-sr` maps to `EnhanceMode.BALANCED` and exposes `fidelity` and `semanticDetail` controls.

- [ ] **Step 2: Run the test and verify failure**

```bash
./gradlew :app:testDebugUnitTest --tests '*ModelManifestParserTest*'
```

Expected: compilation failure because the model types/parser do not exist.

- [ ] **Step 3: Add typed domain models**

```kotlin
enum class EnhanceMode { FIDELITY, BALANCED, DETAIL, ULTRASHARP }
enum class ModelBackend { MNN, NCNN }
enum class ParameterKind { FLOAT, BOOLEAN, CHOICE }

data class ModelParameterDescriptor(
    val id: String,
    val label: String,
    val kind: ParameterKind,
    val defaultValue: String,
    val min: Float? = null,
    val max: Float? = null,
    val step: Float? = null,
    val choices: List<String> = emptyList(),
)

data class ModelDescriptor(
    val id: String,
    val displayName: String,
    val mode: EnhanceMode,
    val version: String,
    val backend: ModelBackend,
    val downloadUrl: String?,
    val fileSizeBytes: Long,
    val sha256: String?,
    val outputScales: List<Int>,
    val estimatedRamMb: Int,
    val minEngineVersion: String,
    val licenseName: String,
    val licenseUrl: String?,
    val attribution: String?,
    val communityModel: Boolean,
    val redistributionAllowed: Boolean,
    val description: String,
    val parameters: List<ModelParameterDescriptor>,
)
```

`ModelInstallState` must be a sealed interface with `NotInstalled`, `Downloading(progress: Float, bytesDownloaded: Long, totalBytes: Long)`, `Installed(version: String, path: String)`, `Corrupt(reason: String)`, and `UpdateAvailable(installedVersion: String, availableVersion: String)`.

- [ ] **Step 4: Implement strict manifest parsing**

Use Moshi and reject entries with blank ids, unsupported scale factors, missing SHA-256 for redistributable downloads, or download URLs that are not HTTPS. A user-supplied-only model is represented by `downloadUrl = null` and `redistributionAllowed = false`.

- [ ] **Step 5: Add the bundled bootstrap manifest**

`model_manifest.json` contains four entries:

- `realesrgan-anime-6b`: Fidelity, NCNN, downloadable metadata placeholder disabled until a verified production URL/hash is entered.
- `pisa-sr`: Balanced, MNN, required product model, fidelity + semantic detail controls.
- `hat-detail`: Detail, MNN, candidate metadata.
- `ultrasharp-4x`: UltraSharp, NCNN, community model, and `redistributionAllowed=false` until license/redistribution verification is complete.

Do not invent real hashes or URLs. For unavailable downloadable binaries, use `downloadUrl=null`, `sha256=null`, and `redistributionAllowed=false`; the UI will show `配布準備中` or user-supplied-only rather than a broken download button.

- [ ] **Step 6: Run parser tests**

```bash
./gradlew :app:testDebugUnitTest --tests '*ModelManifestParserTest*'
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add app/src/main/assets app/src/main/java/com/ikegami99/kiraenhance/domain/model app/src/main/java/com/ikegami99/kiraenhance/data/model app/src/test
 git commit -m "feat: add typed AI model manifest"
```

---

### Task 3: Add verified local model storage

**Files:**
- Create: `app/src/main/java/com/ikegami99/kiraenhance/data/model/Sha256Verifier.kt`
- Create: `app/src/main/java/com/ikegami99/kiraenhance/data/model/ModelStorage.kt`
- Test: `app/src/test/java/com/ikegami99/kiraenhance/data/model/Sha256VerifierTest.kt`
- Test: `app/src/test/java/com/ikegami99/kiraenhance/data/model/ModelStorageTest.kt`

**Interfaces:**
- Produces `Sha256Verifier.sha256(file: File): String` and `ModelStorage` methods `partFile(modelId)`, `activeFile(modelId, version)`, `activateVerifiedPart(...)`, `deleteModel(modelId)`.

- [ ] **Step 1: Write SHA-256 tests**

Test the known UTF-8 payload `abc`, whose SHA-256 is `ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad`.

- [ ] **Step 2: Run and confirm failure**

```bash
./gradlew :app:testDebugUnitTest --tests '*Sha256VerifierTest*'
```

Expected: class not found/compilation failure.

- [ ] **Step 3: Implement streaming SHA-256**

Read in 1 MiB chunks using `DigestInputStream`; never load model files entirely into memory.

- [ ] **Step 4: Write storage tests**

Use a temporary directory and assert:

1. downloads land in `<root>/<modelId>/download.part`;
2. successful activation renames into `<root>/<modelId>/<version>/model.bin`;
3. activation refuses a hash mismatch and leaves the active model untouched;
4. deleting one model does not delete siblings.

- [ ] **Step 5: Implement `ModelStorage`**

Activation order must be: verify part file -> create version directory -> atomic rename/copy+fsync fallback -> write a small `installed.json` marker -> remove old `.part`. Never write `installed.json` before the final model is in place.

- [ ] **Step 6: Run tests**

```bash
./gradlew :app:testDebugUnitTest --tests '*Sha256VerifierTest*' --tests '*ModelStorageTest*'
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add app/src/main/java/com/ikegami99/kiraenhance/data/model app/src/test
 git commit -m "feat: add verified model storage"
```

---

### Task 4: Add resumable WorkManager model downloads

**Files:**
- Create: `app/src/main/java/com/ikegami99/kiraenhance/data/model/ModelDownloadWorker.kt`
- Create: `app/src/main/java/com/ikegami99/kiraenhance/data/model/ModelRepository.kt`
- Modify: `app/src/main/AndroidManifest.xml`
- Test: extend `app/src/test/java/com/ikegami99/kiraenhance/data/model/ModelStorageTest.kt`

**Interfaces:**
- Produces `ModelRepository.models: StateFlow<List<ModelUiState>>`, `enqueueDownload(modelId)`, `cancelDownload(modelId)`, `deleteModel(modelId)`, `refresh()`.
- Work input keys: `model_id`, `version`, `url`, `sha256`, `size_bytes`.
- Work progress keys: `downloaded_bytes`, `total_bytes`.

- [ ] **Step 1: Write a pure range planner test**

Add a small internal function:

```kotlin
internal fun rangeHeader(existingBytes: Long): String? =
    if (existingBytes > 0L) "bytes=$existingBytes-" else null
```

Test `0 -> null` and `1024 -> "bytes=1024-"`.

- [ ] **Step 2: Implement `ModelDownloadWorker`**

Use `HttpURLConnection` with:

- connect/read timeout of 30 seconds;
- `Range` header when `.part` already has bytes;
- append only when server replies `206`; if it replies `200`, truncate and restart;
- update WorkManager progress at most four times per second;
- verify expected byte count when known;
- SHA-256 verify before activation;
- return `Result.retry()` for transient HTTP 408/429/5xx and network I/O failure;
- return `Result.failure()` for 4xx other than 408/429, hash mismatch, or malformed metadata.

The worker must never delete a previously activated version when a new download fails.

- [ ] **Step 3: Add Wi-Fi-only constraints at repository level**

When the user setting is enabled, build the request with `NetworkType.UNMETERED`; otherwise use `NetworkType.CONNECTED`.

- [ ] **Step 4: Implement repository state projection**

`ModelRepository` combines manifest descriptors, storage markers, and WorkManager `WorkInfo` into UI states. A model with `downloadUrl=null` must never enqueue a worker.

- [ ] **Step 5: Run unit tests and build**

```bash
./gradlew :app:testDebugUnitTest :app:assembleDebug
```

Expected: PASS and successful APK build.

- [ ] **Step 6: Commit**

```bash
git add app/src/main/java/com/ikegami99/kiraenhance/data/model app/src/main/AndroidManifest.xml app/src/test
 git commit -m "feat: add resumable model downloads"
```

---

### Task 5: Detect device capabilities and rate model support

**Files:**
- Create: `app/src/main/java/com/ikegami99/kiraenhance/domain/device/DeviceCapabilityDetector.kt`
- Create: `app/src/main/java/com/ikegami99/kiraenhance/domain/device/DeviceSupportEvaluator.kt`
- Test: `app/src/test/java/com/ikegami99/kiraenhance/domain/device/DeviceSupportEvaluatorTest.kt`

**Interfaces:**
- Produces `DeviceCapabilities` and `DeviceSupportEvaluator.evaluate(model, device): SupportLevel` where `SupportLevel` is `RECOMMENDED`, `SUPPORTED`, or `NOT_RECOMMENDED`.

- [ ] **Step 1: Write evaluator tests**

Cover these exact cases:

```text
RAM 16 GiB + Vulkan + model estimate 6 GiB => RECOMMENDED
RAM 8 GiB + Vulkan + model estimate 6 GiB => SUPPORTED
RAM 6 GiB + Vulkan + model estimate 6 GiB => NOT_RECOMMENDED
No Vulkan + NCNN/Vulkan model => NOT_RECOMMENDED
```

- [ ] **Step 2: Implement pure support evaluation**

Use conservative thresholds:

- `RECOMMENDED` when physical RAM >= `estimatedRamMb * 2` and required runtime features are present;
- `SUPPORTED` when RAM >= `estimatedRamMb * 1.25`;
- otherwise `NOT_RECOMMENDED`.

This is guidance only; do not block a model solely because of RAM rating.

- [ ] **Step 3: Implement Android capability collection**

Collect:

- `Build.MANUFACTURER`, `Build.MODEL`, Android SDK;
- total RAM from `ActivityManager.MemoryInfo.totalMem`;
- available internal storage from `StatFs`;
- Vulkan feature presence via `PackageManager.FEATURE_VULKAN_HARDWARE_LEVEL` and hardware version feature.

Do not use fragile `/proc` parsing for SoC identification in this milestone. Record `Build.SOC_MANUFACTURER` and `Build.SOC_MODEL` on API levels where available.

- [ ] **Step 4: Run tests**

```bash
./gradlew :app:testDebugUnitTest --tests '*DeviceSupportEvaluatorTest*'
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/src/main/java/com/ikegami99/kiraenhance/domain/device app/src/test
 git commit -m "feat: add device capability guidance"
```

---

### Task 6: Build the cute beginner-first Home and Model Manager UI

**Files:**
- Create: `app/src/main/java/com/ikegami99/kiraenhance/ui/theme/Color.kt`
- Create: `app/src/main/java/com/ikegami99/kiraenhance/ui/theme/Theme.kt`
- Create: `app/src/main/java/com/ikegami99/kiraenhance/ui/theme/Type.kt`
- Create: `app/src/main/java/com/ikegami99/kiraenhance/ui/home/HomeScreen.kt`
- Create: `app/src/main/java/com/ikegami99/kiraenhance/ui/home/HomeViewModel.kt`
- Create: `app/src/main/java/com/ikegami99/kiraenhance/ui/models/ModelManagerScreen.kt`
- Create: `app/src/main/java/com/ikegami99/kiraenhance/ui/models/ModelManagerViewModel.kt`
- Modify: `app/src/main/java/com/ikegami99/kiraenhance/KiraEnhanceApp.kt`
- Test: `app/src/androidTest/java/com/ikegami99/kiraenhance/ui/ModelManagerScreenTest.kt`

**Interfaces:**
- Home consumes a list of `ModelUiState` and exposes callbacks `onSelectImage`, `onSelectMode`, `onOpenModelManager`.
- Model manager consumes `ModelUiState` and calls repository actions only through its ViewModel.

- [ ] **Step 1: Write the Compose UI test first**

Create an instrumentation test that renders `ModelManagerScreen` with four fake entries and asserts that these text nodes exist:

```text
忠実
おすすめ
高精細
UltraSharp
モデル管理
```

Also assert UltraSharp with `redistributionAllowed=false` shows `ユーザー追加 / 配布確認中` and does not expose an enabled `ダウンロード` button.

- [ ] **Step 2: Run the UI test to confirm failure**

```bash
./gradlew :app:connectedDebugAndroidTest
```

Expected: compilation failure because the screen does not exist. If no emulator/device is connected, compile the test with `./gradlew :app:assembleDebug :app:assembleDebugAndroidTest` and run it once a device is available before merging.

- [ ] **Step 3: Implement the theme**

Design requirements:

- default dark surface is near-black plum, not pure black;
- primary accent is lavender/pink;
- rounded cards 20dp;
- subtle gradient is allowed, but no continuous particle animation;
- light, dark, and system theme hooks must exist;
- Material 3 large touch targets.

Do not copy Kirapara UI assets or logos.

- [ ] **Step 4: Implement Home**

The initial Home screen contains:

1. KiraEnhance wordmark text;
2. a large `画像を選ぶ` button (callback only in this milestone);
3. four mode cards, with the Balanced card visually marked `おすすめ`;
4. a `モデル管理` entry;
5. a privacy line: `画像処理は端末内で完結します`.

No inference button should pretend to work in this milestone. If image processing is not yet implemented, disable it with a clear `推論エンジン準備中` state rather than faking success.

- [ ] **Step 5: Implement Model Manager**

Each card shows:

- user-facing mode;
- technical model name in secondary text;
- version;
- size;
- estimated RAM;
- backend;
- license;
- support rating;
- install/download/update state.

Actions are `ダウンロード`, `キャンセル`, `削除`, or `再試行` only when valid for that state.

- [ ] **Step 6: Run all tests**

```bash
./gradlew :app:testDebugUnitTest :app:assembleDebug :app:assembleDebugAndroidTest
```

Expected: PASS/build success. Run `connectedDebugAndroidTest` on an attached emulator/device before merging.

- [ ] **Step 7: Commit**

```bash
git add app/src
 git commit -m "feat: add KiraEnhance home and model manager UI"
```

---

### Task 7: Wire the app together and document milestone acceptance

**Files:**
- Modify: `app/src/main/java/com/ikegami99/kiraenhance/KiraEnhanceApp.kt`
- Create: `README.md`

**Interfaces:**
- Produces the milestone-1 runnable app and documents where later inference plans attach.

- [ ] **Step 1: Wire one repository instance into ViewModels**

For this small project stage, construct app dependencies in a simple `AppContainer` owned by the application/Compose root. Do not introduce Hilt yet; there is no need to spend a dependency graph framework on half a dozen objects.

- [ ] **Step 2: Add navigation state**

Use a small sealed route (`Home`, `Models`) or Navigation Compose if already required by the UI implementation. Back from Model Manager must return to Home without recreating downloads.

- [ ] **Step 3: Write README milestone instructions**

Document:

```text
Requirements: JDK 17, Android SDK 37, NDK 28.2.13676358
Build: ./gradlew :app:assembleDebug
Unit tests: ./gradlew :app:testDebugUnitTest
Device tests: ./gradlew :app:connectedDebugAndroidTest
APK: app/build/outputs/apk/debug/app-debug.apk
```

Explain that milestone 1 deliberately contains no inference and that models with unverified distribution terms do not expose hosted download URLs.

- [ ] **Step 4: Perform final verification**

Run:

```bash
./gradlew clean :app:testDebugUnitTest :app:assembleDebug :app:assembleDebugAndroidTest
```

Expected: `BUILD SUCCESSFUL` with no test failures.

On the reference Android device, verify manually:

1. app launches;
2. all four mode cards render;
3. model manager opens;
4. a manifest entry with a valid test URL can enter Downloading and be cancelled;
5. corrupt test data fails SHA-256 and is not marked Installed;
6. deleting a test model removes only its own directory;
7. app continues working with networking disabled after the manifest is loaded from bundled assets.

- [ ] **Step 5: Commit**

```bash
git add README.md app/src
 git commit -m "docs: complete foundation milestone"
```

---

## Follow-on plans after this milestone

This specification is intentionally decomposed. After this plan passes review and verification, create separate implementation plans in this order:

1. `ncnn-esrgan-inference`: JNI/NDK engine interface, ncnn/Vulkan, RealESRGAN/UltraSharp-compatible local inference, tiling, 2x/4x output, cancellation.
2. `comparison-and-export`: image picker, EXIF-safe decode, synchronized split/side-by-side/hold comparison, PNG/JPEG/WebP save/share.
3. `pisa-mnn-inference`: PiSA-SR mobile conversion, MNN execution, tile/VAE strategy, model-specific fidelity/semantic controls, memory profiling on Snapdragon 8 Elite.
4. `detail-model-and-release`: final Detail model selection, app updater through GitHub Releases, model update checks, log export, version/about/license surfaces, release signing and acceptance tests.

Each follow-on plan must be written only after the interfaces produced by the previous milestone are present in the repository, so file paths and signatures are based on real code rather than guesses.

## Plan self-review

- Spec coverage for this milestone: Android shell, beginner UI, model manifest, verified download/removal, device guidance, model version/license display are covered.
- Explicitly deferred into separate plans: inference, comparison/export, PiSA-SR, app updater, final logging/release hardening.
- No cloud inference or account system is introduced.
- UltraSharp redistribution remains disabled until terms are verified; the plan does not fabricate a download URL/hash.
- The model UI is capability-driven so later PiSA/HAT/ncnn implementations do not require redesigning the screens.
