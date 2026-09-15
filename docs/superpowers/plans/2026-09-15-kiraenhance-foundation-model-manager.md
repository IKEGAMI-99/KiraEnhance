# KiraEnhance Foundation & Model Manager Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a launchable Android KiraEnhance app shell with the beginner-first visual direction, a typed model registry, device capability detection, verified/resumable model downloads, and a functional model-management screen, without adding AI inference yet.

**Architecture:** Use a single Android app module with Kotlin/Jetpack Compose for UI and domain logic. Model metadata is parsed into immutable descriptors, persisted model state lives under app-private storage, and WorkManager performs resumable downloads into `.part` files before SHA-256 verification and atomic activation. Keep inference behind interfaces introduced by later plans; this milestone must not bake MNN or ncnn assumptions into UI code.

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
- Toolchain: AGP `9.4.0`, Gradle `9.6.0`, compileSdk `37`, targetSdk `36`, minSdk `28`, NDK `28.2.13676358`, Compose BOM `2026.08.00`, WorkManager `2.11.2`, JDK `17`.
- Initial app version: `0.1.0-alpha01`, versionCode `1`.
- Do not bundle AI model binaries in the APK.

## File Structure

```text
KiraEnhance/
├── settings.gradle.kts
├── build.gradle.kts
├── gradle.properties
├── gradle/wrapper/gradle-wrapper.properties
├── app/
│   ├── build.gradle.kts
│   └── src/
│       ├── main/
│       │   ├── AndroidManifest.xml
│       │   ├── java/com/ikegami99/kiraenhance/
│       │   │   ├── MainActivity.kt
│       │   │   ├── KiraEnhanceApp.kt
│       │   │   ├── ui/theme/KiraTheme.kt
│       │   │   ├── ui/home/HomeScreen.kt
│       │   │   ├── ui/models/ModelManagerScreen.kt
│       │   │   ├── ui/models/ModelManagerViewModel.kt
│       │   │   ├── model/ModelDescriptor.kt
│       │   │   ├── model/ModelManifestParser.kt
│       │   │   ├── model/InstalledModelStore.kt
│       │   │   ├── device/DeviceCapabilities.kt
│       │   │   ├── device/DeviceCapabilityDetector.kt
│       │   │   ├── download/ModelDownloadManager.kt
│       │   │   ├── download/ModelDownloadWorker.kt
│       │   │   └── util/Sha256.kt
│       │   └── res/values/strings.xml
│       ├── test/java/com/ikegami99/kiraenhance/
│       │   ├── model/ModelManifestParserTest.kt
│       │   ├── device/DeviceCapabilityDetectorTest.kt
│       │   ├── download/ModelDownloadManagerTest.kt
│       │   └── util/Sha256Test.kt
│       └── androidTest/java/com/ikegami99/kiraenhance/
│           └── ui/models/ModelManagerScreenTest.kt
└── model-manifest.example.json
```

---

### Task 1: Android project scaffold and beginner-first shell

**Files:**
- Create: `settings.gradle.kts`
- Create: `build.gradle.kts`
- Create: `gradle.properties`
- Create: `gradle/wrapper/gradle-wrapper.properties`
- Create: `app/build.gradle.kts`
- Create: `app/src/main/AndroidManifest.xml`
- Create: `app/src/main/java/com/ikegami99/kiraenhance/MainActivity.kt`
- Create: `app/src/main/java/com/ikegami99/kiraenhance/KiraEnhanceApp.kt`
- Create: `app/src/main/java/com/ikegami99/kiraenhance/ui/theme/KiraTheme.kt`
- Create: `app/src/main/java/com/ikegami99/kiraenhance/ui/home/HomeScreen.kt`
- Create: `app/src/main/res/values/strings.xml`

**Interfaces:**
- Produces: `KiraEnhanceApp()` root composable and navigation route `models` reserved for Task 5.

- [ ] **Step 1: Create the Gradle project using the exact toolchain in Global Constraints.**

Root plugin configuration:

```kotlin
plugins {
    id("com.android.application") version "9.4.0" apply false
    id("org.jetbrains.kotlin.android") version "2.2.20" apply false
    id("org.jetbrains.kotlin.plugin.compose") version "2.2.20" apply false
    id("com.google.devtools.ksp") version "2.2.20-2.0.3" apply false
}
```

Set Gradle distribution to `gradle-9.6.0-bin.zip` and JDK target to 17.

- [ ] **Step 2: Configure the app module.**

Use namespace/package `com.ikegami99.kiraenhance`, `compileSdk = 37`, `targetSdk = 36`, `minSdk = 28`, version `0.1.0-alpha01`, versionCode `1`, and `ndkVersion = "28.2.13676358"`. Enable Compose and add Material 3, lifecycle-runtime-compose, activity-compose, navigation-compose, WorkManager `2.11.2`, Moshi, OkHttp, JUnit, and Compose UI test dependencies.

- [ ] **Step 3: Create the theme.**

Define a dark-first Material 3 scheme with soft plum/lavender/pink accents, rounded surfaces, and no copyrighted game assets. Support system/light/dark via `KiraEnhanceTheme(darkTheme: Boolean = isSystemInDarkTheme(), content: @Composable () -> Unit)`.

- [ ] **Step 4: Create HomeScreen.**

The screen must show the app name, a large `画像を選ぶ` primary button, four mode preview cards (`忠実`, `おすすめ`, `高精細`, `UltraSharp`), and a `モデル管理` action. In this milestone the image button may show a disabled/informational state because inference is not implemented.

- [ ] **Step 5: Add root navigation.**

`KiraEnhanceApp()` uses a `NavHost` with `home` as start destination and a placeholder `models` destination so Task 5 can replace the placeholder without changing HomeScreen's callback shape.

- [ ] **Step 6: Build and run unit-test task.**

Run:

```bash
./gradlew :app:assembleDebug :app:testDebugUnitTest
```

Expected: build succeeds and unit-test task completes with no failures.

- [ ] **Step 7: Commit.**

```bash
git add .
git commit -m "feat: scaffold KiraEnhance Android app"
```

---

### Task 2: Typed model manifest and local installed-state store

**Files:**
- Create: `app/src/main/java/com/ikegami99/kiraenhance/model/ModelDescriptor.kt`
- Create: `app/src/main/java/com/ikegami99/kiraenhance/model/ModelManifestParser.kt`
- Create: `app/src/main/java/com/ikegami99/kiraenhance/model/InstalledModelStore.kt`
- Create: `app/src/test/java/com/ikegami99/kiraenhance/model/ModelManifestParserTest.kt`
- Create: `model-manifest.example.json`

**Interfaces:**
- Produces: `ModelDescriptor`, `ModelControlDescriptor`, `ModelManifest`, `ModelManifestParser.parse(json: String): ModelManifest`.
- Produces: `InstalledModelStore.modelsDir()` and `InstalledModelStore.modelFile(modelId: String, version: String): File`.

- [ ] **Step 1: Write parser tests first.**

Test a manifest containing all four user-facing modes, including `community = true` for UltraSharp, multiple supported scales, license metadata, SHA-256, and a PiSA-specific control. Assert invalid backend names and malformed SHA-256 values are rejected.

- [ ] **Step 2: Run tests and verify failure.**

```bash
./gradlew :app:testDebugUnitTest --tests "*ModelManifestParserTest"
```

Expected: fail because parser/types do not exist.

- [ ] **Step 3: Implement immutable descriptor types.**

Use enums:

```kotlin
enum class ModelBackend { MNN, NCNN }
enum class EnhancementMode { FIDELITY, BALANCED, DETAIL, ULTRASHARP }
enum class ControlType { SLIDER, TOGGLE, CHOICE }
```

`ModelDescriptor` must include: `id`, `displayName`, `mode`, `version`, `backend`, `downloadUrl`, `fileSizeBytes`, `sha256`, `supportedScales`, `minAppVersion`, `estimatedRamMb`, `licenseName`, `licenseUrl`, `description`, `community`, and `controls`.

- [ ] **Step 4: Implement parser validation.**

Reject duplicate model ids, empty URLs, non-positive sizes, unsupported scale values outside `{2,4}`, SHA strings not matching 64 lowercase/uppercase hex characters, and slider controls where `min > default > max` is violated.

- [ ] **Step 5: Implement InstalledModelStore.**

Store activated model files below `filesDir/models/<modelId>/<version>/model.bin`. Partial downloads use `cacheDir/model-downloads/<modelId>-<version>.part`. Provide methods only for path ownership and installed checks; download/network logic belongs to Task 4.

- [ ] **Step 6: Add `model-manifest.example.json`.**

Include placeholder example URLs and zero-distribution notes only, not actual model binaries. The four entries are RealESRGAN Anime candidate, PiSA-SR required target, HAT-S candidate, and UltraSharp community model.

- [ ] **Step 7: Run tests.**

```bash
./gradlew :app:testDebugUnitTest --tests "*ModelManifestParserTest"
```

Expected: pass.

- [ ] **Step 8: Commit.**

```bash
git add app/src/main/java/com/ikegami99/kiraenhance/model app/src/test/java/com/ikegami99/kiraenhance/model model-manifest.example.json
git commit -m "feat: add typed model manifest"
```

---

### Task 3: Device capability detection

**Files:**
- Create: `app/src/main/java/com/ikegami99/kiraenhance/device/DeviceCapabilities.kt`
- Create: `app/src/main/java/com/ikegami99/kiraenhance/device/DeviceCapabilityDetector.kt`
- Create: `app/src/test/java/com/ikegami99/kiraenhance/device/DeviceCapabilityDetectorTest.kt`

**Interfaces:**
- Produces: `DeviceCapabilities(totalRamMb, availableStorageMb, vulkanVersion, deviceName, androidApi, supportTier)`.
- Produces: `SupportTier { RECOMMENDED, SUPPORTED, NOT_RECOMMENDED, UNSUPPORTED }`.

- [ ] **Step 1: Write pure classification tests.**

Classification function signature:

```kotlin
fun classifyDevice(totalRamMb: Long, hasVulkan12: Boolean, is64Bit: Boolean): SupportTier
```

Rules for v1:
- `UNSUPPORTED` if not 64-bit or Vulkan 1.2 is unavailable.
- `RECOMMENDED` at `>= 12_000 MB` RAM.
- `SUPPORTED` at `>= 8_000 MB` RAM.
- `NOT_RECOMMENDED` below `8_000 MB` RAM.

- [ ] **Step 2: Run test and verify failure.**

```bash
./gradlew :app:testDebugUnitTest --tests "*DeviceCapabilityDetectorTest"
```

- [ ] **Step 3: Implement classification and Android detector.**

Read memory from `ActivityManager.MemoryInfo.totalMem`, storage from `StatFs(filesDir)`, device name from `Build.MANUFACTURER` + `Build.MODEL`, API from `Build.VERSION.SDK_INT`, 64-bit from `Build.SUPPORTED_64_BIT_ABIS`, and Vulkan capability from `PackageManager.FEATURE_VULKAN_HARDWARE_VERSION` / system feature version.

- [ ] **Step 4: Run tests.**

```bash
./gradlew :app:testDebugUnitTest --tests "*DeviceCapabilityDetectorTest"
```

Expected: pass.

- [ ] **Step 5: Commit.**

```bash
git add app/src/main/java/com/ikegami99/kiraenhance/device app/src/test/java/com/ikegami99/kiraenhance/device
git commit -m "feat: detect device capability tier"
```

---

### Task 4: Verified resumable model downloader

**Files:**
- Create: `app/src/main/java/com/ikegami99/kiraenhance/util/Sha256.kt`
- Create: `app/src/main/java/com/ikegami99/kiraenhance/download/ModelDownloadManager.kt`
- Create: `app/src/main/java/com/ikegami99/kiraenhance/download/ModelDownloadWorker.kt`
- Create: `app/src/test/java/com/ikegami99/kiraenhance/util/Sha256Test.kt`
- Create: `app/src/test/java/com/ikegami99/kiraenhance/download/ModelDownloadManagerTest.kt`

**Interfaces:**
- Consumes: `ModelDescriptor`, `InstalledModelStore`.
- Produces: `ModelDownloadManager.enqueue(model: ModelDescriptor, wifiOnly: Boolean): UUID`.
- Worker progress keys: `bytesDownloaded`, `totalBytes`, `modelId`.

- [ ] **Step 1: Write SHA-256 tests.**

Use the known digest of UTF-8 `abc`: `ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad`. Test both file hashing and constant-time-ish normalized comparison.

- [ ] **Step 2: Write downloader request tests.**

Verify `wifiOnly = true` produces `NetworkType.UNMETERED`; otherwise `NetworkType.CONNECTED`. Verify unique work name is `model-download-<id>-<version>` with `ExistingWorkPolicy.KEEP`.

- [ ] **Step 3: Run tests and verify failure.**

```bash
./gradlew :app:testDebugUnitTest --tests "*Sha256Test" --tests "*ModelDownloadManagerTest"
```

- [ ] **Step 4: Implement Sha256.**

Stream with a fixed-size buffer; never load a model file wholly into RAM.

- [ ] **Step 5: Implement WorkManager scheduling.**

Pass only model metadata required by the worker through `Data` (`id`, `version`, `url`, `sha256`, `fileSizeBytes`). Do not serialize complete manifest state into WorkManager.

- [ ] **Step 6: Implement resumable HTTP download.**

If `.part` exists, send `Range: bytes=<size>-`. Accept `206` for resume. If the server ignores Range and returns `200`, truncate/restart safely. Update WorkManager progress no more often than every 250 ms or 1 MiB, whichever comes later, to avoid database churn.

- [ ] **Step 7: Verify and atomically activate.**

After expected length is reached, compute SHA-256. On mismatch delete the part and fail with an actionable reason. On success create the final version directory and move via `Files.move(..., StandardCopyOption.ATOMIC_MOVE)` when supported, falling back to a same-filesystem rename. Never expose a partial file as installed.

- [ ] **Step 8: Run tests.**

```bash
./gradlew :app:testDebugUnitTest --tests "*Sha256Test" --tests "*ModelDownloadManagerTest"
```

Expected: pass.

- [ ] **Step 9: Commit.**

```bash
git add app/src/main/java/com/ikegami99/kiraenhance/download app/src/main/java/com/ikegami99/kiraenhance/util app/src/test/java/com/ikegami99/kiraenhance/download app/src/test/java/com/ikegami99/kiraenhance/util
git commit -m "feat: add verified resumable model downloads"
```

---

### Task 5: Functional model manager UI

**Files:**
- Create: `app/src/main/java/com/ikegami99/kiraenhance/ui/models/ModelManagerScreen.kt`
- Create: `app/src/main/java/com/ikegami99/kiraenhance/ui/models/ModelManagerViewModel.kt`
- Create: `app/src/androidTest/java/com/ikegami99/kiraenhance/ui/models/ModelManagerScreenTest.kt`
- Modify: `app/src/main/java/com/ikegami99/kiraenhance/KiraEnhanceApp.kt`

**Interfaces:**
- Consumes: `ModelDescriptor`, `InstalledModelStore`, `DeviceCapabilities`, `ModelDownloadManager`.
- Produces: a real `models` destination and state model `ModelCardUiState`.

- [ ] **Step 1: Write Compose UI tests.**

With fake state, assert all four mode labels render, UltraSharp renders a `Community` badge, an uninstalled model has `ダウンロード`, an installed model has `削除`, and a NOT_RECOMMENDED model shows a warning without hard-blocking its download action.

- [ ] **Step 2: Run instrumented test compile.**

```bash
./gradlew :app:assembleDebug :app:assembleDebugAndroidTest
```

Expected before implementation: compile/test setup fails on missing screen types.

- [ ] **Step 3: Implement ModelManagerViewModel.**

Expose `StateFlow<ModelManagerUiState>` containing model cards, device tier, Wi-Fi-only preference (default true), and download progress. Keep Android `Context` out of the composable surface; dependencies enter the ViewModel/factory.

- [ ] **Step 4: Implement ModelManagerScreen.**

Use polished rounded Material 3 cards with mode title, technical model name as secondary text, description, size, RAM estimate, backend, license summary, installed/update/download state, and progress. Provide buttons for download/update/delete/reinstall where applicable. The community badge and license link must be visually obvious for UltraSharp.

- [ ] **Step 5: Replace navigation placeholder.**

Wire HomeScreen `モデル管理` to the real screen and back navigation. Do not expose inference actions yet.

- [ ] **Step 6: Build all milestone targets.**

```bash
./gradlew :app:assembleDebug :app:testDebugUnitTest :app:assembleDebugAndroidTest
```

Expected: all compile/build tasks pass.

- [ ] **Step 7: Manual smoke checklist on the reference device when available.**

Verify launch, dark/light theme, model-manager navigation, device tier, a small test-file download through the same worker path, cancellation/retry state, and deletion. Do not claim POCO F7 Ultra runtime validation unless actually executed on that device.

- [ ] **Step 8: Commit.**

```bash
git add app/src/main/java/com/ikegami99/kiraenhance/ui app/src/androidTest app/src/main/java/com/ikegami99/kiraenhance/KiraEnhanceApp.kt
git commit -m "feat: add model manager experience"
```

---

## Plan self-review

- Spec coverage for this milestone: Android shell, beginner UI direction, model metadata, model download/verification/removal, device guidance, model manager, UltraSharp community treatment.
- Intentionally deferred to later plans: JNI/NDK inference implementation, MNN/ncnn model execution, PiSA-SR conversion/integration, image import/save, 2x/4x processing, comparison viewer, foreground inference service, app updater, logs/export.
- No cloud inference or user account is introduced.
- Model binaries remain external to the APK.
- Task interfaces are monotonic: Task 2 model types feed Tasks 4/5; Task 3 capabilities feed Task 5; Task 4 downloader feeds Task 5.
