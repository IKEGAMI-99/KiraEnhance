# PiSA-SR Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare KiraEnhance for a truthful, selectable PiSA-SR BALANCED engine while preserving the validated UltraSharp production flow.

**Architecture:** Introduce an engine binding/resolver between model descriptors and the shared enhancement processor. Add a typed PiSA/MNN request and engine boundary, keep PiSA unavailable until real converted artifacts exist, and pin the official MNN Android runtime in CI. The first branch proves architecture and compilation; real converted PiSA graph execution is a separate device-validation milestone.

**Tech Stack:** Kotlin, Jetpack Compose, Android Navigation, JNI/C++17, ncnn, MNN 3.6.1, JUnit4, AndroidX Compose tests, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-17-pisa-sr-engine-design.md`

## Global Constraints

- Preserve existing UltraSharp output scale 4 and its production request factory behavior.
- PiSA maps to `EnhancementMode.BALANCED` + `ModelBackend.MNN`.
- PiSA placeholder `example.invalid` artifacts must never be scheduled for download.
- PiSA default mobile package contract is `vae_encoder.mnn`, `unet_default.mnn`, `vae_decoder.mnn`, and `empty_prompt.fp16`.
- No fabricated PiSA output. Until converted graphs exist, native inference must fail explicitly rather than return a passthrough or fake enhancement.
- MNN runtime is pinned to 3.6.1 Android CPU/OpenCL/Vulkan release SHA-256 `46dc7e86d45b8d4e957db81d2603e0b7f6c9ce9b84092ffdcee1b843cbfc9d71`.
- Keep processing fully on-device. No network inference fallback.

---

### Task 1: Prevent placeholder model downloads

**Files:**
- Create: `app/src/main/java/com/ikegami99/kiraenhance/model/ModelDistribution.kt`
- Modify: `app/src/main/java/com/ikegami99/kiraenhance/download/ModelDownloadManager.kt`
- Test: `app/src/test/java/com/ikegami99/kiraenhance/model/ModelDistributionTest.kt`
- Test: `app/src/test/java/com/ikegami99/kiraenhance/download/ModelDownloadManagerTest.kt`

**Interfaces:**
- Produces: `ModelDescriptor.isDownloadableProductionArtifactSet(): Boolean`
- Produces: `ModelDescriptor.requireDownloadable()`

- [ ] **Step 1: Write failing model distribution tests**

```kotlin
@Test
fun `rejects example invalid placeholder artifacts`() {
    assertFalse(model(url = "https://example.invalid/pisa/model.mnn").isDownloadableProductionArtifactSet())
}

@Test
fun `accepts https artifact with size and sha`() {
    assertTrue(model(url = "https://models.example.com/pisa/model.mnn").isDownloadableProductionArtifactSet())
}
```

- [ ] **Step 2: Run CI and verify RED**

Push the tests only. GitHub Actions `Android CI` must fail because `isDownloadableProductionArtifactSet` does not exist.

- [ ] **Step 3: Implement production artifact validation**

```kotlin
fun ModelDescriptor.isDownloadableProductionArtifactSet(): Boolean =
    artifacts.isNotEmpty() && artifacts.all { artifact ->
        val uri = runCatching { URI(artifact.downloadUrl) }.getOrNull()
        uri?.scheme == "https" &&
            !uri.host.isNullOrBlank() &&
            uri.host != "example.invalid" &&
            artifact.fileSizeBytes > 0 &&
            SHA_256.matches(artifact.sha256) &&
            artifact.sha256.toSet().size > 1
    }
```

`ModelDownloadManager.enqueue()` calls `model.requireDownloadable()` before creating WorkManager work.

- [ ] **Step 4: Run CI and verify GREEN**

Expected: unit tests compile and pass with the existing Android build.

- [ ] **Step 5: Commit**

Commit message: `feat: block placeholder model downloads`

---

### Task 2: Introduce engine bindings and resolver

**Files:**
- Create: `app/src/main/java/com/ikegami99/kiraenhance/inference/ModelRequestFactory.kt`
- Create: `app/src/main/java/com/ikegami99/kiraenhance/ui/enhance/EnhanceEngineBinding.kt`
- Create: `app/src/main/java/com/ikegami99/kiraenhance/ui/enhance/EnhanceEngineResolver.kt`
- Modify: `app/src/main/java/com/ikegami99/kiraenhance/inference/ncnn/UltraSharpModelRequestFactory.kt`
- Test: `app/src/test/java/com/ikegami99/kiraenhance/ui/enhance/EnhanceEngineResolverTest.kt`

**Interfaces:**
- Produces: `fun interface ModelRequestFactory { fun create(model: ModelDescriptor, artifactPath: (String) -> String): ModelLoadRequest }`
- Produces: `data class EnhanceEngineBinding(val engine: UpscaleEngine, val requestFactory: ModelRequestFactory, val outputScale: Int, val saveModeName: String)`
- Produces: `EnhanceEngineResolver.resolve(model: ModelDescriptor): EnhanceEngineBinding`

- [ ] **Step 1: Write failing resolver tests**

```kotlin
@Test
fun `maps UltraSharp ncnn to 4x binding`() {
    val binding = resolver.resolve(model(mode = EnhancementMode.ULTRASHARP, backend = ModelBackend.NCNN))
    assertEquals(4, binding.outputScale)
    assertEquals("UltraSharp", binding.saveModeName)
}

@Test(expected = IllegalArgumentException::class)
fun `rejects unsupported backend and mode pairing`() {
    resolver.resolve(model(mode = EnhancementMode.ULTRASHARP, backend = ModelBackend.MNN))
}
```

The test resolver injects engine provider lambdas so JVM tests never load JNI.

- [ ] **Step 2: Run CI and verify RED**

Expected failure: binding/resolver types do not exist.

- [ ] **Step 3: Implement interface and UltraSharp mapping**

`UltraSharpModelRequestFactory` implements `ModelRequestFactory`. `EnhanceEngineResolver` accepts providers:

```kotlin
class EnhanceEngineResolver(
    private val ncnnProvider: () -> UpscaleEngine,
    private val mnnPisaProvider: () -> UpscaleEngine,
    private val pisaRequestFactory: ModelRequestFactory,
) {
    fun resolve(model: ModelDescriptor): EnhanceEngineBinding = when {
        model.mode == EnhancementMode.ULTRASHARP && model.backend == ModelBackend.NCNN ->
            EnhanceEngineBinding(ncnnProvider(), UltraSharpModelRequestFactory, 4, "UltraSharp")
        model.mode == EnhancementMode.BALANCED && model.backend == ModelBackend.MNN ->
            EnhanceEngineBinding(mnnPisaProvider(), pisaRequestFactory, 4, "PiSA-SR")
        else -> throw IllegalArgumentException("Unsupported enhancement model: ${model.id} (${model.mode}/${model.backend})")
    }
}
```

- [ ] **Step 4: Run CI and verify GREEN**

Expected: resolver and existing UltraSharp factory tests pass.

- [ ] **Step 5: Commit**

Commit message: `refactor: resolve enhancement engines by model`

---

### Task 3: Make EnhanceProcessor backend-agnostic

**Files:**
- Modify: `app/src/main/java/com/ikegami99/kiraenhance/ui/enhance/EnhanceProcessor.kt`
- Test: `app/src/test/java/com/ikegami99/kiraenhance/ui/enhance/EnhanceProcessorTest.kt`

**Interfaces:**
- Consumes: `EnhanceEngineBinding`
- Produces: `EnhanceProcessor.run(bitmap, model, binding, artifactPath, onProgress)`

- [ ] **Step 1: Write failing fake-engine processor test**

The fake engine records its `ModelLoadRequest` and returns a deterministic 2x2 RGBA output for a 1x1 input. The request factory records that it was called. Assert that `EnhanceProcessor` uses both injected objects and never references ncnn-specific types.

```kotlin
val result = processor.run(
    bitmap = source,
    model = descriptor,
    binding = EnhanceEngineBinding(fakeEngine, fakeRequestFactory, 4, "Test"),
    artifactPath = { "/models/$it" },
    onProgress = {},
)
assertTrue(result is EnhanceProcessResult.Success)
```

- [ ] **Step 2: Run CI and verify RED**

Expected failure: current `run()` has no `binding` parameter and hard-codes `UltraSharpModelRequestFactory`.

- [ ] **Step 3: Refactor processor**

Remove imports of `NcnnUpscaleEngine` and `UltraSharpModelRequestFactory`. The engine comes from `binding.engine`, the load request comes from `binding.requestFactory.create(...)`, and `UpscaleSettings.outputScale` uses `binding.outputScale`.

Generic error text uses the model display name rather than hard-coded UltraSharp wording.

- [ ] **Step 4: Run CI and verify GREEN**

Expected: fake-engine test and existing unit tests pass.

- [ ] **Step 5: Commit**

Commit message: `refactor: make production processor backend agnostic`

---

### Task 4: Add truthful PiSA/MNN request and engine boundary

**Files:**
- Create: `app/src/main/java/com/ikegami99/kiraenhance/inference/mnn/PisaModelRequestFactory.kt`
- Create: `app/src/main/java/com/ikegami99/kiraenhance/inference/mnn/MnnPisaNativeApi.kt`
- Create: `app/src/main/java/com/ikegami99/kiraenhance/inference/mnn/MnnPisaNativeBridge.kt`
- Create: `app/src/main/java/com/ikegami99/kiraenhance/inference/mnn/MnnPisaUpscaleEngine.kt`
- Test: `app/src/test/java/com/ikegami99/kiraenhance/inference/mnn/PisaModelRequestFactoryTest.kt`
- Test: `app/src/test/java/com/ikegami99/kiraenhance/inference/mnn/MnnPisaUpscaleEngineTest.kt`

**Interfaces:**
- `PisaModelRequestFactory : ModelRequestFactory`
- Required artifacts: `vae_encoder.mnn`, `unet_default.mnn`, `vae_decoder.mnn`, `empty_prompt.fp16`
- `MnnPisaNativeApi.loadModel(...)`, `infer(...)`, `cancel(...)`, `unload(...)`

- [ ] **Step 1: Write failing PiSA request tests**

```kotlin
@Test
fun `creates balanced MNN 4x request with four default artifacts`() {
    val request = PisaModelRequestFactory.create(model(), pathResolver)
    assertEquals(4, request.capabilities.nativeScale)
    assertNotNull(request.artifact("vae_encoder.mnn"))
    assertNotNull(request.artifact("unet_default.mnn"))
    assertNotNull(request.artifact("vae_decoder.mnn"))
    assertNotNull(request.artifact("empty_prompt.fp16"))
}
```

Reject wrong backend/mode, missing artifact, or missing 4x support.

- [ ] **Step 2: Run CI and verify RED**

Expected failure: PiSA factory and engine types do not exist.

- [ ] **Step 3: Implement request factory and native API types**

Use generic blob names `image`/`output`, scale 4, pre-padding 0, GPU support true. Artifact semantic roles are resolved by filename in the MNN engine.

- [ ] **Step 4: Write engine lifecycle tests with fake native API**

Cover successful load state, missing files mapped to `MODEL_ARTIFACT_MISSING`, native load failure mapped to `MODEL_LOAD_FAILED`, explicit cancellation, and the current native placeholder inference returning `INFERENCE_FAILED` rather than fake output.

- [ ] **Step 5: Implement Kotlin MNN engine boundary**

`MnnPisaUpscaleEngine` implements `UpscaleEngine`. Until the native graph executor is implemented, `MnnPisaNativeBridge.infer` returns a typed native `NOT_IMPLEMENTED`/inference failure. The engine must not copy input to output.

- [ ] **Step 6: Run CI and verify GREEN**

Expected: all JVM tests pass before JNI/MNN linking is added.

- [ ] **Step 7: Commit**

Commit message: `feat: add PiSA MNN engine contract`

---

### Task 5: Wire mode selection and model-specific setup

**Files:**
- Modify: `app/src/main/java/com/ikegami99/kiraenhance/KiraEnhanceApp.kt`
- Modify: `app/src/main/java/com/ikegami99/kiraenhance/ui/home/HomeScreen.kt`
- Modify: `app/src/main/java/com/ikegami99/kiraenhance/ui/enhance/EnhanceRoute.kt`
- Test: `app/src/androidTest/java/com/ikegami99/kiraenhance/ui/home/HomeScreenTest.kt`

**Interfaces:**
- Route: `enhance/{mode}` where mode is `balanced` or `ultrasharp`
- Home callback: `onStartEnhance: (EnhancementMode) -> Unit`

- [ ] **Step 1: Add failing Compose selection tests**

Assert `おすすめ` and `UltraSharp` are clickable, `忠実` and `高精細` expose unavailable/disabled semantics, and selecting `おすすめ` calls `EnhancementMode.BALANCED`.

- [ ] **Step 2: Run CI and verify RED**

Expected failure: cards have no click action/callback.

- [ ] **Step 3: Implement selectable cards and route argument**

Use stable lowercase route tokens, parse explicitly, and reject unknown tokens by navigating back rather than silently using UltraSharp.

`EnhanceRoute` looks up the descriptor by selected mode, validates installation/distribution state, resolves the binding, logs backend/mode, and uses `binding.saveModeName` for output naming.

- [ ] **Step 4: Run CI and verify GREEN**

Expected: Compose test sources compile and unit tests remain green.

- [ ] **Step 5: Commit**

Commit message: `feat: select PiSA or UltraSharp enhancement mode`

---

### Task 6: Pin MNN 3.6.1 in native CI without claiming PiSA inference

**Files:**
- Modify: `.github/workflows/android.yml`
- Modify: `app/build.gradle.kts`
- Modify: `app/src/main/cpp/CMakeLists.txt`
- Modify: `app/src/main/cpp/kira_native.cpp`

**Interfaces:**
- Environment variable: `MNN_DIR` points to ABI-specific MNN CMake package.
- Native library links `MNN` in addition to `ncnn`.
- JNI bridge exposes MNN availability/runtime metadata only; image inference remains explicitly unsupported until converted graphs exist.

- [ ] **Step 1: Update CI download step**

Download:

`https://github.com/alibaba/MNN/releases/download/3.6.1/mnn_3.6.1_android_armv7_armv8_cpu_opencl_vulkan.zip`

Verify SHA-256:

`46dc7e86d45b8d4e957db81d2603e0b7f6c9ce9b84092ffdcee1b843cbfc9d71`

Locate the arm64 CMake config and export `MNN_DIR`.

- [ ] **Step 2: Pass MNN_DIR through Gradle/CMake**

```kotlin
providers.environmentVariable("MNN_DIR").orNull
    ?.takeIf { it.isNotBlank() }
    ?.let { arguments += "-DMNN_DIR=$it" }
```

CMake uses `find_package(MNN REQUIRED CONFIG)` and links target `MNN`.

- [ ] **Step 3: Add minimal JNI MNN runtime probe**

Compile against MNN headers and expose a native runtime-available method. Do not add graph inference here.

- [ ] **Step 4: Run full Android CI**

Required command in workflow remains:

`gradle :app:assembleDebug :app:testDebugUnitTest :app:assembleDebugAndroidTest --stacktrace`

Expected: debug APK, JVM tests, androidTest compilation, ncnn native link, and MNN native link all succeed.

- [ ] **Step 5: Commit**

Commit message: `build: link pinned MNN Android runtime`

---

### Task 7: Final verification and handoff for real PiSA conversion

**Files:**
- Modify: `docs/superpowers/specs/2026-09-17-pisa-sr-engine-design.md` only if implementation discoveries require factual corrections.

- [ ] **Step 1: Verify branch diff against PR #3 head**

Confirm no unrelated UltraSharp/native changes and no placeholder PiSA download was enabled.

- [ ] **Step 2: Verify latest Android CI**

Require all workflow jobs/checks on branch HEAD to be successful before any completion claim.

- [ ] **Step 3: Open a PR targeting `feat/production-upscale-flow`**

PR summary must state that this is the PiSA/MNN integration foundation, not a claim of working PiSA image inference.

- [ ] **Step 4: Record next concrete milestone**

Next work is an offline conversion/validation pipeline that produces the four default PiSA artifacts, numerically compares them against upstream PyTorch on representative 512px crops, then performs the first POCO F7 Ultra MNN inference test.
