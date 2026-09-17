# PiSA-SR Native Model Loader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the PiSA native model-load placeholder with a real MNN 3.6.1 loader that owns the three converted graph interpreters plus the empty-prompt artifact, while keeping image inference explicitly unimplemented.

**Architecture:** Keep loading separate from session/backend creation. Kotlin marshals a compact native load result and holds an opaque `Long` handle; C++ owns a heap-allocated `PisaModelBundle` containing three `MNN::Interpreter` instances and the prompt bytes. A later milestone will create OpenCL/Vulkan/CPU sessions from these interpreters and implement VAE/UNet execution.

**Tech Stack:** Kotlin/JNI, C++17, MNN 3.6.1 Android arm64 runtime, Android Gradle Plugin, JUnit4, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-17-pisa-sr-engine-design.md`

## Global Constraints

- Preserve the validated UltraSharp/ncnn path without changing `kira_native.cpp` behavior.
- PiSA package names remain exactly `vae_encoder.mnn`, `unet_default.mnn`, `vae_decoder.mnn`, and `empty_prompt.fp16`.
- Native image inference remains explicitly `NOT_IMPLEMENTED` in this milestone; no fabricated output is allowed.
- MNN runtime stays pinned to 3.6.1 and arm64-v8a in CI.
- `preferGpu` is accepted by the public API but must not report GPU enabled until an actual GPU session is successfully created in a later milestone.
- Every write is kept to a small reviewable unit and followed by CI verification before the next unit.

---

### Task 1: Define and test native load-result decoding

**Files:**
- Create: `app/src/test/java/com/ikegami99/kiraenhance/inference/mnn/MnnPisaNativeLoadResultCodecTest.kt`
- Create: `app/src/main/java/com/ikegami99/kiraenhance/inference/mnn/MnnPisaNativeLoadResultCodec.kt`

**Interfaces:**
- Consumes native `LongArray` payload `[handle, errorOrdinal, gpuEnabledFlag]`.
- Produces `MnnPisaNativeLoadResult`.
- Invalid length, error ordinal, or GPU flag produces `MnnPisaNativeError.INTERNAL` with handle `0L` and `gpuEnabled=false` instead of throwing.

- [ ] **Step 1: Write the failing codec tests**

```kotlin
@Test
fun `decodes successful native load payload`() {
    assertEquals(
        MnnPisaNativeLoadResult(
            handle = 41L,
            errorCode = MnnPisaNativeError.NONE,
            gpuEnabled = false,
        ),
        MnnPisaNativeLoadResultCodec.decode(
            longArrayOf(41L, MnnPisaNativeError.NONE.ordinal.toLong(), 0L),
        ),
    )
}

@Test
fun `rejects malformed native load payload`() {
    assertEquals(
        MnnPisaNativeLoadResult(
            handle = 0L,
            errorCode = MnnPisaNativeError.INTERNAL,
            gpuEnabled = false,
        ),
        MnnPisaNativeLoadResultCodec.decode(longArrayOf(41L)),
    )
}
```

Also cover an out-of-range error ordinal and a GPU flag other than `0` or `1`.

- [ ] **Step 2: Run CI and verify RED**

Expected failure: `MnnPisaNativeLoadResultCodec` is unresolved.

- [ ] **Step 3: Implement the minimal decoder**

```kotlin
object MnnPisaNativeLoadResultCodec {
    fun decode(payload: LongArray): MnnPisaNativeLoadResult {
        if (payload.size != 3) return internalFailure()
        val error = MnnPisaNativeError.entries.getOrNull(payload[1].toInt())
            ?: return internalFailure()
        val gpuEnabled = when (payload[2]) {
            0L -> false
            1L -> true
            else -> return internalFailure()
        }
        val handle = payload[0]
        if (error == MnnPisaNativeError.NONE && handle == 0L) return internalFailure()
        return MnnPisaNativeLoadResult(handle, error, gpuEnabled)
    }

    private fun internalFailure() = MnnPisaNativeLoadResult(
        handle = 0L,
        errorCode = MnnPisaNativeError.INTERNAL,
        gpuEnabled = false,
    )
}
```

- [ ] **Step 4: Run full Android CI and verify GREEN**

Required workflow command remains:

`gradle :app:assembleDebug :app:testDebugUnitTest :app:assembleDebugAndroidTest --stacktrace`

- [ ] **Step 5: Commit**

Commit message: `test: define MNN native load result codec`

---

### Task 2: Add the C++ MNN model-bundle loader

**Files:**
- Create: `app/src/main/cpp/mnn_pisa_loader.cpp`
- Modify: `app/src/main/cpp/CMakeLists.txt`

**Interfaces:**
- JNI load method returns `jlongArray` with exactly three values: `[handle, errorOrdinal, gpuEnabledFlag]`.
- `handle` is a pointer to a heap-allocated `PisaModelBundle` cast through `intptr_t` to `jlong`.
- `PisaModelBundle` owns three `MNN::Interpreter` instances through `std::unique_ptr<MNN::Interpreter, InterpreterDeleter>`.
- `PisaModelBundle` owns `empty_prompt.fp16` as `std::vector<uint8_t>`.
- This task returns `gpuEnabledFlag=0` because no execution session exists yet.

- [ ] **Step 1: Add the loader source to CMake**

Add `mnn_pisa_loader.cpp` next to `kira_native.cpp` and `mnn_runtime_probe.cpp` in the `kiraenhance` shared library source list. Do not modify the ncnn implementation.

- [ ] **Step 2: Implement defensive JNI string conversion**

Use a scoped helper that calls `GetStringUTFChars` and always calls `ReleaseStringUTFChars`. Reject null or empty Java strings with `INVALID_ARGUMENT`.

- [ ] **Step 3: Implement interpreter ownership**

```cpp
struct InterpreterDeleter {
    void operator()(MNN::Interpreter* interpreter) const {
        if (interpreter != nullptr) {
            MNN::Interpreter::destroy(interpreter);
        }
    }
};

using InterpreterPtr = std::unique_ptr<MNN::Interpreter, InterpreterDeleter>;

struct PisaModelBundle {
    InterpreterPtr vaeEncoder;
    InterpreterPtr unet;
    InterpreterPtr vaeDecoder;
    std::vector<uint8_t> emptyPrompt;
};
```

Create each interpreter using `MNN::Interpreter::createFromFile(path.c_str())`. If any interpreter is null, return `LOAD_FAILED`; already-created interpreters are destroyed automatically by RAII.

- [ ] **Step 4: Read and validate the empty-prompt artifact**

Open the file in binary mode, require a positive byte count and an even byte count because it is FP16 data, then read the complete file into `emptyPrompt`. Any open/read/size failure returns `LOAD_FAILED`.

Do not hard-code the final CLIP tensor shape in this task; the later UNet session task will validate the converted graph's encoder-hidden-state input shape against the artifact before inference.

- [ ] **Step 5: Return the opaque handle only after all four artifacts load**

Allocate the completed bundle with `std::make_unique<PisaModelBundle>()`, release ownership only after all fields validate, and return:

```text
[reinterpret_cast<intptr_t>(bundle.release()), NONE.ordinal, 0]
```

The C++ numeric error constants must mirror the Kotlin enum order exactly:

```text
NONE=0
LOAD_FAILED=1
INVALID_ARGUMENT=2
NOT_IMPLEMENTED=3
INFERENCE_FAILED=4
CANCELLED=5
OUT_OF_MEMORY=6
INTERNAL=7
```

- [ ] **Step 6: Implement native unload**

Convert a non-zero `jlong` back to `PisaModelBundle*` through `intptr_t` and `delete` it. Handle `0` as a no-op.

- [ ] **Step 7: Run Android CI and verify native compile/link GREEN**

Expected: C++17 compile succeeds against MNN 3.6.1, APK still contains both `libkiraenhance.so` and `libMNN.so`, JVM tests stay green, and androidTest sources compile.

- [ ] **Step 8: Commit**

Commit message: `feat: add native PiSA MNN model bundle loader`

---

### Task 3: Connect Kotlin bridge to native load/unload

**Files:**
- Modify: `app/src/main/java/com/ikegami99/kiraenhance/inference/mnn/MnnPisaNativeBridge.kt`
- Test: `app/src/test/java/com/ikegami99/kiraenhance/inference/mnn/MnnPisaUpscaleEngineTest.kt` remains the lifecycle contract and must continue passing with its fake API.

**Interfaces:**
- New JNI declarations:

```kotlin
private external fun nativeLoadModel(
    vaeEncoderPath: String,
    unetPath: String,
    vaeDecoderPath: String,
    emptyPromptPath: String,
    preferGpu: Boolean,
): LongArray

private external fun nativeUnloadModel(handle: Long)
```

- `infer(...)` remains the existing pure-Kotlin typed `NOT_IMPLEMENTED` result so JVM tests do not attempt to load Android native libraries.
- `cancel(...)` remains a no-op until sessions/execution exist.

- [ ] **Step 1: Centralize native-library loading**

Replace the runtime-probe-only direct `System.loadLibrary` call with a private function used by both `runtimeInfo()` and `loadModel()`:

```kotlin
private val nativeLibraryLoaded by lazy {
    System.loadLibrary("kiraenhance")
    true
}

private fun ensureNativeLibraryLoaded() {
    nativeLibraryLoaded
}
```

Do not call it from object initialization, `infer()`, or `cancel()`; JVM unit tests must remain able to instantiate/use placeholder inference without an Android `.so`.

- [ ] **Step 2: Implement real bridge load**

Call `ensureNativeLibraryLoaded()`, invoke `nativeLoadModel(...)`, and decode with `MnnPisaNativeLoadResultCodec.decode(...)`.

If native loading or invocation throws, return:

```kotlin
MnnPisaNativeLoadResult(
    handle = 0L,
    errorCode = MnnPisaNativeError.LOAD_FAILED,
    gpuEnabled = false,
)
```

This keeps `MnnPisaUpscaleEngine.load()` on its existing typed failure path instead of crashing the app.

- [ ] **Step 3: Implement bridge unload**

For non-zero handles, call `ensureNativeLibraryLoaded()` and `nativeUnloadModel(handle)` inside `runCatching` so cleanup cannot crash navigation teardown.

- [ ] **Step 4: Run full Android CI**

Expected: all current JVM tests pass, the native loader compiles, androidTest sources compile, and APK native-runtime verification passes.

- [ ] **Step 5: Commit**

Commit message: `feat: connect PiSA MNN native model loading`

---

### Task 4: Add device-executable invalid-package smoke coverage

**Files:**
- Create: `app/src/androidTest/java/com/ikegami99/kiraenhance/inference/mnn/MnnPisaNativeLoaderInstrumentedTest.kt`

**Interfaces:**
- Test calls the real JNI loader with four temporary files that are intentionally not valid MNN graphs.
- Expected result is `LOAD_FAILED`, `handle=0`, `gpuEnabled=false`.

- [ ] **Step 1: Write the instrumentation test**

Use `ApplicationProvider.getApplicationContext<Context>().cacheDir` to create a unique directory. Write one byte to each `.mnn` file and two bytes to `empty_prompt.fp16`, call `MnnPisaNativeBridge.loadModel(...)`, then assert the typed load failure.

- [ ] **Step 2: Compile androidTest in CI**

Expected: `assembleDebugAndroidTest` succeeds. Do not claim this test executed in GitHub Actions because the current workflow has no Android device/emulator.

- [ ] **Step 3: Run on the POCO device during the next APK validation pass**

Expected: invalid graphs fail cleanly rather than crashing or returning a non-zero handle. Record the result in the device test log before moving to converted real graphs.

- [ ] **Step 4: Commit**

Commit message: `test: add PiSA native loader device smoke case`

---

### Task 5: Verify loader milestone and keep inference boundary truthful

**Files:**
- No production file changes unless verification reveals a defect.

**Interfaces:**
- `MnnPisaNativeBridge.loadModel()` is real.
- `MnnPisaNativeBridge.unload()` is real.
- `MnnPisaNativeBridge.infer()` still returns `NOT_IMPLEMENTED`.
- `gpuEnabled` remains false until a real MNN session/backend is created.

- [ ] **Step 1: Run full branch CI at HEAD**

Require success for build/test compile, packaged native runtime verification, and APK upload.

- [ ] **Step 2: Review the PR diff**

Confirm this milestone does not modify `kira_native.cpp`, does not enable the placeholder PiSA manifest download, and does not claim working PiSA image inference.

- [ ] **Step 3: Record the next implementation boundary**

The next plan begins with MNN session creation and backend selection in this order: OpenCL FP16, Vulkan, ARM CPU fallback. Only after sessions and graph I/O shapes are validated should VAE encode, timestep-1 UNet, VAE decode, tiling, stochastic latent sampling, and color alignment be implemented.
