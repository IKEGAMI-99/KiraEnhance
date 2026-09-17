# PiSA-SR MNN Session Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create real MNN sessions for all three PiSA graphs and select one truthful execution backend in the order OpenCL low precision, Vulkan low precision, then ARM CPU fallback.

**Architecture:** Extend the existing native `PisaModelBundle` so model loading and session ownership stay in one C++ file, but keep tensor I/O and graph execution out of scope. A backend attempt succeeds only when all three sessions are created and MNN `Interpreter::getSessionInfo(..., BACKENDS, ...)` reports the requested primary backend for every graph. Kotlin continues to receive only `gpuEnabled`; detailed tensor execution comes in the next milestone.

**Tech Stack:** Kotlin/JNI, C++17 with `-fno-exceptions`, MNN 3.6.1 Android arm64 CPU/OpenCL/Vulkan, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-17-pisa-sr-engine-design.md`

## Global Constraints

- Preserve the validated UltraSharp/ncnn path and do not modify `kira_native.cpp`.
- Keep MNN pinned to 3.6.1.
- Backend preference is OpenCL FP16-style low precision, then Vulkan low precision, then ARM CPU fallback.
- Never report `gpuEnabled=true` unless MNN reports OpenCL or Vulkan as the primary backend for all three PiSA sessions.
- `infer()` remains `NOT_IMPLEMENTED`; this milestone creates sessions only.
- Keep native compilation compatible with the existing `-fno-exceptions` build.
- Each change is small and CI-verified before the next change.

---

### Task 1: Add pure backend preference contract

**Files:**
- Create: `app/src/main/java/com/ikegami99/kiraenhance/inference/mnn/MnnPisaBackend.kt`
- Create: `app/src/test/java/com/ikegami99/kiraenhance/inference/mnn/MnnPisaBackendTest.kt`

**Interfaces:**
- `enum class MnnPisaBackend { OPENCL, VULKAN, CPU }`
- `MnnPisaBackend.preference(preferGpu: Boolean): List<MnnPisaBackend>`
- GPU preference is exactly `[OPENCL, VULKAN, CPU]`; CPU-only is exactly `[CPU]`.

- [ ] **Step 1: Write failing JVM tests**

```kotlin
@Test
fun `gpu preference tries OpenCL then Vulkan then CPU`() {
    assertEquals(
        listOf(MnnPisaBackend.OPENCL, MnnPisaBackend.VULKAN, MnnPisaBackend.CPU),
        MnnPisaBackend.preference(preferGpu = true),
    )
}

@Test
fun `cpu preference skips gpu backends`() {
    assertEquals(
        listOf(MnnPisaBackend.CPU),
        MnnPisaBackend.preference(preferGpu = false),
    )
}
```

- [ ] **Step 2: Run CI and verify RED**

Expected: unresolved `MnnPisaBackend`.

- [ ] **Step 3: Implement the minimal enum**

```kotlin
enum class MnnPisaBackend {
    OPENCL,
    VULKAN,
    CPU;

    companion object {
        fun preference(preferGpu: Boolean): List<MnnPisaBackend> =
            if (preferGpu) listOf(OPENCL, VULKAN, CPU) else listOf(CPU)
    }
}
```

- [ ] **Step 4: Run CI and verify GREEN**

- [ ] **Step 5: Commit**

Commit: `feat: define PiSA MNN backend preference`

---

### Task 2: Create and validate native sessions

**Files:**
- Modify: `app/src/main/cpp/mnn_pisa_loader.cpp`

**Interfaces:**
- `PisaModelBundle` gains three `MNN::Session*` fields plus selected backend metadata.
- Sessions are owned by their respective `MNN::Interpreter`; bundle teardown destroys interpreters and therefore their sessions.
- Backend validation uses `Interpreter::getSessionInfo(session, Interpreter::BACKENDS, &backendType)`.
- `gpuEnabled` is true only for validated OpenCL or Vulkan sessions.

- [ ] **Step 1: Add session fields**

```cpp
struct PisaModelBundle {
    InterpreterPtr vaeEncoder;
    InterpreterPtr unet;
    InterpreterPtr vaeDecoder;
    std::unique_ptr<std::uint8_t[]> emptyPrompt;
    std::size_t emptyPromptBytes = 0;
    MNN::Session* vaeEncoderSession = nullptr;
    MNN::Session* unetSession = nullptr;
    MNN::Session* vaeDecoderSession = nullptr;
    MNNForwardType backend = MNN_FORWARD_CPU;
};
```

- [ ] **Step 2: Add session cleanup for failed backend attempts**

For every non-null session call the matching interpreter's `releaseSession(session)`, then null the pointer. This helper is used before trying the next backend.

- [ ] **Step 3: Build a schedule config per backend**

For all backends:

```cpp
MNN::BackendConfig backendConfig;
backendConfig.precision = MNN::BackendConfig::Precision_Low;
backendConfig.memory = MNN::BackendConfig::Memory_Low;
backendConfig.power = MNN::BackendConfig::Power_High;

MNN::ScheduleConfig config;
config.type = requestedType;
config.backupType = MNN_FORWARD_CPU;
config.backendConfig = &backendConfig;
```

For OpenCL set `config.mode = MNN_GPU_TUNING_FAST`.
For Vulkan set `config.mode = MNN_GPU_TUNING_NONE`.
For CPU set `config.numThread = 4`.

- [ ] **Step 4: Create all three sessions**

Call `createSession(config)` on encoder, UNet, and decoder. If any returns null, release any sessions already created and reject the attempt.

- [ ] **Step 5: Verify the actual primary backend**

For each session:

```cpp
int backendType = -1;
if (!interpreter->getSessionInfo(
        session,
        MNN::Interpreter::BACKENDS,
        &backendType
    ) ||
    backendType != static_cast<int>(requestedType)
) {
    // reject this backend attempt
}
```

This prevents a requested GPU session that actually resolved to CPU from being reported as GPU-enabled.

- [ ] **Step 6: Try backends in required order**

If `preferGpu == JNI_TRUE`, attempt `MNN_FORWARD_OPENCL`, then `MNN_FORWARD_VULKAN`, then `MNN_FORWARD_CPU`.
Otherwise attempt CPU only.

If no attempt succeeds, return `LOAD_FAILED`.
If one succeeds, retain those sessions in the bundle and return the existing handle payload with `gpuEnabled=1` only for OpenCL or Vulkan.

- [ ] **Step 7: Run full Android CI**

Required: C++ compile/link, JVM tests, androidTest compilation, APK native-runtime verification.

- [ ] **Step 8: Commit**

Commit: `feat: create PiSA MNN sessions with backend fallback`

---

### Task 3: Expose selected backend for diagnostics

**Files:**
- Modify: `app/src/main/cpp/mnn_pisa_loader.cpp`
- Modify: `app/src/main/java/com/ikegami99/kiraenhance/inference/mnn/MnnPisaNativeBridge.kt`
- Create: `app/src/main/java/com/ikegami99/kiraenhance/inference/mnn/MnnPisaSessionInfo.kt`
- Create: `app/src/test/java/com/ikegami99/kiraenhance/inference/mnn/MnnPisaSessionInfoTest.kt`

**Interfaces:**
- JNI: `nativeSessionInfo(handle: Long): String`
- Wire format: `backend=opencl|vulkan|cpu;gpu=0|1`
- Kotlin parser rejects malformed strings rather than guessing.

- [ ] **Step 1: Write failing parser tests**

```kotlin
@Test
fun `parses OpenCL session info`() {
    assertEquals(
        MnnPisaSessionInfo(MnnPisaBackend.OPENCL, gpuEnabled = true),
        MnnPisaSessionInfo.parse("backend=opencl;gpu=1"),
    )
}
```

Also test Vulkan, CPU, and malformed data.

- [ ] **Step 2: Run CI and verify RED**

- [ ] **Step 3: Implement parser and native serializer**

The native method looks up the bundle handle and returns the selected backend only. It does not run any graph.

- [ ] **Step 4: Connect the bridge**

Call `ensureNativeLibraryLoaded()` before `nativeSessionInfo(handle)`.

- [ ] **Step 5: Run CI and verify GREEN**

- [ ] **Step 6: Commit**

Commit: `feat: expose PiSA MNN session backend info`

---

### Task 4: Preserve truthful inference boundary and verify milestone

**Files:**
- No production changes unless verification finds a defect.

- [ ] **Step 1: Confirm `MnnPisaNativeBridge.infer()` still returns `NOT_IMPLEMENTED`**

- [ ] **Step 2: Confirm `kira_native.cpp` remains untouched by this milestone**

- [ ] **Step 3: Confirm latest Android CI is fully successful**

- [ ] **Step 4: Record next boundary**

The next milestone is graph I/O inspection and validation: enumerate VAE encoder inputs/outputs, UNet latent/timestep/encoder-hidden-state inputs, and VAE decoder I/O; validate converted graph shapes before implementing any image inference.
