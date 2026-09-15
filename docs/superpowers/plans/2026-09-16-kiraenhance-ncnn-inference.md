# KiraEnhance ncnn/Vulkan Inference Implementation Plan

**Goal:** Add KiraEnhance's first real on-device inference backend using ncnn + Vulkan, targeting arm64-v8a Android first and supporting ESRGAN/RRDB-style `.param + .bin` model bundles such as 4x-UltraSharp and Real-ESRGAN.

**Base:** `feat/ncnn-inference` starts from foundation commit `7ca4c95fccc2b26a08a53ce76d40ea54e4937209`, where app/model-manager CI is green.

**Runtime pin:** ncnn `20260526`, official Android Vulkan prebuilt archive `ncnn-20260526-android-vulkan.zip`, SHA-256 `26909c92eed35afed4a966b5e9e503fcb0a529691ea3f910ec2c94a4fff52804`.

**Platform:** Android API 28+, compile/target API 36, NDK `28.2.13676358`, first ABI `arm64-v8a`. The first reference hardware remains POCO F7 Ultra / Snapdragon 8 Elite class.

**Privacy:** Image inference remains fully local. Model weights stay outside the APK and are loaded from app-private storage.

## Architecture

```text
Compose / ViewModel
      ↓
UpscaleEngine (Kotlin interface)
      ↓
NcnnUpscaleEngine
      ↓ JNI
NativeNcnnEngine (C++)
      ↓
ncnn::Net + Vulkan
      ↓
.param + .bin model bundle
```

The Kotlin layer never depends on a specific ESRGAN model name. Model-specific details such as input/output blob names, scale and padding are capabilities supplied to the engine.

## Task 1: Add reproducible ncnn native dependency and JNI build

Files:
- Modify `app/build.gradle.kts`
- Create `app/src/main/cpp/CMakeLists.txt`
- Create `app/src/main/cpp/kira_native.cpp`
- Modify `.github/workflows/android.yml`

Steps:
1. Add `externalNativeBuild` using CMake and restrict the first inference milestone to `arm64-v8a`.
2. In CI, download the pinned official ncnn Android Vulkan archive, verify the exact SHA-256, extract it, and pass the arm64-v8a `ncnn_DIR` into Gradle/CMake.
3. Configure CMake with `find_package(ncnn REQUIRED)` and link `ncnn`, Android `log`, and JNI/native Android dependencies only where required.
4. Export a tiny JNI function `nativeRuntimeInfo()` that returns an engine/version string without loading a model.
5. Add a Kotlin wrapper and unit-testable runtime-info parser.
6. Run `assembleDebug`, unit tests, and AndroidTest compilation. This task is complete only when CI proves the native `.so` links successfully.

## Task 2: Define common engine contracts before model-specific code

Files:
- Create `app/src/main/java/com/ikegami99/kiraenhance/inference/UpscaleEngine.kt`
- Create `app/src/main/java/com/ikegami99/kiraenhance/inference/EngineCapabilities.kt`
- Create tests under `app/src/test/.../inference/`

Contract:

```kotlin
interface UpscaleEngine : AutoCloseable {
    fun load(request: ModelLoadRequest): ModelLoadResult
    fun isLoaded(): Boolean
    fun upscale(input: UpscaleInput, settings: UpscaleSettings): UpscaleResult
    fun cancel()
    fun unload()
}
```

Rules:
- No UI class imports.
- No model name hard-coding.
- Cancellation is observable by the native implementation.
- Capability data declares native scale, accepted pixel format, input/output blob names, padding and GPU availability.

TDD:
1. Add contract/capability validation tests first.
2. Verify RED.
3. Implement minimal immutable types and validation.
4. Verify GREEN.

## Task 3: Implement native ncnn model loading

Files:
- Modify `kira_native.cpp`
- Create `app/src/main/java/.../inference/ncnn/NcnnNativeBridge.kt`
- Create `app/src/main/java/.../inference/ncnn/NcnnUpscaleEngine.kt`
- Add tests for Kotlin-side path/capability validation.

Requirements:
- Accept explicit `.param` and `.bin` paths from `InstalledModelStore`; never search arbitrary filesystem paths.
- Require both files to exist before JNI is called.
- Create `ncnn::Net` with Vulkan enabled when available.
- Prefer fp16 packed/storage where supported, but keep fp16 arithmetic disabled initially for safer image fidelity.
- Allow CPU fallback only as an explicit engine result/state, not silently hidden from diagnostics.
- `load_param()` and `load_model()` failures must be surfaced as typed errors.
- Native object ownership uses an opaque handle and deterministic `unload()/close()`.

Initial ESRGAN defaults:
- input blob: `data`
- output blob: `output`
- native scale: `4`

These defaults must be overridable by capability metadata.

## Task 4: Implement a small-image 4x inference path

Purpose: prove real pixels can cross Kotlin → JNI → ncnn/Vulkan → Kotlin before building the full tiler.

Files:
- Modify native engine
- Add bitmap/pixel conversion helpers
- Add an internal developer inference screen or callable debug path

Approach:
- Decode with Android/Kotlin APIs.
- Pass RGBA8888 pixels to native code through an Android `Bitmap`/JNI lock rather than serializing millions of boxed values.
- Convert to RGB ncnn Mat, normalized consistently with ESRGAN inputs.
- Run `Extractor.input("data", ...)`, then `extract("output", ...)`.
- Convert/clamp output back to RGBA8888.
- Preserve alpha by resizing the original alpha plane rather than hallucinating alpha through the RGB model.

Guardrails:
- First smoke path limits input dimensions so a full-frame 4x output cannot exhaust Java/native memory.
- Full screenshot support belongs to Task 5 tiling.
- Reject unsupported output dimensions before allocating.

## Task 5: Add tile planning and seam-safe processing

Files:
- Create `TilePlanner.kt` + tests
- Add native per-tile inference support

TDD cases:
- exact tile boundary
- odd image dimensions
- edge/corner tiles
- overlap/padding removal
- decreasing tile size after memory pressure

Implementation:
- Default initial tile size: conservative auto selection from available RAM/GPU info.
- ESRGAN/RRDB initial pre-padding: 10 px, model-overridable.
- Crop model padding after inference before composing the output.
- Process tiles sequentially first; parallel GPU queues are deferred until correctness is proven.
- Progress = completed tiles / total tiles.
- Cancellation checked between tiles and before large allocations.

## Task 6: Wire UltraSharp as the first real downloadable ncnn model

Before changing production manifest URLs, re-verify license, source URLs, exact byte sizes and SHA-256 for every artifact.

Target model:
- `4x-UltraSharp`
- Architecture: ESRGAN
- Native scale: 4x
- FP16 NCNN artifacts from the author's Hugging Face repository
- Known bin size: `33,424,520` bytes
- Known bin SHA-256: `713ce69a8642b1907cc24da7573560dcc933faf0c74e48ab0b3b379245618701`
- Param artifact and its checksum must be independently verified before manifest activation.
- License: `CC BY-NC-SA 4.0`; attribution/non-commercial restrictions remain visible in the model manager.

Do not claim production readiness until the param checksum is verified and a real device successfully loads/runs the model.

## Task 7: Add an Android inference smoke APK artifact

CI:
- Build debug APK with ncnn linked.
- Upload APK as a GitHub Actions artifact for installation on the POCO F7 Ultra.
- Keep AI weights external.

Manual device acceptance test:
1. Install debug APK.
2. Confirm ncnn runtime/Vulkan device is detected.
3. Install/download UltraSharp test bundle.
4. Run a small screenshot crop.
5. Confirm output dimensions are 4x.
6. Inspect for channel swap, black output, seams or crashes.
7. Record runtime, tile size and memory diagnostics.

## Exit Criteria

This milestone is complete only when:
- native ncnn/Vulkan code compiles into the Android APK;
- the common `UpscaleEngine` abstraction exists;
- `.param + .bin` loading is implemented with typed errors;
- a real ESRGAN-compatible model can run a pixel inference path;
- tile planning is covered by unit tests;
- CI is green for build/unit/AndroidTest compilation;
- a debug APK is produced for real-device validation;
- no model weight is bundled in the APK.

PiSA-SR/MNN remains the next separate milestone after this ncnn path is stable.