# Production Upscale Flow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the validated UltraSharp smoke path into a user-facing full-image workflow that imports an image, runs tiled 4x on-device inference, compares original vs enhanced, and saves the result.

**Architecture:** Keep the existing `UpscaleEngine`/`NcnnUpscaleEngine` backend unchanged and add a production orchestration layer around it. A single `EnhanceRoute` owns the selected image and runs an `EnhanceProcessor`; pure helpers handle model-load requests, RGBA conversion, filenames, and UI presentation. The existing smoke route stays available for diagnostics.

**Tech Stack:** Kotlin, Jetpack Compose, Android ImageDecoder/Bitmap, MediaStore, coroutines, existing ncnn/Vulkan backend.

**Spec:** `docs/superpowers/specs/2026-09-15-kiraenhance-design.md`

## Global Constraints

- Android only, API 28+, compile/target API 36.
- Inference remains fully local; image bytes are never uploaded.
- 4x-UltraSharp uses the existing ncnn backend and downloaded `.param + .bin` artifacts.
- Full images use existing automatic tiling and cancellation.
- Output defaults to PNG in `Pictures/KiraEnhance/` where MediaStore relative paths are available.
- Existing ncnn smoke route remains intact as a diagnostic path.

---

### Task 1: Production inference helpers

**Files:**
- Create: `app/src/main/java/com/ikegami99/kiraenhance/inference/ncnn/UltraSharpModelRequestFactory.kt`
- Create: `app/src/main/java/com/ikegami99/kiraenhance/image/RgbaPixelCodec.kt`
- Create: `app/src/main/java/com/ikegami99/kiraenhance/image/EnhanceOutputNamer.kt`
- Test: matching files under `app/src/test/...`

**Interfaces:**
- `UltraSharpModelRequestFactory.create(model, artifactPath): ModelLoadRequest`
- `RgbaPixelCodec.encodeArgb(width, height, argb): UpscaleInput`
- `RgbaPixelCodec.decodeRow(output, y, destination): IntArray`
- `EnhanceOutputNamer.png(timestamp, mode, scale): String`

- [ ] Write failing tests proving UltraSharp requires ncnn + 4x + param/bin, RGBA byte ordering is correct, row-stride decode works, and output filenames are deterministic.
- [ ] Run `:app:testDebugUnitTest` and verify tests fail because production helpers do not exist.
- [ ] Implement the minimum helper code.
- [ ] Re-run unit tests and verify green.

### Task 2: Full-image processor

**Files:**
- Create: `app/src/main/java/com/ikegami99/kiraenhance/ui/enhance/EnhanceProcessor.kt`
- Create: `app/src/main/java/com/ikegami99/kiraenhance/ui/enhance/EnhanceUiState.kt`
- Test: `app/src/test/java/com/ikegami99/kiraenhance/ui/enhance/EnhanceUiStateTest.kt`

**Interfaces:**
- `EnhanceProcessor.run(bitmap, model, artifactPath, onProgress): EnhanceProcessResult`
- `EnhanceProcessor.cancel()`
- UI state distinguishes selecting, ready, processing, completed, saved, and error states.

- [ ] Write failing reducer/state tests for ready -> processing -> completed/error and progress clamping.
- [ ] Implement processor: convert source bitmap to `UpscaleInput`, load model, run `NcnnUpscaleEngine.upscale`, poll tile progress, convert RGBA output to ARGB bitmap row-by-row, and always unload engine.
- [ ] Preserve cancellation through `engine.cancel()` and surface typed engine errors as Japanese actionable messages.
- [ ] Verify unit tests.

### Task 3: User-facing Enhance route and comparison viewer

**Files:**
- Create: `app/src/main/java/com/ikegami99/kiraenhance/ui/enhance/EnhanceRoute.kt`
- Create: `app/src/main/java/com/ikegami99/kiraenhance/ui/enhance/EnhanceScreen.kt`
- Create: `app/src/main/java/com/ikegami99/kiraenhance/ui/enhance/CompareViewer.kt`
- Modify: `app/src/main/java/com/ikegami99/kiraenhance/KiraEnhanceApp.kt`
- Modify: `app/src/main/java/com/ikegami99/kiraenhance/ui/home/HomeScreen.kt`
- AndroidTest: `app/src/androidTest/java/com/ikegami99/kiraenhance/ui/enhance/EnhanceScreenTest.kt`

**Behavior:**
- Home `＋ 画像を選ぶ` opens `enhance` route.
- Route immediately offers system image picker and keeps a replace-image action.
- Ready screen shows source preview/resolution, UltraSharp 4x, and Enhance button.
- Processing shows stage, percentage when tiled progress is available, elapsed time, target resolution, and Cancel.
- Completed screen uses a draggable split divider with shared pinch zoom/pan and provides Save and Re-run.

- [ ] Add failing Compose test that the ready screen exposes Enhance and the processing screen exposes Cancel.
- [ ] Implement screen state rendering and navigation.
- [ ] Implement split comparison with `detectTransformGestures`, bounded zoom, shared transform, and draggable divider.
- [ ] Verify AndroidTest compilation.

### Task 4: Save enhanced image

**Files:**
- Create: `app/src/main/java/com/ikegami99/kiraenhance/image/EnhancedImageSaver.kt`
- Modify: `EnhanceRoute.kt`
- Modify: `app/src/main/AndroidManifest.xml` only if API 28 public-picture fallback requires legacy storage permission.

**Interface:**
- `EnhancedImageSaver.savePng(context, bitmap, displayName): SaveResult`

- [ ] Add pure tests for save-name and destination policy where possible.
- [ ] On API 29+, insert through MediaStore using `RELATIVE_PATH=Pictures/KiraEnhance` and `IS_PENDING`.
- [ ] On API 28, use a safe app-visible fallback without breaking API 29+ behavior.
- [ ] Surface the saved location/state in UI and log save failures without image data.

### Task 5: Verification and integration

- [ ] Run GitHub Actions Android CI for the feature branch.
- [ ] Confirm `assembleDebug`, unit tests, and AndroidTest compilation pass.
- [ ] Inspect changed files for accidental cloud/image-upload code or smoke-route regressions.
- [ ] Produce the debug APK artifact for real-device validation.
