# PiSA-SR Native Latent Math Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and host-test the exact VAE posterior sampling math required after the PiSA VAE encoder produces posterior moments, without coupling it to MNN tensor names or graph execution.

**Architecture:** Put the numerical transform in a pure C++17 helper that has no JNI, Android, MNN, or random-number dependency. The helper consumes posterior moments plus caller-supplied standard-normal noise, applies the Diffusers 0.25.0 `DiagonalGaussianDistribution` math, and applies the SD VAE scaling factor. A Linux host test runs before Android SDK setup, while Android CMake links the same source later.

**Tech Stack:** C++17, host `g++` tests, Android NDK/CMake.

**Spec:** `docs/superpowers/specs/2026-09-17-pisa-sr-engine-design.md`

## Global Constraints

- Do not connect this helper to `MnnPisaNativeBridge.infer()` yet.
- Do not hard-code MNN tensor names or observed converted graph shapes.
- Match Diffusers 0.25.0: split moments in half along channel dimension, clamp log-variance to `[-30, 20]`, compute `std = exp(0.5 * logvar)`, then `sample = mean + std * noise`.
- Apply the VAE scaling factor after sampling.
- Noise is injected by the caller; random generation is a separate milestone.
- Preserve `kira_native.cpp` and the validated UltraSharp path.
- Keep changes small and CI-verified.

---

### Task 1: Establish a failing host-native contract

**Files:**
- Create: `app/src/test/cpp/pisa_latent_math_test.cpp`
- Modify: `.github/workflows/android.yml`

**Interfaces:**

```cpp
namespace kira::pisa {
bool sampleLatentFromMoments(
    const float* moments,
    const float* noise,
    float* output,
    int batch,
    int latentChannels,
    int height,
    int width,
    float scalingFactor
);
}
```

- [ ] Add deterministic tests for normal sampling/scaling, logvar clamps, NCHW batch/channel layout, and invalid arguments.
- [ ] Compile/run the host test with `g++ -std=c++17 -Wall -Wextra -Werror` before Android SDK setup.
- [ ] Verify RED because `pisa_latent_math.h/.cpp` do not yet exist.

### Task 2: Implement pure latent sampling math

**Files:**
- Create: `app/src/main/cpp/pisa_latent_math.h`
- Create: `app/src/main/cpp/pisa_latent_math.cpp`

- [ ] Validate pointers, positive dimensions, and finite positive scaling factor.
- [ ] Treat moments as contiguous NCHW `[batch, 2 * latentChannels, height, width]`.
- [ ] For each batch, first `latentChannels` planes are means and the second half are log-variances.
- [ ] Clamp logvar with `std::clamp(logvar, -30.0f, 20.0f)`.
- [ ] Compute `(mean + exp(0.5f * logvar) * noise) * scalingFactor`.
- [ ] Return `true` only when all arguments are valid.
- [ ] Run CI and verify the host test becomes GREEN.

### Task 3: Link the tested helper into the Android native library

**Files:**
- Modify: `app/src/main/cpp/CMakeLists.txt`

- [ ] Add `pisa_latent_math.cpp` to `kiraenhance`.
- [ ] Run full Android CI and APK native runtime verification.
- [ ] Confirm no JNI or inference behavior changes.

### Task 4: Preserve the boundary

- [ ] Confirm `MnnPisaNativeBridge.infer()` remains `NOT_IMPLEMENTED`.
- [ ] Confirm `mnn_pisa_loader.cpp` does not call the latent helper yet.
- [ ] Confirm `kira_native.cpp` remains untouched.
- [ ] Record the next milestone as deterministic Gaussian noise generation plus graph-execution wiring after a real converted graph contract is observed.
