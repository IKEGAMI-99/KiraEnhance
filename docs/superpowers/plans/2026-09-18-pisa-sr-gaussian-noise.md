# PiSA-SR Deterministic Gaussian Noise Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide a deterministic standard-normal noise source for PiSA VAE posterior sampling without relying on implementation-defined `std::normal_distribution`.

**Architecture:** Implement a small pure C++17 generator using SplitMix64 for deterministic uniform bits and Box-Muller for the Gaussian transform. Keep it separate from VAE latent math so numerical sampling can accept externally supplied reference noise during validation. Host CI locks a seed-42 prefix and basic distribution quality before Android links the same source.

**Tech Stack:** C++17, host `g++`, Android NDK/CMake.

**Spec:** `docs/superpowers/specs/2026-09-17-pisa-sr-engine-design.md`

## Global Constraints

- Do not attempt to reproduce PyTorch's exact RNG bitstream.
- Do not use `std::random_device`, `std::normal_distribution`, or platform RNG APIs in the deterministic generator.
- Seed-to-sequence behavior must be stable in KiraEnhance and covered by a known-prefix host test.
- Gaussian output must be finite and statistically close to mean 0 / variance 1 for a fixed large sample.
- Keep the generator independent from MNN/JNI and from `sampleLatentFromMoments`.
- Do not wire random generation into `infer()` until a real converted MNN graph contract has been inspected.

---

### Task 1: Establish failing host tests

**Files:**
- Create: `app/src/test/cpp/pisa_gaussian_noise_test.cpp`
- Modify: `.github/workflows/android.yml`

**Interface:**

```cpp
namespace kira::pisa {
bool fillGaussianNoise(
    float* output,
    std::size_t count,
    std::uint64_t seed
);
}
```

- [ ] Lock the first six float outputs for seed 42.
- [ ] Verify the same seed repeats exactly and a different seed changes the sequence.
- [ ] Verify 20,000 values have finite outputs, absolute mean below 0.03, and variance between 0.95 and 1.05.
- [ ] Reject null output and zero count.
- [ ] Verify RED because the implementation files do not exist.

### Task 2: Implement SplitMix64 + Box-Muller

**Files:**
- Create: `app/src/main/cpp/pisa_gaussian_noise.h`
- Create: `app/src/main/cpp/pisa_gaussian_noise.cpp`

- [ ] Implement SplitMix64 using fixed unsigned 64-bit constants and overflow semantics.
- [ ] Convert the top 53 random bits to a double strictly inside `(0, 1)`.
- [ ] Generate pairs with Box-Muller using a fixed `2*pi` constant.
- [ ] Cast generated doubles to float in output order cosine then sine.
- [ ] Support odd counts without writing past the output buffer.
- [ ] Run CI and verify host tests GREEN.

### Task 3: Link into Android native library

**Files:**
- Modify: `app/src/main/cpp/CMakeLists.txt`

- [ ] Add `pisa_gaussian_noise.cpp` to `kiraenhance`.
- [ ] Run full Android CI.
- [ ] Confirm no JNI/inference behavior changes.

### Task 4: Preserve validation injection

- [ ] Keep `sampleLatentFromMoments` accepting caller-supplied noise.
- [ ] Do not combine the two helpers yet.
- [ ] The later graph-execution milestone can use deterministic generated noise for app runs and externally supplied noise for PyTorch/MNN numerical comparison.
