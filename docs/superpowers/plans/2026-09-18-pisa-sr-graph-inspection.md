# PiSA-SR MNN Graph Inspection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Inspect the loaded PiSA-SR MNN graphs and expose their input/output tensor names, shapes, data types, and dimension types without running inference.

**Architecture:** Keep graph observation generic because the converted PiSA artifacts are not finalized yet. Native code reads tensor metadata from the already-created encoder, UNet, and decoder sessions using MNN 3.6.1 APIs, serializes a delimiter-safe line protocol, and Kotlin parses it into typed immutable records. No expected PiSA tensor names or shapes are hard-coded until real converted graphs can be inspected.

**Tech Stack:** Kotlin/JVM tests, JNI, C++17 with `-fno-exceptions`, MNN 3.6.1.

**Spec:** `docs/superpowers/specs/2026-09-17-pisa-sr-engine-design.md`

## Global Constraints

- Preserve UltraSharp/ncnn and do not modify `kira_native.cpp`.
- Keep MNN pinned to 3.6.1.
- Do not call `runSession()` in this milestone.
- Do not write image, latent, timestep, or prompt data into tensors.
- Do not hard-code expected converted tensor names or shapes yet.
- Tensor names must be serialized delimiter-safely.
- `infer()` remains `NOT_IMPLEMENTED`.
- Use small CI-verified commits.

---

### Task 1: Define typed graph metadata and wire decoder

**Files:**
- Create: `app/src/main/java/com/ikegami99/kiraenhance/inference/mnn/MnnPisaTensorInfo.kt`
- Create: `app/src/test/java/com/ikegami99/kiraenhance/inference/mnn/MnnPisaTensorInfoTest.kt`

**Interfaces:**

```kotlin
enum class MnnPisaGraph { VAE_ENCODER, UNET, VAE_DECODER }
enum class MnnTensorRole { INPUT, OUTPUT }

data class MnnPisaTensorInfo(
    val graph: MnnPisaGraph,
    val role: MnnTensorRole,
    val name: String,
    val shape: List<Int>,
    val typeCode: Int,
    val typeBits: Int,
    val typeLanes: Int,
    val dimensionType: Int,
)

object MnnPisaTensorInfoCodec {
    fun decode(raw: String): List<MnnPisaTensorInfo>?
}
```

**Wire format:** one tensor per line, exactly seven semicolon-separated fields:

```text
graph=vae_encoder;role=input;nameHex=73616d706c65;shape=1,3,64,64;type=2,32,1;dim=0
```

Allowed graph values: `vae_encoder`, `unet`, `vae_decoder`.
Allowed role values: `input`, `output`.
`nameHex` is the UTF-8 tensor name encoded as lowercase hexadecimal, so tensor names cannot break delimiters.
`shape` may be empty for a scalar tensor.
`type` is exactly MNN/Halide `code,bits,lanes`.
`dim` is the integer value of MNN `Tensor::DimensionType`.

- [ ] **Step 1: Write failing decoder tests**

Test one record for each graph, multiple dimensions, an empty scalar shape, and rejection of malformed hex, unknown graph/role, malformed type, duplicate/missing fields, and non-integer dimensions.

- [ ] **Step 2: Run CI and verify RED**

Expected: unresolved `MnnPisaTensorInfo` / `MnnPisaTensorInfoCodec`.

- [ ] **Step 3: Implement the minimal strict decoder**

Decode hex pairs into UTF-8 bytes, require an even non-empty hex string, require exactly the seven named fields, and return `null` for any malformed line. An empty raw string represents a valid empty list only at the codec level.

- [ ] **Step 4: Run CI and verify GREEN**

- [ ] **Step 5: Commit**

Commit: `feat: decode PiSA MNN graph metadata`

---

### Task 2: Serialize native MNN tensor metadata

**Files:**
- Modify: `app/src/main/cpp/mnn_pisa_loader.cpp`

**Interfaces:**
- JNI: `nativeGraphInfo(handle: Long): String`
- Native reads only existing sessions.
- Tensor enumeration uses:
  - `Interpreter::getSessionInputAll(session)`
  - `Interpreter::getSessionOutputAll(session)`
  - `Tensor::shape()`
  - `Tensor::getType()`
  - `Tensor::getDimensionType()`

- [ ] **Step 1: Add UTF-8-name hex encoding**

For every byte in the MNN tensor name, emit exactly two lowercase hex characters.

- [ ] **Step 2: Add one tensor serializer**

Serialize graph, role, encoded name, comma-separated shape, `type.code,type.bits,type.lanes`, and numeric dimension type using the exact Task 1 protocol.

- [ ] **Step 3: Enumerate all six maps**

Append, in this order:
1. VAE encoder inputs
2. VAE encoder outputs
3. UNet inputs
4. UNet outputs
5. VAE decoder inputs
6. VAE decoder outputs

The MNN maps already provide deterministic name ordering.

- [ ] **Step 4: Add JNI entrypoint**

`nativeGraphInfo(0)` returns an empty string.
A valid handle serializes its loaded sessions without running them.
When MNN is not linked, return an empty string.

- [ ] **Step 5: Run full Android CI**

Required: native compile/link, JVM tests, androidTest compile, APK native runtime verification.

- [ ] **Step 6: Commit**

Commit: `feat: inspect PiSA MNN graph tensors`

---

### Task 3: Connect graph inspection to Kotlin

**Files:**
- Modify: `app/src/main/java/com/ikegami99/kiraenhance/inference/mnn/MnnPisaNativeBridge.kt`

**Interfaces:**

```kotlin
private external fun nativeGraphInfo(handle: Long): String

fun graphInfo(handle: Long): List<MnnPisaTensorInfo>? 
```

- [ ] **Step 1: Declare the JNI method**

- [ ] **Step 2: Add safe public wrapper**

Return `null` for handle 0, native-load failure, JNI failure, or malformed metadata. Otherwise return the strict codec result.

- [ ] **Step 3: Confirm JVM inference tests still do not load the native library**

Calling `infer()` must remain safe in JVM unit tests and continue to return `NOT_IMPLEMENTED`.

- [ ] **Step 4: Run full Android CI**

- [ ] **Step 5: Commit**

Commit: `feat: expose PiSA MNN graph inspection`

---

### Task 4: Verify the observation-only boundary

**Files:**
- No production changes unless verification finds a defect.

- [ ] **Step 1:** Confirm `MnnPisaNativeBridge.infer()` still returns `NOT_IMPLEMENTED`.
- [ ] **Step 2:** Confirm native graph inspection contains no `runSession()`, tensor host/device writes, resize calls, or image buffers.
- [ ] **Step 3:** Confirm `kira_native.cpp` remains untouched.
- [ ] **Step 4:** Confirm latest Android CI is fully successful.
- [ ] **Step 5:** Record the next milestone as real converted-artifact inspection, followed by an explicit PiSA graph contract based on observed names/shapes/types.
