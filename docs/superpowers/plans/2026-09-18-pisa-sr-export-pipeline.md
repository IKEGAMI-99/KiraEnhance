# PiSA-SR Reproducible MNN Export Pipeline Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible offline tool that starts from the official PiSA-SR checkpoint plus a local Stable Diffusion 2.1-base directory and produces the four Android artifacts `vae_encoder.mnn`, `unet_default.mnn`, `vae_decoder.mnn`, and `empty_prompt.fp16` plus a verification manifest.

**Architecture:** Export three ONNX graphs with the official PiSA-SR custom model classes, fuse pixel and semantic LoRA into the default UNet exactly in upstream order, export VAE encoder moments instead of embedding stochastic sampling into the graph, precompute the empty-prompt CLIP hidden state, then invoke a user-supplied MNNConvert binary. Keep heavy ML imports lazy so contract/unit tests run in normal CI without downloading models.

**Tech Stack:** Python 3.10, PyTorch 2.0.1, Diffusers 0.25.0, Transformers 4.28.1, PEFT 0.9.0, ONNX, MNNConvert, Python unittest.

**Spec:** `docs/superpowers/specs/2026-09-17-pisa-sr-engine-design.md`

## Global Constraints

- Do not download or redistribute PiSA-SR, SD 2.1, or converted model weights from CI.
- PiSA-SR source is Apache 2.0, while SD 2.1 model weights have their own model license; do not enable app distribution until the combined redistribution terms are reviewed.
- Default PiSA inference uses empty prompt, timestep 1, one UNet pass, and `x_denoised = encoded_control - model_pred`.
- Default UNet fusion order must match upstream: merge pixel LoRA first, then merge semantic LoRA at weight 1.0.
- VAE encoder export must output posterior moments; Android native code will perform stochastic latent sampling later.
- Exported MNN models are correctness-first FP16 baselines. Quantization is a later milestone.
- MNNConvert path/version must be explicit and recorded in the manifest.
- Do not change the app model manifest from placeholder/unavailable state in this plan.

---

### Task 1: Add pure export contract and CI tests

**Files:**
- Create: `tools/pisa_export/export_contract.py`
- Create: `tools/pisa_export/tests/test_export_contract.py`
- Modify: `.github/workflows/android.yml`

**Interfaces:**

```python
ARTIFACT_NAMES = (
    "vae_encoder.mnn",
    "unet_default.mnn",
    "vae_decoder.mnn",
    "empty_prompt.fp16",
)

def build_mnnconvert_command(
    binary: str,
    onnx_path: str,
    mnn_path: str,
) -> list[str]

def sha256_file(path: pathlib.Path) -> str

def write_export_manifest(
    output_path: pathlib.Path,
    *,
    source_checkpoint: pathlib.Path,
    sd21_path: pathlib.Path,
    mnnconvert_version: str,
    metadata: dict,
    artifact_paths: list[pathlib.Path],
) -> None
```

MNN conversion command must be:

```text
MNNConvert -f ONNX --modelFile <onnx> --MNNModel <mnn> --bizCode KiraEnhancePiSA --fp16
```

Manifest must include:
- schemaVersion = 1
- source PiSA checkpoint path and SHA-256
- local SD2.1 path
- MNNConvert version string
- timestep = 1
- prompt = ""
- VAE scaling factor
- empty prompt tensor shape
- each produced artifact filename, byte size, SHA-256
- export timestamp in UTC ISO-8601

- [ ] Write unit tests for exact artifact names, exact MNNConvert argument ordering, SHA-256, and deterministic manifest field structure excluding timestamp.
- [ ] Run CI and verify RED.
- [ ] Implement the pure helper module.
- [ ] Add `python3 -m unittest discover -s tools/pisa_export/tests -v` to Android CI before Gradle.
- [ ] Run CI and verify GREEN.
- [ ] Commit in a maximum of three files.

---

### Task 2: Implement official PiSA default-UNet fusion

**Files:**
- Create: `tools/pisa_export/export_pisa_sr.py`

**Interfaces:**

```python
def load_official_components(
    pisa_repo: pathlib.Path,
    sd21_path: pathlib.Path,
    pisa_checkpoint: pathlib.Path,
    device: str,
):
    # returns fused_default_unet, vae, tokenizer, text_encoder

def merge_default_unet(unet, checkpoint: dict):
    # pixel adapters -> copy pixel LoRA -> merge
    # semantic adapters -> copy semantic LoRA -> merge
    # returns the same UNet with no active LoRA adapters
```

- [ ] Parse CLI arguments without importing torch/diffusers/transformers/peft.
- [ ] Insert `--pisa-repo` into `sys.path` only after validating it contains `src/models/unet_2d_condition.py` and `src/models/autoencoder_kl.py`.
- [ ] Import the upstream custom UNet/VAE classes and the exact pinned dependency APIs.
- [ ] Load the official `pisa_sr.pkl` with `map_location="cpu"`.
- [ ] Recreate upstream pixel adapter configs from checkpoint module lists/rank, copy only pixel LoRA parameters, activate all three pixel adapters at 1.0, and `merge_and_unload()`.
- [ ] Recreate semantic adapters, copy semantic LoRA parameters, activate all three semantic adapters at 1.0, and `merge_and_unload()`.
- [ ] Assert no parameter name contains `lora_` after final merge.
- [ ] Keep this code callable without constructing the upstream `PiSASR_eval`, because that class hard-codes CUDA and installs VAE tiling hooks.
- [ ] Run Python syntax/unit CI; real fusion is workstation-only.
- [ ] Commit one file.

---

### Task 3: Export VAE moments, fused UNet, decoder, and empty prompt

**Files:**
- Modify: `tools/pisa_export/export_pisa_sr.py`

**Interfaces:**

```python
class VaeEncoderMoments(torch.nn.Module):
    def forward(self, image):
        h = self.vae.encoder(image)
        return self.vae.quant_conv(h)

class VaeDecoder(torch.nn.Module):
    def forward(self, latent):
        latent = self.vae.post_quant_conv(latent)
        return self.vae.decoder(latent)

class DefaultUnet(torch.nn.Module):
    def forward(self, latent, timestep, encoder_hidden_states):
        return self.unet(
            latent,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
        ).sample
```

- [ ] Export with batch size 1 and opset 17.
- [ ] Use dynamic spatial axes for image/latent height and width.
- [ ] Keep timestep shape `[1]` and dtype int64.
- [ ] Derive text hidden size and tokenizer max length from loaded components rather than hard-coding 77/1024.
- [ ] Generate empty-prompt hidden state with the official tokenizer/text encoder, convert to little-endian FP16, and write `empty_prompt.fp16`.
- [ ] Record empty prompt shape and VAE scaling factor for the manifest.
- [ ] Export ONNX files as `vae_encoder.onnx`, `unet_default.onnx`, and `vae_decoder.onnx`.
- [ ] Run `onnx.checker.check_model` on all three.
- [ ] Do not call VAE `latent_dist.sample()` during export.
- [ ] Commit one file.

---

### Task 4: Convert ONNX to MNN and record provenance

**Files:**
- Modify: `tools/pisa_export/export_pisa_sr.py`
- Create: `tools/pisa_export/README.md`

- [ ] Require `--mnnconvert /absolute/path/to/MNNConvert`.
- [ ] Call `MNNConvert --version` and record its exact output.
- [ ] Convert all three ONNX graphs with `--fp16` using Task 1 command construction.
- [ ] Verify every MNN artifact exists and is non-empty.
- [ ] Write `export_manifest.json` with source hashes and artifact hashes.
- [ ] Document the pinned upstream PiSA dependency versions and a CUDA conversion environment because PyTorch 2.0.1 CPU FP16 execution is not a reliable export path for this very large UNet.
- [ ] Document that Android model distribution remains disabled until license/provenance and real-device validation are complete.
- [ ] Run full CI.
- [ ] Commit at most two files.

---

### Task 5: Verification boundary

- [ ] Confirm app `model-manifest.json` still does not point to newly generated PiSA downloads.
- [ ] Confirm no model binaries, checkpoints, ONNX files, or MNN files were committed.
- [ ] Confirm exporter uses VAE moments and not sampled latent.
- [ ] Confirm final UNet merge order is pixel then semantic.
- [ ] Confirm timestep 1 and empty prompt are recorded in manifest.
- [ ] Confirm latest Android CI plus Python export-contract tests are GREEN.

The following milestone will run the exporter on a conversion workstation, feed the generated MNN files to the existing Android graph inspector, record the observed tensor contract, and only then implement native latent sampling and graph execution.
