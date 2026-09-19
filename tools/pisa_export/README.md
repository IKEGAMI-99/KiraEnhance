# PiSA-SR MNN export

This directory contains the offline conversion tooling for the KiraEnhance
PiSA-SR backend.

The exporter intentionally does **not** download or commit model weights.
Provide the official PiSA-SR checkout, the official PiSA checkpoint, and a
local Diffusers-format Stable Diffusion 2.1-base directory yourself.

The Android application must continue to treat PiSA-SR distribution as
unavailable until the converted artifacts have passed provenance/license
review, numerical validation, and real-device testing.

## What the exporter produces

A successful conversion creates:

- `vae_encoder.mnn`
- `unet_default.mnn`
- `vae_decoder.mnn`
- `empty_prompt.fp16`
- `export_manifest.json`

It also keeps the intermediate ONNX files:

- `vae_encoder.onnx`
- `unet_default.onnx`
- `vae_decoder.onnx`

Do not commit generated ONNX, MNN, checkpoint, or Stable Diffusion weight
files to this repository.

## Upstream behavior preserved by this exporter

The exporter follows the official PiSA-SR default one-step inference path:

1. Load the Stable Diffusion 2.1-base UNet.
2. Add the PiSA pixel LoRA adapters, copy their checkpoint weights, activate
   all three pixel adapter groups at weight 1.0, then merge them into the
   base UNet.
3. Add the semantic LoRA adapters, copy their checkpoint weights, activate
   all three semantic adapter groups at weight 1.0, then merge them.
4. Use the empty text prompt.
5. Use diffusion timestep `1`.
6. Run one UNet prediction.
7. The Android inference milestone will later compute
   `x_denoised = encoded_control - model_pred`.

The VAE encoder graph ends at posterior `moments`
(`encoder -> quant_conv`). It deliberately does not export
`latent_dist.sample()`. Stochastic latent sampling belongs in the later
Android native inference implementation so the random source and sampling
formula can be tested independently from MNN graph conversion.

## Recommended conversion environment

Start from the versions pinned by the official PiSA-SR project:

- Python 3.10
- PyTorch 2.0.1
- diffusers 0.25.0
- transformers 4.28.1
- peft 0.9.0
- numpy 1.23.5
- the remaining packages from the upstream PiSA-SR `requirements.txt`
- ONNX with an opset-17-capable PyTorch exporter
- MNNConvert from the MNN version used for Android validation

The current Android runtime is pinned to MNN 3.6.1. Prefer an MNNConvert
3.6.1 build for the first correctness baseline and record its exact
`--version` output in `export_manifest.json`.

The exporter supports two tracing paths. With `--export-precision auto`
(the default), CUDA devices trace the ONNX graphs in FP16, while CPU and MPS
devices trace in FP32. Explicit FP16 tracing is restricted to CUDA because
CPU/MPS FP16 operator coverage is not the baseline being validated here.

CPU export is therefore supported as the portable fallback. It is slower and
uses more memory for the fused SD2.1 UNet, but it avoids requiring an NVIDIA
GPU. MPS may also be selected when the installed PyTorch build reports it as
available; if an ONNX tracing operator is unsupported there, use CPU instead.

Regardless of ONNX tracing precision, the final baseline MNN conversion still
uses `MNNConvert --fp16` for weight storage. Quantization below FP16 remains
a later milestone after numerical comparison.

## Build MNNConvert 3.6.1

From a separate MNN checkout at tag `3.6.1`:

```bash
mkdir -p build
cd build
cmake .. -DMNN_BUILD_CONVERTER=ON
cmake --build . -j
```

Use the resulting absolute `MNNConvert` path with `--mnnconvert`.

## Run the exporter

Example:

```bash
python tools/pisa_export/export_pisa_sr.py \
  --pisa-repo /abs/path/PiSA-SR \
  --sd21-base /abs/path/stable-diffusion-2-1-base \
  --pisa-checkpoint /abs/path/pisa_sr.pkl \
  --mnnconvert /abs/path/MNN/build/MNNConvert \
  --output-dir /abs/path/kiraenhance-pisa-export \
  --device cuda \
  --export-precision auto
```

With CUDA and `auto`, tracing uses FP16. For a machine without CUDA, use:

```bash
python tools/pisa_export/export_pisa_sr.py \
  --pisa-repo /abs/path/PiSA-SR \
  --sd21-base /abs/path/stable-diffusion-2-1-base \
  --pisa-checkpoint /abs/path/pisa_sr.pkl \
  --mnnconvert /abs/path/MNN/build/MNNConvert \
  --output-dir /abs/path/kiraenhance-pisa-export \
  --device cpu \
  --export-precision fp32
```

The default dummy export image is `512x512`. The ONNX image/latent spatial
axes are dynamic, so this size is only the tracing sample. Both dimensions
must be positive multiples of 8.

MNN conversion uses the equivalent of:

```text
MNNConvert -f ONNX --modelFile <graph.onnx> --MNNModel <graph.mnn> --bizCode KiraEnhancePiSA --fp16
```

The `--fp16` switch is the correctness-first storage baseline. It is not
the final mobile-size target.

## Provenance manifest

`export_manifest.json` records at least:

- PiSA checkpoint absolute path and SHA-256
- local SD2.1 directory path
- PiSA repository path and Git commit when available
- MNNConvert version
- ONNX opset
- export device and ONNX tracing precision
- timestep `1`
- empty prompt
- VAE scaling factor
- empty-prompt tensor shape
- sample tracing shapes
- final artifact byte sizes and SHA-256 hashes
- UTC generation timestamp

Keep this manifest with every candidate export. Two files named
`unet_default.mnn` are not necessarily the same model, despite humanity's
long-running campaign against useful filenames.

## Deterministic official validation reference

The upstream `test_pisasr.py` exposes `--seed`, but the current inference
script does not apply that value before `latent_dist.sample()`. A direct
comparison against that script can therefore change between runs even when the
source image and model weights are identical.

For parity work, generate the official-side reference with KiraEnhance's
validation runner instead:

```bash
python tools/pisa_export/run_official_validation.py \
  --pisa-repo /abs/path/PiSA-SR \
  --sd21-base /abs/path/stable-diffusion-2-1-base \
  --pisa-checkpoint /abs/path/pisa_sr.pkl \
  --input-image /path/to/source.png \
  --output-image /path/to/official-deterministic.png \
  --seed 42
```

This keeps the official PiSA-SR model, VAE hook, UNet tiling, preprocessing,
AdaIN, and final resize path. Only the posterior random draw is replaced with
the same SplitMix64 + Box-Muller float32 sequence used by the Android native
validation path.

Use `--dump-npz /path/to/reference-stages.npz` to retain the posterior
moments, validation noise, encoded latent, UNet prediction, denoised latent,
and decoder output for later stage-by-stage diagnosis. The official runner
currently requires CUDA because upstream `PiSASR_eval` hardcodes CUDA.

To compare those intermediate stages with the Android `PiSAInfer` line,
generate the same canonical float32 FNV-1a fingerprints:

```bash
python tools/pisa_export/stage_fingerprints.py \
  /path/to/reference-stages.npz
```

The command prints `momentsFp`, `sampledLatentFp`, `modelPredFp`,
`decoderLatentFp`, and `decodedFp` in the same fixed-width hexadecimal
format as Android. The first differing field identifies the earliest observed
stage divergence without dumping multi-megabyte tensors from the phone.

## Compare official and Android output

After exporting the same source image from the official PiSA-SR pipeline and
KiraEnhance, compare their final RGB pixels with:

```bash
python tools/pisa_export/compare_outputs.py \
  --reference /path/to/official.png \
  --candidate /path/to/kiraenhance.png
```

The report includes exact-pixel and exact-channel ratios, mean absolute error
(MAE), RMSE, maximum byte error, and PSNR. A perfect byte-for-byte match reports
`psnr_db: null` because mathematical PSNR is infinite.

Optional thresholds make the command suitable for repeatable validation:

```bash
python tools/pisa_export/compare_outputs.py \
  --reference /path/to/official.png \
  --candidate /path/to/kiraenhance.png \
  --max-mae 1.0 \
  --max-error 8
```

The command exits with status 2 when a supplied threshold is exceeded. Pillow
is required only when running the image comparison CLI; the metric unit tests
use the Python standard library only.

## Validation order after export

Do not enable downloads in `app/src/main/assets/model-manifest.json` yet.

The next validation sequence is:

1. Load the four generated artifacts with the existing Android PiSA loader.
2. Read `MnnPisaNativeBridge.sessionInfo(handle)` to confirm the selected
   backend.
3. Read `MnnPisaNativeBridge.graphInfo(handle)` and record the actual input
   and output tensor names, shapes, types, and dimension formats.
4. Compare that observed MNN contract with the ONNX export contract.
5. Run the bounded Android inference path on a real device with the same
   source image used by the official PiSA-SR pipeline.
6. Compare the final images with `compare_outputs.py`, then inspect
   intermediate tensors if the final error is larger than expected.
7. Validate or replace the high-resolution tiled VAE strategy before removing
   the current monolithic VAE safety cap.
8. Review redistribution terms and provenance before enabling app-hosted
   model downloads.
