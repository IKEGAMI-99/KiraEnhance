# PiSA-SR Android Integration Design

## Goal

Add PiSA-SR as KiraEnhance's BALANCED on-device enhancement engine without regressing the validated UltraSharp 4x path.

The first device milestone is a real PiSA-SR default-mode proof of concept on arm64 Android. It must preserve the upstream one-step diffusion data flow and must not pretend that the placeholder single `model.mnn` file in the current manifest is sufficient.

## Upstream facts that constrain the design

PiSA-SR is a CVPR 2025 one-step diffusion super-resolution pipeline built on Stable Diffusion 2.1. Upstream inference performs a 4x resize, VAE encode, one UNet denoising step at timestep 1, VAE decode, and color alignment. The official test path uses an empty text prompt by default. Upstream also provides tiled latent and tiled VAE execution for memory control.

PiSA-SR merges the pixel LoRA into the base UNet and then applies the semantic LoRA for the default model. Adjustable mode additionally evaluates the pixel-fused UNet and combines pixel/semantic predictions with `lambda_pix` and `lambda_sem`.

MNN 3.6.1 provides Android arm64 CPU/OpenCL/Vulkan libraries. KiraEnhance will use the packaged Android runtime rather than an obsolete Maven artifact.

## Scope of this milestone

1. Generalize production enhancement so the processor does not hard-code ncnn or UltraSharp.
2. Select an engine and model-request factory from `ModelDescriptor.backend` and `ModelDescriptor.mode`.
3. Keep the existing UltraSharp behavior byte-for-byte equivalent at the public interface level.
4. Define a truthful PiSA Android model package contract using multiple artifacts.
5. Add a PiSA MNN engine boundary and native API contract that can be unit-tested without the real converted weights.
6. Wire BALANCED vs ULTRASHARP selection through navigation and production enhancement setup.
7. Update CI to fetch the pinned MNN 3.6.1 Android CPU/OpenCL/Vulkan SDK with its published SHA-256 digest.
8. Do not enable PiSA download in the production manifest until converted artifacts and their redistribution terms are verified.

Actual PiSA model conversion and device validation are the next milestone after this foundation compiles and tests cleanly.

## Android PiSA package contract

The first default-mode mobile package contains:

- `vae_encoder.mnn` — Stable Diffusion 2.1 VAE encoder exported for FP16-capable MNN execution.
- `unet_default.mnn` — SD 2.1 UNet with PiSA pixel LoRA merged and PiSA semantic LoRA merged, matching upstream default inference.
- `vae_decoder.mnn` — Stable Diffusion 2.1 VAE decoder.
- `empty_prompt.fp16` — precomputed empty-prompt CLIP embedding used by the official default test path. This avoids shipping/running the tokenizer and CLIP text encoder for the first on-device milestone.

An optional later artifact, `unet_pixel.mnn`, enables adjustable pixel/semantic blending without changing the engine interface.

The current placeholder `pisa-sr/model.mnn` manifest entry is not considered installable production data.

## Runtime architecture

### Engine selection

`EnhanceEngineResolver` accepts a `ModelDescriptor` and returns an `EnhanceEngineBinding` containing:

- an `UpscaleEngine`
- a model request factory
- the output scale
- a display/save mode name

Mappings for this milestone:

- `ULTRASHARP + NCNN` -> `NcnnUpscaleEngine` + `UltraSharpModelRequestFactory`, scale 4.
- `BALANCED + MNN` -> `MnnPisaUpscaleEngine` + `PisaModelRequestFactory`, scale 4.

Any other model/backend combination fails before processing with an actionable `INVALID_INPUT` error.

### Processor

`EnhanceProcessor` receives an `EnhanceEngineBinding` instead of constructing `NcnnUpscaleEngine` and `UltraSharpModelRequestFactory` itself. Bitmap/RGBA conversion, progress polling, result bitmap creation, cancellation, and generic error presentation remain shared.

### PiSA MNN engine

The Kotlin engine mirrors the existing ncnn boundary:

- `MnnPisaNativeApi` is injectable and returns typed load/inference results.
- `MnnPisaNativeBridge` owns JNI calls.
- `MnnPisaUpscaleEngine` owns lifecycle, validation, progress, cancellation, and maps native failures to `EngineErrorCode`.

The first compile-safe native implementation may expose capability/runtime information and reject inference with `MODEL_LOAD_FAILED` until converted model artifacts are available. It must never return a fabricated enhanced image.

### Native execution target

MNN backend preference for the device PoC is:

1. OpenCL FP16
2. Vulkan
3. ARM CPU FP16 fallback

Backend selection and actual PiSA graph execution are device-validation work, not guessed in JVM code.

## UI and navigation

The home AI MODE cards become selectable for implemented modes only.

For this milestone:

- `おすすめ` maps to `EnhancementMode.BALANCED` / PiSA-SR.
- `UltraSharp` maps to `EnhancementMode.ULTRASHARP`.
- `忠実` and `高精細` remain visible but disabled/unavailable until their engines are implemented.

The selected mode is encoded in the enhance navigation route so a process recreation does not silently switch engines.

If the selected model is not installed or is a placeholder/unavailable package, the enhance screen shows the model-specific reason and links to Model Manager instead of falling back to UltraSharp without consent.

## Model manifest policy

The PiSA descriptor remains present for product planning but must be marked unavailable until real artifacts exist. Placeholder URLs must not be presented as downloadable.

A descriptor is downloadable only when all artifact URLs are HTTPS, non-placeholder URLs and each artifact has a non-zero realistic size and a non-placeholder SHA-256.

The model manager will surface PiSA as `準備中` rather than attempting `example.invalid` downloads.

## Diagnostics

Logs include:

- model id/version/backend/mode
- input/output dimensions
- selected engine
- requested scale
- GPU backend reported by native runtime when available
- elapsed time
- cancellation and typed failure code

No image pixels or source file contents are logged.

## Testing

Unit tests cover:

- engine resolver mapping and unsupported combinations
- PiSA request validation and artifact contract
- placeholder/unavailable manifest detection
- `EnhanceProcessor` working through an injected binding rather than ncnn directly
- existing UltraSharp request behavior remains unchanged

Compose tests cover selectable implemented modes and disabled future modes.

CI compiles Kotlin/JNI and unit/androidTest sources against both ncnn and pinned MNN SDKs. Real PiSA image quality and runtime performance require POCO F7 Ultra device validation and are not claimed by CI.

## Non-goals

- Shipping converted PiSA weights in this milestone.
- Claiming PiSA device inference works before converted graphs are tested.
- Implementing Fidelity or Detail engines.
- Shipping the CLIP tokenizer/text encoder for arbitrary prompts.
- Using PiSA's training-only RAM path.
- Silent cloud inference or server fallback.

## References

- PiSA-SR upstream: https://github.com/csslc/PiSA-SR
- MNN upstream: https://github.com/alibaba/MNN
- MNN 3.6.1 Android release asset: `mnn_3.6.1_android_armv7_armv8_cpu_opencl_vulkan.zip`
