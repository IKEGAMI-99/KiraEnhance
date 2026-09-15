# KiraEnhance v1 Design Specification

Date: 2026-09-15
Status: Draft for user review
Repository: `IKEGAMI-99/KiraEnhance`

## 1. Product goal

KiraEnhance is an Android-only, fully on-device AI image enhancement app focused on high-quality game screenshots, especially Kirapara-style 3D anime screenshots. The product goal is to make AI upscaling usable by non-technical users while still exposing model-specific tuning for advanced users.

Primary user flow:

1. Select an image.
2. Choose an enhancement mode.
3. Optionally adjust advanced parameters.
4. Run local AI inference.
5. Compare original and enhanced images with synchronized zoom/pan.
6. Save or share the result.

Images must never be uploaded for inference. Network access is used only for app updates, model metadata, and model downloads.

## 2. Target platform and baseline device

- Platform: Android only for v1.
- UI: Kotlin + Jetpack Compose.
- Native inference core: C++ via Android NDK/JNI.
- Primary inference runtime: MNN.
- Optional secondary runtime: ncnn/Vulkan where it offers better compatibility or performance for ESRGAN-family models.
- Initial reference device class: POCO F7 Ultra / Snapdragon 8 Elite class hardware.
- Initial optimization target: high-end Android devices first. Broader SoC support comes after stable operation on the reference class.

## 3. Distribution model

### App

- Distributed as APK through GitHub Releases.
- No Google Play dependency for v1.
- In-app update checker reads release metadata from GitHub.
- Update flow:
  1. Check latest compatible release.
  2. Show version and changelog.
  3. Download APK.
  4. Verify SHA-256.
  5. Launch the Android package installer.
- All app versions must use the same signing key.

### AI models

- Model binaries are not bundled into the APK.
- Model metadata/manifest is hosted with the project on GitHub.
- Large model binaries are hosted on Hugging Face.
- Model updates are independent of app updates.
- Each downloadable model entry includes:
  - model id
  - display name
  - model version
  - backend/runtime
  - download URL
  - file size
  - SHA-256
  - supported scale factors
  - required app/engine version
  - estimated RAM requirement
  - license metadata
  - description
  - adjustable parameters and defaults

## 4. Model strategy

KiraEnhance v1 presents user-facing enhancement modes rather than forcing users to understand model names.

### Mode 1: Fidelity

Goal: fast, structure-preserving enhancement with minimal hallucination.

Initial candidate: RealESRGAN Anime 6B or another Android-friendly ESRGAN-family model.

Priority:
- fast inference
- stable tile processing
- minimal changes to face, costume geometry, lace, fishnets, hair lines, accessories

### Mode 2: Balanced

Goal: best default balance between faithfulness and generated detail.

Required model family: PiSA-SR.

PiSA-SR is the only model family treated as a required target for v1 architecture. If full mobile conversion requires staged optimization, the application architecture must still reserve this mode and its model-specific controls.

Expected advanced controls include, where supported by the converted implementation:
- fidelity / pixel preservation strength
- semantic detail strength
- denoise
- sharpen
- texture/detail strength
- output scale

### Mode 3: Detail

Goal: higher-detail reconstruction for users who prioritize texture and sharpness over speed.

Initial candidate: HAT-S / Real-HAT-style mobile-compatible model. Final model may change if another model provides better Android performance and image quality.

Priority:
- cloth and hair micro-detail
- clean edges
- higher visual sharpness than Fidelity mode
- acceptable performance on Snapdragon 8 Elite class devices

### Mode 4: UltraSharp

Goal: preserve an existing familiar visual baseline using 4x-UltraSharp.

- Presented as a dedicated optional model/mode.
- Treated internally as a Community Model so licensing and attribution can be displayed clearly.
- Not bundled with the APK.
- License and redistribution terms must be re-verified before public release of downloadable model files.
- If redistribution is not permitted under the final verified terms, KiraEnhance must support user-supplied installation instead of hosting the model itself.

## 5. Model abstraction

The UI and business logic must not depend directly on a specific inference framework or model implementation.

Conceptual interface:

```text
UpscaleEngine
  loadModel()
  unloadModel()
  upscale(input, settings)
  cancel()
  getProgress()
  getCapabilities()
```

Backends:

```text
UpscaleEngine
├── MnnBackend
│   ├── PiSA
│   └── HAT / future models
└── NcnnBackend
    ├── RealESRGAN
    └── UltraSharp
```

A model descriptor maps a downloaded model to a backend and declares its capabilities. This allows later model replacement without redesigning the UI.

## 6. Image processing pipeline

```text
Image import
  ↓
Decode + orientation correction
  ↓
Model capability check
  ↓
Memory / tile plan
  ↓
AI inference
  ↓
Tile blending if required
  ↓
Optional post-processing
  ↓
Preview + comparison
  ↓
Encode + save/share
```

Requirements:

- Support JPEG, PNG, and WebP input in v1.
- Support PNG, JPEG, and WebP output.
- Primary scale options: 2x and 4x.
- 4x must use automatic tiling when the full frame cannot fit safely in memory.
- Tile overlap and blending must avoid visible seams.
- Processing must be cancellable.
- OOM conditions must be handled gracefully and should trigger smaller-tile retry where safe.
- The app must not silently downgrade to cloud inference.

## 7. Beginner-first UI

The default interface hides technical model parameters.

User-facing mode labels:

- `忠実` / Fidelity
- `おすすめ` / Balanced
- `高精細` / Detail
- `UltraSharp`

Each mode card includes:
- short plain-language description
- expected speed category
- expected faithfulness/detail trade-off
- download state
- model size if not installed

Model names and technical information are shown in secondary text or in a `詳しく見る` section.

Default workflow should require no more than:

1. Choose image.
2. Choose mode.
3. Choose 2x or 4x.
4. Tap Enhance.

## 8. Advanced model controls

Advanced controls appear only when `詳細設定` is expanded.

Common controls where supported:
- scale
- denoise
- sharpen
- texture/detail
- color preservation
- face preservation
- memory-saving tile mode

Model-specific controls are declared by the model manifest/capabilities rather than hard-coded into the main UI.

Each adjustable model also provides simple presets such as:
- Natural
- Sharp
- Detailed

Unsupported controls must not be shown.

## 9. Processing screen

Display:
- current stage
- progress percentage where measurable
- selected mode/model
- source and target resolution
- elapsed time
- estimated remaining time when reliable
- cancel button

Suggested stages:
- Preparing image
- Loading model
- Enhancing
- Blending tiles
- Finalizing

Long-running inference should use an Android foreground service so the process can continue when the app moves to the background, subject to Android lifecycle restrictions.

## 10. Original vs enhanced comparison

The comparison viewer is a core feature.

Required modes:

### Split view
A movable vertical divider reveals original on one side and enhanced on the other.

### Side-by-side
Original and enhanced images are displayed simultaneously.

### Hold-to-compare
Enhanced image is shown normally; pressing and holding temporarily shows the original.

Interaction requirements:
- synchronized zoom
- synchronized pan
- pinch zoom
- double-tap zoom
- fit-to-screen
- 100%, 200%, and 400% quick zoom options
- stable coordinate mapping even when source and output resolutions differ

## 11. Model manager

A dedicated model management screen shows, for every model:
- installed / not installed
- version
- size
- estimated RAM requirement
- runtime/backend
- description
- license summary
- update state

Actions:
- download
- pause/resume if technically supported by the download implementation
- retry
- update
- delete
- reinstall

Downloads must verify SHA-256 before activation. Partially downloaded or corrupted files must never be treated as installed.

Optional setting:
- download models over Wi-Fi only

## 12. Device capability detection

The app detects and records:
- device model
- Android version
- SoC where available
- RAM
- GPU/Vulkan capability
- available storage

The model selector may show:
- Recommended
- Supported
- Not recommended

These are guidance states, not arbitrary hard blocks, unless a required API/runtime feature is missing.

## 13. Storage and privacy

- AI inference is fully local.
- Images are not uploaded.
- Diagnostic logs are not uploaded automatically.
- Model and app update requests contain no image data.
- Output should not copy unnecessary location or personal EXIF metadata by default.
- Metadata retention may be added as an explicit user setting.

Suggested output folder:
`Pictures/KiraEnhance/`

Suggested filename pattern:
`KiraEnhance_YYYYMMDD_HHMMSS_<mode>_<scale>x.<ext>`

## 14. Logging and diagnostics

The app keeps local diagnostic logs sufficient for troubleshooting.

Record where available:
- app version
- build number
- engine version
- Android version
- device / SoC / RAM
- backend
- model id and version
- input resolution
- output resolution
- effective parameters
- tile size
- inference duration
- peak memory estimate/measurement where feasible
- warnings
- errors and native stack information where available

Do not include image content in logs.
Avoid full source file paths where they may expose personal information.

User actions:
- view recent logs
- export logs
- clear logs

Export filename example:
`KiraEnhance_Log_20260915_211500.txt`

## 15. Version display

Settings/About shows separately:
- App Version
- Build Number
- Engine Version
- Model versions

Example:

```text
KiraEnhance 0.1.0-beta
Build 1
Engine 0.1.0

Models
PiSA-SR      0.x
RealESRGAN   x.x
HAT          x.x
UltraSharp   x.x
```

## 16. Visual design

Direction: cute, polished, slightly gothic/fashion-oriented, but not a direct copy of Kirapara UI or proprietary visual assets.

Design language:
- soft lavender / pink / plum accents
- optional dark base
- rounded cards
- restrained translucency
- subtle gradients
- minimal decorative particles/light effects
- large touch targets
- clear typography

Themes:
- System
- Light
- Dark

Performance takes priority over decorative animation. UI effects must not meaningfully compete with inference resources.

## 17. Primary screens

1. Home
2. Image editor / setup
3. Mode/model selection
4. Model details
5. Processing
6. Original vs enhanced comparison
7. Save/share
8. Model manager
9. Settings
10. About/update
11. Logs

## 18. v1 MVP scope

Required for v1:
- Android app shell in Kotlin/Compose
- native JNI/NDK engine abstraction
- at least one fully working local model backend
- architecture ready for PiSA-SR
- PiSA-SR integration target retained as required product goal
- 2x and 4x output workflow
- automatic tiling
- model download/verification/removal
- beginner mode cards
- advanced controls driven by capabilities
- comparison viewer with synchronized zoom/pan
- save/share
- GitHub APK update checker/downloader
- model update mechanism
- local logs and export
- version display

UltraSharp is optional at first-run but supported as a first-class downloadable/community model once licensing and redistribution are validated.

## 19. Explicit non-goals for v1

- Cloud inference
- iOS support
- account/login system
- social feed
- server-side image history
- batch processing of large folders
- automatic face regeneration
- generative background replacement
- video upscaling
- Google Play distribution

These may be considered after v1 stability.

## 20. Error handling

The app must provide actionable errors for:
- insufficient storage
- insufficient memory
- unsupported Vulkan/runtime feature
- corrupted model download
- incompatible model version
- network failure during model/app download
- image decode failure
- native inference failure
- save/export failure

For memory errors, the app should attempt smaller tiles when safe before failing. Retried processing must be visible in logs.

## 21. Testing strategy

### Unit tests
- model manifest parsing
- version comparison
- SHA-256 verification
- capability-to-control mapping
- output filename generation
- update decision logic

### Android/UI tests
- image picker flow
- mode selection
- model download states
- advanced setting visibility
- comparison viewer gestures
- save/share flow

### Native/inference tests
- model load/unload
- cancellation
- tile planning
- seam-free tile blending
- repeated inference without leaks
- OOM fallback

### Golden image tests
Maintain a small test set of representative screenshots containing:
- face and eyes
- hair strands
- lace
- fishnet patterns
- dark fabric
- metallic accessories
- detailed background

Compare model changes for geometry preservation, artifacts, seams, and detail consistency.

## 22. Release acceptance criteria

KiraEnhance v1 is ready for release when:

1. The reference Snapdragon 8 Elite class device can complete repeated on-device inference without crashes or unbounded memory growth.
2. At least one production-quality model works end-to-end through the common engine interface.
3. PiSA-SR has either a working mobile implementation or a documented blocking compatibility issue plus an integration path that does not require redesigning the app architecture.
4. 2x and 4x processing work with automatic tile fallback.
5. Original/enhanced comparison remains coordinate-synchronized during pan and zoom.
6. Model files are verified before use.
7. App update verification is implemented.
8. Log export contains enough information to diagnose native/model failures without including image data.
9. A clean install can reach a successful enhancement result using only the documented beginner flow.
10. License/attribution requirements are verified for every model file that KiraEnhance itself redistributes.

## 23. Future extensions

Post-v1 candidates:
- Kirapara-specific fine-tuned weights
- custom ESRGAN-compatible model import
- hardware benchmark and automatic recommended model settings
- broader Snapdragon generation support
- QNN/NPU backend experiments
- batch processing
- video/frame-sequence enhancement

The architecture must allow adding or replacing models without changing the basic user flow.