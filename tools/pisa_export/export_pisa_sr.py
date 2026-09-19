#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
from typing import Any

from export_contract import (
    ARTIFACT_NAMES,
    build_mnnconvert_command,
    write_export_manifest,
)
from vae_tile_contract import (
    build_vae_group_norm_affine_contract,
    build_vae_tile_barrier_contract,
    build_vae_tile_execution_contract,
    build_vae_tile_segment_contract,
    validate_vae_tile_execution_contract,
)


PIXEL_ADAPTERS = (
    ("default_encoder_pix", "unet_lora_encoder_modules_pix"),
    ("default_decoder_pix", "unet_lora_decoder_modules_pix"),
    ("default_others_pix", "unet_lora_others_modules_pix"),
)

SEMANTIC_ADAPTERS = (
    ("default_encoder_sem", "unet_lora_encoder_modules_sem"),
    ("default_decoder_sem", "unet_lora_decoder_modules_sem"),
    ("default_others_sem", "unet_lora_others_modules_sem"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export the official PiSA-SR default one-step model to ONNX and "
            "FP16 MNN artifacts for KiraEnhance."
        )
    )
    parser.add_argument(
        "--pisa-repo",
        type=pathlib.Path,
        required=True,
        help="Local checkout of the official csslc/PiSA-SR repository.",
    )
    parser.add_argument(
        "--sd21-base",
        type=pathlib.Path,
        required=True,
        help="Local Diffusers-format Stable Diffusion 2.1-base directory.",
    )
    parser.add_argument(
        "--pisa-checkpoint",
        type=pathlib.Path,
        required=True,
        help="Official pisa_sr.pkl checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        required=True,
        help="Directory for ONNX, MNN, prompt, and manifest outputs.",
    )
    parser.add_argument(
        "--mnnconvert",
        type=pathlib.Path,
        required=True,
        help="Absolute path to the MNNConvert executable.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="PyTorch export device such as cuda, cuda:1, cpu, or mps.",
    )
    parser.add_argument(
        "--export-precision",
        choices=("auto", "fp16", "fp32"),
        default="auto",
        help=(
            "ONNX tracing precision. auto uses fp16 on CUDA and fp32 on "
            "CPU/MPS. MNNConvert still stores baseline weights with --fp16."
        ),
    )
    parser.add_argument(
        "--sample-height",
        type=int,
        default=512,
        help="Dummy export image height. Must be a positive multiple of 8.",
    )
    parser.add_argument(
        "--sample-width",
        type=int,
        default=512,
        help="Dummy export image width. Must be a positive multiple of 8.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version.",
    )
    return parser


def resolve_export_precision(device: str, requested: str) -> str:
    if requested == "auto":
        return "fp16" if device.startswith("cuda") else "fp32"
    if requested == "fp32":
        return "fp32"
    if requested == "fp16":
        if not device.startswith("cuda"):
            raise ValueError(
                "FP16 ONNX tracing is currently restricted to CUDA; "
                "use --export-precision fp32 on CPU/MPS"
            )
        return "fp16"
    raise ValueError(f"Unsupported export precision: {requested!r}")


def _validate_inputs(args: argparse.Namespace) -> None:
    pisa_repo = args.pisa_repo.resolve()
    sd21 = args.sd21_base.resolve()
    checkpoint = args.pisa_checkpoint.resolve()
    mnnconvert = args.mnnconvert.resolve()

    required_upstream = (
        pisa_repo / "src/models/unet_2d_condition.py",
        pisa_repo / "src/models/autoencoder_kl.py",
    )
    missing = [path for path in required_upstream if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "PiSA-SR checkout is missing required custom model files: "
            + ", ".join(str(path) for path in missing)
        )
    if not sd21.is_dir():
        raise NotADirectoryError(sd21)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not mnnconvert.is_file():
        raise FileNotFoundError(mnnconvert)
    if args.sample_height <= 0 or args.sample_height % 8 != 0:
        raise ValueError("--sample-height must be a positive multiple of 8")
    if args.sample_width <= 0 or args.sample_width % 8 != 0:
        raise ValueError("--sample-width must be a positive multiple of 8")
    if args.opset < 17:
        raise ValueError("--opset must be 17 or newer")


def _prepend_pisa_repo(pisa_repo: pathlib.Path) -> None:
    value = str(pisa_repo.resolve())
    if value not in sys.path:
        sys.path.insert(0, value)


def _add_adapters(
    unet: Any,
    checkpoint: dict[str, Any],
    *,
    rank_key: str,
    adapter_specs: tuple[tuple[str, str], ...],
) -> None:
    from peft import LoraConfig

    rank = int(checkpoint[rank_key])
    for adapter_name, modules_key in adapter_specs:
        modules = list(checkpoint[modules_key])
        if not modules:
            raise ValueError(f"Checkpoint field {modules_key!r} is empty")
        unet.add_adapter(
            LoraConfig(
                r=rank,
                init_lora_weights="gaussian",
                target_modules=modules,
            ),
            adapter_name=adapter_name,
        )


def _copy_lora_parameters(
    unet: Any,
    checkpoint: dict[str, Any],
    *,
    name_token: str | None,
) -> int:
    state_dict = checkpoint["state_dict_unet"]
    copied = 0
    for name, parameter in unet.named_parameters():
        if "lora" not in name:
            continue
        if name_token is not None and name_token not in name:
            continue
        if name not in state_dict:
            raise KeyError(f"PiSA checkpoint is missing LoRA parameter {name!r}")
        source = state_dict[name].to(
            device=parameter.device,
            dtype=parameter.dtype,
        )
        parameter.data.copy_(source)
        copied += 1

    if copied == 0:
        label = name_token if name_token is not None else "semantic"
        raise ValueError(f"No {label} LoRA parameters were copied")
    return copied


def merge_default_unet(unet: Any, checkpoint: dict[str, Any]) -> Any:
    from diffusers.utils.peft_utils import set_weights_and_activate_adapters

    _add_adapters(
        unet,
        checkpoint,
        rank_key="lora_rank_unet_pix",
        adapter_specs=PIXEL_ADAPTERS,
    )
    _copy_lora_parameters(
        unet,
        checkpoint,
        name_token="pix",
    )
    set_weights_and_activate_adapters(
        unet,
        [name for name, _ in PIXEL_ADAPTERS],
        [1.0, 1.0, 1.0],
    )
    unet.merge_and_unload()

    _add_adapters(
        unet,
        checkpoint,
        rank_key="lora_rank_unet_sem",
        adapter_specs=SEMANTIC_ADAPTERS,
    )
    _copy_lora_parameters(
        unet,
        checkpoint,
        name_token=None,
    )
    set_weights_and_activate_adapters(
        unet,
        [name for name, _ in SEMANTIC_ADAPTERS],
        [1.0, 1.0, 1.0],
    )
    unet.merge_and_unload()

    remaining = [
        name for name, _ in unet.named_parameters()
        if "lora_" in name
    ]
    if remaining:
        raise RuntimeError(
            "Default UNet still contains LoRA parameters after merge: "
            + ", ".join(remaining[:10])
        )

    unet.eval()
    unet.requires_grad_(False)
    return unet


def load_official_components(
    pisa_repo: pathlib.Path,
    sd21_path: pathlib.Path,
    pisa_checkpoint: pathlib.Path,
    device: str,
):
    _prepend_pisa_repo(pisa_repo)

    import torch
    from transformers import AutoTokenizer, CLIPTextModel
    from src.models.autoencoder_kl import AutoencoderKL
    from src.models.unet_2d_condition import UNet2DConditionModel

    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA export requested via {device!r}, "
                "but torch.cuda.is_available() is false"
            )
    elif device == "mps":
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError(
                "MPS export requested, but torch.backends.mps.is_available() is false"
            )
    elif device != "cpu":
        raise ValueError(
            f"Unsupported export device {device!r}; "
            "use cuda, cuda:N, cpu, or mps"
        )

    checkpoint = torch.load(
        str(pisa_checkpoint.resolve()),
        map_location="cpu",
    )
    if not isinstance(checkpoint, dict):
        raise TypeError("PiSA checkpoint must contain a dictionary")

    unet = UNet2DConditionModel.from_pretrained(
        str(sd21_path.resolve()),
        subfolder="unet",
    )
    merge_default_unet(unet, checkpoint)

    vae = AutoencoderKL.from_pretrained(
        str(sd21_path.resolve()),
        subfolder="vae",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(sd21_path.resolve()),
        subfolder="tokenizer",
    )
    text_encoder = CLIPTextModel.from_pretrained(
        str(sd21_path.resolve()),
        subfolder="text_encoder",
    )

    for model in (vae, text_encoder):
        model.eval()
        model.requires_grad_(False)

    return unet, vae, tokenizer, text_encoder


def _clear_device_cache(torch: Any, device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    elif device == "mps":
        mps = getattr(torch, "mps", None)
        if mps is not None and hasattr(mps, "empty_cache"):
            mps.empty_cache()


def _empty_prompt_embedding(
    torch: Any,
    tokenizer: Any,
    text_encoder: Any,
    *,
    device: str,
    compute_dtype: Any,
    output_path: pathlib.Path,
):
    import numpy as np

    text_encoder.to(device=device, dtype=compute_dtype)
    tokens = tokenizer(
        "",
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tokens.input_ids.to(device)

    with torch.no_grad():
        embedding = text_encoder(input_ids)[0]
        embedding = embedding.to(
            dtype=torch.float16,
            device="cpu",
        ).contiguous()

    array = embedding.numpy().astype(np.dtype("<f2"), copy=False)
    output_path.write_bytes(array.tobytes(order="C"))
    shape = list(array.shape)

    text_encoder.to("cpu")
    del input_ids, embedding
    _clear_device_cache(torch, device)
    return shape


def _make_vae_encoder_wrapper(torch: Any, vae: Any):
    class VaeEncoderMoments(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.encoder = model.encoder
            self.quant_conv = model.quant_conv

        def forward(self, image):
            hidden = self.encoder(image)
            return self.quant_conv(hidden)

    return VaeEncoderMoments(vae).eval()


def _make_vae_decoder_wrapper(torch: Any, vae: Any):
    class VaeDecoder(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.post_quant_conv = model.post_quant_conv
            self.decoder = model.decoder

        def forward(self, latent):
            latent = self.post_quant_conv(latent)
            return self.decoder(latent)

    return VaeDecoder(vae).eval()


def _make_unet_wrapper(torch: Any, unet: Any):
    class DefaultUnet(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, latent, timestep, encoder_hidden_states):
            return self.model(
                latent,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
            ).sample

    return DefaultUnet(unet).eval()


def _export_onnx(
    torch: Any,
    *,
    model: Any,
    args: tuple[Any, ...],
    output_path: pathlib.Path,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]],
    opset: int,
) -> None:
    torch.onnx.export(
        model,
        args,
        str(output_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )


def export_onnx_graphs(
    *,
    unet: Any,
    vae: Any,
    tokenizer: Any,
    text_encoder: Any,
    device: str,
    output_dir: pathlib.Path,
    sample_height: int,
    sample_width: int,
    opset: int,
    export_precision: str,
) -> dict[str, Any]:
    import onnx
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    dtype = (
        torch.float16
        if export_precision == "fp16"
        else torch.float32
    )
    latent_height = sample_height // 8
    latent_width = sample_width // 8

    empty_prompt_path = output_dir / "empty_prompt.fp16"
    empty_prompt_shape = _empty_prompt_embedding(
        torch,
        tokenizer,
        text_encoder,
        device=device,
        compute_dtype=dtype,
        output_path=empty_prompt_path,
    )

    vae.to(device=device, dtype=dtype)
    image = torch.zeros(
        (1, 3, sample_height, sample_width),
        device=device,
        dtype=dtype,
    )
    latent = torch.zeros(
        (1, int(vae.config.latent_channels), latent_height, latent_width),
        device=device,
        dtype=dtype,
    )

    vae_encoder_path = output_dir / "vae_encoder.onnx"
    with torch.no_grad():
        _export_onnx(
            torch,
            model=_make_vae_encoder_wrapper(torch, vae).to(device),
            args=(image,),
            output_path=vae_encoder_path,
            input_names=["image"],
            output_names=["moments"],
            dynamic_axes={
                "image": {2: "image_height", 3: "image_width"},
                "moments": {2: "latent_height", 3: "latent_width"},
            },
            opset=opset,
        )

    vae_decoder_path = output_dir / "vae_decoder.onnx"
    with torch.no_grad():
        _export_onnx(
            torch,
            model=_make_vae_decoder_wrapper(torch, vae).to(device),
            args=(latent,),
            output_path=vae_decoder_path,
            input_names=["latent"],
            output_names=["image"],
            dynamic_axes={
                "latent": {2: "latent_height", 3: "latent_width"},
                "image": {2: "image_height", 3: "image_width"},
            },
            opset=opset,
        )

    vae_scaling_factor = float(vae.config.scaling_factor)
    vae.to("cpu")
    del image
    _clear_device_cache(torch, device)

    unet.to(device=device, dtype=dtype)
    latent = latent.to(device=device, dtype=dtype)
    timestep = torch.tensor([1], device=device, dtype=torch.int64)

    prompt_values = pathlib.Path(empty_prompt_path).read_bytes()
    expected_values = 1
    for dimension in empty_prompt_shape:
        expected_values *= int(dimension)
    if len(prompt_values) != expected_values * 2:
        raise RuntimeError("empty_prompt.fp16 size does not match its tensor shape")

    import numpy as np

    prompt_array = np.frombuffer(prompt_values, dtype="<f2").reshape(
        empty_prompt_shape
    )
    encoder_hidden_states = torch.from_numpy(
        prompt_array.copy()
    ).to(device=device, dtype=dtype)

    unet_path = output_dir / "unet_default.onnx"
    with torch.no_grad():
        _export_onnx(
            torch,
            model=_make_unet_wrapper(torch, unet).to(device),
            args=(latent, timestep, encoder_hidden_states),
            output_path=unet_path,
            input_names=[
                "latent",
                "timestep",
                "encoder_hidden_states",
            ],
            output_names=["model_pred"],
            dynamic_axes={
                "latent": {2: "latent_height", 3: "latent_width"},
                "model_pred": {2: "latent_height", 3: "latent_width"},
            },
            opset=opset,
        )

    unet.to("cpu")
    del latent, timestep, encoder_hidden_states
    _clear_device_cache(torch, device)

    for path in (vae_encoder_path, unet_path, vae_decoder_path):
        onnx.checker.check_model(str(path))

    return {
        "onnxPaths": {
            "vaeEncoder": vae_encoder_path,
            "unetDefault": unet_path,
            "vaeDecoder": vae_decoder_path,
        },
        "emptyPromptPath": empty_prompt_path,
        "emptyPromptShape": empty_prompt_shape,
        "vaeScalingFactor": vae_scaling_factor,
        "sampleImageShape": [1, 3, sample_height, sample_width],
        "sampleLatentShape": [
            1,
            int(vae.config.latent_channels),
            latent_height,
            latent_width,
        ],
    }


def _command_output(command: list[str]) -> str:
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return (result.stdout or result.stderr).strip()


def _git_commit(path: pathlib.Path) -> str:
    try:
        value = _command_output(
            ["git", "-C", str(path.resolve()), "rev-parse", "HEAD"]
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return value or "unknown"


def convert_to_mnn(
    *,
    mnnconvert: pathlib.Path,
    output_dir: pathlib.Path,
    onnx_paths: dict[str, pathlib.Path],
) -> tuple[str, list[pathlib.Path]]:
    binary = str(mnnconvert.resolve())
    version = _command_output([binary, "--version"])

    conversion_pairs = (
        (onnx_paths["vaeEncoder"], output_dir / "vae_encoder.mnn"),
        (onnx_paths["unetDefault"], output_dir / "unet_default.mnn"),
        (onnx_paths["vaeDecoder"], output_dir / "vae_decoder.mnn"),
    )

    mnn_paths = []
    for onnx_path, mnn_path in conversion_pairs:
        subprocess.run(
            build_mnnconvert_command(
                binary,
                str(onnx_path.resolve()),
                str(mnn_path.resolve()),
            ),
            check=True,
        )
        if not mnn_path.is_file() or mnn_path.stat().st_size <= 0:
            raise RuntimeError(f"MNNConvert did not produce {mnn_path}")
        mnn_paths.append(mnn_path)

    return version, mnn_paths


def run(args: argparse.Namespace) -> pathlib.Path:
    _validate_inputs(args)
    export_precision = resolve_export_precision(
        args.device,
        args.export_precision,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    unet, vae, tokenizer, text_encoder = load_official_components(
        args.pisa_repo,
        args.sd21_base,
        args.pisa_checkpoint,
        args.device,
    )

    export = export_onnx_graphs(
        unet=unet,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        device=args.device,
        output_dir=output_dir,
        sample_height=args.sample_height,
        sample_width=args.sample_width,
        opset=args.opset,
        export_precision=export_precision,
    )

    mnnconvert_version, mnn_paths = convert_to_mnn(
        mnnconvert=args.mnnconvert,
        output_dir=output_dir,
        onnx_paths=export["onnxPaths"],
    )

    vae_tile_barrier_contract = build_vae_tile_barrier_contract(vae)
    vae_group_norm_affine_contract = build_vae_group_norm_affine_contract(vae)
    vae_tile_execution_contract = build_vae_tile_execution_contract(vae)
    vae_tile_segment_contract = build_vae_tile_segment_contract(
        vae_tile_execution_contract
    )
    vae_tile_resolved_module_count = validate_vae_tile_execution_contract(
        vae,
        vae_tile_execution_contract,
    )
    if (
        vae_tile_barrier_contract["encoderBarrierCount"] != 22 or
        vae_tile_barrier_contract["decoderBarrierCount"] != 30
    ):
        raise RuntimeError(
            "Unexpected SD2.1 VAE GroupNorm barrier contract: "
            f"encoder={vae_tile_barrier_contract['encoderBarrierCount']} "
            f"decoder={vae_tile_barrier_contract['decoderBarrierCount']}"
        )

    artifact_paths = [
        mnn_paths[0],
        mnn_paths[1],
        mnn_paths[2],
        export["emptyPromptPath"],
    ]
    if tuple(path.name for path in artifact_paths) != ARTIFACT_NAMES:
        raise RuntimeError("Internal artifact order no longer matches contract")

    manifest_path = output_dir / "export_manifest.json"
    write_export_manifest(
        manifest_path,
        source_checkpoint=args.pisa_checkpoint,
        sd21_path=args.sd21_base,
        mnnconvert_version=mnnconvert_version,
        metadata={
            "timestep": 1,
            "prompt": "",
            "vaeScalingFactor": export["vaeScalingFactor"],
            "emptyPromptShape": export["emptyPromptShape"],
            "sampleImageShape": export["sampleImageShape"],
            "sampleLatentShape": export["sampleLatentShape"],
            "onnxOpset": args.opset,
            "exportDevice": args.device,
            "onnxExportPrecision": export_precision,
            "pisaRepoPath": str(args.pisa_repo.resolve()),
            "pisaRepoCommit": _git_commit(args.pisa_repo),
            "vaeTileBarrierContract": vae_tile_barrier_contract,
            "vaeGroupNormAffineContract": vae_group_norm_affine_contract,
            "vaeTileExecutionContract": vae_tile_execution_contract,
            "vaeTileSegmentContract": vae_tile_segment_contract,
            "vaeTileResolvedModuleCount": vae_tile_resolved_module_count,
        },
        artifact_paths=artifact_paths,
    )
    return manifest_path


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        manifest_path = run(args)
    except Exception as error:
        print(f"PiSA export failed: {error}", file=sys.stderr)
        return 1

    print(f"PiSA export complete: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
