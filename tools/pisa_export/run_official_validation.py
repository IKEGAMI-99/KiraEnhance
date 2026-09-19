#!/usr/bin/env python3
"""Run a deterministic official PiSA-SR validation reference.

The upstream test script accepts --seed but does not apply it before
latent_dist.sample(). This runner keeps the official model/pre/post-processing
path while replacing only the VAE posterior random draw with KiraEnhance's
deterministic validation noise.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from types import SimpleNamespace
from typing import Any

from validation_noise import gaussian_noise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a deterministic official PiSA-SR reference image using "
            "the same validation noise sequence as KiraEnhance."
        )
    )
    parser.add_argument("--pisa-repo", type=pathlib.Path, required=True)
    parser.add_argument("--sd21-base", type=pathlib.Path, required=True)
    parser.add_argument("--pisa-checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--input-image", type=pathlib.Path, required=True)
    parser.add_argument("--output-image", type=pathlib.Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--process-size", type=int, default=512)
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--vae-decoder-tiled-size", type=int, default=224)
    parser.add_argument("--vae-encoder-tiled-size", type=int, default=1024)
    parser.add_argument("--latent-tiled-size", type=int, default=96)
    parser.add_argument("--latent-tiled-overlap", type=int, default=32)
    parser.add_argument(
        "--mixed-precision",
        choices=("fp16", "bf16", "fp32"),
        default="fp16",
    )
    parser.add_argument(
        "--dump-npz",
        type=pathlib.Path,
        default=None,
        help=(
            "Optional compressed NumPy dump containing moments, validation "
            "noise, encoded latent, UNet prediction, denoised latent, "
            "decoder input, and decoder output."
        ),
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    required_files = (
        args.pisa_repo / "pisasr.py",
        args.pisa_repo / "src/models/autoencoder_kl.py",
        args.pisa_repo / "src/models/unet_2d_condition.py",
        args.pisa_checkpoint,
        args.input_image,
    )
    missing = [path for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing required validation input: "
            + ", ".join(str(path) for path in missing)
        )
    if not args.sd21_base.is_dir():
        raise NotADirectoryError(args.sd21_base)
    if args.seed < 0 or args.seed >= 1 << 64:
        raise ValueError("--seed must fit uint64")
    if args.process_size <= 0:
        raise ValueError("--process-size must be positive")
    if args.upscale <= 0:
        raise ValueError("--upscale must be positive")


def _prepend_pisa_repo(path: pathlib.Path) -> None:
    value = str(path.resolve())
    if value not in sys.path:
        sys.path.insert(0, value)


def _model_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        pretrained_model_path=str(args.sd21_base.resolve()),
        pretrained_path=str(args.pisa_checkpoint.resolve()),
        mixed_precision=args.mixed_precision,
        vae_decoder_tiled_size=args.vae_decoder_tiled_size,
        vae_encoder_tiled_size=args.vae_encoder_tiled_size,
        latent_tiled_size=args.latent_tiled_size,
        latent_tiled_overlap=args.latent_tiled_overlap,
        lambda_pix=1.0,
        lambda_sem=1.0,
    )


def _prepare_input_image(
    image: Any,
    *,
    process_size: int,
    upscale: int,
) -> tuple[Any, tuple[int, int], bool]:
    from PIL import Image

    input_image = image.convert("RGB")
    original_size = input_image.size
    original_width, original_height = original_size
    resize_flag = False

    minimum_side = process_size // upscale
    if minimum_side <= 0:
        raise ValueError("process-size // upscale must be positive")

    if original_width < minimum_side or original_height < minimum_side:
        scale = minimum_side / min(original_width, original_height)
        input_image = input_image.resize(
            (
                int(scale * original_width),
                int(scale * original_height),
            )
        )
        resize_flag = True

    input_image = input_image.resize(
        (
            input_image.size[0] * upscale,
            input_image.size[1] * upscale,
        )
    )
    new_width = input_image.width - input_image.width % 8
    new_height = input_image.height - input_image.height % 8
    if new_width <= 0 or new_height <= 0:
        raise ValueError("aligned PiSA input dimensions must be positive")
    input_image = input_image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS,
    )
    return input_image, original_size, resize_flag


def _deterministic_latent(
    torch: Any,
    posterior: Any,
    *,
    scaling_factor: float,
    seed: int,
) -> tuple[Any, Any]:
    mean = posterior.mean.to(dtype=torch.float32)
    log_variance = posterior.logvar.to(dtype=torch.float32).clamp(-30.0, 20.0)
    values = gaussian_noise(mean.numel(), seed=seed)
    noise = torch.tensor(
        values,
        dtype=torch.float32,
        device=mean.device,
    ).reshape(mean.shape)

    sample = mean + torch.exp(0.5 * log_variance) * noise
    return sample * float(scaling_factor), noise


def _to_numpy_float32(tensor: Any):
    return tensor.detach().to(dtype=tensor.new_zeros(()).float().dtype).cpu().numpy()


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    _prepend_pisa_repo(args.pisa_repo)

    import numpy as np
    import torch
    import torchvision.transforms.functional as F
    from PIL import Image
    from torchvision import transforms

    from pisasr import PiSASR_eval
    from src.my_utils.wavelet_color_fix import adain_color_fix

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Official PiSASR_eval currently requires CUDA for validation"
        )

    model = PiSASR_eval(_model_args(args))
    model.set_eval()

    with Image.open(args.input_image) as opened:
        input_image, original_size, resize_flag = _prepare_input_image(
            opened,
            process_size=args.process_size,
            upscale=args.upscale,
        )

    with torch.no_grad():
        c_t = F.to_tensor(input_image).unsqueeze(0).cuda() * 2 - 1
        c_t = c_t.to(dtype=model.weight_dtype)

        posterior = model.vae.encode(c_t).latent_dist
        encoded_control_fp32, noise = _deterministic_latent(
            torch,
            posterior,
            scaling_factor=float(model.vae.config.scaling_factor),
            seed=args.seed,
        )
        encoded_control = encoded_control_fp32.to(dtype=model.weight_dtype)

        prompt_embeds = model.encode_prompt([""]).to(dtype=model.weight_dtype)
        model_pred = model._process_latents(
            encoded_control,
            prompt_embeds,
            True,
        )
        x_denoised = encoded_control - model_pred
        decoder_input = x_denoised / float(model.vae.config.scaling_factor)
        decoder_output = model.vae.decode(decoder_input).sample.clamp(-1, 1)

        output_image = decoder_output * 0.5 + 0.5
        output_image = torch.clip(output_image, 0, 1)
        output_pil = transforms.ToPILImage()(output_image[0].cpu())

    output_pil = adain_color_fix(
        target=output_pil,
        source=input_image,
    )

    if resize_flag:
        original_width, original_height = original_size
        output_pil = output_pil.resize(
            (
                int(args.upscale * original_width),
                int(args.upscale * original_height),
            )
        )

    args.output_image.parent.mkdir(parents=True, exist_ok=True)
    output_pil.save(args.output_image)

    if args.dump_npz is not None:
        parameters = getattr(posterior, "parameters", None)
        if parameters is None:
            parameters = torch.cat(
                [
                    posterior.mean,
                    posterior.logvar,
                ],
                dim=1,
            )

        args.dump_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.dump_npz,
            moments=parameters.detach().float().cpu().numpy(),
            noise=noise.detach().float().cpu().numpy(),
            encoded_control=encoded_control_fp32.detach().float().cpu().numpy(),
            model_pred=model_pred.detach().float().cpu().numpy(),
            x_denoised=x_denoised.detach().float().cpu().numpy(),
            decoder_input=decoder_input.detach().float().cpu().numpy(),
            decoder_output=decoder_output.detach().float().cpu().numpy(),
        )


def main() -> int:
    args = build_parser().parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
