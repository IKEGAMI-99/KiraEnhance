#!/usr/bin/env python3
"""Compare official PiSA-SR and KiraEnhance RGB outputs."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ImageDiffMetrics:
    width: int
    height: int
    pixel_count: int
    sample_count: int
    exact_sample_count: int
    exact_pixel_count: int
    exact_sample_ratio: float
    exact_pixel_ratio: float
    mae: float
    rmse: float
    max_abs_error: int
    psnr_db: Optional[float]


def compare_rgb_bytes(
    reference: bytes,
    candidate: bytes,
    width: int,
    height: int,
) -> ImageDiffMetrics:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")

    pixel_count = width * height
    expected = pixel_count * 3
    if len(reference) != expected or len(candidate) != expected:
        raise ValueError(
            f"expected {expected} RGB bytes for {width}x{height}, "
            f"got reference={len(reference)} candidate={len(candidate)}"
        )

    absolute_sum = 0
    squared_sum = 0
    max_abs_error = 0
    exact_sample_count = 0
    exact_pixel_count = 0

    for pixel_offset in range(0, expected, 3):
        pixel_exact = True
        for channel in range(3):
            index = pixel_offset + channel
            difference = abs(reference[index] - candidate[index])
            absolute_sum += difference
            squared_sum += difference * difference
            max_abs_error = max(max_abs_error, difference)
            if difference == 0:
                exact_sample_count += 1
            else:
                pixel_exact = False
        if pixel_exact:
            exact_pixel_count += 1

    sample_count = expected
    mae = absolute_sum / sample_count
    mse = squared_sum / sample_count
    rmse = math.sqrt(mse)
    psnr_db = None if mse == 0.0 else 20.0 * math.log10(255.0 / rmse)

    return ImageDiffMetrics(
        width=width,
        height=height,
        pixel_count=pixel_count,
        sample_count=sample_count,
        exact_sample_count=exact_sample_count,
        exact_pixel_count=exact_pixel_count,
        exact_sample_ratio=exact_sample_count / sample_count,
        exact_pixel_ratio=exact_pixel_count / pixel_count,
        mae=mae,
        rmse=rmse,
        max_abs_error=max_abs_error,
        psnr_db=psnr_db,
    )


def load_rgb(path: Path) -> tuple[tuple[int, int], bytes]:
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError(
            "Pillow is required for image comparison. "
            "Install it with: python -m pip install Pillow"
        ) from error

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return rgb.size, rgb.tobytes()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare an official PiSA-SR reference image with a "
            "KiraEnhance Android output image."
        )
    )
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--max-mae",
        type=float,
        default=None,
        help="Return a failing exit code when MAE exceeds this value.",
    )
    parser.add_argument(
        "--max-error",
        type=int,
        default=None,
        help="Return a failing exit code when any RGB byte error exceeds this value.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    reference_size, reference = load_rgb(args.reference)
    candidate_size, candidate = load_rgb(args.candidate)
    if reference_size != candidate_size:
        raise SystemExit(
            "Image dimensions differ: "
            f"reference={reference_size[0]}x{reference_size[1]} "
            f"candidate={candidate_size[0]}x{candidate_size[1]}"
        )

    metrics = compare_rgb_bytes(
        reference,
        candidate,
        width=reference_size[0],
        height=reference_size[1],
    )
    payload = asdict(metrics)
    payload["reference"] = str(args.reference)
    payload["candidate"] = str(args.candidate)
    print(json.dumps(payload, indent=2, sort_keys=True))

    failed = False
    if args.max_mae is not None and metrics.mae > args.max_mae:
        failed = True
    if args.max_error is not None and metrics.max_abs_error > args.max_error:
        failed = True
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
