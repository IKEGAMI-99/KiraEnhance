#!/usr/bin/env python3
"""Fingerprint deterministic PiSA validation stages for Android comparison."""

from __future__ import annotations

import argparse
import pathlib
from collections.abc import Iterable

FNV1A64_OFFSET = 0xCBF29CE484222325
FNV1A64_PRIME = 0x100000001B3
FNV1A64_MASK = 0xFFFFFFFFFFFFFFFF

STAGE_KEYS = (
    ("momentsFp", "moments"),
    ("sampledLatentFp", "encoded_control"),
    ("modelPredFp", "model_pred"),
    ("decoderLatentFp", "decoder_input"),
    ("decodedFp", "decoder_output"),
)


def fnv1a64_bytes(data: Iterable[int]) -> int:
    """Return the 64-bit FNV-1a digest for a byte iterable."""
    value = FNV1A64_OFFSET
    for byte in data:
        if byte < 0 or byte > 0xFF:
            raise ValueError("byte values must be in [0, 255]")
        value ^= int(byte)
        value = (value * FNV1A64_PRIME) & FNV1A64_MASK
    return value


def fingerprint_float32(array) -> int:
    """Hash C-order IEEE-754 float32 bytes in canonical little-endian order."""
    import numpy as np

    canonical = np.ascontiguousarray(array, dtype=np.dtype("<f4"))
    return fnv1a64_bytes(memoryview(canonical).cast("B"))


def format_fingerprint(value: int) -> str:
    if value < 0 or value > FNV1A64_MASK:
        raise ValueError("fingerprint must fit uint64")
    return f"{value:016x}"


def fingerprints_from_npz(path: pathlib.Path) -> list[tuple[str, int]]:
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        missing = [key for _, key in STAGE_KEYS if key not in archive]
        if missing:
            raise KeyError(
                "Missing validation stage(s): " + ", ".join(missing)
            )
        return [
            (label, fingerprint_float32(archive[key]))
            for label, key in STAGE_KEYS
        ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Print PiSA reference-stage fingerprints in the same format as "
            "the Android PiSAInfer diagnostics."
        )
    )
    parser.add_argument(
        "reference_npz",
        type=pathlib.Path,
        help="NPZ created by run_official_validation.py --dump-npz",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.reference_npz.is_file():
        raise FileNotFoundError(args.reference_npz)

    values = fingerprints_from_npz(args.reference_npz)
    print(
        " ".join(
            f"{label}={format_fingerprint(value)}"
            for label, value in values
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
