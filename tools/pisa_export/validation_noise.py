#!/usr/bin/env python3
"""Deterministic Gaussian noise matching KiraEnhance native PiSA validation."""

from __future__ import annotations

import argparse
import math
import pathlib
import struct
from typing import Iterable

_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_MUL_1 = 0xBF58476D1CE4E5B9
_SPLITMIX_MUL_2 = 0x94D049BB133111EB
_TWO_PI = 6.283185307179586476925286766559
_TWO_POW_53 = 9007199254740992.0


def _next_splitmix64(state: int) -> tuple[int, int]:
    state = (state + _SPLITMIX_GAMMA) & _MASK64
    value = state
    value = ((value ^ (value >> 30)) * _SPLITMIX_MUL_1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_MUL_2) & _MASK64
    value ^= value >> 31
    return state, value & _MASK64


def _uniform_open_01(bits: int) -> float:
    top53 = bits >> 11
    return (float(top53) + 0.5) / _TWO_POW_53


def _float32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def gaussian_noise(count: int, seed: int = 42) -> list[float]:
    if count <= 0:
        raise ValueError("count must be positive")
    if seed < 0 or seed > _MASK64:
        raise ValueError("seed must fit uint64")

    state = seed
    values: list[float] = []
    while len(values) < count:
        state, radius_bits = _next_splitmix64(state)
        state, angle_bits = _next_splitmix64(state)

        uniform_radius = _uniform_open_01(radius_bits)
        uniform_angle = _uniform_open_01(angle_bits)
        magnitude = math.sqrt(-2.0 * math.log(uniform_radius))
        angle = _TWO_PI * uniform_angle

        values.append(_float32(magnitude * math.cos(angle)))
        if len(values) < count:
            values.append(_float32(magnitude * math.sin(angle)))
    return values


def write_float32_le(path: pathlib.Path, values: Iterable[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        for value in values:
            output.write(struct.pack("<f", float(value)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the deterministic Gaussian validation noise used by "
            "KiraEnhance PiSA-SR."
        )
    )
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    values = gaussian_noise(args.count, args.seed)
    write_float32_le(args.output, values)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
