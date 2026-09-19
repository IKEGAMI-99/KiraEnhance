from __future__ import annotations

import pathlib
import struct
from typing import Any


MAGIC = b"KRVSEG01"
VERSION = 1


def _write_u32(output, value: int) -> None:
    if not isinstance(value, int) or value < 0 or value > 0xFFFFFFFF:
        raise ValueError(f"value does not fit uint32: {value!r}")
    output.write(struct.pack("<I", value))


def _write_u64(output, value: int) -> None:
    if not isinstance(value, int) or value < 0 or value > 0xFFFFFFFFFFFFFFFF:
        raise ValueError(f"value does not fit uint64: {value!r}")
    output.write(struct.pack("<Q", value))


def _write_f32(output, value: float) -> None:
    output.write(struct.pack("<f", float(value)))


def _validate_side(
    side_name: str,
    segment_paths: list[pathlib.Path],
    affine_records: list[dict[str, Any]],
) -> None:
    if not segment_paths:
        raise ValueError(f"{side_name} has no VAE segment models")
    if len(affine_records) != len(segment_paths) - 1:
        raise ValueError(
            f"{side_name} GroupNorm count must be one less than segment count"
        )

    for expected_index, path in enumerate(segment_paths):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)
        expected_name = f"vae_{side_name}_segment_{expected_index:02d}.mnn"
        if path.name != expected_name:
            raise ValueError(
                f"{side_name} segment order changed: "
                f"expected {expected_name!r}, got {path.name!r}"
            )

    for expected_index, record in enumerate(affine_records):
        if record.get("index") != expected_index:
            raise ValueError(f"{side_name} affine indices are not contiguous")
        groups = record.get("groups")
        channels = record.get("channels")
        epsilon = record.get("epsilon")
        weight = record.get("weight")
        bias = record.get("bias")
        if (
            not isinstance(groups, int) or
            groups <= 0 or
            not isinstance(channels, int) or
            channels <= 0 or
            channels % groups != 0 or
            not isinstance(epsilon, (int, float)) or
            float(epsilon) <= 0.0 or
            not isinstance(weight, list) or
            not isinstance(bias, list) or
            len(weight) != channels or
            len(bias) != channels
        ):
            raise ValueError(
                f"{side_name} affine record {expected_index} is invalid"
            )


def write_vae_segment_pack(
    output_path: pathlib.Path,
    *,
    segment_mnn_paths: dict[str, list[pathlib.Path]],
    affine_contract: dict[str, Any],
) -> pathlib.Path:
    if affine_contract.get("schemaVersion") != 1:
        raise ValueError("unsupported VAE GroupNorm affine schema")

    encoder_segments = list(segment_mnn_paths.get("encoder", ()))
    decoder_segments = list(segment_mnn_paths.get("decoder", ()))
    encoder_affine = list(affine_contract.get("encoder", ()))
    decoder_affine = list(affine_contract.get("decoder", ()))

    _validate_side("encoder", encoder_segments, encoder_affine)
    _validate_side("decoder", decoder_segments, decoder_affine)

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as output:
        output.write(MAGIC)
        _write_u32(output, VERSION)
        _write_u32(output, len(encoder_segments))
        _write_u32(output, len(decoder_segments))
        _write_u32(output, len(encoder_affine))
        _write_u32(output, len(decoder_affine))

        for records in (encoder_affine, decoder_affine):
            for record in records:
                groups = int(record["groups"])
                channels = int(record["channels"])
                _write_u32(output, groups)
                _write_u32(output, channels)
                _write_f32(output, float(record["epsilon"]))
                for value in record["weight"]:
                    _write_f32(output, float(value))
                for value in record["bias"]:
                    _write_f32(output, float(value))

        for paths in (encoder_segments, decoder_segments):
            for path in paths:
                data = path.read_bytes()
                _write_u64(output, len(data))
                output.write(data)

    if output_path.stat().st_size <= len(MAGIC) + 5 * 4:
        raise RuntimeError("VAE segment pack is unexpectedly empty")
    return output_path
