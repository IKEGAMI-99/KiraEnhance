from __future__ import annotations

import hashlib
import json
import pathlib
from datetime import datetime, timezone
from typing import Iterable, Mapping, Any


ARTIFACT_NAMES = (
    "vae_encoder.mnn",
    "unet_default.mnn",
    "vae_decoder.mnn",
    "empty_prompt.fp16",
    "vae_segments.pack",
)


def build_mnnconvert_command(
    binary: str,
    onnx_path: str,
    mnn_path: str,
) -> list[str]:
    return [
        binary,
        "-f",
        "ONNX",
        "--modelFile",
        onnx_path,
        "--MNNModel",
        mnn_path,
        "--bizCode",
        "KiraEnhancePiSA",
        "--fp16",
    ]


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_record(path: pathlib.Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "fileName": path.name,
        "sizeBytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_export_manifest(
    output_path: pathlib.Path,
    *,
    source_checkpoint: pathlib.Path,
    sd21_path: pathlib.Path,
    mnnconvert_version: str,
    metadata: Mapping[str, Any],
    artifact_paths: Iterable[pathlib.Path],
) -> None:
    source_checkpoint = source_checkpoint.resolve()
    sd21_path = sd21_path.resolve()
    artifacts = [path.resolve() for path in artifact_paths]

    artifact_names = tuple(path.name for path in artifacts)
    if artifact_names != ARTIFACT_NAMES:
        raise ValueError(
            "artifact paths must match ARTIFACT_NAMES in canonical order: "
            f"{ARTIFACT_NAMES!r}"
        )
    if not source_checkpoint.is_file():
        raise FileNotFoundError(source_checkpoint)
    if not sd21_path.is_dir():
        raise NotADirectoryError(sd21_path)

    reserved = {
        "schemaVersion",
        "generatedAtUtc",
        "source",
        "mnnconvertVersion",
        "artifacts",
    }
    conflict = reserved.intersection(metadata)
    if conflict:
        raise ValueError(
            f"metadata contains reserved manifest keys: {sorted(conflict)!r}"
        )

    manifest = {
        "schemaVersion": 1,
        "generatedAtUtc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "source": {
            "pisaCheckpointPath": str(source_checkpoint),
            "pisaCheckpointSha256": sha256_file(source_checkpoint),
            "sd21Path": str(sd21_path),
        },
        "mnnconvertVersion": mnnconvert_version,
        **dict(metadata),
        "artifacts": [_artifact_record(path) for path in artifacts],
    }

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
