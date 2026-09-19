import hashlib
import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from export_contract import (
    ARTIFACT_NAMES,
    build_mnnconvert_command,
    sha256_file,
    write_export_manifest,
)


class ExportContractTest(unittest.TestCase):
    def test_artifact_names_are_fixed(self):
        self.assertEqual(
            (
                "vae_encoder.mnn",
                "unet_default.mnn",
                "vae_decoder.mnn",
                "empty_prompt.fp16",
                "vae_segments.pack",
            ),
            ARTIFACT_NAMES,
        )

    def test_builds_exact_fp16_mnnconvert_command(self):
        self.assertEqual(
            [
                "/opt/MNNConvert",
                "-f",
                "ONNX",
                "--modelFile",
                "/tmp/unet_default.onnx",
                "--MNNModel",
                "/tmp/unet_default.mnn",
                "--bizCode",
                "KiraEnhancePiSA",
                "--fp16",
            ],
            build_mnnconvert_command(
                "/opt/MNNConvert",
                "/tmp/unet_default.onnx",
                "/tmp/unet_default.mnn",
            ),
        )

    def test_sha256_file_hashes_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            path = pathlib.Path(temp) / "artifact.bin"
            path.write_bytes(b"kiraenhance")
            self.assertEqual(
                hashlib.sha256(b"kiraenhance").hexdigest(),
                sha256_file(path),
            )

    def test_manifest_records_provenance_and_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            checkpoint = root / "pisa_sr.pkl"
            checkpoint.write_bytes(b"official-pisa-checkpoint")
            sd21 = root / "stable-diffusion-2-1-base"
            sd21.mkdir()

            artifacts = []
            for index, name in enumerate(ARTIFACT_NAMES, start=1):
                path = root / name
                path.write_bytes(bytes([index]) * index)
                artifacts.append(path)

            manifest_path = root / "export_manifest.json"
            write_export_manifest(
                manifest_path,
                source_checkpoint=checkpoint,
                sd21_path=sd21,
                mnnconvert_version="MNNConvert 3.6.1",
                metadata={
                    "timestep": 1,
                    "prompt": "",
                    "vaeScalingFactor": 0.18215,
                    "emptyPromptShape": [1, 77, 1024],
                },
                artifact_paths=artifacts,
            )

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            generated_at = manifest.pop("generatedAtUtc")
            self.assertTrue(generated_at.endswith("Z"))

            self.assertEqual(
                {
                    "schemaVersion": 1,
                    "source": {
                        "pisaCheckpointPath": str(checkpoint.resolve()),
                        "pisaCheckpointSha256": hashlib.sha256(
                            b"official-pisa-checkpoint"
                        ).hexdigest(),
                        "sd21Path": str(sd21.resolve()),
                    },
                    "mnnconvertVersion": "MNNConvert 3.6.1",
                    "timestep": 1,
                    "prompt": "",
                    "vaeScalingFactor": 0.18215,
                    "emptyPromptShape": [1, 77, 1024],
                    "artifacts": [
                        {
                            "fileName": path.name,
                            "sizeBytes": path.stat().st_size,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        }
                        for path in artifacts
                    ],
                },
                manifest,
            )


if __name__ == "__main__":
    unittest.main()
