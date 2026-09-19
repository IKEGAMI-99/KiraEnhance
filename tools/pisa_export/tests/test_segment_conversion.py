import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import export_pisa_sr


class SegmentConversionTest(unittest.TestCase):
    def test_converts_core_and_segment_onnx_graphs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            onnx_paths = {
                "vaeEncoder": root / "vae_encoder.onnx",
                "unetDefault": root / "unet_default.onnx",
                "vaeDecoder": root / "vae_decoder.onnx",
            }
            for path in onnx_paths.values():
                path.write_bytes(b"onnx")

            segment_dir = root / "vae_segments"
            segment_dir.mkdir()
            encoder_segment = segment_dir / "vae_encoder_segment_00.onnx"
            decoder_segment = segment_dir / "vae_decoder_segment_00.onnx"
            encoder_segment.write_bytes(b"encoder-segment")
            decoder_segment.write_bytes(b"decoder-segment")

            def fake_run(command, check):
                self.assertTrue(check)
                output_index = command.index("--MNNModel") + 1
                pathlib.Path(command[output_index]).write_bytes(b"mnn")

            with (
                patch.object(
                    export_pisa_sr,
                    "_command_output",
                    return_value="MNNConvert 3.6.1",
                ),
                patch.object(
                    export_pisa_sr.subprocess,
                    "run",
                    side_effect=fake_run,
                ),
            ):
                version, core, segments = export_pisa_sr.convert_to_mnn(
                    mnnconvert=root / "MNNConvert",
                    output_dir=root,
                    onnx_paths=onnx_paths,
                    segment_onnx_paths={
                        "encoder": [encoder_segment],
                        "decoder": [decoder_segment],
                    },
                )

            self.assertEqual("MNNConvert 3.6.1", version)
            self.assertEqual(
                [
                    "vae_encoder.mnn",
                    "unet_default.mnn",
                    "vae_decoder.mnn",
                ],
                [path.name for path in core],
            )
            self.assertEqual(
                ["vae_encoder_segment_00.mnn"],
                [path.name for path in segments["encoder"]],
            )
            self.assertEqual(
                ["vae_decoder_segment_00.mnn"],
                [path.name for path in segments["decoder"]],
            )
            self.assertTrue(segments["encoder"][0].is_file())
            self.assertTrue(segments["decoder"][0].is_file())


if __name__ == "__main__":
    unittest.main()
