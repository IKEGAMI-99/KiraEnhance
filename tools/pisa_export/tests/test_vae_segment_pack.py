import pathlib
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import vae_segment_pack


def affine(index, channels=2):
    return {
        "index": index,
        "modulePath": f"norm.{index}",
        "groups": 1,
        "channels": channels,
        "epsilon": 1.0e-6,
        "weight": [1.0 + index] * channels,
        "bias": [-0.5 - index] * channels,
    }


class VaeSegmentPackTest(unittest.TestCase):
    def test_writes_affine_records_and_model_blobs_in_stable_order(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            encoder = []
            for index, payload in enumerate((b"enc0", b"enc1")):
                path = root / f"vae_encoder_segment_{index:02d}.mnn"
                path.write_bytes(payload)
                encoder.append(path)
            decoder = []
            for index, payload in enumerate((b"dec0", b"dec1", b"dec2")):
                path = root / f"vae_decoder_segment_{index:02d}.mnn"
                path.write_bytes(payload)
                decoder.append(path)

            output = vae_segment_pack.write_vae_segment_pack(
                root / "vae_segments.pack",
                segment_mnn_paths={
                    "encoder": encoder,
                    "decoder": decoder,
                },
                affine_contract={
                    "schemaVersion": 1,
                    "encoder": [affine(0)],
                    "decoder": [affine(0), affine(1)],
                },
            )

            data = output.read_bytes()
            self.assertEqual(vae_segment_pack.MAGIC, data[:8])
            header = struct.unpack_from("<5I", data, 8)
            self.assertEqual((1, 2, 3, 1, 2), header)
            self.assertTrue(data.endswith(b"dec2"))
            self.assertIn(b"enc0", data)
            self.assertIn(b"dec1", data)

    def test_rejects_missing_segment_and_affine_count_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            segment = root / "vae_encoder_segment_00.mnn"
            segment.write_bytes(b"x")
            decoder = root / "vae_decoder_segment_00.mnn"
            decoder.write_bytes(b"y")

            with self.assertRaisesRegex(ValueError, "one less"):
                vae_segment_pack.write_vae_segment_pack(
                    root / "bad.pack",
                    segment_mnn_paths={
                        "encoder": [segment],
                        "decoder": [decoder],
                    },
                    affine_contract={
                        "schemaVersion": 1,
                        "encoder": [affine(0)],
                        "decoder": [],
                    },
                )

            segment.unlink()
            with self.assertRaises(FileNotFoundError):
                vae_segment_pack.write_vae_segment_pack(
                    root / "missing.pack",
                    segment_mnn_paths={
                        "encoder": [segment],
                        "decoder": [decoder],
                    },
                    affine_contract={
                        "schemaVersion": 1,
                        "encoder": [],
                        "decoder": [],
                    },
                )


if __name__ == "__main__":
    unittest.main()
