import pathlib
import struct
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import stage_fingerprints


class StageFingerprintsTest(unittest.TestCase):
    def test_float32_byte_hash_matches_native_cpp_fixture(self):
        raw = struct.pack("<3f", 0.0, 1.0, -2.5)

        actual = stage_fingerprints.fnv1a64_bytes(raw)

        self.assertEqual(0xE1C67AD3BEA76428, actual)
        self.assertEqual(
            "e1c67ad3bea76428",
            stage_fingerprints.format_fingerprint(actual),
        )

    def test_stage_mapping_matches_android_diagnostic_names(self):
        self.assertEqual(
            (
                ("momentsFp", "moments"),
                ("sampledLatentFp", "encoded_control"),
                ("modelPredFp", "model_pred"),
                ("decoderLatentFp", "decoder_input"),
                ("decodedFp", "decoder_output"),
            ),
            stage_fingerprints.STAGE_KEYS,
        )

    def test_hash_is_order_sensitive(self):
        forward = stage_fingerprints.fnv1a64_bytes(b"abcd")
        reverse = stage_fingerprints.fnv1a64_bytes(b"dcba")
        self.assertNotEqual(forward, reverse)

    def test_rejects_invalid_byte_and_fingerprint_ranges(self):
        with self.assertRaises(ValueError):
            stage_fingerprints.fnv1a64_bytes([256])
        with self.assertRaises(ValueError):
            stage_fingerprints.format_fingerprint(-1)
        with self.assertRaises(ValueError):
            stage_fingerprints.format_fingerprint(1 << 64)


if __name__ == "__main__":
    unittest.main()
