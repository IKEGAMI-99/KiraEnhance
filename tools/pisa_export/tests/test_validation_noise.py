import math
import pathlib
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import validation_noise


class ValidationNoiseTest(unittest.TestCase):
    def test_seed_42_prefix_matches_native_cpp(self):
        actual = validation_noise.gaussian_noise(6, seed=42)
        expected = [
            0.4147197604,
            0.6526812315,
            -0.8918862343,
            1.3268336058,
            1.7295930386,
            -1.8834167719,
        ]

        for left, right in zip(actual, expected):
            self.assertAlmostEqual(left, right, places=6)

    def test_same_seed_repeats_and_different_seed_changes(self):
        first = validation_noise.gaussian_noise(9, seed=123456789)
        second = validation_noise.gaussian_noise(9, seed=123456789)
        different = validation_noise.gaussian_noise(9, seed=987654321)

        self.assertEqual(first, second)
        self.assertNotEqual(first, different)

    def test_distribution_is_reasonably_standard_normal(self):
        values = validation_noise.gaussian_noise(20000, seed=42)

        mean = sum(values) / len(values)
        variance = (
            sum(value * value for value in values) / len(values)
            - mean * mean
        )

        self.assertLess(abs(mean), 0.03)
        self.assertGreater(variance, 0.95)
        self.assertLess(variance, 1.05)
        self.assertTrue(all(math.isfinite(value) for value in values))

    def test_writes_little_endian_float32_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "noise.fp32"
            values = validation_noise.gaussian_noise(3, seed=42)

            validation_noise.write_float32_le(path, values)

            raw = path.read_bytes()
            self.assertEqual(12, len(raw))
            decoded = list(struct.unpack("<3f", raw))
            self.assertEqual(values, decoded)

    def test_rejects_invalid_count_and_seed(self):
        with self.assertRaises(ValueError):
            validation_noise.gaussian_noise(0, seed=42)
        with self.assertRaises(ValueError):
            validation_noise.gaussian_noise(1, seed=-1)
        with self.assertRaises(ValueError):
            validation_noise.gaussian_noise(1, seed=1 << 64)


if __name__ == "__main__":
    unittest.main()
