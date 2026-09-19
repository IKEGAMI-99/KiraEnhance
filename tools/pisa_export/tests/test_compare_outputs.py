import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import compare_outputs


class CompareOutputsTest(unittest.TestCase):
    def test_identical_rgb_is_exact(self):
        rgb = bytes([0, 10, 255, 64, 128, 192])

        metrics = compare_outputs.compare_rgb_bytes(rgb, rgb, 2, 1)

        self.assertEqual(2, metrics.pixel_count)
        self.assertEqual(6, metrics.exact_sample_count)
        self.assertEqual(2, metrics.exact_pixel_count)
        self.assertEqual(1.0, metrics.exact_sample_ratio)
        self.assertEqual(1.0, metrics.exact_pixel_ratio)
        self.assertEqual(0.0, metrics.mae)
        self.assertEqual(0.0, metrics.rmse)
        self.assertEqual(0, metrics.max_abs_error)
        self.assertIsNone(metrics.psnr_db)

    def test_known_error_metrics_are_reported(self):
        reference = bytes([0, 0, 0, 10, 20, 30])
        candidate = bytes([3, 4, 0, 10, 18, 36])

        metrics = compare_outputs.compare_rgb_bytes(
            reference,
            candidate,
            2,
            1,
        )

        self.assertAlmostEqual(15.0 / 6.0, metrics.mae)
        self.assertAlmostEqual((65.0 / 6.0) ** 0.5, metrics.rmse)
        self.assertEqual(6, metrics.max_abs_error)
        self.assertEqual(2, metrics.exact_sample_count)
        self.assertEqual(0, metrics.exact_pixel_count)
        self.assertIsNotNone(metrics.psnr_db)

    def test_rejects_dimension_or_byte_count_mismatch(self):
        with self.assertRaises(ValueError):
            compare_outputs.compare_rgb_bytes(b"\x00\x00\x00", b"\x00\x00", 1, 1)

        with self.assertRaises(ValueError):
            compare_outputs.compare_rgb_bytes(b"", b"", 0, 1)

    def test_cli_exposes_optional_thresholds(self):
        args = compare_outputs.build_parser().parse_args(
            [
                "--reference",
                "official.png",
                "--candidate",
                "android.png",
                "--max-mae",
                "1.5",
                "--max-error",
                "8",
            ]
        )

        self.assertEqual(pathlib.Path("official.png"), args.reference)
        self.assertEqual(pathlib.Path("android.png"), args.candidate)
        self.assertEqual(1.5, args.max_mae)
        self.assertEqual(8, args.max_error)


if __name__ == "__main__":
    unittest.main()
