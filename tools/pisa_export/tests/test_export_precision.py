import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import export_pisa_sr


class ExportPrecisionTest(unittest.TestCase):
    def test_auto_uses_fp16_only_for_cuda(self):
        self.assertEqual(
            "fp16",
            export_pisa_sr.resolve_export_precision("cuda", "auto"),
        )
        self.assertEqual(
            "fp16",
            export_pisa_sr.resolve_export_precision("cuda:1", "auto"),
        )
        self.assertEqual(
            "fp32",
            export_pisa_sr.resolve_export_precision("cpu", "auto"),
        )
        self.assertEqual(
            "fp32",
            export_pisa_sr.resolve_export_precision("mps", "auto"),
        )

    def test_explicit_fp32_is_portable(self):
        for device in ("cpu", "mps", "cuda"):
            with self.subTest(device=device):
                self.assertEqual(
                    "fp32",
                    export_pisa_sr.resolve_export_precision(device, "fp32"),
                )

    def test_fp16_is_rejected_for_non_cuda_export(self):
        for device in ("cpu", "mps"):
            with self.subTest(device=device):
                with self.assertRaises(ValueError):
                    export_pisa_sr.resolve_export_precision(device, "fp16")

    def test_cli_defaults_to_auto_export_precision(self):
        parser = export_pisa_sr.build_parser()
        args = parser.parse_args(
            [
                "--pisa-repo",
                "/tmp/pisa",
                "--sd21-base",
                "/tmp/sd21",
                "--pisa-checkpoint",
                "/tmp/pisa_sr.pkl",
                "--output-dir",
                "/tmp/output",
                "--mnnconvert",
                "/tmp/MNNConvert",
            ]
        )
        self.assertEqual("auto", args.export_precision)


if __name__ == "__main__":
    unittest.main()
