import pathlib
import subprocess
import sys
import unittest


class ExportCliTest(unittest.TestCase):
    def test_help_works_without_ml_dependencies(self):
        script = pathlib.Path(__file__).resolve().parents[1] / "export_pisa_sr.py"
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("--pisa-repo", result.stdout)
        self.assertIn("--sd21-base", result.stdout)
        self.assertIn("--pisa-checkpoint", result.stdout)
        self.assertIn("--output-dir", result.stdout)
        self.assertIn("--mnnconvert", result.stdout)
        self.assertIn("--device", result.stdout)


if __name__ == "__main__":
    unittest.main()
