import pathlib
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import run_official_validation


class OfficialValidationRunnerTest(unittest.TestCase):
    def test_cli_defaults_match_android_validation_contract(self):
        args = run_official_validation.build_parser().parse_args(
            [
                "--pisa-repo",
                "/tmp/pisa",
                "--sd21-base",
                "/tmp/sd21",
                "--pisa-checkpoint",
                "/tmp/pisa_sr.pkl",
                "--input-image",
                "/tmp/input.png",
                "--output-image",
                "/tmp/output.png",
            ]
        )

        self.assertEqual(42, args.seed)
        self.assertEqual(512, args.process_size)
        self.assertEqual(4, args.upscale)
        self.assertEqual(224, args.vae_decoder_tiled_size)
        self.assertEqual(1024, args.vae_encoder_tiled_size)
        self.assertEqual(96, args.latent_tiled_size)
        self.assertEqual(32, args.latent_tiled_overlap)
        self.assertEqual("fp16", args.mixed_precision)
        self.assertIsNone(args.dump_npz)

    def test_model_args_select_default_one_step_configuration(self):
        args = SimpleNamespace(
            sd21_base=pathlib.Path("/tmp/sd21"),
            pisa_checkpoint=pathlib.Path("/tmp/pisa_sr.pkl"),
            mixed_precision="fp16",
            vae_decoder_tiled_size=224,
            vae_encoder_tiled_size=1024,
            latent_tiled_size=96,
            latent_tiled_overlap=32,
        )

        model_args = run_official_validation._model_args(args)

        self.assertEqual(1.0, model_args.lambda_pix)
        self.assertEqual(1.0, model_args.lambda_sem)
        self.assertEqual(224, model_args.vae_decoder_tiled_size)
        self.assertEqual(96, model_args.latent_tiled_size)


if __name__ == "__main__":
    unittest.main()
