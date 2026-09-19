import pathlib
import sys
import types
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import export_pisa_sr


class FakeModule:
    def eval(self):
        return self

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class FakeTorch:
    nn = types.SimpleNamespace(
        Module=FakeModule,
        functional=types.SimpleNamespace(
            silu=lambda value: value,
        ),
    )


class UnaryOp:
    def __init__(self, label, events):
        self.label = label
        self.events = events

    def __call__(self, value):
        self.events.append((self.label, value))
        return f"{self.label}({value})"


class FakeUnet:
    def __init__(self):
        self.calls = []

    def __call__(self, latent, timestep, *, encoder_hidden_states):
        self.calls.append(
            {
                "latent": latent,
                "timestep": timestep,
                "encoder_hidden_states": encoder_hidden_states,
            }
        )
        return types.SimpleNamespace(sample="model_pred")


class ExportWrapperTest(unittest.TestCase):
    def test_vae_encoder_exports_moments_without_sampling(self):
        events = []
        vae = types.SimpleNamespace(
            encoder=UnaryOp("encoder", events),
            quant_conv=UnaryOp("quant_conv", events),
        )

        wrapper = export_pisa_sr._make_vae_encoder_wrapper(FakeTorch, vae)
        result = wrapper("image")

        self.assertEqual("quant_conv(encoder(image))", result)
        self.assertEqual(
            [
                ("encoder", "image"),
                ("quant_conv", "encoder(image)"),
            ],
            events,
        )

    def test_vae_decoder_applies_post_quant_before_decoder(self):
        events = []
        vae = types.SimpleNamespace(
            post_quant_conv=UnaryOp("post_quant_conv", events),
            decoder=UnaryOp("decoder", events),
        )

        wrapper = export_pisa_sr._make_vae_decoder_wrapper(FakeTorch, vae)
        result = wrapper("latent")

        self.assertEqual(
            "decoder(post_quant_conv(latent))",
            result,
        )
        self.assertEqual(
            [
                ("post_quant_conv", "latent"),
                ("decoder", "post_quant_conv(latent)"),
            ],
            events,
        )

    def test_vae_segment_wrapper_respects_residual_contract(self):
        events = []
        vae = types.SimpleNamespace(
            encoder=types.SimpleNamespace(
                conv_in=UnaryOp("conv_in", events),
            )
        )
        segment = {
            "requiresResidualInput": False,
            "producesResidualOutput": True,
            "operations": [
                {
                    "kind": "module",
                    "modulePath": "encoder.conv_in",
                    "residualKey": None,
                    "shortcut": None,
                },
                {
                    "kind": "store_residual",
                    "modulePath": None,
                    "residualKey": "r0",
                    "shortcut": "identity",
                },
            ],
        }

        wrapper = export_pisa_sr._make_vae_segment_wrapper(
            FakeTorch,
            vae,
            segment,
            silu=lambda value: value,
            attention=lambda module, value: module(value),
        )
        activation, residual = wrapper("image")

        self.assertEqual("conv_in(image)", activation)
        self.assertEqual("conv_in(image)", residual)
        self.assertEqual([("conv_in", "image")], events)

    def test_vae_segment_wrapper_requires_declared_residual(self):
        vae = types.SimpleNamespace()
        segment = {
            "requiresResidualInput": True,
            "producesResidualOutput": False,
            "operations": [
                {
                    "kind": "add_residual",
                    "modulePath": None,
                    "residualKey": "r0",
                    "shortcut": None,
                }
            ],
        }
        wrapper = export_pisa_sr._make_vae_segment_wrapper(
            FakeTorch,
            vae,
            segment,
            silu=lambda value: value,
            attention=lambda module, value: value,
        )

        with self.assertRaisesRegex(ValueError, "requires residual"):
            wrapper("activation")

        self.assertEqual(
            "activationresidual",
            wrapper("activation", "residual"),
        )

    def test_vae_segment_onnx_spec_tracks_residual_io(self):
        spec = export_pisa_sr._vae_segment_onnx_spec(
            "encoder",
            {
                "index": 3,
                "requiresResidualInput": True,
                "producesResidualOutput": True,
            },
        )

        self.assertEqual("vae_encoder_segment_03.onnx", spec["fileName"])
        self.assertEqual(
            ["activation", "residual"],
            spec["inputNames"],
        )
        self.assertEqual(
            ["activation_out", "residual_out"],
            spec["outputNames"],
        )
        self.assertEqual(
            {2: "vae_encoder_segment_03_height", 3: "vae_encoder_segment_03_width"},
            spec["dynamicAxes"]["activation"],
        )

    def test_collects_segment_examples_across_external_norm(self):
        events = []
        vae = types.SimpleNamespace(
            encoder=types.SimpleNamespace(
                conv_in=UnaryOp("conv_in", events),
                norm=UnaryOp("norm", events),
            )
        )
        side_contract = {
            "segments": [
                {
                    "index": 0,
                    "entryBarrier": None,
                    "requiresResidualInput": False,
                    "producesResidualOutput": True,
                    "operations": [
                        {
                            "kind": "module",
                            "modulePath": "encoder.conv_in",
                            "residualKey": None,
                            "shortcut": None,
                        },
                        {
                            "kind": "store_residual",
                            "modulePath": None,
                            "residualKey": "r0",
                            "shortcut": "identity",
                        },
                    ],
                },
                {
                    "index": 1,
                    "entryBarrier": "encoder.norm",
                    "requiresResidualInput": True,
                    "producesResidualOutput": False,
                    "operations": [
                        {
                            "kind": "add_residual",
                            "modulePath": None,
                            "residualKey": "r0",
                            "shortcut": None,
                        }
                    ],
                },
            ]
        }

        examples, result = export_pisa_sr._collect_vae_segment_examples(
            FakeTorch,
            vae,
            side_contract,
            "image",
        )

        self.assertEqual(2, len(examples))
        self.assertEqual(("image",), examples[0]["args"])
        self.assertEqual(
            ("norm(conv_in(image))", "conv_in(image)"),
            examples[1]["args"],
        )
        self.assertEqual(
            "norm(conv_in(image))conv_in(image)",
            result,
        )

    def test_default_unet_forwards_all_three_inputs_and_returns_sample(self):
        unet = FakeUnet()
        wrapper = export_pisa_sr._make_unet_wrapper(FakeTorch, unet)

        result = wrapper(
            "latent",
            "timestep",
            "empty_prompt_embedding",
        )

        self.assertEqual("model_pred", result)
        self.assertEqual(
            [
                {
                    "latent": "latent",
                    "timestep": "timestep",
                    "encoder_hidden_states": "empty_prompt_embedding",
                }
            ],
            unet.calls,
        )


if __name__ == "__main__":
    unittest.main()
