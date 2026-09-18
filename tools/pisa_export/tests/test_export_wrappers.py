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
    nn = types.SimpleNamespace(Module=FakeModule)


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
