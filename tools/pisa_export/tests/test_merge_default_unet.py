import pathlib
import sys
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import export_pisa_sr


class FakeSource:
    def __init__(self, name):
        self.name = name

    def to(self, *, device, dtype):
        return self


class FakeParameter:
    def __init__(self, name, events):
        self.name = name
        self.events = events
        self.device = "cpu"
        self.dtype = "float32"
        self.data = self

    def copy_(self, source):
        self.events.append(f"copy:{self.name}")
        return self


class FakeLoraConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeUnet:
    def __init__(self, events):
        self.events = events
        self._parameters = []

    def add_adapter(self, config, *, adapter_name):
        self.events.append(f"add:{adapter_name}")
        self._parameters.append(
            (
                f"layer.lora_A.{adapter_name}.weight",
                FakeParameter(
                    f"layer.lora_A.{adapter_name}.weight",
                    self.events,
                ),
            )
        )

    def named_parameters(self):
        return list(self._parameters)

    def merge_and_unload(self):
        self.events.append("merge")
        self._parameters.clear()
        return self

    def eval(self):
        self.events.append("eval")
        return self

    def requires_grad_(self, enabled):
        self.events.append(f"requires_grad:{enabled}")
        return self


def fake_ml_modules(events):
    peft = types.ModuleType("peft")
    peft.LoraConfig = FakeLoraConfig

    peft_utils = types.ModuleType("diffusers.utils.peft_utils")

    def activate(unet, names, weights):
        events.append(
            "activate:"
            + ",".join(names)
            + ":"
            + ",".join(str(weight) for weight in weights)
        )

    peft_utils.set_weights_and_activate_adapters = activate

    diffusers = types.ModuleType("diffusers")
    diffusers_utils = types.ModuleType("diffusers.utils")
    diffusers.utils = diffusers_utils
    diffusers_utils.peft_utils = peft_utils

    return {
        "peft": peft,
        "diffusers": diffusers,
        "diffusers.utils": diffusers_utils,
        "diffusers.utils.peft_utils": peft_utils,
    }


class MergeDefaultUnetTest(unittest.TestCase):
    def test_pixel_is_merged_before_semantic(self):
        events = []
        unet = FakeUnet(events)

        adapter_names = [
            name
            for name, _ in (
                export_pisa_sr.PIXEL_ADAPTERS
                + export_pisa_sr.SEMANTIC_ADAPTERS
            )
        ]
        state_dict = {
            f"layer.lora_A.{name}.weight": FakeSource(name)
            for name in adapter_names
        }
        checkpoint = {
            "lora_rank_unet_pix": 4,
            "lora_rank_unet_sem": 8,
            "unet_lora_encoder_modules_pix": ["pixel.encoder"],
            "unet_lora_decoder_modules_pix": ["pixel.decoder"],
            "unet_lora_others_modules_pix": ["pixel.other"],
            "unet_lora_encoder_modules_sem": ["semantic.encoder"],
            "unet_lora_decoder_modules_sem": ["semantic.decoder"],
            "unet_lora_others_modules_sem": ["semantic.other"],
            "state_dict_unet": state_dict,
        }

        with patch.dict(sys.modules, fake_ml_modules(events)):
            result = export_pisa_sr.merge_default_unet(unet, checkpoint)

        self.assertIs(unet, result)

        first_merge = events.index("merge")
        second_merge = events.index("merge", first_merge + 1)

        pixel_names = [name for name, _ in export_pisa_sr.PIXEL_ADAPTERS]
        semantic_names = [name for name, _ in export_pisa_sr.SEMANTIC_ADAPTERS]

        for name in pixel_names:
            self.assertLess(events.index(f"add:{name}"), first_merge)
            self.assertLess(
                events.index(f"copy:layer.lora_A.{name}.weight"),
                first_merge,
            )

        for name in semantic_names:
            self.assertGreater(events.index(f"add:{name}"), first_merge)
            self.assertLess(events.index(f"add:{name}"), second_merge)
            self.assertGreater(
                events.index(f"copy:layer.lora_A.{name}.weight"),
                first_merge,
            )
            self.assertLess(
                events.index(f"copy:layer.lora_A.{name}.weight"),
                second_merge,
            )

        self.assertEqual(
            [
                "activate:"
                + ",".join(pixel_names)
                + ":1.0,1.0,1.0",
                "activate:"
                + ",".join(semantic_names)
                + ":1.0,1.0,1.0",
            ],
            [event for event in events if event.startswith("activate:")],
        )
        self.assertEqual(2, events.count("merge"))
        self.assertEqual(["eval", "requires_grad:False"], events[-2:])


if __name__ == "__main__":
    unittest.main()
