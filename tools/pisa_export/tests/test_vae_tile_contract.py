import pathlib
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import vae_tile_contract


def norm(groups=32, channels=32, epsilon=1.0e-6):
    return SimpleNamespace(
        num_groups=groups,
        num_channels=channels,
        eps=epsilon,
        weight=[1.0 + index * 0.01 for index in range(channels)],
        bias=[-0.5 + index * 0.02 for index in range(channels)],
    )


def resnet(groups=32, in_channels=4, out_channels=4):
    shortcut = object() if in_channels != out_channels else None
    return SimpleNamespace(
        norm1=norm(groups),
        norm2=norm(groups),
        conv1=object(),
        conv2=object(),
        in_channels=in_channels,
        out_channels=out_channels,
        use_in_shortcut=in_channels != out_channels,
        conv_shortcut=shortcut,
        nin_shortcut=None,
    )


def attention(groups=32):
    return SimpleNamespace(group_norm=norm(groups))


def fake_vae():
    encoder = SimpleNamespace(
        down_blocks=[
            SimpleNamespace(
                resnets=[resnet(), resnet()],
                downsamplers=[object()],
            ),
            SimpleNamespace(resnets=[resnet()]),
        ],
        mid_block=SimpleNamespace(
            resnets=[resnet(), resnet()],
            attentions=[attention()],
        ),
        conv_in=object(),
        conv_norm_out=norm(),
        conv_out=object(),
    )
    decoder = SimpleNamespace(
        mid_block=SimpleNamespace(
            resnets=[resnet(), resnet()],
            attentions=[attention()],
        ),
        up_blocks=[
            SimpleNamespace(
                resnets=[resnet(), resnet(), resnet()],
                upsamplers=[object()],
            ),
            SimpleNamespace(resnets=[resnet()]),
        ],
        conv_in=object(),
        conv_norm_out=norm(),
        conv_out=object(),
    )
    return SimpleNamespace(
        encoder=encoder,
        decoder=decoder,
        quant_conv=object(),
        post_quant_conv=object(),
    )


class VaeTileContractTest(unittest.TestCase):
    def test_builds_barrier_order_and_residual_contract(self):
        contract = vae_tile_contract.build_vae_tile_barrier_contract(fake_vae())

        self.assertEqual(1, contract["schemaVersion"])
        self.assertEqual(32, contract["groupCount"])
        self.assertEqual(12, contract["encoderBarrierCount"])
        self.assertEqual(14, contract["decoderBarrierCount"])

        encoder = contract["encoder"]
        self.assertEqual(
            "encoder.down_blocks.0.resnets.0.norm1",
            encoder[0]["module_path"],
        )
        self.assertEqual("resnet_norm1", encoder[0]["kind"])
        self.assertEqual("open", encoder[0]["residual_action"])
        self.assertEqual(
            "encoder.down_blocks.0.resnets.0",
            encoder[0]["residual_key"],
        )

        self.assertEqual(
            "encoder.down_blocks.0.resnets.0.norm2",
            encoder[1]["module_path"],
        )
        self.assertEqual(
            "consume_after_segment",
            encoder[1]["residual_action"],
        )
        self.assertEqual(
            "encoder.mid_block.attentions.0.group_norm",
            encoder[8]["module_path"],
        )
        self.assertEqual(
            "open_and_consume_after_segment",
            encoder[8]["residual_action"],
        )
        self.assertEqual("encoder.conv_norm_out", encoder[-1]["module_path"])
        self.assertIsNone(encoder[-1]["residual_key"])

        decoder = contract["decoder"]
        self.assertEqual(
            "decoder.mid_block.resnets.0.norm1",
            decoder[0]["module_path"],
        )
        self.assertEqual(
            "decoder.mid_block.attentions.0.group_norm",
            decoder[2]["module_path"],
        )
        self.assertEqual(
            "decoder.up_blocks.1.resnets.0.norm2",
            decoder[-2]["module_path"],
        )
        self.assertEqual("decoder.conv_norm_out", decoder[-1]["module_path"])

    def test_builds_group_norm_affine_contract(self):
        contract = vae_tile_contract.build_vae_group_norm_affine_contract(
            fake_vae()
        )

        self.assertEqual(1, contract["schemaVersion"])
        self.assertEqual(12, contract["encoderCount"])
        self.assertEqual(14, contract["decoderCount"])

        first = contract["encoder"][0]
        self.assertEqual(0, first["index"])
        self.assertEqual(
            "encoder.down_blocks.0.resnets.0.norm1",
            first["modulePath"],
        )
        self.assertEqual(32, first["groups"])
        self.assertEqual(32, first["channels"])
        self.assertAlmostEqual(1.0e-6, first["epsilon"])
        self.assertEqual(32, len(first["weight"]))
        self.assertEqual(32, len(first["bias"]))
        self.assertAlmostEqual(1.0, first["weight"][0])
        self.assertAlmostEqual(-0.5, first["bias"][0])

    def test_affine_contract_rejects_missing_weight(self):
        model = fake_vae()
        model.encoder.down_blocks[0].resnets[0].norm1.weight = None

        with self.assertRaisesRegex(ValueError, "affine weight"):
            vae_tile_contract.build_vae_group_norm_affine_contract(model)

    def test_builds_execution_recipe_around_group_norm_barriers(self):
        contract = vae_tile_contract.build_vae_tile_execution_contract(fake_vae())

        encoder = contract["encoder"]
        self.assertEqual(
            [
                {
                    "kind": "module",
                    "modulePath": "encoder.conv_in",
                    "residualKey": None,
                    "shortcut": None,
                },
                {
                    "kind": "store_residual",
                    "modulePath": None,
                    "residualKey": "encoder.down_blocks.0.resnets.0",
                    "shortcut": "identity",
                },
            ],
            encoder["prelude"],
        )
        self.assertEqual(12, len(encoder["stages"]))
        self.assertEqual(
            "encoder.down_blocks.0.resnets.0.norm1",
            encoder["stages"][0]["barrier"]["module_path"],
        )
        self.assertEqual(
            ["silu", "module"],
            [op["kind"] for op in encoder["stages"][0]["after"]],
        )
        self.assertEqual(
            "encoder.down_blocks.0.resnets.0.conv1",
            encoder["stages"][0]["after"][1]["modulePath"],
        )

        norm2_after = encoder["stages"][1]["after"]
        self.assertEqual(
            ["silu", "module", "add_residual", "store_residual"],
            [op["kind"] for op in norm2_after],
        )
        self.assertEqual(
            "encoder.down_blocks.0.resnets.1",
            norm2_after[-1]["residualKey"],
        )

        self.assertIn(
            "encoder.down_blocks.0.downsamplers.0",
            [
                op["modulePath"]
                for stage in encoder["stages"]
                for op in stage["after"]
                if op["kind"] == "module"
            ],
        )
        self.assertEqual(
            ["silu", "module", "module"],
            [op["kind"] for op in encoder["stages"][-1]["after"]],
        )
        self.assertEqual(
            "encoder.conv_out",
            encoder["stages"][-1]["after"][-2]["modulePath"],
        )
        self.assertEqual(
            "quant_conv",
            encoder["stages"][-1]["after"][-1]["modulePath"],
        )

        decoder = contract["decoder"]
        self.assertEqual(14, len(decoder["stages"]))
        self.assertEqual(
            ["post_quant_conv", "decoder.conv_in"],
            [
                decoder["prelude"][0]["modulePath"],
                decoder["prelude"][1]["modulePath"],
            ],
        )
        self.assertIn(
            "decoder.up_blocks.0.upsamplers.0",
            [
                op["modulePath"]
                for stage in decoder["stages"]
                for op in stage["after"]
                if op["kind"] == "module"
            ],
        )
        self.assertEqual(
            "decoder.conv_out",
            decoder["stages"][-1]["after"][-1]["modulePath"],
        )

    def test_records_channel_changing_residual_shortcut(self):
        model = fake_vae()
        model.encoder.down_blocks[0].resnets[0] = resnet(
            in_channels=4,
            out_channels=8,
        )

        contract = vae_tile_contract.build_vae_tile_execution_contract(model)

        residual = contract["encoder"]["prelude"][1]
        self.assertEqual("module", residual["shortcut"])
        self.assertEqual(
            "encoder.down_blocks.0.resnets.0.conv_shortcut",
            residual["modulePath"],
        )

    def test_builds_segment_io_contract_from_residual_liveness(self):
        execution = vae_tile_contract.build_vae_tile_execution_contract(
            fake_vae()
        )
        contract = vae_tile_contract.build_vae_tile_segment_contract(
            execution
        )

        encoder = contract["encoder"]
        self.assertEqual(13, encoder["segmentCount"])
        self.assertEqual("encoder_prelude", encoder["segments"][0]["id"])
        self.assertFalse(
            encoder["segments"][0]["requiresResidualInput"]
        )
        self.assertTrue(
            encoder["segments"][0]["producesResidualOutput"]
        )
        self.assertEqual(
            "encoder.down_blocks.0.resnets.0.norm1",
            encoder["segments"][0]["exitBarrier"],
        )

        norm1_segment = encoder["segments"][1]
        self.assertTrue(norm1_segment["requiresResidualInput"])
        self.assertTrue(norm1_segment["producesResidualOutput"])
        self.assertEqual(
            "encoder.down_blocks.0.resnets.0.norm1",
            norm1_segment["entryBarrier"],
        )
        self.assertEqual(
            "encoder.down_blocks.0.resnets.0.norm2",
            norm1_segment["exitBarrier"],
        )

        final_segment = encoder["segments"][-1]
        self.assertFalse(final_segment["requiresResidualInput"])
        self.assertFalse(final_segment["producesResidualOutput"])
        self.assertIsNone(final_segment["exitBarrier"])
        self.assertEqual(
            ["silu", "module", "module"],
            [op["kind"] for op in final_segment["operations"]],
        )
        self.assertEqual(
            "quant_conv",
            final_segment["operations"][-1]["modulePath"],
        )

        decoder = contract["decoder"]
        self.assertEqual(15, decoder["segmentCount"])
        self.assertEqual("decoder_prelude", decoder["segments"][0]["id"])
        self.assertFalse(
            decoder["segments"][-1]["producesResidualOutput"]
        )

    def test_segment_contract_rejects_residual_liveness_mismatch(self):
        execution = vae_tile_contract.build_vae_tile_execution_contract(
            fake_vae()
        )
        execution["encoder"]["prelude"][-1]["residualKey"] = "wrong"

        with self.assertRaisesRegex(ValueError, "first barrier"):
            vae_tile_contract.build_vae_tile_segment_contract(execution)

    def test_resolves_and_validates_all_execution_module_paths(self):
        model = fake_vae()
        contract = vae_tile_contract.build_vae_tile_execution_contract(model)

        resolved = vae_tile_contract.resolve_vae_module_path(
            model,
            "encoder.down_blocks.0.resnets.1.norm2",
        )
        self.assertIs(
            model.encoder.down_blocks[0].resnets[1].norm2,
            resolved,
        )

        resolved_count = (
            vae_tile_contract.validate_vae_tile_execution_contract(
                model,
                contract,
            )
        )
        self.assertGreater(resolved_count, 20)

    def test_contract_validation_rejects_stale_module_path(self):
        model = fake_vae()
        contract = vae_tile_contract.build_vae_tile_execution_contract(model)
        contract["encoder"]["stages"][0]["after"][1]["modulePath"] = (
            "encoder.down_blocks.0.resnets.0.missing_conv"
        )

        with self.assertRaisesRegex(ValueError, "missing_conv"):
            vae_tile_contract.validate_vae_tile_execution_contract(
                model,
                contract,
            )

    def test_contract_validation_rejects_changed_group_count(self):
        model = fake_vae()
        contract = vae_tile_contract.build_vae_tile_execution_contract(model)
        model.encoder.down_blocks[0].resnets[0].norm1.num_groups = 16

        with self.assertRaisesRegex(ValueError, "group count changed"):
            vae_tile_contract.validate_vae_tile_execution_contract(
                model,
                contract,
            )

    def test_rejects_non_32_group_contract(self):
        model = fake_vae()
        model.encoder.down_blocks[0].resnets[0].norm1.num_groups = 16

        with self.assertRaisesRegex(ValueError, "does not use 32 groups"):
            vae_tile_contract.build_vae_tile_barrier_contract(model)

    def test_rejects_missing_mid_attention(self):
        model = fake_vae()
        model.decoder.mid_block.attentions = []

        with self.assertRaisesRegex(ValueError, "exactly one attention"):
            vae_tile_contract.build_vae_tile_barrier_contract(model)


if __name__ == "__main__":
    unittest.main()
