import pathlib
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import vae_segment_runtime


class Op:
    def __init__(self, function):
        self.function = function

    def __call__(self, value):
        return self.function(value)


def fake_vae():
    return SimpleNamespace(
        encoder=SimpleNamespace(
            conv_in=Op(lambda value: value + 1),
            block=SimpleNamespace(
                conv1=Op(lambda value: value * 3),
                shortcut=Op(lambda value: value + 10),
            ),
            attention=Op(lambda value: value),
        )
    )


class VaeSegmentRuntimeTest(unittest.TestCase):
    def test_runs_module_silu_and_identity_residual_flow(self):
        model = fake_vae()
        operations = [
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
            {
                "kind": "silu",
                "modulePath": None,
                "residualKey": None,
                "shortcut": None,
            },
            {
                "kind": "module",
                "modulePath": "encoder.block.conv1",
                "residualKey": None,
                "shortcut": None,
            },
            {
                "kind": "add_residual",
                "modulePath": None,
                "residualKey": "r0",
                "shortcut": None,
            },
        ]

        result = vae_segment_runtime.run_segment_operations(
            model,
            operations,
            activation=2,
            residual=None,
            silu=lambda value: value * 2,
            attention=lambda module, value: module(value),
        )

        self.assertEqual(21, result.activation)
        self.assertIsNone(result.residual)

    def test_runs_module_residual_shortcut(self):
        model = fake_vae()
        operations = [
            {
                "kind": "store_residual",
                "modulePath": "encoder.block.shortcut",
                "residualKey": "r0",
                "shortcut": "module",
            },
        ]

        result = vae_segment_runtime.run_segment_operations(
            model,
            operations,
            activation=5,
            residual=None,
            silu=lambda value: value,
            attention=lambda module, value: module(value),
        )

        self.assertEqual(5, result.activation)
        self.assertEqual(15, result.residual)

    def test_attention_uses_injected_upstream_compatible_function(self):
        model = fake_vae()
        calls = []

        result = vae_segment_runtime.run_segment_operations(
            model,
            [
                {
                    "kind": "attention",
                    "modulePath": "encoder.attention",
                    "residualKey": None,
                    "shortcut": None,
                }
            ],
            activation=4,
            residual=None,
            silu=lambda value: value,
            attention=lambda module, value: calls.append(module) or value + 7,
        )

        self.assertEqual(11, result.activation)
        self.assertEqual([model.encoder.attention], calls)

    def test_runs_complete_partitioned_side_with_external_norm(self):
        model = SimpleNamespace(
            encoder=SimpleNamespace(
                conv_in=Op(lambda value: value + 1),
                norm=Op(lambda value: value * 10),
                conv_out=Op(lambda value: value + 3),
            )
        )
        side_contract = {
            "prelude": [
                {
                    "kind": "module",
                    "modulePath": "encoder.conv_in",
                    "residualKey": None,
                    "shortcut": None,
                }
            ],
            "stages": [
                {
                    "index": 0,
                    "barrier": {
                        "module_path": "encoder.norm",
                    },
                    "after": [
                        {
                            "kind": "silu",
                            "modulePath": None,
                            "residualKey": None,
                            "shortcut": None,
                        },
                        {
                            "kind": "module",
                            "modulePath": "encoder.conv_out",
                            "residualKey": None,
                            "shortcut": None,
                        },
                    ],
                }
            ],
        }

        result = vae_segment_runtime.run_partitioned_side(
            model,
            side_contract,
            activation=2,
            normalize=lambda module, value: module(value),
            silu=lambda value: value * 2,
            attention=lambda module, value: module(value),
        )

        self.assertEqual(63, result)

    def test_runs_tiles_in_lockstep_across_group_norm_barrier(self):
        model = SimpleNamespace(
            encoder=SimpleNamespace(
                conv_in=Op(lambda value: value + 1),
                norm=Op(lambda value: value * 100),
                conv_out=Op(lambda value: value + 3),
            )
        )
        side_contract = {
            "prelude": [
                {
                    "kind": "module",
                    "modulePath": "encoder.conv_in",
                    "residualKey": None,
                    "shortcut": None,
                }
            ],
            "stages": [
                {
                    "index": 0,
                    "barrier": {
                        "module_path": "encoder.norm",
                    },
                    "after": [
                        {
                            "kind": "silu",
                            "modulePath": None,
                            "residualKey": None,
                            "shortcut": None,
                        },
                        {
                            "kind": "module",
                            "modulePath": "encoder.conv_out",
                            "residualKey": None,
                            "shortcut": None,
                        },
                    ],
                }
            ],
        }
        calls = []

        def normalize_tiles(module, values):
            calls.append((module, list(values)))
            total = sum(values)
            return [value + total for value in values]

        result = vae_segment_runtime.run_partitioned_tiles(
            model,
            side_contract,
            activations=[1, 3],
            normalize_tiles=normalize_tiles,
            silu=lambda value: value * 2,
            attention=lambda module, value: module(value),
        )

        self.assertEqual([19, 23], result)
        self.assertEqual([(model.encoder.norm, [2, 4])], calls)

    def test_lockstep_runtime_rejects_normalization_tile_count_change(self):
        model = SimpleNamespace(
            encoder=SimpleNamespace(
                norm=Op(lambda value: value),
            )
        )
        side_contract = {
            "prelude": [],
            "stages": [
                {
                    "index": 0,
                    "barrier": {
                        "module_path": "encoder.norm",
                    },
                    "after": [],
                }
            ],
        }

        with self.assertRaisesRegex(ValueError, "changed tile count"):
            vae_segment_runtime.run_partitioned_tiles(
                model,
                side_contract,
                activations=[1, 2],
                normalize_tiles=lambda module, values: values[:1],
                silu=lambda value: value,
                attention=lambda module, value: value,
            )

    def test_rejects_invalid_residual_state_and_unknown_operation(self):
        model = fake_vae()

        with self.assertRaisesRegex(ValueError, "already live"):
            vae_segment_runtime.run_segment_operations(
                model,
                [
                    {
                        "kind": "store_residual",
                        "modulePath": None,
                        "residualKey": "r1",
                        "shortcut": "identity",
                    }
                ],
                activation=1,
                residual=2,
                silu=lambda value: value,
                attention=lambda module, value: value,
            )

        with self.assertRaisesRegex(ValueError, "unsupported"):
            vae_segment_runtime.run_segment_operations(
                model,
                [{"kind": "teleport"}],
                activation=1,
                residual=None,
                silu=lambda value: value,
                attention=lambda module, value: value,
            )


if __name__ == "__main__":
    unittest.main()
