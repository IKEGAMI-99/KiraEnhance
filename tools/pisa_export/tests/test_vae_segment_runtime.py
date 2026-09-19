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
