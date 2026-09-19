from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from vae_tile_contract import resolve_vae_module_path


@dataclass(frozen=True)
class SegmentRuntimeResult:
    activation: Any
    residual: Any | None


def run_segment_operations(
    vae: Any,
    operations: list[dict[str, Any]],
    activation: Any,
    residual: Any | None,
    *,
    silu: Callable[[Any], Any],
    attention: Callable[[Any, Any], Any],
) -> SegmentRuntimeResult:
    current = activation
    live_residual = residual

    for operation in operations:
        kind = operation.get("kind")
        module_path = operation.get("modulePath")
        residual_key = operation.get("residualKey")

        if kind == "module":
            if not isinstance(module_path, str):
                raise ValueError("module operation requires modulePath")
            module = resolve_vae_module_path(vae, module_path)
            if not callable(module):
                raise TypeError(f"{module_path} is not callable")
            current = module(current)
            continue

        if kind == "silu":
            current = silu(current)
            continue

        if kind == "store_residual":
            if live_residual is not None:
                raise ValueError(
                    f"residual is already live before storing {residual_key!r}"
                )
            shortcut = operation.get("shortcut")
            if shortcut == "identity":
                live_residual = current
            elif shortcut == "module":
                if not isinstance(module_path, str):
                    raise ValueError(
                        "module residual shortcut requires modulePath"
                    )
                module = resolve_vae_module_path(vae, module_path)
                if not callable(module):
                    raise TypeError(f"{module_path} is not callable")
                live_residual = module(current)
            else:
                raise ValueError(
                    f"unsupported residual shortcut {shortcut!r}"
                )
            continue

        if kind == "add_residual":
            if live_residual is None:
                raise ValueError(
                    f"cannot add residual {residual_key!r}; none is live"
                )
            current = current + live_residual
            live_residual = None
            continue

        if kind == "attention":
            if not isinstance(module_path, str):
                raise ValueError("attention operation requires modulePath")
            module = resolve_vae_module_path(vae, module_path)
            current = attention(module, current)
            continue

        raise ValueError(f"unsupported VAE segment operation {kind!r}")

    return SegmentRuntimeResult(
        activation=current,
        residual=live_residual,
    )


def run_partitioned_side(
    vae: Any,
    side_contract: dict[str, Any],
    activation: Any,
    *,
    normalize: Callable[[Any, Any], Any],
    silu: Callable[[Any], Any],
    attention: Callable[[Any, Any], Any],
) -> Any:
    prelude = side_contract.get("prelude")
    stages = side_contract.get("stages")
    if not isinstance(prelude, list) or not isinstance(stages, list):
        raise ValueError("partitioned VAE side contract is incomplete")

    result = run_segment_operations(
        vae,
        prelude,
        activation,
        None,
        silu=silu,
        attention=attention,
    )
    current = result.activation
    residual = result.residual

    for expected_index, stage in enumerate(stages):
        if stage.get("index") != expected_index:
            raise ValueError("partitioned VAE stage indices are not contiguous")
        barrier = stage.get("barrier")
        if not isinstance(barrier, dict):
            raise ValueError(
                f"partitioned VAE stage {expected_index} has no barrier"
            )
        barrier_path = barrier.get("module_path")
        if not isinstance(barrier_path, str):
            raise ValueError(
                f"partitioned VAE stage {expected_index} has invalid barrier path"
            )
        barrier_module = resolve_vae_module_path(vae, barrier_path)
        current = normalize(barrier_module, current)

        after = stage.get("after")
        if not isinstance(after, list):
            raise ValueError(
                f"partitioned VAE stage {expected_index} has no operations"
            )
        result = run_segment_operations(
            vae,
            after,
            current,
            residual,
            silu=silu,
            attention=attention,
        )
        current = result.activation
        residual = result.residual

    if residual is not None:
        raise ValueError("partitioned VAE side finished with a live residual")
    return current
