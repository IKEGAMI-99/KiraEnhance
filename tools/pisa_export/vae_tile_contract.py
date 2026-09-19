from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable


@dataclass(frozen=True)
class VaeGroupNormBarrier:
    index: int
    module_path: str
    kind: str
    groups: int
    residual_key: str | None
    residual_action: str


def _groups(module: Any, path: str) -> int:
    value = getattr(module, "num_groups", None)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{path} does not expose a valid num_groups")
    return value


def _resnet_barriers(
    prefix: str,
    resnet: Any,
) -> Iterable[tuple[str, Any, str, str | None, str]]:
    residual_key = prefix
    yield (
        f"{prefix}.norm1",
        resnet.norm1,
        "resnet_norm1",
        residual_key,
        "open",
    )
    yield (
        f"{prefix}.norm2",
        resnet.norm2,
        "resnet_norm2",
        residual_key,
        "consume_after_segment",
    )


def _attention_barrier(
    prefix: str,
    attention: Any,
) -> tuple[str, Any, str, str | None, str]:
    return (
        f"{prefix}.group_norm",
        attention.group_norm,
        "attention_group_norm",
        prefix,
        "open_and_consume_after_segment",
    )


def _enumerate_encoder(encoder: Any):
    for block_index, block in enumerate(encoder.down_blocks):
        for resnet_index, resnet in enumerate(block.resnets):
            prefix = f"encoder.down_blocks.{block_index}.resnets.{resnet_index}"
            yield from _resnet_barriers(prefix, resnet)

    for resnet_index, resnet in enumerate(encoder.mid_block.resnets):
        prefix = f"encoder.mid_block.resnets.{resnet_index}"
        yield from _resnet_barriers(prefix, resnet)

        if resnet_index == 0:
            attentions = list(encoder.mid_block.attentions)
            if len(attentions) != 1:
                raise ValueError(
                    "PiSA encoder mid block must contain exactly one attention"
                )
            yield _attention_barrier(
                "encoder.mid_block.attentions.0",
                attentions[0],
            )

    yield (
        "encoder.conv_norm_out",
        encoder.conv_norm_out,
        "output_group_norm",
        None,
        "none",
    )


def _enumerate_decoder(decoder: Any):
    for resnet_index, resnet in enumerate(decoder.mid_block.resnets):
        prefix = f"decoder.mid_block.resnets.{resnet_index}"
        yield from _resnet_barriers(prefix, resnet)

        if resnet_index == 0:
            attentions = list(decoder.mid_block.attentions)
            if len(attentions) != 1:
                raise ValueError(
                    "PiSA decoder mid block must contain exactly one attention"
                )
            yield _attention_barrier(
                "decoder.mid_block.attentions.0",
                attentions[0],
            )

    for block_index, block in enumerate(decoder.up_blocks):
        for resnet_index, resnet in enumerate(block.resnets):
            prefix = f"decoder.up_blocks.{block_index}.resnets.{resnet_index}"
            yield from _resnet_barriers(prefix, resnet)

    yield (
        "decoder.conv_norm_out",
        decoder.conv_norm_out,
        "output_group_norm",
        None,
        "none",
    )


def _build_barriers(
    entries: Iterable[tuple[str, Any, str, str | None, str]],
) -> list[VaeGroupNormBarrier]:
    result: list[VaeGroupNormBarrier] = []
    for index, (
        path,
        module,
        kind,
        residual_key,
        residual_action,
    ) in enumerate(entries):
        result.append(
            VaeGroupNormBarrier(
                index=index,
                module_path=path,
                kind=kind,
                groups=_groups(module, path),
                residual_key=residual_key,
                residual_action=residual_action,
            )
        )
    return result


def _float_vector(value: Any, path: str, label: str) -> list[float]:
    if value is None:
        raise ValueError(f"{path} does not expose affine {label}")

    current = value
    for method_name in ("detach", "float", "cpu"):
        method = getattr(current, method_name, None)
        if callable(method):
            current = method()
    reshape = getattr(current, "reshape", None)
    if callable(reshape):
        current = reshape(-1)
    tolist = getattr(current, "tolist", None)
    if callable(tolist):
        current = tolist()

    try:
        values = [float(item) for item in current]
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{path} exposes invalid affine {label}"
        ) from error

    if not values or any(not math.isfinite(item) for item in values):
        raise ValueError(f"{path} exposes invalid affine {label}")
    return values


def _group_norm_affine_record(
    index: int,
    path: str,
    module: Any,
) -> dict[str, Any]:
    groups = _groups(module, path)
    channels = getattr(module, "num_channels", None)
    epsilon = getattr(module, "eps", None)
    if (
        not isinstance(channels, int) or
        channels <= 0 or
        channels % groups != 0
    ):
        raise ValueError(f"{path} does not expose valid channel dimensions")
    if (
        not isinstance(epsilon, (int, float)) or
        not math.isfinite(float(epsilon)) or
        float(epsilon) <= 0.0
    ):
        raise ValueError(f"{path} does not expose a valid epsilon")

    weight = _float_vector(getattr(module, "weight", None), path, "weight")
    bias = _float_vector(getattr(module, "bias", None), path, "bias")
    if len(weight) != channels or len(bias) != channels:
        raise ValueError(f"{path} affine vector length does not match channels")

    return {
        "index": index,
        "modulePath": path,
        "groups": groups,
        "channels": channels,
        "epsilon": float(epsilon),
        "weight": weight,
        "bias": bias,
    }


def build_vae_group_norm_affine_contract(vae: Any) -> dict[str, Any]:
    encoder_entries = list(_enumerate_encoder(vae.encoder))
    decoder_entries = list(_enumerate_decoder(vae.decoder))

    encoder = [
        _group_norm_affine_record(index, path, module)
        for index, (path, module, _, _, _) in enumerate(encoder_entries)
    ]
    decoder = [
        _group_norm_affine_record(index, path, module)
        for index, (path, module, _, _, _) in enumerate(decoder_entries)
    ]

    return {
        "schemaVersion": 1,
        "encoderCount": len(encoder),
        "decoderCount": len(decoder),
        "encoder": encoder,
        "decoder": decoder,
    }


def build_vae_tile_barrier_contract(vae: Any) -> dict[str, Any]:
    encoder = _build_barriers(_enumerate_encoder(vae.encoder))
    decoder = _build_barriers(_enumerate_decoder(vae.decoder))

    for label, barriers in (("encoder", encoder), ("decoder", decoder)):
        if not barriers:
            raise ValueError(f"{label} contains no GroupNorm barriers")
        if any(barrier.groups != 32 for barrier in barriers):
            raise ValueError(
                f"{label} contains a GroupNorm barrier that does not use 32 groups"
            )

    return {
        "schemaVersion": 1,
        "groupCount": 32,
        "encoderBarrierCount": len(encoder),
        "decoderBarrierCount": len(decoder),
        "encoder": [asdict(barrier) for barrier in encoder],
        "decoder": [asdict(barrier) for barrier in decoder],
    }


def _operation(
    kind: str,
    *,
    module_path: str | None = None,
    residual_key: str | None = None,
    shortcut: str | None = None,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "modulePath": module_path,
        "residualKey": residual_key,
        "shortcut": shortcut,
    }


def _residual_shortcut(prefix: str, resnet: Any) -> tuple[str, str | None]:
    in_channels = getattr(resnet, "in_channels", None)
    out_channels = getattr(resnet, "out_channels", None)
    if not isinstance(in_channels, int) or not isinstance(out_channels, int):
        raise ValueError(f"{prefix} does not expose channel dimensions")

    if in_channels == out_channels:
        return "identity", None

    if bool(getattr(resnet, "use_in_shortcut", False)):
        if getattr(resnet, "conv_shortcut", None) is None:
            raise ValueError(f"{prefix} requires conv_shortcut but it is missing")
        return "module", f"{prefix}.conv_shortcut"

    if getattr(resnet, "nin_shortcut", None) is not None:
        return "module", f"{prefix}.nin_shortcut"

    if getattr(resnet, "conv_shortcut", None) is not None:
        return "module", f"{prefix}.conv_shortcut"

    raise ValueError(f"{prefix} changes channels without a residual shortcut")


def _append_resnet_tokens(tokens: list[dict[str, Any]], prefix: str, resnet: Any) -> None:
    shortcut, shortcut_path = _residual_shortcut(prefix, resnet)
    tokens.append(
        _operation(
            "store_residual",
            module_path=shortcut_path,
            residual_key=prefix,
            shortcut=shortcut,
        )
    )
    tokens.append(
        _operation(
            "group_norm_barrier",
            module_path=f"{prefix}.norm1",
            residual_key=prefix,
        )
    )
    tokens.append(_operation("silu"))
    tokens.append(_operation("module", module_path=f"{prefix}.conv1"))
    tokens.append(
        _operation(
            "group_norm_barrier",
            module_path=f"{prefix}.norm2",
            residual_key=prefix,
        )
    )
    tokens.append(_operation("silu"))
    tokens.append(_operation("module", module_path=f"{prefix}.conv2"))
    tokens.append(_operation("add_residual", residual_key=prefix))


def _append_attention_tokens(
    tokens: list[dict[str, Any]],
    prefix: str,
    attention: Any,
) -> None:
    if getattr(attention, "group_norm", None) is None:
        raise ValueError(f"{prefix} does not expose group_norm")
    tokens.append(
        _operation(
            "store_residual",
            residual_key=prefix,
            shortcut="identity",
        )
    )
    tokens.append(
        _operation(
            "group_norm_barrier",
            module_path=f"{prefix}.group_norm",
            residual_key=prefix,
        )
    )
    tokens.append(_operation("attention", module_path=prefix))
    tokens.append(_operation("add_residual", residual_key=prefix))


def _encoder_execution_tokens(vae: Any) -> list[dict[str, Any]]:
    encoder = vae.encoder
    tokens = [_operation("module", module_path="encoder.conv_in")]

    down_blocks = list(encoder.down_blocks)
    for block_index, block in enumerate(down_blocks):
        for resnet_index, resnet in enumerate(block.resnets):
            _append_resnet_tokens(
                tokens,
                f"encoder.down_blocks.{block_index}.resnets.{resnet_index}",
                resnet,
            )
        if block_index != len(down_blocks) - 1:
            downsamplers = list(getattr(block, "downsamplers", ()) or ())
            if len(downsamplers) != 1:
                raise ValueError(
                    f"encoder.down_blocks.{block_index} must expose one downsampler"
                )
            tokens.append(
                _operation(
                    "module",
                    module_path=f"encoder.down_blocks.{block_index}.downsamplers.0",
                )
            )

    mid_resnets = list(encoder.mid_block.resnets)
    attentions = list(encoder.mid_block.attentions)
    if len(mid_resnets) != 2 or len(attentions) != 1:
        raise ValueError(
            "PiSA encoder mid block must contain two resnets and one attention"
        )
    _append_resnet_tokens(tokens, "encoder.mid_block.resnets.0", mid_resnets[0])
    _append_attention_tokens(
        tokens,
        "encoder.mid_block.attentions.0",
        attentions[0],
    )
    _append_resnet_tokens(tokens, "encoder.mid_block.resnets.1", mid_resnets[1])

    tokens.append(
        _operation(
            "group_norm_barrier",
            module_path="encoder.conv_norm_out",
        )
    )
    tokens.append(_operation("silu"))
    tokens.append(_operation("module", module_path="encoder.conv_out"))
    tokens.append(_operation("module", module_path="quant_conv"))
    return tokens


def _decoder_execution_tokens(vae: Any) -> list[dict[str, Any]]:
    decoder = vae.decoder
    tokens = [
        _operation("module", module_path="post_quant_conv"),
        _operation("module", module_path="decoder.conv_in"),
    ]

    mid_resnets = list(decoder.mid_block.resnets)
    attentions = list(decoder.mid_block.attentions)
    if len(mid_resnets) != 2 or len(attentions) != 1:
        raise ValueError(
            "PiSA decoder mid block must contain two resnets and one attention"
        )
    _append_resnet_tokens(tokens, "decoder.mid_block.resnets.0", mid_resnets[0])
    _append_attention_tokens(
        tokens,
        "decoder.mid_block.attentions.0",
        attentions[0],
    )
    _append_resnet_tokens(tokens, "decoder.mid_block.resnets.1", mid_resnets[1])

    up_blocks = list(decoder.up_blocks)
    for block_index, block in enumerate(up_blocks):
        for resnet_index, resnet in enumerate(block.resnets):
            _append_resnet_tokens(
                tokens,
                f"decoder.up_blocks.{block_index}.resnets.{resnet_index}",
                resnet,
            )
        if block_index != len(up_blocks) - 1:
            upsamplers = list(getattr(block, "upsamplers", ()) or ())
            if len(upsamplers) != 1:
                raise ValueError(
                    f"decoder.up_blocks.{block_index} must expose one upsampler"
                )
            tokens.append(
                _operation(
                    "module",
                    module_path=f"decoder.up_blocks.{block_index}.upsamplers.0",
                )
            )

    tokens.append(
        _operation(
            "group_norm_barrier",
            module_path="decoder.conv_norm_out",
        )
    )
    tokens.append(_operation("silu"))
    tokens.append(_operation("module", module_path="decoder.conv_out"))
    return tokens


def _build_execution_side(
    tokens: list[dict[str, Any]],
    barriers: list[dict[str, Any]],
) -> dict[str, Any]:
    by_path = {barrier["module_path"]: barrier for barrier in barriers}
    prelude: list[dict[str, Any]] = []
    stages: list[dict[str, Any]] = []
    current_after: list[dict[str, Any]] | None = None
    seen_paths: list[str] = []

    for token in tokens:
        if token["kind"] == "group_norm_barrier":
            path = token["modulePath"]
            barrier = by_path.get(path)
            if barrier is None:
                raise ValueError(
                    f"execution token references unknown GroupNorm barrier {path}"
                )
            seen_paths.append(path)
            stage = {
                "index": len(stages),
                "barrier": barrier,
                "after": [],
            }
            stages.append(stage)
            current_after = stage["after"]
        elif current_after is None:
            prelude.append(token)
        else:
            current_after.append(token)

    if seen_paths != [barrier["module_path"] for barrier in barriers]:
        raise ValueError("execution GroupNorm order does not match barrier contract")

    return {
        "prelude": prelude,
        "stages": stages,
    }


def build_vae_tile_execution_contract(vae: Any) -> dict[str, Any]:
    barriers = build_vae_tile_barrier_contract(vae)
    return {
        "schemaVersion": 1,
        "encoder": _build_execution_side(
            _encoder_execution_tokens(vae),
            barriers["encoder"],
        ),
        "decoder": _build_execution_side(
            _decoder_execution_tokens(vae),
            barriers["decoder"],
        ),
    }


def resolve_vae_module_path(vae: Any, path: str) -> Any:
    if not isinstance(path, str) or not path:
        raise ValueError("module path must be a non-empty string")

    current: Any = vae
    for token in path.split("."):
        if token.isdigit():
            try:
                current = current[int(token)]
            except (IndexError, KeyError, TypeError) as error:
                raise ValueError(
                    f"cannot resolve index {token!r} in VAE module path {path!r}"
                ) from error
            continue

        if not hasattr(current, token):
            raise ValueError(
                f"cannot resolve attribute {token!r} in VAE module path {path!r}"
            )
        current = getattr(current, token)
    return current


def validate_vae_tile_execution_contract(
    vae: Any,
    contract: dict[str, Any],
) -> int:
    if contract.get("schemaVersion") != 1:
        raise ValueError("unsupported VAE tile execution contract schema")

    resolved_paths: set[str] = set()
    for side_name in ("encoder", "decoder"):
        side = contract.get(side_name)
        if not isinstance(side, dict):
            raise ValueError(f"missing {side_name} execution contract")

        operations = list(side.get("prelude", ()))
        stages = side.get("stages")
        if not isinstance(stages, list) or not stages:
            raise ValueError(f"{side_name} execution contract has no stages")

        for expected_index, stage in enumerate(stages):
            if stage.get("index") != expected_index:
                raise ValueError(
                    f"{side_name} stage indices are not contiguous"
                )
            barrier = stage.get("barrier")
            if not isinstance(barrier, dict):
                raise ValueError(f"{side_name} stage {expected_index} has no barrier")

            barrier_path = barrier.get("module_path")
            if not isinstance(barrier_path, str):
                raise ValueError(
                    f"{side_name} stage {expected_index} has invalid barrier path"
                )
            barrier_module = resolve_vae_module_path(vae, barrier_path)
            expected_groups = barrier.get("groups")
            actual_groups = getattr(barrier_module, "num_groups", None)
            if actual_groups != expected_groups:
                raise ValueError(
                    f"{barrier_path} group count changed: "
                    f"contract={expected_groups} model={actual_groups}"
                )
            resolved_paths.add(barrier_path)
            operations.extend(stage.get("after", ()))

        for operation in operations:
            if not isinstance(operation, dict):
                raise ValueError(
                    f"{side_name} execution contract contains a non-object operation"
                )
            module_path = operation.get("modulePath")
            if module_path is None:
                continue
            if not isinstance(module_path, str):
                raise ValueError(
                    f"{side_name} operation contains an invalid module path"
                )
            resolve_vae_module_path(vae, module_path)
            resolved_paths.add(module_path)

    return len(resolved_paths)


def _apply_residual_operations(
    operations: list[dict[str, Any]],
    live_residual: str | None,
) -> str | None:
    current = live_residual
    for operation in operations:
        kind = operation.get("kind")
        key = operation.get("residualKey")
        if kind == "store_residual":
            if current is not None:
                raise ValueError(
                    f"residual {current!r} is still live before storing {key!r}"
                )
            if not isinstance(key, str) or not key:
                raise ValueError("store_residual requires a residual key")
            current = key
        elif kind == "add_residual":
            if current != key:
                raise ValueError(
                    f"cannot consume residual {key!r}; live residual is {current!r}"
                )
            current = None
    return current


def _build_segment_side(
    side_name: str,
    side: dict[str, Any],
) -> dict[str, Any]:
    prelude = side.get("prelude")
    stages = side.get("stages")
    if not isinstance(prelude, list) or not isinstance(stages, list) or not stages:
        raise ValueError(f"{side_name} execution recipe is incomplete")

    live_residual = _apply_residual_operations(prelude, None)
    first_barrier = stages[0]["barrier"]
    first_key = first_barrier.get("residual_key")
    if first_key != live_residual:
        raise ValueError(
            f"{side_name} prelude residual does not match first barrier"
        )

    segments: list[dict[str, Any]] = [
        {
            "index": 0,
            "id": f"{side_name}_prelude",
            "entryBarrier": None,
            "exitBarrier": first_barrier["module_path"],
            "requiresResidualInput": False,
            "producesResidualOutput": live_residual is not None,
            "residualKey": live_residual,
            "operations": prelude,
        }
    ]

    for stage_index, stage in enumerate(stages):
        barrier = stage["barrier"]
        barrier_key = barrier.get("residual_key")
        if barrier_key != live_residual:
            raise ValueError(
                f"{side_name} barrier {barrier['module_path']} residual "
                f"{barrier_key!r} does not match live residual {live_residual!r}"
            )

        requires_residual = live_residual is not None
        after = stage.get("after")
        if not isinstance(after, list):
            raise ValueError(
                f"{side_name} stage {stage_index} operations are missing"
            )
        live_residual = _apply_residual_operations(after, live_residual)

        next_barrier = (
            stages[stage_index + 1]["barrier"]["module_path"]
            if stage_index + 1 < len(stages)
            else None
        )
        if next_barrier is not None:
            expected_key = stages[stage_index + 1]["barrier"].get(
                "residual_key"
            )
            if expected_key != live_residual:
                raise ValueError(
                    f"{side_name} stage {stage_index} output residual does "
                    f"not match next barrier"
                )
        elif live_residual is not None:
            raise ValueError(
                f"{side_name} final segment leaves residual {live_residual!r}"
            )

        segments.append(
            {
                "index": stage_index + 1,
                "id": f"{side_name}_stage_{stage_index:02d}",
                "entryBarrier": barrier["module_path"],
                "exitBarrier": next_barrier,
                "requiresResidualInput": requires_residual,
                "producesResidualOutput": live_residual is not None,
                "residualKey": live_residual,
                "operations": after,
            }
        )

    return {
        "segmentCount": len(segments),
        "segments": segments,
    }


def build_vae_tile_segment_contract(
    execution_contract: dict[str, Any],
) -> dict[str, Any]:
    if execution_contract.get("schemaVersion") != 1:
        raise ValueError("unsupported VAE tile execution contract schema")

    return {
        "schemaVersion": 1,
        "encoder": _build_segment_side(
            "encoder",
            execution_contract["encoder"],
        ),
        "decoder": _build_segment_side(
            "decoder",
            execution_contract["decoder"],
        ),
    }
