from __future__ import annotations

from dataclasses import asdict, dataclass
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


def _encoder_execution_tokens(encoder: Any) -> list[dict[str, Any]]:
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
    return tokens


def _decoder_execution_tokens(decoder: Any) -> list[dict[str, Any]]:
    tokens = [_operation("module", module_path="decoder.conv_in")]

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
            _encoder_execution_tokens(vae.encoder),
            barriers["encoder"],
        ),
        "decoder": _build_execution_side(
            _decoder_execution_tokens(vae.decoder),
            barriers["decoder"],
        ),
    }
