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
