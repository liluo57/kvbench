"""Primary KV-cache slicing and RoPE realignment for CacheClip."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from transformers.cache_utils import DynamicCache

from helpers.backends.TransformersHelper import CacheLayerPairs


def _cache_layer_pairs(cache: Any) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Return layer K/V pairs from tuple and supported cache layouts."""
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        if len(cache.key_cache) != len(cache.value_cache):
            raise ValueError("primary cache has mismatched key/value layer counts")
        return list(zip(cache.key_cache, cache.value_cache))
    return list(CacheLayerPairs(cache))


def _new_dynamic_cache(config: Any) -> DynamicCache:
    """Construct DynamicCache across pre-5 and current Transformers APIs."""
    try:
        return DynamicCache(config=config)
    except (AttributeError, TypeError):
        return DynamicCache()


def _as_dynamic_cache(cache: Any, config: Any = None) -> DynamicCache:
    """Convert a legacy tuple to a cache exposing ``get_seq_length``."""
    if hasattr(cache, "get_seq_length"):
        return cache
    converter = getattr(DynamicCache, "from_legacy_cache", None)
    if callable(converter):
        try:
            return converter(cache)
        except (AttributeError, TypeError):
            pass
    dynamic_cache = _new_dynamic_cache(config)
    for layer_index, (key, value) in enumerate(_cache_layer_pairs(cache)):
        dynamic_cache.update(key, value, layer_index)
    return dynamic_cache


def _layer_rotary_embedding(model: Any, layer_index: int) -> Any:
    """Get RoPE from the model or the layer-local Qwen2 attention module."""
    base_model = model.model
    rotary_emb = getattr(base_model, "rotary_emb", None)
    if rotary_emb is not None:
        return rotary_emb
    try:
        rotary_emb = base_model.layers[layer_index].self_attn.rotary_emb
    except (AttributeError, IndexError) as exc:
        raise RuntimeError(
            f"CacheClip cannot find rotary embedding for layer {layer_index}"
        ) from exc
    return rotary_emb


def _rotary_cos_sin(rotary_emb: Any, x: torch.Tensor, positions: torch.Tensor):
    """Call both position-id and sequence-length RoPE APIs."""
    try:
        return rotary_emb(x, position_ids=positions)
    except TypeError:
        seq_len = getattr(rotary_emb, "max_position_embeddings", None)
        if seq_len is None:
            seq_len = int(positions.max().item()) + 1
        cos, sin = rotary_emb(x, seq_len=seq_len)
        return cos[positions], sin[positions]


def _rotate_key(
    key: torch.Tensor,
    rotary_emb: Any,
    old_positions: torch.Tensor,
    new_positions: torch.Tensor,
    nope_dim: int | None,
) -> torch.Tensor:
    delta = new_positions - old_positions
    if hasattr(rotary_emb, "attention_scaling") and rotary_emb.attention_scaling != 1.0:
        raise RuntimeError("CacheClip RoPE re-alignment does not support attention scaling")
    if nope_dim is None:
        prefix = None
        rope = key
    else:
        prefix = key[..., :nope_dim]
        rope = key[..., nope_dim:]
    cos, sin = _rotary_cos_sin(rotary_emb, rope, delta)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    first_half = rope[..., : rope.shape[-1] // 2]
    second_half = rope[..., rope.shape[-1] // 2 :]
    rotated_half = torch.cat((-second_half, first_half), dim=-1)
    rotated = (rope * cos) + (rotated_half * sin)
    return rotated if prefix is None else torch.cat([prefix, rotated], dim=-1)


@dataclass
class PrimaryAssembly:
    cache: DynamicCache
    context_ids: torch.Tensor


def _sequence_ids(ids: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.size(0) != 1:
        raise ValueError(f"{name} must have shape [1, sequence]")
    return ids


def _checked_layers(cache: Any, row_count: int, name: str):
    layers = _cache_layer_pairs(cache)
    if not layers:
        raise ValueError(f"{name} has no cache layers")
    for layer_index, (key, value) in enumerate(layers):
        if key.ndim != 4 or value.ndim != 4:
            raise ValueError(f"{name} layer {layer_index} must have rank-4 K/V tensors")
        if key.size(-2) != row_count or value.size(-2) != row_count:
            raise ValueError(
                f"{name} layer {layer_index} rows do not match its token IDs "
                f"({key.size(-2)}/{value.size(-2)} != {row_count})"
            )
    return layers


def _cache_length(cache: Any) -> int:
    layers = _cache_layer_pairs(cache)
    if not layers:
        return 0
    lengths = {key.size(-2) for key, _ in layers}
    value_lengths = {value.size(-2) for _, value in layers}
    if len(lengths) != 1 or lengths != value_lengths:
        raise ValueError("primary cache has inconsistent layer row counts")
    return next(iter(lengths))


def _validate_assembly(assembly: PrimaryAssembly) -> list[tuple[torch.Tensor, torch.Tensor]]:
    ids = _sequence_ids(assembly.context_ids, "assembly context_ids")
    layers = _checked_layers(assembly.cache, ids.size(1), "assembled primary cache")
    return layers


def assemble_primary_cache(
    model: Any,
    prefix_cache: DynamicCache | None,
    prefix_ids: torch.Tensor,
    document_caches: Sequence[DynamicCache],
    document_ids: Sequence[torch.Tensor],
    *,
    old_start: int = 0,
) -> PrimaryAssembly:
    """Join runtime-prefix IDs and cached document bodies without ID edits.

    Each document cache contains only its task-provided body IDs. Its K/V rows
    were prefetched starting at ``old_start`` (for example after a synthetic
    warmup prefix); only the key positions are realigned for their runtime
    positions. No synthetic IDs are inserted and no document IDs are removed.
    """
    if isinstance(old_start, bool) or not isinstance(old_start, int) or old_start < 0:
        raise ValueError("old_start must be a non-negative integer")
    prefix_ids = _sequence_ids(prefix_ids, "prefix_ids")
    if len(document_caches) != len(document_ids):
        raise ValueError("document caches and IDs must have equal lengths")

    segments: list[tuple[list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor, int]] = []
    if prefix_cache is None:
        if prefix_ids.size(1):
            raise ValueError("prefix IDs were provided without a runtime prefix cache")
    else:
        prefix_layers = _checked_layers(prefix_cache, prefix_ids.size(1), "runtime prefix cache")
        segments.append((prefix_layers, prefix_ids, 0))

    for index, (cache, raw_ids) in enumerate(zip(document_caches, document_ids)):
        ids = _sequence_ids(raw_ids, f"document_ids[{index}]")
        layers = _checked_layers(cache, ids.size(1), f"document cache {index}")
        segments.append((layers, ids, old_start))
    if not segments:
        raise ValueError("CacheClip cannot assemble an empty primary cache")

    layer_count = len(segments[0][0])
    if any(len(layers) != layer_count for layers, _, _ in segments):
        raise ValueError("primary cache segments have mismatched layer counts")

    total_length = sum(ids.size(1) for _, ids, _ in segments)
    assembled = _new_dynamic_cache(model.config)
    nope_dim = getattr(model.config, "qk_nope_head_dim", None)
    for layer_index in range(layer_count):
        rotary_emb = _layer_rotary_embedding(model, layer_index)
        keys = []
        values = []
        cursor = 0
        for layers, ids, segment_old_start in segments:
            key, value = layers[layer_index]
            length = ids.size(1)
            old_positions = torch.arange(
                segment_old_start, segment_old_start + length, device=key.device
            ).unsqueeze(0)
            new_positions = torch.arange(
                cursor, cursor + length, device=key.device
            ).unsqueeze(0)
            keys.append(_rotate_key(key, rotary_emb, old_positions, new_positions, nope_dim))
            values.append(value)
            cursor += length
        assembled.update(torch.cat(keys, dim=2), torch.cat(values, dim=2), layer_index)

    output_layers = _cache_layer_pairs(assembled)
    if not output_layers:
        raise RuntimeError("CacheClip assembled cache has no layers")
    cache_device = output_layers[0][0].device
    context_ids = torch.cat([ids.to(cache_device) for _, ids, _ in segments], dim=1)
    if context_ids.size(1) != total_length:
        raise RuntimeError("CacheClip primary cache/token assembly mismatch")
    result = PrimaryAssembly(cache=assembled, context_ids=context_ids)
    _validate_assembly(result)
    return result


def append_cached_primary_segment(
    model: Any,
    assembly: PrimaryAssembly,
    document_cache: DynamicCache,
    document_ids: torch.Tensor,
    *,
    old_start: int = 0,
) -> PrimaryAssembly:
    """Append one document body while preserving its provided IDs exactly."""
    if isinstance(old_start, bool) or not isinstance(old_start, int) or old_start < 0:
        raise ValueError("old_start must be a non-negative integer")
    current_layers = _validate_assembly(assembly)
    ids = _sequence_ids(document_ids, "document_ids")
    document_layers = _checked_layers(document_cache, ids.size(1), "document cache")
    if len(document_layers) != len(current_layers):
        raise ValueError("primary cache layers are missing or mismatched")
    if ids.size(1) == 0:
        return assembly

    current_length = assembly.context_ids.size(1)
    nope_dim = getattr(model.config, "qk_nope_head_dim", None)
    result_cache = _new_dynamic_cache(model.config)
    for layer_index, ((old_key, old_value), (key, value)) in enumerate(
        zip(current_layers, document_layers)
    ):
        rotary_emb = _layer_rotary_embedding(model, layer_index)
        segment_length = ids.size(1)
        old_positions = torch.arange(
            old_start, old_start + segment_length, device=key.device
        ).unsqueeze(0)
        new_positions = torch.arange(
            current_length, current_length + segment_length, device=key.device
        ).unsqueeze(0)
        realigned_key = _rotate_key(
            key, rotary_emb, old_positions, new_positions, nope_dim
        )
        result_cache.update(
            torch.cat([old_key, realigned_key], dim=2),
            torch.cat([old_value, value], dim=2),
            layer_index,
        )

    device = _cache_layer_pairs(result_cache)[0][0].device
    context_ids = torch.cat([assembly.context_ids.to(device), ids.to(device)], dim=1)
    result = PrimaryAssembly(cache=result_cache, context_ids=context_ids)
    _validate_assembly(result)
    return result


def select_cache_sequence(
    cache: DynamicCache,
    start: int,
    end: int | None = None,
) -> DynamicCache:
    """Copy a sequence slice from a DynamicCache without mutating its owner."""
    selected = _new_dynamic_cache(getattr(cache, "config", None))
    for layer_index, (keys, values) in enumerate(_cache_layer_pairs(cache)):
        selected.update(
            keys[:, :, start:end, :],
            values[:, :, start:end, :],
            layer_index,
        )
    return selected


__all__ = [
    "PrimaryAssembly",
    "_as_dynamic_cache",
    "_cache_layer_pairs",
    "_layer_rotary_embedding",
    "_new_dynamic_cache",
    "_rotate_key",
    "_rotary_cos_sin",
    "append_cached_primary_segment",
    "assemble_primary_cache",
    "select_cache_sequence",
]
