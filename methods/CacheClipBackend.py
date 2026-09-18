"""CacheClip-only CPU selection and primary-cache helpers."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
import inspect
import math
import os
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from helpers.backends.TransformersHelper import CacheLayerPairs


Offset = tuple[int, int]


def _offsets(value: Any) -> list[Offset]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list) and value[0] and isinstance(value[0][0], (list, tuple)):
        value = value[0]
    return [tuple(pair) for pair in value]


def shared_prefix_token_length(
    token_ids_by_document: Sequence[Sequence[int]],
    offsets_by_document: Sequence[Sequence[Offset]],
    prefix_char_length: int,
) -> int:
    """Return a common token prefix that does not cross the text boundary."""
    if isinstance(prefix_char_length, bool) or not isinstance(prefix_char_length, int):
        raise ValueError("prefix character length must be an integer")
    if prefix_char_length < 0:
        raise ValueError("prefix character length must be non-negative")
    if not token_ids_by_document or len(token_ids_by_document) != len(offsets_by_document):
        raise ValueError("token IDs and offsets must be non-empty and document-aligned")

    safe_limits: list[int] = []
    normalized_ids: list[list[int]] = []
    for token_ids, offsets in zip(token_ids_by_document, offsets_by_document):
        if len(token_ids) != len(offsets):
            raise ValueError("token IDs and offsets must have equal lengths")
        safe_limit = 0
        for index, (_, end) in enumerate(offsets):
            if end > prefix_char_length:
                break
            safe_limit = index + 1
        safe_limits.append(safe_limit)
        normalized_ids.append(list(token_ids))
    common_limit = min(safe_limits)
    shared_length = 0
    for index in range(common_limit):
        if any(ids[index] != normalized_ids[0][index] for ids in normalized_ids[1:]):
            break
        shared_length = index + 1
    return shared_length


def continuation_offsets_from_full_sequence(
    full_offsets: Sequence[Offset],
    shared_prefix_length: int,
    prefix_char_length: int,
) -> list[Offset]:
    """Return continuation offsets relative to the raw document text."""
    if shared_prefix_length < 0 or shared_prefix_length >= len(full_offsets):
        raise ValueError("shared prefix must leave continuation tokens")
    if prefix_char_length < 0:
        raise ValueError("prefix character length must be non-negative")
    result = []
    for start, end in full_offsets[shared_prefix_length:]:
        if end <= prefix_char_length:
            result.append((0, 0))
        else:
            result.append((max(start, prefix_char_length) - prefix_char_length, end - prefix_char_length))
    return result


def map_auxiliary_to_primary_indices(
    auxiliary_offsets: Sequence[Offset],
    primary_offsets: Sequence[Offset],
    selected_auxiliary_indices: Sequence[int],
) -> list[int]:
    """Map auxiliary selections through overlapping character spans."""
    spans = []
    for index in selected_auxiliary_indices:
        if index < 0 or index >= len(auxiliary_offsets):
            raise ValueError("selected auxiliary index lies outside offsets")
        start, end = auxiliary_offsets[index]
        if start != end:
            spans.append((start, end))
    return [
        primary_index
        for primary_index, (start, end) in enumerate(primary_offsets)
        if start != end and any(end > aux_start and start < aux_end for aux_start, aux_end in spans)
    ]


def select_top_k_candidates(
    scores_by_document: Sequence[Sequence[float]], candidate_ratio: float
) -> list[list[int]]:
    """Select globally ranked auxiliary token positions."""
    if isinstance(candidate_ratio, bool) or not isinstance(candidate_ratio, (int, float)):
        raise ValueError("candidate_ratio must be a number in [0, 1]")
    if not 0 <= float(candidate_ratio) <= 1:
        raise ValueError("candidate_ratio must be in [0, 1]")
    flattened = [
        (float(score), document_index, token_index)
        for document_index, document_scores in enumerate(scores_by_document)
        for token_index, score in enumerate(document_scores)
        if math.isfinite(float(score))
    ]
    if len(flattened) != sum(len(scores) for scores in scores_by_document):
        raise ValueError("auxiliary attention scores must be finite")
    selected = [[] for _ in scores_by_document]
    count = math.floor(float(candidate_ratio) * len(flattened))
    for _, document_index, token_index in sorted(
        flattened, key=lambda item: (-item[0], item[1], item[2])
    )[:count]:
        selected[document_index].append(token_index)
    return selected


def group_candidate_windows(
    candidates_by_document: Sequence[Sequence[int]],
    document_lengths: Sequence[int],
    window_size: int = 8,
    density_threshold: int = 5,
) -> list[list[int]]:
    """Expand dense candidates into document-local, non-crossing windows."""
    if len(candidates_by_document) != len(document_lengths):
        raise ValueError("candidate and document lists must have equal length")
    if window_size <= 0 or density_threshold <= 0:
        raise ValueError("window size and density threshold must be positive")
    grouped = []
    for candidates, document_length in zip(candidates_by_document, document_lengths):
        if document_length < 0:
            raise ValueError("document lengths must be non-negative")
        candidate_set = set(candidates)
        if any(index < 0 or index >= document_length for index in candidate_set):
            raise ValueError("candidate index lies outside its document")
        candidate_indices = sorted(candidate_set)
        intervals = []
        right = 0
        for left, start in enumerate(candidate_indices):
            end = min(document_length, start + window_size)
            while right < len(candidate_indices) and candidate_indices[right] < end:
                right += 1
            if right - left >= density_threshold:
                if intervals and start <= intervals[-1][1]:
                    intervals[-1] = (intervals[-1][0], max(intervals[-1][1], end))
                else:
                    intervals.append((start, end))
        grouped.append([
            index
            for start, end in intervals
            for index in range(start, end)
        ])
    return grouped


@dataclass
class AuxiliaryDocumentCache:
    cache: Any
    continuation_offsets: list[Offset]
    prefix_length: int


def _cache_layer_pairs(cache: Any) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Return layer K/V pairs from tuple and all supported cache layouts."""
    if hasattr(cache, "layers"):
        return list(CacheLayerPairs(cache))
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
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


def _bucket_indices(caches: Sequence[AuxiliaryDocumentCache]) -> list[list[int]]:
    buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for index, document_cache in enumerate(caches):
        layer_pairs = _cache_layer_pairs(document_cache.cache)
        lengths = {key.size(-2) for key, _ in layer_pairs}
        if len(lengths) != 1:
            raise ValueError("auxiliary cache has inconsistent layer lengths")
        buckets[(next(iter(lengths)), document_cache.prefix_length, len(document_cache.continuation_offsets))].append(index)
    return list(buckets.values())


def _cached_query_attention_kwargs(input_ids: torch.Tensor, cache: Any) -> dict[str, torch.Tensor]:
    """Build the explicit mask/positions needed when querying a prefilled KV cache.

    Transformers 4.40 does not infer the cached prefix length when
    use_cache=False. The attention module still appends the supplied past KV,
    though, so the default causal mask is too short. Supplying the full
    attention mask and absolute cache positions keeps the mask aligned with
    the cached keys while leaving the cache unchanged.
    """
    past_length = int(cache.get_seq_length())
    batch_size, query_length = input_ids.shape
    total_length = past_length + query_length
    return {
        "attention_mask": torch.ones(
            (batch_size, total_length), dtype=torch.long, device=input_ids.device
        ),
        "cache_position": torch.arange(
            past_length, total_length, dtype=torch.long, device=input_ids.device
        ),
    }


def _copy_dynamic_cache(cache: Any, config: Any) -> DynamicCache:
    """Copy cache metadata while retaining read-only references to document K/V."""
    copied = _new_dynamic_cache(config)
    for layer_index, (key, value) in enumerate(_cache_layer_pairs(cache)):
        copied.update(key, value, layer_index)
    return copied


class AuxiliaryRuntime:
    """Small CPU causal LM used by CacheClip's online selector."""

    def __init__(self, model: Any, tokenizer: Any, num_threads: int):
        if num_threads <= 0:
            raise ValueError("auxiliary num_threads must be positive")
        self.model = model.eval().to("cpu")
        self.tokenizer = tokenizer
        self.num_threads = num_threads
        torch.set_num_threads(num_threads)
        setter = getattr(self.model, "set_attn_implementation", None)
        if callable(setter):
            setter("eager")
        self.model.config._attn_implementation = "eager"

    @classmethod
    def load(cls, model_path: str, num_threads: int) -> "AuxiliaryRuntime":
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float32, low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "right"
        return cls(model, tokenizer, num_threads)

    def _encode(self, text: str, *, offsets: bool = False):
        return self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
            return_offsets_mapping=offsets,
        )

    def prefill_documents(self, prefix: str, documents: Sequence[str]) -> list[AuxiliaryDocumentCache]:
        encoded = [self._encode(prefix + document, offsets=True) for document in documents]
        ids = [item["input_ids"][0].tolist() for item in encoded]
        offsets = [_offsets(item["offset_mapping"]) for item in encoded]
        prefix_length = shared_prefix_token_length(ids, offsets, len(prefix)) if prefix else 0
        result = []
        with torch.no_grad():
            for item, item_offsets in zip(encoded, offsets):
                input_ids = item["input_ids"]
                if input_ids.size(1) <= prefix_length:
                    raise ValueError("auxiliary document has no continuation tokens")
                cache = _as_dynamic_cache(
                    self.model(input_ids=input_ids, use_cache=True).past_key_values,
                    self.model.config,
                )
                result.append(AuxiliaryDocumentCache(
                    cache=cache,
                    continuation_offsets=continuation_offsets_from_full_sequence(
                        item_offsets, prefix_length, len(prefix)
                    ),
                    prefix_length=prefix_length,
                ))
        return result

    @staticmethod
    def _document_scores(attention: torch.Tensor, cache: AuxiliaryDocumentCache) -> list[list[float]]:
        start = cache.prefix_length
        end = start + len(cache.continuation_offsets)
        return attention[:, :, :, start:end].float().mean(dim=(1, 2)).cpu().tolist()

    def score_documents(self, query: str, caches: Sequence[AuxiliaryDocumentCache]) -> list[list[float]]:
        if not query:
            raise ValueError("CacheClip query must be non-empty")
        query_ids = self._encode(query)["input_ids"]
        scores: list[list[float] | None] = [None] * len(caches)
        with torch.no_grad():
            for bucket in _bucket_indices(caches):
                bucket_caches = [
                    AuxiliaryDocumentCache(
                        cache=_as_dynamic_cache(caches[index].cache, self.model.config),
                        continuation_offsets=caches[index].continuation_offsets,
                        prefix_length=caches[index].prefix_length,
                    )
                    for index in bucket
                ]
                if len(bucket) == 1:
                    scoring_cache = _copy_dynamic_cache(
                        bucket_caches[0].cache, self.model.config
                    )
                    query_kwargs = _cached_query_attention_kwargs(
                        query_ids, scoring_cache
                    )
                    output = self.model(
                        input_ids=query_ids,
                        past_key_values=scoring_cache,
                        use_cache=False,
                        output_attentions=True,
                        return_dict=True,
                        **query_kwargs,
                    )
                    attention = output.attentions[-1]
                    bucket_scores = self._document_scores(attention, bucket_caches[0])
                else:
                    layer_pairs = [
                        _cache_layer_pairs(document_cache.cache)
                        for document_cache in bucket_caches
                    ]
                    batched_cache = _new_dynamic_cache(self.model.config)
                    for layer_index in range(len(layer_pairs[0])):
                        batched_cache.update(
                            torch.cat([pairs[layer_index][0] for pairs in layer_pairs]),
                            torch.cat([pairs[layer_index][1] for pairs in layer_pairs]),
                            layer_index,
                        )
                    batched_query = query_ids.expand(len(bucket), -1)
                    query_kwargs = _cached_query_attention_kwargs(
                        batched_query, batched_cache
                    )
                    output = self.model(
                        input_ids=batched_query,
                        past_key_values=batched_cache,
                        use_cache=False,
                        output_attentions=True,
                        return_dict=True,
                        **query_kwargs,
                    )
                    bucket_scores = self._document_scores(output.attentions[-1], bucket_caches[0])
                for index, value in zip(bucket, bucket_scores):
                    scores[index] = value
        if any(value is None for value in scores):
            raise RuntimeError("auxiliary selector did not score every document")
        return [value for value in scores if value is not None]


def tokenize_with_offsets(tokenizer: Any, text: str) -> tuple[torch.Tensor, list[Offset]]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    return encoded["input_ids"], _offsets(encoded["offset_mapping"])


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


def _apply_qwen2_rotary(
    model: Any,
    layer_index: int,
    attention: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    hidden_states: torch.Tensor,
    token_indices: torch.Tensor,
    total_length: int,
    apply_rotary_pos_emb: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Qwen2 RoPE across Transformers' layer-local and model-level APIs."""
    rotary_emb = _layer_rotary_embedding(model, layer_index)
    positions = token_indices.unsqueeze(0)
    rotary_parameters = inspect.signature(rotary_emb.forward).parameters
    apply_parameters = inspect.signature(apply_rotary_pos_emb).parameters
    if "position_ids" in rotary_parameters:
        cos, sin = rotary_emb(hidden_states, position_ids=positions)
        if "position_ids" in apply_parameters:
            return apply_rotary_pos_emb(query, key, cos, sin, positions)
        return apply_rotary_pos_emb(query, key, cos, sin)

    cos, sin = rotary_emb(value, seq_len=total_length)
    if "position_ids" in apply_parameters:
        return apply_rotary_pos_emb(query, key, cos, sin, positions)
    cos = cos.index_select(-2, token_indices)
    sin = sin.index_select(-2, token_indices)
    return apply_rotary_pos_emb(query, key, cos, sin)


def _rotate_key(key: torch.Tensor, rotary_emb: Any, old_positions: torch.Tensor, new_positions: torch.Tensor, nope_dim: int | None) -> torch.Tensor:
    delta = new_positions - old_positions
    if hasattr(rotary_emb, "attention_scaling") and rotary_emb.attention_scaling != 1.0:
        raise RuntimeError("CacheClip RoPE re-alignment does not support attention scaling")
    if nope_dim is None:
        rope = key
        prefix = None
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


def _set_eager(model: Any) -> None:
    setter = getattr(model, "set_attn_implementation", None)
    if callable(setter):
        setter("eager")
    model.config._attn_implementation = "eager"
    for layer in getattr(model.model, "layers", []):
        config = getattr(getattr(layer, "self_attn", None), "config", None)
        if config is not None:
            config._attn_implementation = "eager"


@dataclass
class PrimaryAssembly:
    cache: DynamicCache
    context_ids: torch.Tensor


def assemble_primary_cache(
    model: Any,
    prefix_cache: DynamicCache | None,
    prefix_ids: torch.Tensor,
    shared_prefix_cache: DynamicCache | None,
    shared_prefix_ids: torch.Tensor,
    shared_prefix_length_value: int,
    document_caches: Sequence[DynamicCache],
    document_ids: Sequence[torch.Tensor],
) -> PrimaryAssembly:
    """Concatenate independently prefetched caches and re-align their RoPE."""
    segments: list[tuple[list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor, torch.Tensor]] = []
    if prefix_cache is not None and prefix_ids.numel():
        segments.append((list(CacheLayerPairs(prefix_cache)), prefix_ids, torch.arange(prefix_ids.size(1), device=prefix_ids.device)))
    if shared_prefix_cache is not None and shared_prefix_ids.numel():
        segments.append((list(CacheLayerPairs(shared_prefix_cache)), shared_prefix_ids, torch.arange(shared_prefix_ids.size(1), device=shared_prefix_ids.device)))
    for document_cache, ids in zip(document_caches, document_ids):
        if shared_prefix_length_value:
            layers = [
                (key[:, :, shared_prefix_length_value:, :], value[:, :, shared_prefix_length_value:, :])
                for key, value in CacheLayerPairs(document_cache)
            ]
            ids = ids[:, shared_prefix_length_value:]
            old = torch.arange(shared_prefix_length_value, shared_prefix_length_value + ids.size(1), device=ids.device)
        else:
            layers = list(CacheLayerPairs(document_cache))
            old = torch.arange(ids.size(1), device=ids.device)
        segments.append((layers, ids, old))
    if not segments:
        raise ValueError("CacheClip cannot assemble an empty primary cache")
    total_length = sum(ids.size(1) for _, ids, _ in segments)
    cursor = 0
    assembled = _new_dynamic_cache(model.config)
    nope_dim = getattr(model.config, "qk_nope_head_dim", None)
    layer_count = len(segments[0][0])
    for layer_index in range(layer_count):
        rotary_emb = _layer_rotary_embedding(model, layer_index)
        keys = []
        values = []
        cursor = 0
        for layers, ids, old_positions in segments:
            key, value = layers[layer_index]
            new_positions = torch.arange(cursor, cursor + ids.size(1), device=key.device).unsqueeze(0)
            old_positions = old_positions.to(key.device).unsqueeze(0)
            keys.append(_rotate_key(key, rotary_emb, old_positions, new_positions, nope_dim))
            values.append(value)
            cursor += ids.size(1)
        assembled.update(torch.cat(keys, dim=2), torch.cat(values, dim=2), layer_index)
    # Token ids produced during Prepare live on CPU, while online prefix ids
    # and the assembled cache live on the primary model device.  Normalize the
    # ids before concatenating; the final ``context_ids.to(...)`` below is too
    # late because ``torch.cat`` itself requires one device.
    cache_device = _cache_layer_pairs(assembled)[0][0].device
    context_ids = torch.cat(
        [ids.to(cache_device) for _, ids, _ in segments], dim=1
    )
    if context_ids.size(1) != total_length:
        raise RuntimeError("CacheClip primary cache/token assembly mismatch")
    assembled_layers = _cache_layer_pairs(assembled)
    if not assembled_layers:
        raise RuntimeError("CacheClip assembled cache has no layers")
    context_ids = context_ids.to(assembled_layers[0][0].device)
    return PrimaryAssembly(cache=assembled, context_ids=context_ids)


def append_cached_primary_segment(
    model: Any,
    assembly: PrimaryAssembly,
    document_cache: DynamicCache,
    document_ids: torch.Tensor,
    *,
    old_start: int = 0,
) -> PrimaryAssembly:
    """Append one independently prefetched segment at its new global position.

    ``document_cache`` was prefetched from position ``old_start`` (normally
    zero), while ``assembly`` already contains the preceding cached/fresh
    segments.  Keys therefore need RoPE re-alignment before concatenation;
    values and token ids are copied unchanged.  This small primitive lets
    CacheClip build a prompt with fresh spans between reused documents without
    changing the core KVBench workflow.
    """
    if document_ids.ndim != 2 or document_ids.size(0) != 1:
        raise ValueError("document_ids must have shape [1, sequence]")
    document_pairs = _cache_layer_pairs(document_cache)
    current_pairs = _cache_layer_pairs(assembly.cache)
    if not document_pairs or len(document_pairs) != len(current_pairs):
        raise ValueError("primary cache layers are missing or mismatched")
    segment_length = document_ids.size(1)
    if segment_length <= 0:
        return assembly
    current_length = assembly.context_ids.size(1)
    nope_dim = getattr(model.config, "qk_nope_head_dim", None)
    result = _new_dynamic_cache(model.config)
    for layer_index, ((old_key, old_value), (key, value)) in enumerate(
        zip(current_pairs, document_pairs)
    ):
        rotary_emb = _layer_rotary_embedding(model, layer_index)
        if key.size(-2) != segment_length:
            raise ValueError("document cache and token ids have different lengths")
        old_positions = torch.arange(
            old_start, old_start + segment_length, device=key.device
        ).unsqueeze(0)
        new_positions = torch.arange(
            current_length, current_length + segment_length, device=key.device
        ).unsqueeze(0)
        realigned_key = _rotate_key(
            key,
            rotary_emb,
            old_positions,
            new_positions,
            nope_dim,
        )
        result.update(
            torch.cat([old_key, realigned_key], dim=2),
            torch.cat([old_value, value], dim=2),
            layer_index,
        )
    device = _cache_layer_pairs(result)[0][0].device
    ids = document_ids.to(device)
    context_ids = torch.cat([assembly.context_ids.to(device), ids], dim=1)
    return PrimaryAssembly(cache=result, context_ids=context_ids)


def select_cache_sequence(cache: DynamicCache, start: int, end: int | None = None) -> DynamicCache:
    """Copy a sequence slice from a DynamicCache without mutating its owner."""
    selected = _new_dynamic_cache(getattr(cache, "config", None))
    for layer_index, (keys, values) in enumerate(_cache_layer_pairs(cache)):
        selected.update(
            keys[:, :, start:end, :],
            values[:, :, start:end, :],
            layer_index,
        )
    return selected


def _recompute_mask(token_indices: list[int], key_length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    rows = torch.tensor(token_indices, device=device).unsqueeze(1)
    columns = torch.arange(key_length, device=device).unsqueeze(0)
    allowed = columns <= rows
    mask = torch.zeros((1, 1, len(token_indices), key_length), device=device, dtype=dtype)
    return mask.masked_fill(~allowed.unsqueeze(0).unsqueeze(0), torch.finfo(dtype).min)


def _local_eager_attention_forward(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fallback eager attention for Transformers versions without the helper."""
    del kwargs
    query_heads = query.size(1)
    key_heads = key.size(1)
    if query_heads % key_heads:
        raise ValueError("CacheClip attention heads are not divisible for GQA")
    repeat_count = query_heads // key_heads
    if repeat_count > 1:
        key = key[:, :, None, :, :].expand(
            key.size(0), key_heads, repeat_count, key.size(2), key.size(3)
        ).reshape(query.size(0), query_heads, key.size(2), key.size(3))
        value = value[:, :, None, :, :].expand(
            value.size(0), key_heads, repeat_count, value.size(2), value.size(3)
        ).reshape(query.size(0), query_heads, value.size(2), value.size(3))
    weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        weights = weights + attention_mask
    weights = torch.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
    weights = torch.nn.functional.dropout(weights, p=dropout, training=module.training)
    return torch.matmul(weights, value), weights


def _resolve_attention_backend() -> str:
    """Resolve CacheClip's explicitly selected attention implementation."""
    backend = os.environ.get("CACHECLIP_ATTENTION_BACKEND", "torch").strip().lower()
    if backend in {"torch", "reference"}:
        return "torch"
    if backend in {"flashinfer", "flashinfer_single"}:
        return backend
    raise ValueError(
        "CACHECLIP_ATTENTION_BACKEND must be one of: torch, reference, flashinfer, flashinfer_single"
    )


def _flashinfer_causal_bsr_metadata(
    attention_mask: torch.Tensor,
    block_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Build block BSR metadata from CacheClip's selected-row causal mask."""
    if attention_mask.ndim != 4 or attention_mask.size(0) != 1 or attention_mask.size(1) != 1:
        raise RuntimeError(
            "CacheClip FlashInfer requires a [1, 1, selected_rows, key_length] mask"
        )
    if block_size <= 0:
        raise ValueError("FlashInfer block_size must be positive")
    allowed = attention_mask[0, 0] > torch.finfo(attention_mask.dtype).min / 2
    key_length = allowed.size(1)
    padded_length = ((key_length + block_size - 1) // block_size) * block_size
    indptr = [0]
    indices = []
    block_masks = []
    for row in allowed:
        row_indices = torch.nonzero(row, as_tuple=False).flatten()
        if row_indices.numel() == 0:
            raise RuntimeError("CacheClip FlashInfer cannot represent an empty causal row")
        expected = torch.arange(row_indices[-1].item() + 1, device=row.device)
        if not torch.equal(row_indices, expected):
            raise RuntimeError(
                "CacheClip FlashInfer requires contiguous causal prefixes in attention_mask"
            )
        last_block = row_indices[-1].item() // block_size
        row_blocks = torch.arange(last_block + 1, device=row.device)
        indices.append(row_blocks)
        for block in range(last_block + 1):
            valid = block_size if block < last_block else (row_indices[-1].item() % block_size) + 1
            block_masks.append(
                torch.arange(block_size, device=row.device).lt(valid).view(1, block_size)
            )
        indptr.append(indptr[-1] + row_blocks.numel())
    return (
        torch.tensor(indptr, device=attention_mask.device, dtype=torch.int32),
        torch.cat(indices).to(dtype=torch.int32),
        torch.cat(block_masks).view(-1, 1, block_size),
        padded_length,
    )


def _flashinfer_attention_forward(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Run selected-query attention through FlashInfer, failing closed if unsupported."""
    del module
    flashinfer_state = kwargs.pop("flashinfer_state", None)
    if flashinfer_state is None:
        flashinfer_state = {}
    del kwargs
    try:
        from flashinfer.sparse import BlockSparseAttentionWrapper
    except Exception as exc:
        raise RuntimeError(
            "CacheClip FlashInfer backend requested, but "
            "flashinfer.sparse.BlockSparseAttentionWrapper is unavailable"
        ) from exc
    if not callable(BlockSparseAttentionWrapper):
        raise RuntimeError(
            "CacheClip FlashInfer backend found no callable "
            "flashinfer.sparse.BlockSparseAttentionWrapper"
        )
    if dropout != 0.0:
        raise RuntimeError("CacheClip FlashInfer backend only supports dropout=0")
    if attention_mask is None:
        raise RuntimeError("CacheClip FlashInfer backend requires a causal attention mask")
    if not query.is_cuda or not key.is_cuda or not value.is_cuda:
        raise RuntimeError("CacheClip FlashInfer backend requires CUDA query/key/value tensors")
    if query.size(0) != 1 or key.size(0) != 1 or value.size(0) != 1:
        raise RuntimeError("CacheClip FlashInfer backend currently supports batch size 1 only")
    if key.shape != value.shape:
        raise RuntimeError("CacheClip FlashInfer key and value shapes must match")
    if query.size(-1) != key.size(-1):
        raise RuntimeError("CacheClip FlashInfer query/key head dimensions must match")
    if query.size(1) % key.size(1):
        raise RuntimeError("CacheClip FlashInfer requires divisible GQA head counts")

    block_size = 16
    query_rows = query.size(2)
    key_length = key.size(2)
    try:
        wrapper = flashinfer_state.get("wrapper")
        padded_key_length = flashinfer_state.get("padded_key_length")
        if wrapper is None:
            indptr, indices, block_mask, padded_key_length = _flashinfer_causal_bsr_metadata(
                attention_mask, block_size=block_size
            )
            workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=query.device)
            wrapper = BlockSparseAttentionWrapper(workspace)
            wrapper.plan(
                indptr,
                indices,
                query_rows,
                padded_key_length,
                1,
                block_size,
                query.size(1),
                key.size(1),
                query.size(-1),
                mask=block_mask,
                causal=False,
                sm_scale=scaling,
                q_data_type=query.dtype,
                kv_data_type=key.dtype,
                o_data_type=query.dtype,
                )
            flashinfer_state["wrapper"] = wrapper
            flashinfer_state["padded_key_length"] = padded_key_length
        if padded_key_length != key_length:
            key = torch.nn.functional.pad(key, (0, 0, 0, padded_key_length - key_length))
            value = torch.nn.functional.pad(value, (0, 0, 0, padded_key_length - key_length))
        attended = wrapper.run(
            query[0].transpose(0, 1).contiguous(),
            key[0].transpose(0, 1).contiguous(),
            value[0].transpose(0, 1).contiguous(),
        )
    except Exception as exc:
        raise RuntimeError(
            "CacheClip FlashInfer BlockSparseAttentionWrapper failed for the "
            "selected-query BSR layout; use the default torch backend"
        ) from exc
    # FlashInfer returns [query_rows, heads, head_dim], while the Transformers
    # eager-attention contract used by ``recompute_selected_tokens`` is
    # [batch, query_rows, heads, head_dim].  Do not transpose the query/head
    # axes here: the caller reshapes this tensor directly before ``o_proj``.
    return attended.unsqueeze(0), None


def _flashinfer_single_attention_forward(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Run selected rows through ProphetKV's direct FlashInfer path.

    ``single_prefill_with_kv_cache`` accepts compact query rows and an
    arbitrary boolean mask directly.  This avoids the BSR metadata/paged
    tensor-core path used by :func:`_flashinfer_attention_forward`, which is
    poorly matched to a few hundred non-contiguous causal query rows.
    """
    del module, scaling, kwargs
    try:
        import flashinfer
    except Exception as exc:
        raise RuntimeError(
            "CacheClip single FlashInfer backend requested, but flashinfer is unavailable"
        ) from exc
    if dropout != 0.0:
        raise RuntimeError("CacheClip FlashInfer backend only supports dropout=0")
    if attention_mask is None:
        raise RuntimeError("CacheClip FlashInfer backend requires a causal attention mask")
    if not query.is_cuda or not key.is_cuda or not value.is_cuda:
        raise RuntimeError("CacheClip FlashInfer backend requires CUDA query/key/value tensors")
    if query.size(0) != 1 or key.size(0) != 1 or value.size(0) != 1:
        raise RuntimeError("CacheClip FlashInfer backend currently supports batch size 1 only")
    if key.shape != value.shape:
        raise RuntimeError("CacheClip FlashInfer key/value shapes must match")
    allowed = attention_mask[0, 0] > torch.finfo(attention_mask.dtype).min / 2
    attended = flashinfer.single_prefill_with_kv_cache(
        query[0].transpose(0, 1).contiguous(),
        key[0].transpose(0, 1).contiguous(),
        value[0].transpose(0, 1).contiguous(),
        custom_mask=allowed.contiguous(),
        causal=False,
        kv_layout="NHD",
        pos_encoding_mode="NONE",
        backend="auto",
    )
    return attended.unsqueeze(0), None


@torch.inference_mode()
def recompute_selected_tokens(model: Any, assembly: PrimaryAssembly, selected_indices: list[int]) -> None:
    """Update only selected context positions through every supported decoder layer."""
    if not selected_indices:
        return
    if selected_indices != sorted(set(selected_indices)):
        raise ValueError("selected recomputation indices must be sorted and unique")
    if selected_indices[0] < 0 or selected_indices[-1] >= assembly.context_ids.size(1):
        raise ValueError("selected recomputation index lies outside the primary cache")
    _set_eager(model)
    attention_backend = _resolve_attention_backend()
    model_name = type(model).__name__
    if model_name not in {"LlamaForCausalLM", "Qwen2ForCausalLM", "Qwen3ForCausalLM"}:
        raise RuntimeError(
            f"CacheClip sparse recomputation is not migrated for {model_name}; "
            "a model-specific decoder-layer adapter is required"
        )
    if model_name == "LlamaForCausalLM":
        from transformers.models.llama import modeling_llama

        apply_rotary_pos_emb = modeling_llama.apply_rotary_pos_emb
        eager_attention_forward = getattr(modeling_llama, "eager_attention_forward", _local_eager_attention_forward)
    elif model_name == "Qwen2ForCausalLM":
        from transformers.models.qwen2 import modeling_qwen2

        apply_rotary_pos_emb = modeling_qwen2.apply_rotary_pos_emb
        eager_attention_forward = getattr(modeling_qwen2, "eager_attention_forward", _local_eager_attention_forward)
    else:
        from transformers.models.qwen3 import modeling_qwen3

        apply_rotary_pos_emb = modeling_qwen3.apply_rotary_pos_emb
        eager_attention_forward = getattr(modeling_qwen3, "eager_attention_forward", _local_eager_attention_forward)

    base_model = model.model
    cache_layers = _cache_layer_pairs(assembly.cache)
    token_indices = torch.tensor(selected_indices, device=assembly.context_ids.device)
    hidden_states = base_model.embed_tokens(assembly.context_ids[:, token_indices])
    total_length = assembly.context_ids.size(1)
    if model_name == "Qwen2ForCausalLM":
        cos = sin = None
    else:
        position_ids = torch.arange(total_length, device=hidden_states.device).unsqueeze(0)
        cos, sin = base_model.rotary_emb(hidden_states, position_ids=position_ids)
        cos = cos[:, token_indices]
        sin = sin[:, token_indices]
    mask = _recompute_mask(selected_indices, total_length, hidden_states.device, hidden_states.dtype)
    flashinfer_state = {} if attention_backend == "flashinfer" else None
    for layer_index, decoder_layer in enumerate(base_model.layers):
        residual = hidden_states
        normalized = decoder_layer.input_layernorm(hidden_states)
        attention = decoder_layer.self_attn
        input_shape = normalized.size()[:-1]
        hidden_shape = (*input_shape, -1, attention.head_dim)
        query = attention.q_proj(normalized).view(hidden_shape).transpose(1, 2)
        key = attention.k_proj(normalized).view(hidden_shape).transpose(1, 2)
        value = attention.v_proj(normalized).view(hidden_shape).transpose(1, 2)
        if model_name == "Qwen3ForCausalLM":
            query = attention.q_norm(query.transpose(1, 2)).transpose(1, 2)
            key = attention.k_norm(key.transpose(1, 2)).transpose(1, 2)
        if model_name == "Qwen2ForCausalLM":
            query, key = _apply_qwen2_rotary(
                model,
                layer_index,
                attention,
                query,
                key,
                value,
                normalized,
                token_indices,
                total_length,
                apply_rotary_pos_emb,
            )
        else:
            query, key = apply_rotary_pos_emb(query, key, cos, sin)
        cache_keys, cache_values = cache_layers[layer_index]
        cache_keys[:, :, token_indices, :] = key
        cache_values[:, :, token_indices, :] = value
        kwargs = {
            "attention_mask": mask,
            "dropout": 0.0,
            "scaling": getattr(attention, "scaling", attention.head_dim ** -0.5),
        }
        if model_name == "LlamaForCausalLM":
            kwargs["sliding_window"] = getattr(attention, "sliding_window", None)
        if attention_backend == "flashinfer":
            attended, _ = _flashinfer_attention_forward(
                attention,
                query,
                cache_keys,
                cache_values,
                flashinfer_state=flashinfer_state,
                **kwargs,
            )
        elif attention_backend == "flashinfer_single":
            attended, _ = _flashinfer_single_attention_forward(
                attention,
                query,
                cache_keys,
                cache_values,
                **kwargs,
            )
        else:
            attended, _ = eager_attention_forward(
                attention, query, cache_keys, cache_values, **kwargs
            )
        hidden_states = residual + attention.o_proj(attended.reshape(*input_shape, -1).contiguous())
        residual = hidden_states
        hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
        hidden_states = decoder_layer.mlp(hidden_states)
        hidden_states = residual + hidden_states


__all__ = [
    "AuxiliaryDocumentCache",
    "AuxiliaryRuntime",
    "PrimaryAssembly",
    "assemble_primary_cache",
    "continuation_offsets_from_full_sequence",
    "group_candidate_windows",
    "map_auxiliary_to_primary_indices",
    "recompute_selected_tokens",
    "select_top_k_candidates",
    "select_cache_sequence",
    "shared_prefix_token_length",
    "tokenize_with_offsets",
]
