"""CacheClip-only CPU selection and primary-cache helpers."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
import math
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
        selected = set()
        for start in sorted(candidate_set):
            end = min(document_length, start + window_size)
            if sum(start <= candidate < end for candidate in candidate_set) >= density_threshold:
                selected.update(range(start, end))
        grouped.append(sorted(selected))
    return grouped


@dataclass
class AuxiliaryDocumentCache:
    cache: DynamicCache
    continuation_offsets: list[Offset]
    prefix_length: int


def _bucket_indices(caches: Sequence[AuxiliaryDocumentCache]) -> list[list[int]]:
    buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for index, document_cache in enumerate(caches):
        lengths = {layer.keys.size(2) for layer in document_cache.cache.layers}
        if len(lengths) != 1:
            raise ValueError("auxiliary cache has inconsistent layer lengths")
        buckets[(next(iter(lengths)), document_cache.prefix_length, len(document_cache.continuation_offsets))].append(index)
    return list(buckets.values())


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
                cache = self.model(input_ids=input_ids, use_cache=True).past_key_values
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
                bucket_caches = [caches[index] for index in bucket]
                if len(bucket) == 1:
                    output = self.model(
                        input_ids=query_ids,
                        past_key_values=bucket_caches[0].cache,
                        use_cache=False,
                        output_attentions=True,
                        return_dict=True,
                    )
                    attention = output.attentions[-1]
                    bucket_scores = self._document_scores(attention, bucket_caches[0])
                else:
                    batched_cache = DynamicCache(config=self.model.config)
                    for layer_index in range(len(bucket_caches[0].cache.layers)):
                        batched_cache.update(
                            torch.cat([cache.cache.layers[layer_index].keys for cache in bucket_caches]),
                            torch.cat([cache.cache.layers[layer_index].values for cache in bucket_caches]),
                            layer_index,
                        )
                    output = self.model(
                        input_ids=query_ids.expand(len(bucket), -1),
                        past_key_values=batched_cache,
                        use_cache=False,
                        output_attentions=True,
                        return_dict=True,
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
    cos, sin = rotary_emb(rope, position_ids=delta)
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
    assembled = DynamicCache(config=model.config)
    rotary_emb = model.model.rotary_emb
    nope_dim = getattr(model.config, "qk_nope_head_dim", None)
    layer_count = len(segments[0][0])
    for layer_index in range(layer_count):
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
    context_ids = torch.cat([ids for _, ids, _ in segments], dim=1)
    if context_ids.size(1) != total_length:
        raise RuntimeError("CacheClip primary cache/token assembly mismatch")
    return PrimaryAssembly(cache=assembled, context_ids=context_ids)


def select_cache_sequence(cache: DynamicCache, start: int, end: int | None = None) -> DynamicCache:
    """Copy a sequence slice from a DynamicCache without mutating its owner."""
    selected = DynamicCache(config=getattr(cache, "config", None))
    for layer_index, layer in enumerate(cache.layers):
        selected.update(
            layer.keys[:, :, start:end, :],
            layer.values[:, :, start:end, :],
            layer_index,
        )
    return selected


def _recompute_mask(token_indices: list[int], key_length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    rows = torch.tensor(token_indices, device=device).unsqueeze(1)
    columns = torch.arange(key_length, device=device).unsqueeze(0)
    allowed = columns <= rows
    mask = torch.zeros((1, 1, len(token_indices), key_length), device=device, dtype=dtype)
    return mask.masked_fill(~allowed.unsqueeze(0).unsqueeze(0), torch.finfo(dtype).min)


def recompute_selected_tokens(model: Any, assembly: PrimaryAssembly, selected_indices: list[int]) -> None:
    """Update only selected context positions through every supported decoder layer."""
    if not selected_indices:
        return
    if selected_indices != sorted(set(selected_indices)):
        raise ValueError("selected recomputation indices must be sorted and unique")
    if selected_indices[0] < 0 or selected_indices[-1] >= assembly.context_ids.size(1):
        raise ValueError("selected recomputation index lies outside the primary cache")
    _set_eager(model)
    model_name = type(model).__name__
    if model_name not in {"LlamaForCausalLM", "Qwen2ForCausalLM", "Qwen3ForCausalLM"}:
        raise RuntimeError(
            f"CacheClip sparse recomputation is not migrated for {model_name}; "
            "a model-specific decoder-layer adapter is required"
        )
    if model_name == "LlamaForCausalLM":
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, eager_attention_forward
    elif model_name == "Qwen2ForCausalLM":
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, eager_attention_forward
    else:
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, eager_attention_forward

    base_model = model.model
    token_indices = torch.tensor(selected_indices, device=assembly.context_ids.device)
    hidden_states = base_model.embed_tokens(assembly.context_ids[:, token_indices])
    total_length = assembly.context_ids.size(1)
    position_ids = torch.arange(total_length, device=hidden_states.device).unsqueeze(0)
    cos, sin = base_model.rotary_emb(hidden_states, position_ids=position_ids)
    cos = cos[:, token_indices]
    sin = sin[:, token_indices]
    mask = _recompute_mask(selected_indices, total_length, hidden_states.device, hidden_states.dtype)
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
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        layer = assembly.cache.layers[layer_index]
        layer.keys[:, :, token_indices, :] = key
        layer.values[:, :, token_indices, :] = value
        kwargs = {
            "attention_mask": mask,
            "dropout": 0.0,
            "scaling": attention.scaling,
        }
        if model_name == "LlamaForCausalLM":
            kwargs["sliding_window"] = getattr(attention, "sliding_window", None)
        attended, _ = eager_attention_forward(attention, query, layer.keys, layer.values, **kwargs)
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
