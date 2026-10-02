"""Auxiliary selection and token-text alignment for CacheClip."""

from __future__ import annotations

import codecs
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
import math
import threading
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .Cache import _as_dynamic_cache, _cache_layer_pairs, _new_dynamic_cache


Offset = tuple[int, int]


def _bytelevel_inverse() -> dict[str, int]:
    """Build the standard reversible GPT-2/ByteLevel alphabet map."""
    byte_values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    unicode_values = byte_values[:]
    next_codepoint = 0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            unicode_values.append(256 + next_codepoint)
            next_codepoint += 1
    return {chr(codepoint): value for value, codepoint in zip(byte_values, unicode_values)}


_BYTELEVEL_INVERSE = _bytelevel_inverse()
_UTF8_REPLACEMENT_ERRORS = threading.local()
_UTF8_REPLACEMENT_HANDLER = "kvbench_cacheclip_utf8_replace"


def _capture_utf8_replacement(error: UnicodeDecodeError):
    spans = getattr(_UTF8_REPLACEMENT_ERRORS, "spans", None)
    if spans is not None:
        spans.append((error.start, error.end))
    return "\ufffd", error.end


codecs.register_error(_UTF8_REPLACEMENT_HANDLER, _capture_utf8_replacement)


def _decoded_character_byte_spans(raw: bytes, text: str, errors: Sequence[Offset]) -> list[Offset]:
    """Map decoded code points to source byte ranges, including lossy UTF-8."""
    spans: list[Offset] = []
    byte_position = 0

    def append_valid(start: int, end: int) -> None:
        part = raw[start:end].decode("utf-8", errors="strict")
        cursor = start
        for char in part:
            next_cursor = cursor + len(char.encode("utf-8"))
            spans.append((cursor, next_cursor))
            cursor = next_cursor

    for start, end in errors:
        append_valid(byte_position, start)
        spans.append((start, end))
        byte_position = end
    append_valid(byte_position, len(raw))
    if len(spans) != len(text):
        raise RuntimeError("UTF-8 text and source-span counts differ")
    return spans


def token_text_and_offsets(tokenizer: Any, ids: Sequence[int]) -> tuple[str, list[Offset]]:
    """Decode ByteLevel IDs to canonical text and map each ID to character spans.

    The supplied IDs are authoritative and are never re-tokenized or changed.
    This supports Hugging Face fast tokenizers with a ByteLevel decoder. Every
    token piece follows the decoder's byte-alphabet mapping, including special
    and added-token pieces; malformed UTF-8 byte streams use the same
    replacement-character policy as normal HF decoding.
    """
    backend = getattr(tokenizer, "backend_tokenizer", None)
    decoder = getattr(backend, "decoder", None)
    if not getattr(tokenizer, "is_fast", False) or decoder is None:
        raise ValueError("CacheClip token alignment requires a fast ByteLevel tokenizer")
    if type(decoder).__name__ != "ByteLevel":
        raise ValueError(
            "CacheClip token alignment currently supports ByteLevel decoders; "
            f"got {type(decoder).__name__}"
        )

    token_ids = [int(token_id) for token_id in ids]
    if not token_ids:
        return "", []
    pieces = tokenizer.convert_ids_to_tokens(token_ids)
    byte_stream = bytearray()
    id_byte_spans: list[Offset] = []
    for token_id, piece in zip(token_ids, pieces, strict=True):
        if piece is None:
            raise ValueError(f"invalid tokenizer ID: {token_id}")
        start = len(byte_stream)
        for char in piece:
            byte_value = _BYTELEVEL_INVERSE.get(char)
            if byte_value is None:
                byte_stream.extend(char.encode("utf-8"))
            else:
                byte_stream.append(byte_value)
        id_byte_spans.append((start, len(byte_stream)))

    raw = bytes(byte_stream)
    errors: list[Offset] = []
    _UTF8_REPLACEMENT_ERRORS.spans = errors
    try:
        text = raw.decode("utf-8", errors=_UTF8_REPLACEMENT_HANDLER)
    finally:
        del _UTF8_REPLACEMENT_ERRORS.spans
    char_byte_spans = _decoded_character_byte_spans(raw, text, errors)

    id_char_spans: list[Offset] = []
    char_index = 0
    for start, end in id_byte_spans:
        while char_index < len(char_byte_spans) and char_byte_spans[char_index][1] <= start:
            char_index += 1
        first = char_index
        scan = char_index
        while scan < len(char_byte_spans) and char_byte_spans[scan][0] < end:
            scan += 1
        id_char_spans.append((first, scan))
    return text, id_char_spans


def _offsets(value: Any) -> list[Offset]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list) and value[0] and isinstance(value[0][0], (list, tuple)):
        value = value[0]
    return [tuple(pair) for pair in value]


def map_auxiliary_to_primary_indices(
    auxiliary_offsets: Sequence[Offset],
    primary_offsets: Sequence[Offset],
    selected_auxiliary_indices: Sequence[int],
) -> list[int]:
    """Map selected auxiliary spans to overlapping primary token spans."""
    spans = []
    for index in selected_auxiliary_indices:
        if index < 0 or index >= len(auxiliary_offsets):
            raise ValueError("selected auxiliary index lies outside offsets")
        start, end = auxiliary_offsets[index]
        if start != end:
            spans.append((start, end))
    spans.sort()
    merged: list[Offset] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    result: list[int] = []
    span_index = 0
    previous_start = -1
    for primary_index, (start, end) in enumerate(primary_offsets):
        if start < previous_start:
            raise ValueError("primary token spans must be ordered")
        previous_start = start
        if start == end:
            continue
        while span_index < len(merged) and merged[span_index][1] <= start:
            span_index += 1
        if span_index < len(merged) and merged[span_index][0] < end:
            result.append(primary_index)
    return result


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
    for candidates, document_length in zip(candidates_by_document, document_lengths, strict=True):
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
        grouped.append([index for start, end in intervals for index in range(start, end)])
    return grouped


@dataclass
class AuxiliaryDocumentCache:
    cache: Any
    continuation_offsets: list[Offset]
    prefix_length: int


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
    """Build the attention mask/positions for a query over a cached prefix."""
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


def _copy_dynamic_cache(cache: Any, config: Any) -> Any:
    """Copy cache metadata while retaining read-only references to its K/V."""
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

    def prefill_documents(
        self, prefix: str, documents: Sequence[str]
    ) -> list[AuxiliaryDocumentCache]:
        """Prefill a separately encoded warmup prefix and per-document IDs."""
        prefix_ids = self._encode(prefix)["input_ids"][0].tolist() if prefix else []
        encoded_documents = [self._encode(document, offsets=True) for document in documents]
        result = []
        with torch.no_grad():
            for encoded in encoded_documents:
                document_ids = encoded["input_ids"][0].tolist()
                if not document_ids:
                    raise ValueError("auxiliary document has no continuation tokens")
                offsets = _offsets(encoded["offset_mapping"])
                if len(offsets) != len(document_ids):
                    raise ValueError("auxiliary IDs and character offsets must align")
                input_ids = torch.tensor(
                    [prefix_ids + document_ids], dtype=torch.long
                )
                cache = _as_dynamic_cache(
                    self.model(input_ids=input_ids, use_cache=True).past_key_values,
                    self.model.config,
                )
                result.append(AuxiliaryDocumentCache(
                    cache=cache,
                    continuation_offsets=offsets,
                    prefix_length=len(prefix_ids),
                ))
        return result

    @staticmethod
    def _document_scores(
        attention: torch.Tensor, cache: AuxiliaryDocumentCache
    ) -> list[list[float]]:
        start = cache.prefix_length
        end = start + len(cache.continuation_offsets)
        return attention[:, :, :, start:end].float().mean(dim=(1, 2)).cpu().tolist()

    def score_documents(
        self, query: str, caches: Sequence[AuxiliaryDocumentCache]
    ) -> list[list[float]]:
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
                    query_kwargs = _cached_query_attention_kwargs(query_ids, scoring_cache)
                    output = self.model(
                        input_ids=query_ids,
                        past_key_values=scoring_cache,
                        use_cache=False,
                        output_attentions=True,
                        return_dict=True,
                        **query_kwargs,
                    )
                    bucket_scores = self._document_scores(
                        output.attentions[-1], bucket_caches[0]
                    )
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
                    query_kwargs = _cached_query_attention_kwargs(batched_query, batched_cache)
                    output = self.model(
                        input_ids=batched_query,
                        past_key_values=batched_cache,
                        use_cache=False,
                        output_attentions=True,
                        return_dict=True,
                        **query_kwargs,
                    )
                    bucket_scores = self._document_scores(
                        output.attentions[-1], bucket_caches[0]
                    )
                for index, value in zip(bucket, bucket_scores, strict=True):
                    scores[index] = value
        if any(value is None for value in scores):
            raise RuntimeError("auxiliary selector did not score every document")
        return [value for value in scores if value is not None]


__all__ = [
    "AuxiliaryDocumentCache",
    "AuxiliaryRuntime",
    "group_candidate_windows",
    "map_auxiliary_to_primary_indices",
    "select_top_k_candidates",
    "token_text_and_offsets",
]
