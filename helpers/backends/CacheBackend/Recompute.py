"""Selective dense-decoder forwards over the shared, immutable cache format.

The decoder equations follow KV Packet's MIT-licensed baseline kernels. A
selected row attends to the complete causal KV prefix, and replaces that row
at every layer. Fresh rows and the final prompt row must always be selected.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .Cache import KVCache, KeyValue
from .Types import ForwardOutput

_QUERY_CHUNK_SIZE = 512


@dataclass(frozen=True)
class CacheHit:
    start: int
    end: int
    cache: KVCache


def assemble_cache(backend, input_ids: torch.Tensor, hits: list[CacheHit]) -> KVCache:
    """Align independent cached chunks without mutating any prepared tensors."""
    backend.assert_rerotatable()
    length = input_ids.shape[1]
    positions = torch.arange(length, device=input_ids.device).unsqueeze(0)
    previous_end = 0
    shifted = []
    for hit in hits:
        if not previous_end <= hit.start < hit.end <= length:
            raise ValueError("cache hits must be ordered, disjoint prompt spans")
        if hit.cache.seq_len != hit.end - hit.start or hit.cache.doc_len != hit.cache.seq_len:
            raise ValueError("recomputation requires complete, uncompressed chunk caches")
        shifted.append((hit, backend.shift_cache(hit.cache, hit.start)))
        previous_end = hit.end
    if not shifted:
        raise ValueError("cache assembly requires at least one cache hit")
    layers = []
    for layer_index in range(backend.num_layers):
        source = shifted[0][1].layers[layer_index]
        shape = (*source.key.shape[:2], length, source.key.shape[-1])
        key = source.key.new_zeros(shape)
        value = source.value.new_zeros((*shape[:3], source.value.shape[-1]))
        for hit, cache in shifted:
            key[:, :, hit.start:hit.end] = cache.layers[layer_index].key
            value[:, :, hit.start:hit.end] = cache.layers[layer_index].value
        layers.append(KeyValue(key, value, positions))
    return KVCache(layers)


@torch.inference_mode()
def selective_forward(backend, input_ids: torch.Tensor, cache: KVCache,
                      token_indices: list[int]) -> ForwardOutput:
    """Compute selected rows and return the final prompt logits and full KV."""
    length = input_ids.shape[1]
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or not length:
        raise ValueError("selective recomputation requires one non-empty token sequence")
    if token_indices != sorted(set(token_indices)) or not token_indices:
        raise ValueError("recomputation positions must be sorted, unique and non-empty")
    if token_indices[0] < 0 or token_indices[-1] != length - 1:
        raise ValueError("recomputation must include the final prompt token")
    if cache.seq_len != length or cache.doc_len != length:
        raise ValueError("assembled cache must cover the complete prompt")
    model = backend.model
    if model.config.model_type not in {"llama", "mistral", "qwen2", "qwen3"}:
        raise ValueError(f"selective recomputation does not support {model.config.model_type}")
    base = model.model
    indices = torch.tensor(token_indices, dtype=torch.long, device=input_ids.device)
    hidden = base.embed_tokens(input_ids[:, indices])
    positions = indices.unsqueeze(0)
    cos, sin = base.rotary_emb(hidden, position_ids=positions)
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    columns = torch.arange(length, device=input_ids.device).unsqueeze(0)
    # Own the online cache: neither repair nor later decode can change Prepare.
    working = cache.clone()
    for layer_index, layer in enumerate(base.layers):
        residual = hidden
        normalized = layer.input_layernorm(hidden)
        attention = layer.self_attn
        shape = (*normalized.shape[:-1], -1, attention.head_dim)
        query = attention.q_proj(normalized).view(shape)
        key = attention.k_proj(normalized).view(shape)
        value = attention.v_proj(normalized).view(shape)
        if hasattr(attention, "q_norm"):
            query = attention.q_norm(query)
            key = attention.k_norm(key)
        query, key, value = (tensor.transpose(1, 2) for tensor in (query, key, value))

        def rotate(states):
            half = states.shape[-1] // 2
            rotated = torch.cat((-states[..., half:], states[..., :half]), dim=-1)
            return states * cos + rotated * sin

        query, key = rotate(query), rotate(key)
        kv = working.layers[layer_index]
        kv.key[:, :, indices] = key
        kv.value[:, :, indices] = value
        window = getattr(attention, "sliding_window", None)
        attended_chunks = []
        # Bound the attention workspace even when an arbitrary selected-row
        # mask makes PyTorch choose its math implementation on long inputs.
        for start in range(0, len(token_indices), _QUERY_CHUNK_SIZE):
            end = start + _QUERY_CHUNK_SIZE
            rows = indices[start:end].unsqueeze(1)
            allowed = columns <= rows
            if window is not None:
                allowed = allowed & (columns > rows - window)
            attended_chunks.append(F.scaled_dot_product_attention(
                query[:, :, start:end], kv.key, kv.value,
                attn_mask=allowed[None, None], dropout_p=0.0, is_causal=False,
                scale=getattr(attention, "scaling", attention.head_dim ** -0.5),
                enable_gqa=query.shape[1] != kv.key.shape[1],
            ))
        attended = torch.cat(attended_chunks, dim=2)
        attended = attended.transpose(1, 2).reshape(*normalized.shape[:-1], -1)
        hidden = residual + attention.o_proj(attended)
        hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
    # Generation consumes only the last row; avoid projecting all repair rows.
    logits = model.lm_head(base.norm(hidden[:, -1:]))
    return ForwardOutput(logits=logits, cache=working)
