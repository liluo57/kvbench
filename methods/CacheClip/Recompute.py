"""Sparse primary-cache recomputation kernels for CacheClip."""

from __future__ import annotations

import inspect
from typing import Any
import os

import torch

from .Cache import (
    PrimaryAssembly,
    _cache_layer_pairs,
    _layer_rotary_embedding,
    _rotate_key,
)


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


def _set_eager(model: Any) -> None:
    """Force eager attention for CacheClip's selected-query recomputation."""
    setter = getattr(model, "set_attn_implementation", None)
    if callable(setter):
        setter("eager")
    model.config._attn_implementation = "eager"
    for layer in getattr(model.model, "layers", []):
        config = getattr(getattr(layer, "self_attn", None), "config", None)
        if config is not None:
            config._attn_implementation = "eager"

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


__all__ = ["recompute_selected_tokens"]
