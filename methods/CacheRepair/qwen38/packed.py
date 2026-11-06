"""Zero-state variable-length chunk prefill with all boundaries explicit.

Attention is causal only within each chunk, convolution uses seq_idx resets,
and FLA receives cu_seqlens to reset the GDN recurrence for every chunk.
The target's module attributes are restored before any joint forward.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import partial

import torch

from .cache import canonicalize_target_kv, pack_target_kv, reposition_target_kv
from .target import text_backbone


def _conv(original, boundaries, **kwargs):
    kwargs["seq_idx"] = boundaries
    return original(**kwargs)


def _gdn(original, cumulative, cumulative_cpu, *args, **kwargs):
    kwargs.update(cu_seqlens=cumulative, cu_seqlens_cpu=cumulative_cpu)
    return original(*args, **kwargs)


@contextmanager
def chunk_boundaries(target, chunk_ids, lengths):
    cumulative_cpu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    tables, saved = {}, []
    try:
        for layer in text_backbone(target).layers:
            if not hasattr(layer, "linear_attn"):
                continue
            module = layer.linear_attn
            device = module.in_proj_qkv.weight.device
            if device not in tables:
                tables[device] = (chunk_ids.to(device=device, dtype=torch.int32), cumulative_cpu.to(device))
            ids, cumulative = tables[device]
            if module.causal_conv1d_fn is None:
                raise RuntimeError("Packed prefill requires fused convolution with seq_idx support")
            saved.append((module, module.causal_conv1d_fn, module.chunk_gated_delta_rule))
            module.causal_conv1d_fn = partial(_conv, module.causal_conv1d_fn, ids)
            module.chunk_gated_delta_rule = partial(_gdn, module.chunk_gated_delta_rule, cumulative, cumulative_cpu)
        yield
    finally:
        for module, conv, gdn in saved:
            module.causal_conv1d_fn, module.chunk_gated_delta_rule = conv, gdn


@torch.no_grad()
def packed_stale(target, chunks, geometry):
    device = target.get_input_embeddings().weight.device
    lengths = list(map(len, chunks))
    tokens = torch.tensor([sum(chunks, [])], device=device, dtype=torch.long)
    positions = torch.cat([torch.arange(n, device=device) for n in lengths]).unsqueeze(0)
    chunk_ids = torch.repeat_interleave(torch.arange(len(chunks), device=device),
                                         torch.tensor(lengths, device=device)).unsqueeze(0)
    length = tokens.shape[1]
    physical = torch.arange(length, device=device)
    visible = ((chunk_ids.unsqueeze(-1) == chunk_ids.unsqueeze(-2)) &
               (physical[:, None] >= physical[None, :]))
    stride = (length + 15) // 16 * 16
    mask = torch.full((1, 1, length, stride), float("-inf"), device=device, dtype=torch.bfloat16)[..., :length]
    mask.masked_fill_(visible.unsqueeze(1), 0)
    with chunk_boundaries(target, chunk_ids, lengths):
        output = target(input_ids=tokens, attention_mask=mask, position_ids=positions,
                        use_cache=True, logits_to_keep=1)
    packed = pack_target_kv(output.past_key_values, geometry, device=device)
    del output
    parts, offset = [], 0
    rotary = text_backbone(target).rotary_emb
    for n in lengths:
        native = packed[:, offset:offset + n].contiguous()
        local = positions[:, offset:offset + n]
        global_positions = torch.arange(offset, offset + n, device=device).unsqueeze(0)
        if offset:
            native = reposition_target_kv(native, local, global_positions, rotary)
        parts.append(canonicalize_target_kv(native, global_positions, rotary))
        offset += n
    return torch.cat(parts, dim=1), tokens, chunk_ids
