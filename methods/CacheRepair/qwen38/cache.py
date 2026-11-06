"""Full-attention cache boundary; recurrent GDN state is never packed or repaired.

All packed tensors use [B,T,L_attn,H_kv,2,D]. Positions must be supplied by
the caller: [B,T] for text, or the native [3,B,T] MRoPE axes. Never infer
chunk-local positions from an assembled prefix's length.
"""

from __future__ import annotations

import torch

from .target import TargetGeometry


def _attention_layers(cache, geometry):
    layers = getattr(cache, "layers", None)
    if layers is None or len(layers) != geometry.num_hidden_layers:
        raise ValueError("Expected a native hybrid cache indexed by all text-backbone layers")
    indices = set(geometry.full_attention_layer_indices)
    for index, layer in enumerate(layers):
        is_kv = isinstance(getattr(layer, "keys", None), torch.Tensor)
        is_kv &= isinstance(getattr(layer, "values", None), torch.Tensor)
        if is_kv != (index in indices):
            raise ValueError(f"Runtime cache/full-attention mapping differs at layer {index}")
    selected = [layers[index] for index in geometry.full_attention_layer_indices]
    shape = selected[0].keys.shape
    if len(shape) != 4 or shape[1] != geometry.n_kv_heads or shape[-1] != geometry.head_dim:
        raise ValueError("Runtime full-attention cache geometry differs")
    if any(layer.keys.shape != shape or layer.values.shape != shape for layer in selected):
        raise ValueError("Full-attention K/V must share [B,H_kv,T,D] geometry")
    return selected


def pack_target_kv(cache, geometry: TargetGeometry, *, device=None):
    """Extract native post-K-norm, post-RoPE K and V from full-attention layers."""
    selected = _attention_layers(cache, geometry)
    device = device or selected[0].keys.device
    layers = [torch.stack((layer.keys.to(device), layer.values.to(device)), dim=-2) for layer in selected]
    return torch.stack(layers, dim=3).permute(0, 2, 3, 1, 4, 5).contiguous()


@torch.no_grad()
def replace_target_kv(cache, native_kv, geometry: TargetGeometry):
    """Replace K/V in an existing complete native cache, retaining all GDN state.

    This does not synthesize recurrent states for a concatenation of chunks.
    The caller must already have a valid GDN path for the exact prefix.
    Shape checks complete before mutation; cache length/position metadata and
    native tensor objects remain unchanged.
    """
    geometry.validate_kv(native_kv)
    selected = _attention_layers(cache, geometry)
    pending = []
    for packed_index, layer in enumerate(selected):
        key, value = native_kv[:, :, packed_index].permute(3, 0, 2, 1, 4).unbind(0)
        if key.shape != layer.keys.shape:
            raise ValueError("Replacement length or batch differs from the existing native cache")
        pending.append((layer, key.to(layer.keys), value.to(layer.values)))
    for layer, key, value in pending:
        layer.keys.copy_(key)
        layer.values.copy_(value)
    return cache


def _tables(kv, position_ids, rotary):
    if kv.ndim != 6 or kv.shape[-2] != 2 or not kv.is_floating_point():
        raise ValueError("Expected floating-point KV shaped [B,T,L_attn,H_kv,2,D]")
    batch, length = kv.shape[:2]
    if position_ids.dtype not in (torch.int32, torch.int64) or position_ids.device != kv.device:
        raise ValueError("Target position_ids must be integer tensors on the KV device")
    if tuple(position_ids.shape) not in {(batch, length), (3, batch, length)}:
        raise ValueError("Target position_ids must have shape [B,T] or [3,B,T]")
    if (position_ids < 0).any():
        raise ValueError("Target positions must be nonnegative")
    if getattr(rotary, "rope_type", None) != "default":
        raise ValueError("Only native default target RoPE is supported")
    # FP32 probe keeps canonicalization independent of the target cache dtype;
    # the native module supplies partial dimensions, MRoPE mixing and scaling.
    probe = torch.empty(batch, length, 1, device=kv.device, dtype=torch.float32)
    with torch.no_grad():
        cos, sin = rotary(probe, position_ids)
    dim = 2 * rotary.inv_freq.numel()
    if cos.shape != (batch, length, dim) or sin.shape != cos.shape or dim > kv.shape[-1]:
        raise ValueError("Native rotary tables differ from target KV geometry")
    return cos[:, :, None, None, :].float(), sin[:, :, None, None, :].float()


def _rotate(keys, cos, sin):
    dim = cos.shape[-1]
    rotated, passthrough = keys.float()[..., :dim], keys.float()[..., dim:]
    first, second = rotated.chunk(2, dim=-1)
    half = torch.cat((-second, first), dim=-1)
    return torch.cat((rotated * cos + half * sin, passthrough), dim=-1)


def canonicalize_target_kv(native_kv, position_ids, rotary):
    """Invert native partial MRoPE on K only, returning FP32 canonical KV."""
    cos, sin = _tables(native_kv, position_ids, rotary)
    norm = cos.square() + sin.square()
    key = _rotate(native_kv[..., 0, :], cos / norm, -sin / norm)
    return torch.stack((key, native_kv[..., 1, :].float()), dim=-2)


def materialize_target_kv(canonical_kv, position_ids, rotary, *, dtype):
    """Apply native partial MRoPE to K, then cast K/V to the serving dtype."""
    if not dtype.is_floating_point:
        raise ValueError("Native cache dtype must be floating point")
    cos, sin = _tables(canonical_kv, position_ids, rotary)
    key = _rotate(canonical_kv[..., 0, :], cos, sin)
    return torch.stack((key, canonical_kv[..., 1, :].float()), dim=-2).to(dtype)


def reposition_target_kv(native_kv, local_position_ids, global_position_ids, rotary):
    """Reposition native chunk K, rounding to cache dtype before canonicalization.

    Uses a single delta rotation, as in CR's rerotate_chunk. V and non-rotary
    K are preserved. Call canonicalize_target_kv at the global positions next.
    """
    local_cos, local_sin = _tables(native_kv, local_position_ids, rotary)
    global_cos, global_sin = _tables(native_kv, global_position_ids, rotary)
    norm = local_cos.square() + local_sin.square()
    cos = (global_cos * local_cos + global_sin * local_sin) / norm
    sin = (global_sin * local_cos - global_cos * local_sin) / norm
    key = _rotate(native_kv[..., 0, :], cos, sin).to(native_kv.dtype)
    return torch.stack((key, native_kv[..., 1, :]), dim=-2)
