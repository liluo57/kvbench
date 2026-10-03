"""Opt-in PIC cache-entry dump for semantic diagnostics.

This module is deliberately inert unless ``PIC_CACHE_DEBUG_DUMP`` is set.  It
copies the entry's token ids, full-attention KV, and every mamba-pool field at
the entry slot to CPU after the normal cache commit.  It does not alter slot
allocation, cache keys, or forward execution.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch


def _tensor_cpu(t: torch.Tensor) -> torch.Tensor:
    return t.detach().clone().cpu()


def _tensor_stats(t: torch.Tensor) -> dict[str, Any]:
    x = t.detach().to(torch.float32)
    if x.numel() == 0:
        return {"shape": list(t.shape), "dtype": str(t.dtype), "numel": 0}
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "numel": int(t.numel()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "mean": float(x.mean().item()),
        "norm": float(x.norm().item()),
    }


def maybe_dump_entry(cache, entry) -> None:
    """Dump one committed entry when its hash matches the optional filter."""
    out_dir = os.environ.get("PIC_CACHE_DEBUG_DUMP", "").strip()
    if not out_dir:
        return
    wanted = os.environ.get("PIC_CACHE_DEBUG_HASH", "").strip().lower()
    if wanted and entry.seg_hash.hex().lower() != wanted:
        return

    payload: dict[str, Any] = {
        "label": os.environ.get("PIC_CACHE_DEBUG_LABEL", "run"),
        "seg_hash": entry.seg_hash.hex(),
        "token_ids": entry.token_ids.detach().cpu(),
        "full_kv_slots": entry.full_kv_slots.detach().cpu(),
        "mamba_state_slot": int(entry.mamba_state_slot),
    }
    payload["token_stats"] = _tensor_stats(entry.token_ids)

    # HybridLinearKVPool exposes full-attention buffers through the allocator's
    # backing KV cache.  Use the model's actual full-layer ids, not assumptions
    # about the interleaving pattern.
    kv_pool = cache.token_to_kv_pool_allocator.get_kvcache()
    full_layer_ids = list(
        getattr(kv_pool, "full_attention_layer_id_mapping", {}).keys()
    )
    slots = entry.full_kv_slots.to(entry.full_kv_slots.device)
    full_kv: dict[str, Any] = {}
    for layer_id in full_layer_ids:
        k = kv_pool.get_key_buffer(layer_id).index_select(0, slots)
        v = kv_pool.get_value_buffer(layer_id).index_select(0, slots)
        full_kv[str(layer_id)] = {
            "k": _tensor_cpu(k),
            "v": _tensor_cpu(v),
            "k_stats": _tensor_stats(k),
            "v_stats": _tensor_stats(v),
        }
    payload["full_kv"] = full_kv

    mamba_cache = cache.req_to_token_pool.mamba_pool.mamba_cache
    mamba: dict[str, Any] = {}
    for name in ("conv", "temporal", "transition", "conv_tails"):
        value = getattr(mamba_cache, name, None)
        if value is None:
            mamba[name] = None
        elif isinstance(value, (list, tuple)):
            rows = [x[:, int(entry.mamba_state_slot)] for x in value]
            mamba[name] = {
                "values": [_tensor_cpu(x) for x in rows],
                "stats": [_tensor_stats(x) for x in rows],
            }
        else:
            row = value[:, int(entry.mamba_state_slot)]
            mamba[name] = {
                "values": _tensor_cpu(row),
                "stats": _tensor_stats(row),
            }
    payload["mamba"] = mamba

    # Include a deterministic payload digest so comparisons can distinguish
    # exact equality from floating-point tolerance equality without loading all
    # tensors in the reporting script.
    digest = hashlib.sha256()
    digest.update(
        payload["token_ids"].contiguous().view(torch.uint8).numpy().tobytes()
    )
    for layer_id in sorted(full_kv):
        for key in ("k", "v"):
            value = full_kv[layer_id][key].contiguous()
            digest.update(value.view(torch.uint8).numpy().tobytes())
    for name in ("conv", "temporal", "transition", "conv_tails"):
        item = mamba[name]
        if item is None:
            continue
        values = item["values"] if isinstance(item["values"], list) else [item["values"]]
        for value in values:
            value = value.contiguous()
            digest.update(value.view(torch.uint8).numpy().tobytes())
    payload["value_sha256"] = digest.hexdigest()

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    filename = f"{payload['label']}_{entry.seg_hash.hex()}.pt"
    torch.save(payload, Path(out_dir) / filename)


def maybe_dump_match(req, segments, entries, hit_tokens: int) -> None:
    """Append the resolved hit/miss segment map when explicitly requested."""
    path = os.environ.get("PIC_CACHE_DEBUG_MATCH", "").strip()
    if not path:
        return
    rows = []
    origin = list(getattr(req, "origin_input_ids", []) or [])
    from methods.hypic.backend.kvbench_hypic.srt.pic.segmenter import segment_hash

    for (start, end), entry in zip(segments, entries):
        token_ids = origin[start:end]
        rows.append(
            {
                "start": int(start),
                "end": int(end),
                "tokens": int(end - start),
                "hash": segment_hash(token_ids).hex(),
                "hit": entry is not None,
            }
        )
    record = {
        "label": os.environ.get("PIC_CACHE_DEBUG_LABEL", "request"),
        "rid": str(getattr(req, "rid", "")),
        "mode": str(getattr(getattr(req, "pic_policy", None), "name", "")),
        "hit_tokens": int(hit_tokens),
        "segments": rows,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a") as fh:
        fh.write(json.dumps(record) + "\n")
