"""Qwen3.8 full-attention repairer body; training/serving integration is separate."""

from .cache import (
    canonicalize_target_kv,
    materialize_target_kv,
    pack_target_kv,
    replace_target_kv,
    reposition_target_kv,
)
from .model import QwenCacheRepair, RepairConfig, chunk_causal_mask, load_repairer, save_repairer
from .target import TargetGeometry, text_backbone

__all__ = [
    "QwenCacheRepair", "RepairConfig", "TargetGeometry", "text_backbone",
    "chunk_causal_mask", "load_repairer", "save_repairer", "pack_target_kv",
    "replace_target_kv", "canonicalize_target_kv", "materialize_target_kv",
    "reposition_target_kv",
]
