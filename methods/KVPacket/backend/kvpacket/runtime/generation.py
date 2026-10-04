"""Provide framework-owned token selection and incremental decoding state."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from .types import PacketCacheState
from helpers.backends.CacheBackend.Sampling import (
    SamplingConfig,
    make_generator,
    select_next_token,
)

__all__ = [
    "SamplingConfig",
    "make_generator",
    "PacketDecodeState",
    "PacketDecodeStep",
    "select_next_token",
]




@dataclass(frozen=True)
class PacketDecodeState:
    """Caller-owned cache and logits preceding the next generated token."""

    cache_state: PacketCacheState
    next_token_logits: Tensor


@dataclass(frozen=True)
class PacketDecodeStep:
    """One selected token and the state after committing it to the cache."""

    token_id: int
    state: PacketDecodeState
