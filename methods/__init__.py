"""Concrete KV optimization methods and their public class names.

Each constructor declares the strict ``gpuNums`` requirement and a relative
``perfWeight``.  The Engine binds concrete GPU ids inside a worker process.

- :class:`CacheblendLmcache` — CacheBlend via vLLM 0.25 + LMCache in-process
  blending: ``Prepare`` stores per-chunk KV segments (token-level ``' # # '``
  sep separators), ``Run`` reuses them and blends the fresh query KV in, or
  generates the whole prompt when reuse is impossible. All runtime wiring lives
  in the framework (``helpers.VllmCacheblendPatches``), so no host-level
  ``sitecustomize.py`` or modified vLLM is required.
- :class:`CacheblendRepo` — CacheBlend through the bundled backend in
  ``methods.Cacheblend``. A worker sharing the project Python owns collected KV and
  fuses fresh queries; sampling and reuse metrics follow the shared contracts.
- :class:`A3Repo` — A^3 through the bundled ``ragkv`` runtime in
  ``methods.A3Repo``. An isolated worker uses the project Python, with
  integrated precompute, selection, fusion and dense Qwen3 support.
- :class:`FullPrefillTransformer` / :class:`FullPrefillVllm` — the recompute
  baselines: answer the full prompt from scratch every query, over plain
  transformers / the system vLLM.
- :class:`NaiveTransformer` — plain KV reuse without repair: cache the shared
  context once and answer every query against that cache (per-chunk KV concat
  over transformers).
- :class:`HypicMethod` — position-independent segment reuse through the HYPIC
  SGLang fork, including hybrid linear/full-attention state composition.
"""

from .CacheblendLmcache import CacheblendLmcache
from .Cacheblend import CacheblendRepo, NaiveCacheblendRepo
from .A3Repo import A3Repo
from .FullPrefill import FullPrefillTransformer, FullPrefillVllm
from .hypic import HypicMethod, LinearKVMethod
from .Naive import NaiveTransformer
from .KVComm import KVCommTransformer
from .ProphetKV import ProphetKV
from .CacheClip import CacheClip
from .DependencyAnalysis import DependencyAnalysisMethod
from .Rollout import RolloutMethod
from .KVPacket import KVPacket
from .KVPacketRecompute import (
    EPIC,
    FullRecomputeTransformer,
    RandRecomputeTransformer,
)

__all__ = [
    "CacheblendLmcache",
    "CacheblendRepo",
    "NaiveCacheblendRepo",
    "A3Repo",
    "FullPrefillTransformer",
    "FullPrefillVllm",
    "HypicMethod",
    "LinearKVMethod",
    "NaiveTransformer",
    "KVCommTransformer",
    "ProphetKV",
    "CacheClip",
    "DependencyAnalysisMethod",
    "RolloutMethod",
    "KVPacket",
    "EPIC",
    "RandRecomputeTransformer",
    "FullRecomputeTransformer",
]
