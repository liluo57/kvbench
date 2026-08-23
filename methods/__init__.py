"""Concrete KV optimization methods. One module per method.

Each constructor declares the strict ``gpuNums`` requirement and a relative
``perfWeight``.  The Engine binds concrete GPU ids inside a worker process.

- :class:`CacheblendLmcache` — CacheBlend via vLLM 0.25 + LMCache in-process
  blending: ``Prepare`` stores per-chunk KV segments (token-level ``' # # '``
  sep separators), ``Run`` reuses them and blends the fresh query KV in, or
  generates the whole prompt when reuse is impossible. All runtime wiring lives
  in the framework (``helpers.VllmCacheblendPatches``), so no host-level
  ``sitecustomize.py`` or modified vLLM is required.
- :class:`CacheblendRepo` — CacheBlend via the *original* repo: a worker
  subprocess under the repo's venv (patched vLLM 0.4.1, ``helpers.
  CacheblendRepoHelper``) collects the context KV once and fuses each fresh
  query against it; ``reuse_ratio`` is reported by the worker. Repo/model paths
  come from ``config.yaml`` (``Cacheblend.Repo.*``).
- :class:`FullPrefillTransformer` / :class:`FullPrefillVllm` — the recompute
  baselines: answer the full prompt from scratch every query, over plain
  transformers / the system vLLM.
- :class:`NaiveTransformer` — plain KV reuse without repair: cache the shared
  context once and answer every query against that cache (per-chunk KV concat
  over transformers).
"""

from .CacheblendLmcache import CacheblendLmcache
from .CacheblendRepo import CacheblendRepo, NaiveCacheblendRepo
from .FullPrefill import FullPrefillTransformer, FullPrefillVllm
from .Naive import NaiveTransformer

__all__ = [
    "CacheblendLmcache",
    "CacheblendRepo",
    "NaiveCacheblendRepo",
    "FullPrefillTransformer",
    "FullPrefillVllm",
    "NaiveTransformer",
]
