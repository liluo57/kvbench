"""Concrete KV optimization methods. One module per method.

Each method takes the task's available GPUs (``gpu_ids``, equivalent to
``CUDA_VISIBLE_DEVICES``) in its constructor:

- :class:`CacheBlendMethod` — CacheBlend via vLLM 0.25 + LMCache in-process
  blending: ``Prepare`` stores per-chunk KV segments (token-level ``' # # '``
  sep separators), ``Run`` reuses them and blends the fresh query KV in, or
  generates the whole prompt when reuse is impossible. All runtime wiring lives
  in the framework (``helpers.VllmCacheblendPatches``), so no host-level
  ``sitecustomize.py`` or modified vLLM is required.
- :class:`FullPrefillTransformer` / :class:`FullPrefillVllm` — the recompute
  baselines: answer the full prompt from scratch every query, over plain
  transformers / the system vLLM.
- :class:`NaiveTransformer` — plain KV reuse without repair: cache the shared
  context once and answer every query against that cache (per-chunk KV concat
  over transformers).
"""

from .Cacheblend import CacheBlendMethod
from .FullPrefill import FullPrefillTransformer, FullPrefillVllm
from .Naive import NaiveTransformer

__all__ = [
    "CacheBlendMethod",
    "FullPrefillTransformer",
    "FullPrefillVllm",
    "NaiveTransformer",
]
