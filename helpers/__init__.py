"""Backend helpers: one module per backend, no grab-bag.

- :mod:`helpers.VllmHelper` — driving the project vLLM (>= 0.23): backend
  probe, LLM construction (with a sanitized model dir for the 1M model's
  dual-chunk config), and a Generate() that returns TTFT.
- :mod:`helpers.TransformersHelper` — the plain HF ``transformers`` generator
  (owned model + tokenizer, manual greedy decode with real TTFT).
- :mod:`helpers.Prompt` — splitting a case's complete prompt into the shared
  context and the fresh suffix for the reuse methods.
- :mod:`helpers.Gpu` — NVML-backed GPU discovery and explicit pool resolution.
- :mod:`helpers.VllmCacheblendPatches` — runtime wiring for vLLM + LMCache
  CacheBlend.
- :mod:`helpers.CacheblendRepoHelper` — JSON-lines worker for the original
  CacheBlend repository and its isolated virtual environment.
- :mod:`helpers.Qwen3ForCacheBlendRepo` — Qwen3 model adapter used by that
  original-repository worker.
"""
