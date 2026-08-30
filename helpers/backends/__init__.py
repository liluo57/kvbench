"""Backend adapters — every per-runtime / per-arch helper for talking to a
model server.

Modules:

- :mod:`helpers.backends.VllmHelper` — project vLLM (>= 0.23) backend probe,
  LLM construction (with a sanitized model dir for the 1M model's
  dual-chunk config), and ``Generate()`` returning TTFT.
- :mod:`helpers.backends.TransformersHelper` — plain HF ``transformers``
  generator (owned model + tokenizer, manual greedy decode with real TTFT).
- :mod:`helpers.backends.VllmCacheblendPatches` — runtime wiring for
  vLLM + LMCache CacheBlend.
- :mod:`helpers.backends.ModelAdapter` — the **single chat-template truth
  source**: per-arch detection (Qwen3 / Muse Glimmer), tokenizer-driven
  prompt rendering, and vLLM-native tool-call / reasoning parsers. Every
  task family routes its prompts via this module.
- :mod:`helpers.backends.Prompt` — splitting a case's complete prompt into
  the shared context and the fresh suffix for the reuse methods.

These five modules are unrelated to :mod:`helpers.benchflow` and
:mod:`helpers.cacheblend_repo`; they live together only because they share
the same audience (Methods / Tasks) and the same dependency direction (→
:mod:`core` only).
"""