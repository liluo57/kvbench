"""Backend helpers: one module per backend, no grab-bag.

- :mod:`helpers.VllmHelper` — driving the project vLLM (>= 0.23): backend
  probe, LLM construction (with a sanitized model dir for the 1M model's
  dual-chunk config), and a Generate() that returns TTFT.
- :mod:`helpers.TransformersHelper` — the plain HF ``transformers`` generator
  (owned model + tokenizer, manual greedy decode with real TTFT).
- :mod:`helpers.Prompt` — splitting a case's complete prompt into the shared
  context and the fresh suffix for the reuse methods.
"""
