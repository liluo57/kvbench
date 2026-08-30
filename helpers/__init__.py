"""Cross-cutting helpers grouped by concern.

This package is a deliberate ``helpers/`` catch-all: it is the only place
outside :mod:`core` that tasks, methods, and workloads depend on. Three
loosely-coupled subsystems live here today:

1. **Backend adapters** — talk to model runtimes.

   - :mod:`helpers.VllmHelper` — driving the project vLLM (>= 0.23): backend
     probe, LLM construction (with a sanitized model dir for the 1M model's
     dual-chunk config), and a Generate() that returns TTFT.
   - :mod:`helpers.TransformersHelper` — the plain HF ``transformers``
     generator (owned model + tokenizer, manual greedy decode with real TTFT).
   - :mod:`helpers.VllmCacheblendPatches` — runtime wiring for vLLM + LMCache
     CacheBlend.
   - :mod:`helpers.ModelAdapter` — the **single chat-template truth source**:
     per-arch detection (Qwen3 / Muse Glimmer), tokenizer-driven prompt
     rendering, and vLLM-native tool-call / reasoning parsers. Every task
     family routes its prompts via this module.
   - :mod:`helpers.Prompt` — splitting a case's complete prompt into the
     shared context and the fresh suffix for the reuse methods.

2. **CacheBlend (original repo)** — JSONL worker that drives the upstream
   CacheBlend vLLM fork.

   - :mod:`helpers.CacheblendRepoHelper` — JSON-lines worker for the
     original CacheBlend repository and its isolated virtual environment.
   - :mod:`helpers.Qwen3ForCacheBlendRepo` — Qwen3 model adapter registered
     via side-effect (``AutoConfig.register`` + ``ModelRegistry.register_model``)
     for that worker.

3. **BenchFlow (SkillsBench)** — agent-in-sandbox orchestration.

   - :mod:`helpers.SkillInjector` — skill body feed into the system region.
   - :mod:`helpers.benchflow` — HTTP endpoint, sandbox staging, and
     watchdog that runs the agent + verifier in an apptainer SIF.

The split is documentation-only for now; see the helpers/ subpackage
split tracked separately for the structural cleanup.
"""