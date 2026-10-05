# Tests

Install the project's `requirement.txt` dependencies, then run from the repository
root:

```bash
python -m pytest -q
python -m pytest -q tests/methods/test_epic.py
python -m pytest -q tests/core tests/workflows
```

The default suite uses CPU tensors, tiny randomly initialized models, in-memory
tokenizers, temporary files, and local HTTP/subprocess doubles. It needs no model
checkpoint, dataset, Docker daemon, GPU, or external service.

## Layout

| Directory | Responsibility |
| --- | --- |
| `methods/` | One suite per method family, including backend integration, retained output, matching, metrics, sampling and lifecycle |
| `backends/` | Shared cache representation/recomputation, generation helpers, prompt matching, tokenization and subprocess transport |
| `core/` | Configuration, public contracts, metrics, worker execution, scheduling, engine, GPU selection and TUI |
| `workflows/` | RAG, multi-agent and AgentBenchFlow action sequencing |
| `tasks/` | Dataset loading, chunking, answer parsing and evaluation |
| `helpers/` | OpenAI endpoint and local/remote BenchFlow runners |
| `scripts/` | Environment setup, dataset preparation and remote runtime server |
| `support/` | Reusable test doubles and tiny-model factories; no collected tests |

HYPIC/LinearKV, the two FullPrefill backends, and CacheBlend's repository/naive
variants share a family suite. LMCache has its own suite because its execution
and retention contracts differ from the repository backend.

## Optional backends

Compiled backend tests stay in their method suites and carry the `native` marker.
CUDA kernel tests carry `gpu`. They are skipped unless explicitly enabled:

```bash
python -m pytest -q --run-native -m native
python -m pytest -q --run-gpu -m gpu
python -m pytest -q --run-native --run-gpu
```

These tests use the project's selected interpreter and built backends. Missing
optional packages/kernels or an unavailable GPU produce an explanatory skip;
failures in an available backend are test failures.

## Adding coverage

- Put a method-specific regression in that method's existing file, including
  `retainOutput` and generation-budget behavior. Put shared backend behavior in
  `backends/`.
- Import shared utilities from `tests.support`; never import a `test_*.py` module.
  Use `conftest.py` for shared pytest fixtures.
- Prefer observable behavior and numerical references: authoritative token IDs,
  native KV rows, immutable prepared state, cache-hit coverage, returned results,
  and failure propagation. Avoid source-text checks and assertions that only
  repeat a class's declarations.
- Use the `clock` fixture for timing arithmetic. Real waits belong to transport,
  timeout and concurrency tests that actually exercise those boundaries.
  If the implementation imports `perf_counter` directly, patch that module's
  binding to `clock` as well.
- Construct test inputs locally. Do not hardcode a developer's filesystem,
  downloaded tokenizer IDs, checkpoint versions or live services.

CPU Torch/Python random state is restored between tests. CPU Torch threads are
limited to one for tiny numerical workloads and restored when the session ends.
