# Method Interface and Extension

A Method defines how inference runs and how cache state is built or reused. KVBench does not require a shared KV-cache data structure; each Method owns its cache format, model backend, and reuse algorithm.

## Interface

All methods inherit from `core.Method.Method` and implement this lifecycle:

| Method | Input and responsibility |
| --- | --- |
| `Initialize(gpuIds)` | Bind the GPUs assigned by the Engine and initialize the model/backend inside the worker process. Constructors should only store configuration; they must not load a model or initialize CUDA. |
| `Prepare(data)` | Build reusable state for a batch of PREPARE actions. `data[i]` is the list of text segments for action `i`; an empty list means that action has no warm-up segments. |
| `Run(data, retainOutput=None, maxNewTokens=None)` | Run inference on a batch of complete prompts. Results must preserve input order. The Engine passes the generation limit from the owning Task. |
| `Reset()` | Clear state for the current batch. Stateless methods can use the default no-op implementation. |
| `Close()` | Release model and backend resources before the worker exits normally. |

Each `Result` returned by `Run` should contain the generated `output` and timing fields in `performance`. System metrics use these keys from `core.Result`:

- `ttft`: seconds. Timing starts when `Run` begins and includes request-dependent lookup, cache loading/assembly/transformation, selection, repair, recomputation, and transfers.
- `num_output_tokens`: number of generated tokens.
- `total_time`: total time spent in `Run`, in seconds.

Model initialization and reusable state built in `Prepare` are excluded from TTFT. Per-request method metrics belong in `Result.metadata`; keys declared in the class attribute `method_metrics` are aggregated by the Engine into the report's `method_metrics` section.

## Existing methods

Refer to each class constructor for its implementation and configurable parameters.

| Method | Purpose |
| --- | --- |
| `FullPrefillTransformer`, `FullPrefillVllm` | Full-prefill baselines using Transformers and vLLM, respectively. |
| `NaiveTransformer`, `NaiveCacheblendRepo` | Controls that directly reuse cached state without repairing context changes. |
| `CacheblendRepo`, `CacheblendLmcache` | CacheBlend recomputation and fusion implementations; the former runs the backend from a separate CacheBlend checkout. |
| `HypicMethod` | HYPIC position-independent reuse. `picMode` supports `addition`, `transition`, `transition_rope`, and `transition_rope_recompute`. |
| `A3Repo` | Adapter for the A3 repository implementation. |
| `ProphetKV` | ProphetKV implementation. |
| `CacheClip` | CacheClip implementation with online token selection by an auxiliary model. |
| `KVCommTransformer` | KVCOMM-style cache reuse for multi-agent workloads. |
| `DependencyAnalysisMethod` | Computes cache dependency analysis metrics. |
| `RolloutMethod` | Decorator that repeats `Run` on a base Method and aggregates the results. |
| `KVPacket` | Reuses prepared text through KV Packet wrappers and packet-aware matching. |
| `EPIC`, `RandRecomputeTransformer` | KVPacket-artifact cache-combination methods with token-count or ratio-based recomputation. |
| `FullRecomputeTransformer` | Full-prompt baseline through the KVPacket recomputation artifact. |

## Using a method in an evaluation

Usually, instantiate methods in `Main.py` and pass them to the Engine:

```python
from core.engine import Engine
from methods import CacheblendRepo, FullPrefillTransformer
from metrics import TTFTMetric

methods = [
    FullPrefillTransformer(gpuNums=1),
    CacheblendRepo(gpuNums=1, recompRatio=0.15, tag="0.15"),
]
report = Engine().Evaluate(tasks=tasks, methods=methods, metrics=[TTFTMetric()])
```

`tag` is appended to the report label (for example, `cacheblend_repo(0.15)`). `gpuNums` declares how many GPUs the method requires; `perfWeight` controls its scheduling weight. The model path, global sampling settings, and backend paths come from `config.yaml`.

## Adding a method

1. Add a class under `methods/` that inherits from `Method` and declares a stable `name`. Call `super().__init__(...)` in the constructor and store algorithm parameters. Do not create models or occupy GPUs in the constructor.
2. In `Initialize`, call `super().Initialize(gpuIds)` before loading the model and backend resources.
3. Implement `Prepare` to turn each action's text segments into the cache or index required by the method.
4. Implement `Run` to return one `Result` per prompt in input order. Honor `maxNewTokens` and record TTFT, generated token count, and total time.
5. For per-request algorithm metrics, write numeric values to `Result.metadata` and declare their keys in `method_metrics`. Set `maxCaseBatchSize = 1` if state cannot be shared safely across Cases in a batch.
6. Clear Case/batch state in `Reset` and release backend resources in `Close`. Export the class from `methods/__init__.py`, then add an instance to the `methods` list in `Main.py`.

A Workflow may provide multiple text segments or preserve state across multiple RUN actions. Handle these inputs according to the method's algorithm, and prevent one Case's cache from leaking into another when cross-Case reuse is not intended.
