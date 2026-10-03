# Method Interface and Extension

A Method defines how inference runs and how cache state is built or reused. KVBench does not require a shared KV-cache data structure; each Method owns its cache format, model backend, and reuse algorithm.

## Interface

All methods inherit from `core.Method.Method` and implement this lifecycle:

| Method | Input and responsibility |
| --- | --- |
| `Initialize(gpuIds)` | Bind the GPUs assigned by the Engine and initialize the model/backend inside the worker process. Constructors should only store configuration; they must not load a model or initialize CUDA. |
| `Prepare(data)` | Build reusable state for a batch of PREPARE actions. `data[i]` is a list of token-ID chunks for action `i`; each chunk is tokenized independently. An empty list means that action has no warm-up segments. |
| `Run(data, retainOutput=None, maxNewTokens=None)` | Run inference on a batch of complete prompts represented as token-ID lists. Results must preserve input order. The Engine passes the generation limit from the owning Task. |
| `Reset()` | Clear state for the current batch. Stateless methods can use the default no-op implementation. |
| `Close()` | Release model and backend resources before the worker exits normally. |

Each `Result` returned by `Run` should contain decoded generated `output` and timing fields in `performance`. Set `output_ids` to the authoritative generated token fragment when available: it excludes terminal stop IDs but retains interior special IDs. Multi-turn workflows pass this exact sequence into a later prompt and retained KV must use the same fragment; never decode and re-encode `output` as a substitute for cache identity. A text-only backend without `output_ids` cannot provide reusable output to a later turn. System metrics use these keys from `core.Result`:

- `ttft`: seconds. Timing starts when `Run` begins and includes request-dependent lookup, cache loading/assembly/transformation, selection, repair, recomputation, and transfers. Task-side prompt tokenization is completed before `Run` and is excluded.
- `num_output_tokens`: number of generated tokens.
- `total_time`: total time spent in `Run`, in seconds.

Model initialization and reusable state built in `Prepare` are excluded from TTFT. Per-request method metrics belong in `Result.metadata`; keys declared in the class attribute `method_metrics` are aggregated by the Engine into the report's `method_metrics` section.

## Existing methods

Refer to each class constructor for its implementation and configurable parameters.

| Method | Purpose |
| --- | --- |
| `FullPrefillTransformer`, `FullPrefillVllm` | Full-prefill baselines using Transformers and vLLM, respectively. |
| `NaiveTransformer`, `NaiveCacheblendRepo` | Controls that directly reuse cached state without repairing context changes. |
| `CacheblendRepo`, `CacheblendLmcache` | CacheBlend recomputation and fusion implementations; the former runs the bundled backend in `methods/Cacheblend` through a worker sharing the main project's Python environment. |
| `HypicMethod`, `LinearKVMethod` | Built-in hybrid position-independent reuse in `methods/hypic`; both accept token IDs and explicit segment boundaries. HYPIC defaults to transition/RoPE/seam repair; LinearKV uses the last matched GDN state and static chunk-prefix repair. |
| `A3Repo` | Adapter for the A3 repository implementation. |
| `ProphetKV` | ProphetKV implementation. |
| `CacheClip` | CPU auxiliary-token selection with sparse repair of primary KV caches. Prepared chunks and prompts remain authoritative primary-model token IDs; selector text and character spans are derived locally from each prepared ID chunk. |
| `KVCommTransformer` | KVCOMM-style cache reuse for multi-agent workloads. |
| `DependencyAnalysisMethod` | Computes cache dependency analysis metrics. |
| `RolloutMethod` | Decorator that repeats `Run` on a base Method and aggregates the results. |
| `KVPacket` | Reuses prepared token sequences through KV Packet wrappers and token-aware matching. |
| `EPIC`, `RandRecomputeTransformer`, `FullRecomputeTransformer` | KVPacket recomputation artifact methods. Their token-ID input support is temporarily deferred. |

CacheClip's current primary-ID alignment supports fast ByteLevel tokenizers and has been validated with Qwen. `sharedPrefixText` optionally supplies an explicit warmup prefix: preparation tokenizes `P` independently, prefills `P_ids + D_ids`, then stores only the cache rows for `D_ids` and records the length of `P_ids`. Run never inserts or removes IDs for this prefix; without a matching prefix in the runtime prompt, this warmup is a calibration variant rather than strict paper-style prefix retention. The auxiliary selector view is decoded from primary IDs, so Unicode normalization may differ from the original source text.

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

Import public Method classes from `methods`. Each implementation has one class
name: `RandRecomputeTransformer`, `DependencyAnalysisMethod`, and `KVPacket`
are their respective public names.

## Adding a method

1. Add a class under `methods/` that inherits from `Method` and declares a stable `name`. Call `super().__init__(...)` in the constructor and store algorithm parameters. Do not create models or occupy GPUs in the constructor.
2. In `Initialize`, call `super().Initialize(gpuIds)` before loading the model and backend resources.
3. Implement `Prepare` to turn each action's token-ID chunks into the cache or index required by the method.
4. Implement `Run` to return one `Result` per prompt in input order. Honor `maxNewTokens` and record TTFT, generated token count, and total time.
5. For per-request algorithm metrics, write numeric values to `Result.metadata` and declare their keys in `method_metrics`. Set `maxCaseBatchSize = 1` if state cannot be shared safely across Cases in a batch.
6. Clear Case/batch state in `Reset` and release backend resources in `Close`. Export the class from `methods/__init__.py`, then add an instance to the `methods` list in `Main.py`.

A Workflow may provide multiple token chunks or preserve state across multiple RUN actions. Handle these inputs according to the method's algorithm, and prevent one Case's cache from leaking into another when cross-Case reuse is not intended.

HYPIC and LinearKV share the project environment. Run `./setup.py --hypic` to
prepare their dependencies and compile the bundled CUDA ops against the active
PyTorch ABI. They do not require an external HYPIC checkout or an installed
SGLang package. Building requires a CUDA 12.9+ development toolkit (set
`CUDA_HOME`); setup reuses these compiler settings for shape-specialized JIT
kernels in workers. The retained native ops serve unquantized FP16/BF16
checkpoints on NVIDIA GPUs. `Hypic.MaxMambaCacheSize` caps per-rank recurrent state slots
(default 32; -1 selects auto-fit).

`HypicMethod(picMode="transition_rope_recompute", seamTokens=8)` exposes the
upstream HYPIC modes. `LinearKVMethod(recompRatio=0.2)` uses the branch's
paper-based LinearKV implementation and static per-chunk prefix selector;
its recurrent replay currently supports Qwen3.5/Qwen3.8 GDN models. Both accept
`fullPrefill=True` for the same-backend no-PIC control, which keeps ordinary
prefix caching enabled. Each case owns an isolated prepare/run/reset lifecycle.
The Task's input IDs pass through unchanged. Retained outputs are registered
from authoritative generated IDs with terminal stop IDs removed, outside
online TTFT/total time.
