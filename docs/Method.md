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
- `num_output_tokens`: number of tokens actually generated, including terminal EOS/stop tokens. Count the backend's raw generated IDs or completion-token metadata, not the decoded text or the stop-trimmed `output_ids` fragment.
- `total_time`: total time spent in `Run`, in seconds.

Model initialization and reusable state built in `Prepare` are excluded from TTFT. Per-request method metrics belong in `Result.metadata`; keys declared in the class attribute `method_metrics` are aggregated by the Engine into the report's `method_metrics` section.

`retainOutput[i]=True` requests the native generated KV matching
`Result.output_ids` as reusable state. Running a separate warmup/prefill on
the generated output does not satisfy this contract. Missing flags default to
false. Completing the final sampled token's missing KV row is part of the
original generation; completion, copying and registration count toward
`total_time`, after first-token latency has already been recorded. An empty
output fragment creates no retained chunk.

| Methods | `retainOutput=True` behavior |
| --- | --- |
| `NaiveTransformer`, `NaiveCacheblendRepo`, `CacheblendRepo`, `KVCommTransformer` | Preserve native output KV. |
| `EPIC`, `RandRecomputeTransformer`, `A3Repo`, `ProphetKV`, `CacheClip`, `KVPacket` | Preserve native output KV as a reusable output chunk, subject to each method's existing matching and model restrictions. |
| `HypicMethod`, `LinearKVMethod`, `CacheblendLmcache` | Raise `NotImplementedError` before generation; native output retention requires backend changes. |
| `FullPrefillTransformer`, `FullPrefillVllm`, `FullRecomputeTransformer`, `DependencyAnalysisMethod` | Ignore the flag and keep baseline behavior without an error. |
| `RolloutMethod` | Forward the flag to the base method for the final returned rollout only. |

`reuse_ratio` is cache-hit coverage: the number of canonical input token
positions matched to cached state divided by `n_input`, the complete RUN
prompt length. Count each matched occurrence, including repeated chunks.
Repairing a hit or replaying the final prompt token does not remove it from
the numerator. Artificial wrappers, separators and cache-isolation salts are
excluded. The Engine weights the aggregate mean by `n_input`, so it represents
total cache-hit tokens divided by total input tokens across RUN actions.
Full HF prefill has zero cache hits; a backend that uses native prefix caching
reports its actual hits.

Recomputation and reuse are separate concepts. A fixed `recompRatio` or
`recomputeRatio` is an algorithm parameter, not an aggregated method metric,
and `reuse_ratio` is not one minus that parameter. Methods that determine the
repair amount dynamically may report `recompute_ratio`. CacheClip reports
the number of repaired cached positions (including a mandatory final-token
replay) divided by cached input positions. Fresh prefix/query positions do not
enter that denominator. A CacheClip full-prefill fallback records
`reuse_ratio=0`, `recompute_ratio=1`, `full_prefill=True`, and `fallback_rate=1`;
ordinary reuse records `fallback_rate=0`. Both paths contribute to the report.

`MaxModelLen` bounds the context window. RUN prompts must leave room for at
least one generated token. Generation-budget handling is backend-specific:
the HF generation paths cap the budget at the remaining window, while
HYPIC/LinearKV reject requests whose input plus requested generation budget
exceeds the backend's permitted request length. Choose a Task budget that
passes the selected backend's validation. KVPacket counts inserted wrapper
positions as part of that window even when its physical KV cache is compressed.

## Existing methods

Refer to each class constructor for its implementation and configurable parameters.

| Method | Purpose |
| --- | --- |
| `FullPrefillTransformer`, `FullPrefillVllm` | Full-prefill baselines using Transformers and vLLM, respectively. |
| `NaiveTransformer`, `NaiveCacheblendRepo` | Controls that directly reuse cached state without repairing context changes. |
| `CacheblendRepo`, `CacheblendLmcache` | CacheBlend recomputation and fusion implementations; the former runs the bundled backend in `methods/Cacheblend` through a worker sharing the main project's Python environment. |
| `HypicMethod`, `LinearKVMethod` | Built-in hybrid position-independent reuse in `methods/hypic`; both accept token IDs and explicit segment boundaries. HYPIC defaults to transition/RoPE/seam repair; LinearKV uses the last matched GDN state with EPIC, CacheBlend, or ProphetKV selection. |
| `A3Repo` | A^3 with bundled precompute, selection, fusion and decoding; Llama/Mistral/Qwen2/dense Qwen3. |
| `ProphetKV` | ProphetKV implementation. |
| `CacheClip` | CPU auxiliary-token selection with sparse repair of primary KV caches. Prepared chunks and prompts remain authoritative primary-model token IDs; selector text and character spans are derived locally from each prepared ID chunk. |
| `KVCommTransformer` | KVCOMM-style cache reuse for multi-agent workloads. |
| `DependencyAnalysisMethod` | Computes cache dependency analysis metrics. |
| `RolloutMethod` | Decorator that repeats `Run` on a base Method and aggregates the results. |
| `KVPacket` | Reuses prepared token sequences through KV Packet wrappers and token-aware matching. |
| `RandRecomputeTransformer`, `FullRecomputeTransformer` | KVPacket package controls using the shared local HF/cache backend; random selective repair and full recomputation. |
| `EPIC` | Independent EPIC/LegoLink method: repairs a fixed prefix of relocated cached chunks using the shared local HF/cache backend. |

CacheClip's current primary-ID alignment supports fast ByteLevel tokenizers and has been validated with Qwen. `sharedPrefixText` optionally supplies an explicit warmup prefix: preparation tokenizes `P` independently, prefills `P_ids + D_ids`, then stores only the cache rows for `D_ids` and records the length of `P_ids`. Run never inserts or removes IDs for this prefix; without a matching prefix in the runtime prompt, this warmup is a calibration variant rather than strict paper-style prefix retention. The auxiliary selector view is decoded from primary IDs, so Unicode normalization may differ from the original source text.

CacheClip's sparse repair currently has decoder adapters for Llama, Qwen2 and
dense Qwen3 only. Tokenizer alignment support alone does not imply support for
another primary-model architecture.

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

KVPacket, its random/full-recompute controls, and EPIC use the same cache, RoPE,
HF execution and selective-recompute implementation in
`helpers/backends/CacheBackend`. The KVPacket package exports the control
classes alongside `KVPacket`; EPIC lives in `methods/EPIC.py` and imports only
the shared backend. These methods accept the Task's authoritative token IDs,
including reordered/repeated chunks and fresh spans between cached chunks.
Prepared caches are immutable. EPIC and Rand retain only the output rows,
normalized to local positions for later relocation. KVPacket registers native
outputs as unwrapped packets, separate from prepared wrapper packets. TTFT
includes matching, cache alignment, repair, fresh rows and first-token
selection; GPU precompute finishes before Prepare returns.

EPIC's `recomputeTokens` (default 32) repairs the first tokens of each cached
chunk except a chunk already at prompt position zero. Rand's `recomputeRatio`
(default 0.2) samples that same eligible cached region with an optional local
`seed`. Every fresh row and the final prompt row is forwarded. FullRecompute
ignores prepared state and recomputes the whole prompt. These controls require
no trained KVPacket wrapper and no external `kv_packet` checkout. The selective
kernel supports dense Llama/Mistral/Qwen2/Qwen3 and complete, uncompressed
standard-attention caches; cache relocation retains the backend's RoPE checks.
EPIC follows the baseline LegoLink policy and is not the authors' vLLM serving
system. Sources and adaptations are recorded in the shared backend's UPSTREAM.

A3Repo includes the upstream ragkv causal-LM precompute, attention selection,
selective recomputation and KV fusion in `methods/A3Repo`. Its decode and
matched vanilla paths use the shared `ModelConfig` sampling policy: greedy or
sampled selection, temperature, top-p/top-k/min-p/typical-p, repetition,
presence and frequency penalties, request seed, and EOS override. Penalties
use the complete prompt and generated-token history. EOS overrides replace
the model EOS; ragkv's BOS termination convention remains active. Unsupported
resolved sampling fields are rejected explicitly.
Run `./setup.py --a3` for the optional dependencies; its worker shares project
Python, with an optional `A3.Python` override. No external checkout is used.
Llama, Mistral, Qwen2 and dense Qwen3 are supported. The integrated Qwen3
projection bridge currently requires `head_dim == hidden_size / num_attention_heads`.
Prepared chunks must form a contiguous prefix followed by a fresh query; other
prompts fall back to full prefill through the same runtime. If the entire prompt
is covered by cached chunks, its final chunk is kept fresh for query-based
selection (or the whole prompt is recomputed when only one chunk matches). Task generation
budgets are sent per request. The pinned source commit, retained files and
adaptations are recorded in `methods/A3Repo/UPSTREAM`.
Prepare finishes queued CUDA work before returning, and TTFT is recorded
after the first generated token can be read.
A3 output retention completes the final token's KV and converts native
RoPE-encoded keys back to the pre-RoPE format used by its chunk store, without
output prefill. It supports static, invertible RoPE and retains the same
contiguous-prefix matching restriction. ProphetKV stores its generated raw
K/V rows directly. CacheClip keeps native primary-model output KV and builds
the CPU auxiliary selector view for that chunk; this auxiliary work counts
toward `total_time`.

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
its recurrent replay requires a compatible Qwen3_5 GDN architecture. Both accept
`fullPrefill=True` for the same-backend no-PIC control, which keeps ordinary
prefix caching enabled. Each case owns an isolated prepare/run/reset lifecycle.
The Task's input IDs pass through unchanged. Both reject `retainOutput=True`,
including with `fullPrefill=True`: their backend releases generated decode
state, and warming up the output again would not preserve native output KV.

### LinearKV selectors on the bundled HYPIC backend

```python
from methods import LinearKVMethod

methods = [
    LinearKVMethod(gpuNums=2, selector="epic", recompRatio=0.2),
    LinearKVMethod(gpuNums=2, selector="cacheblend", recompRatio=0.2),
    LinearKVMethod(gpuNums=2, selector="prophetkv", recompRatio=0.2),
]
```

These adapters share the existing LinearKV last-block (LB) initializer and
selective replay. At each GDN layer, LB restores the last matched document's
independently cached recurrent state and paired convolution history. Fresh
positions, including Query, always run; selected cached positions are replayed
in their original prompt order. FA reuses the other cached KV rows with the
backend's position relocation. ProphetKV requires a fresh terminal Query segment and rejects a cached terminal segment.

| Selector | Selection and budget |
| --- | --- |
| `epic` (default) | Repair each matched chunk's prefix. Partial repair excludes the chunk beginning at prompt position zero, without reallocating its budget. The count is `floor(r * chunk_length)`, at least one for a positive budget. |
| `cacheblend` | Probe the full prompt through the first FA layer, then rank squared differences between current and cached V over all matched positions. `selectorLayer` optionally chooses an absolute FA layer index. |
| `prophetkv` | Probe fresh positions against the document KV and LB state. Score only the terminal Query's context attention, average over heads and FA layers, then select one shared set of document positions. |

Dynamic selectors use a global `floor(r * matched_tokens)` budget (at least one
when positive), with ties resolved by original position. All TP ranks reduce
scores before selection. A ratio of zero skips document repair; a ratio of one
replays all matched rows, including the first chunk, and bypasses probing.
Full replay still uses LB initialization; it is not ordinary full prefill.

Probe changes to request-private recurrent state are restored before formal
replay, including on failure. Prepared document state and public KV stay frozen.
The adapter reports `selector`, `target_recompute_ratio`, `recompute_ratio`,
`num_recomputed_tokens`, and `num_matched_tokens` separately from cache-hit
coverage. In particular, EPIC's first-chunk exclusion can make its actual ratio
lower than the requested ratio.

The bundled LinearKV model path requires a `Qwen3_5` architecture, including
compatible Qwen3.6 checkpoints. OLMo is not supported by this backend. Dynamic
probes support one request and tensor parallelism only (`PP=DP=1`), disable
scheduler overlap, and reject embedding overrides and speculative batches.
PD multiplexing / split prefill is explicitly rejected for dynamic selectors.
They require an explicit ratio rather than a fixed token count.

CacheBlend uses a separate probe and replay here; this does not implement its
author runtime's fused layer-wise pruning optimization. The selector adapters
support method comparisons within KVBench's input/scoring protocol and do not
by themselves establish complete reproduction of the original papers.

CPU regressions: `python -m pytest -q tests/methods/test_hypic.py`.
CUDA/model validation remains separate from these dependency-isolated tests.
