
**KVBench: A Unified Evaluation Framework for KV Cache Optimization Methods**

KVBench is a lightweight and extensible evaluation framework for KV cache optimization research.

The goal of KVBench is to provide a unified experimental protocol for evaluating different KV optimization methods, including:

- KV reuse methods
- KV cache compression methods
- KV quantization methods
- KV eviction methods
- KV offloading methods
    
KVBench focuses on **benchmarking and fair comparison**, rather than implementing a new KV cache runtime.

---

## Design Philosophy

KV optimization methods are highly diverse.

Different methods may have completely different internal implementations:

- prefix caching
- cache blending
- token selection
- KV compression
- quantization
- memory management

Therefore, KVBench does **not** attempt to unify internal KV cache representations.

Instead, KVBench abstracts the **evaluation workflow**:

```
                 KVBench Engine
                       |
        +--------------+--------------+
      Task / Case                  Method
          |                           |
       Workload                 KV optimization
      (Actions)                     logic
          +-------------+-------------+
                        |
                      Result
                        |
              Task + system metrics
```

The framework only defines how experiments are executed.
The method implementation remains fully customizable.

---

# Core Components

## 1. Method

A `Method` represents a KV optimization algorithm.
Examples:

- Prefix Cache
- CacheBlend
- SnapKV
- KVQuant
- PyramidKV
    

The framework treats methods as black boxes.

The base interface is batch-oriented. Constructors are lightweight because the
Engine may pickle one configuration into several spawned workers:

```python
class Method:

    name = "method"
    tag = None
    method_metrics = ()
    maxCaseBatchSize = None

    def __init__(
        self,
        *,
        gpuNums=1,
        perfWeight=1.0,
        maxGpuNums=None,
        tag=None,
    ):
        # Lightweight configuration only; no CUDA/model loading here.
        pass

    def Initialize(self, gpuIds):
        # Called by Engine in the spawned worker with exactly gpuNums ids.
        pass

    @property
    def Label(self):
        # name or name(tag), used in reports.
        ...

    def Prepare(self, data: list[list[str]]) -> None:
        # data[i] contains the reusable segments for one PREPARE Action.
        ...

    def Run(
        self,
        data: list[str],
        retainOutput: list[bool] | None = None,
    ) -> list[Result]:
        # Return one Result per prompt, preserving input order.
        ...

    def Reset(self) -> None:
        # Clear state after one batch.
        ...

    def Close(self) -> None:
        # Release backend resources before worker exit.
        ...
```

`retainOutput[i]` is a future-reuse hint. A method may register the generated
output as another reusable segment or ignore the hint. Method-specific numeric
metadata is written to `Result.metadata`; only keys declared by
`method_metrics` are aggregated into the method-metrics report.

`maxCaseBatchSize` is a Method capability limit. `None` accepts the Engine's
requested `batchSize`; a request-sequential, stateful method sets it to `1` so
each Case completes and is reset before the next Case starts. The effective
size is `min(Engine.batchSize, Method.maxCaseBatchSize)` and is recorded in the
run manifest.

A Method is responsible for:

- model interaction
- KV cache management
- optimization algorithm logic
- runtime-specific implementation

KVBench does not impose restrictions on how a Method manages KV cache internally.

---

# 2. Task

A `Task` defines what should be evaluated.

Included tasks cover RULER NIAH / variable tracking / common-words extraction,
the Musique / WikimQA / Samsum knowledge-base workloads, FreshGap, and the
KVComm MMLU / GSM8K / HumanEval / Copy multi-agent workloads.

A task produces evaluation cases.

---

## Case

Each sample is represented as:

```python
@dataclass
class Case:
    input: Any
    workload: Workload
    metadata: dict[str, Any]
```

Meaning:

|Field|Purpose|
|---|---|
|`input`|Raw benchmark data whose type is defined by the Workload|
|`workload`|Stateful policy that turns the input and prior results into Actions|
|`metadata`|Information required by `Task.Evaluate`|

Example:

```python
data = RAGInput(
    prepare_input=[document_chunk],
    run_input=complete_prompt,
)
Case(
    input=data,
    workload=RAGWorkload(case_id=sample_id, data=data),
    metadata={"answer": expected_answer},
)
```

`prepare_input` and `run_input` are fields of the built-in `RAGInput`, not
fields of `Case`. Other Workloads can define different input types and produce
multiple rounds of execution dynamically.

## Workload Interface

```python
class Workload:

    def next(self) -> list[Action] | None:
        # Return the next PREPARE or RUN step, or None when done.
        ...

    def observe(self, results: list[ActionResult]) -> None:
        # Update state and choose later actions from prior outputs.
        ...

    @property
    def finished(self) -> bool:
        ...
```

Every Action in one step must have the same `ActionKind`. A `PREPARE` Action
holds a list of reusable text segments. A `RUN` Action holds one complete
prompt plus a `retainOutput` hint. This supports both the fixed RAG
Prepare→Run path and output-dependent multi-agent conversations.

---

## Task Interface

```python
class Task:

    def Cases(self) -> Iterator[Case]:
        # Generate one Case per benchmark sample.
        ...

    def Evaluate(
        self,
        result: Result,
        metadata: dict[str, Any],
    ) -> dict[str, float]:
        # Return task metric name -> score.
        ...
```

Task is responsible for:
- generating workloads
- checking correctness
    
Task is not responsible for:
- latency measurement
- memory measurement
    

---

# 3. Result

`Method.Run` returns one `Result` for every input prompt in its batch.

```python
@dataclass
class Result:
    output: Any = None
    performance: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
```

Example:

```python
result = Result(
    output="answer",
    performance={
        "ttft": 0.5,
        "num_output_tokens": 16,
        "total_time": 2.0,
    },
    metadata={"reuse_ratio": 0.8},
)
```

---

# 4. Metrics

Metrics measure system-level performance.
KVBench separates:

## Task Metrics

Examples:
- Accuracy
- F1
- Exact Match

These belong to `Task.Evaluate()`.

---

## Method Metrics

Method-specific measurements describe *how the method ran*, e.g. the KV reuse
rate (share of the run stream served from cached KV). Unlike system metrics,
these are owned by the method itself: the method records a per-RUN value in
`Result.metadata` and declares which keys are method metrics via the
`Method.method_metrics` class attribute. The engine aggregates them (mean /
min / max / percentiles) into the report's `method_metrics` section, per
(method, task) pair.

A method with no declared method metrics (empty `method_metrics`) gets no
`method_metrics` section in the report, even if it keeps undeclared diagnostic
values in `Result.metadata`.

---

## System Metrics

The included system metrics are TTFT and throughput. Additional `Metric`
subclasses can consume fields recorded in `Result.performance`.

Interface:

```python
class Metric:

    def Update(self, result: Result) -> None:
        # Called once per RUN Action, not once per Case.
        ...

    def Summary(self) -> dict[str, Any]:
        ...

    def Reset(self) -> None:
        # Called before one (method, task) pair.
        ...
```

---

# 5. Engine

Engine is a GPU scheduler and process supervisor. `Evaluate(tasks, methods,
metrics)` evaluates the Cartesian product of methods and tasks. The coordinator
never loads a model. Each spawned worker owns one initialized Method and exactly
`gpuNums` GPUs; a method can have several workers, while each worker handles its
assigned tasks sequentially.

```python
methods = [
    CacheblendRepo(gpuNums=1, perfWeight=3),
    FullPrefillVllm(gpuNums=2, perfWeight=1),
]

engine = Engine(
    availableGpuIds="auto",
    batchSize=4,
    initializeTimeout=1800,
    taskTimeout=3600,
    shutdownGracePeriod=30,
    gpuReleaseTimeout=30,
    gpuReleaseStableSeconds=1,
    gpuReleaseMemoryToleranceMiB=256,
    pairRetries=1,
)
report = engine.Evaluate(tasks, methods, metrics)
```

## HYPIC with Qwen3.8

`HypicMethod` drives the HYPIC SGLang fork configured by `Hypic.RepoPath` and
uses the global `ModelPath`. Qwen3.8-27B reports the Qwen3.5 dense runtime
architecture and a hybrid 3-linear/1-full attention layer pattern, so the
adapter uses HYPIC's native PIC path:

```python
from methods import HypicMethod

methods = [
    HypicMethod(
        gpuNums=1,
        maxNewTokens=64,
        maxModelLen=25600,
        memFractionStatic=0.80,
        picMode="addition",
    ),
    HypicMethod(
        gpuNums=1,
        maxNewTokens=64,
        maxModelLen=25600,
        memFractionStatic=0.80,
        fullPrefill=True,
        tag="full_prefill",
    ),
]
```

`Prepare` caches the declared text segments; `Run` detects them even when they
are reordered or separated by fresh text. Results expose actual
`num_cached_tokens` and `reuse_ratio` reported by HYPIC. HYPIC v1 accepts one
text request at a time, so this method declares `maxCaseBatchSize = 1` even if
the Engine is configured with a larger batch. A direct and an end-to-end worker
smoke test are available through `scripts/SmokeHypic.py`.
HYPIC's last PIC segment is reserved for the fresh query row that seeds decode;
therefore `FreshGap` represents the scenario as `A + B + C + Q`, with `A` and
`C` prepared, `B` fresh, and the small `Q` tail intentionally unprepared. This
is what allows the benchmark to measure reuse of C without invalidating the
generation query.
The `hypic(full_prefill)` control uses the same HYPIC runtime and model but
disables both PIC and the ordinary radix prefix cache, and skips `Prepare`.

Within one `(method, task)` pair, `Task.Cases()` is split into batches of at
most that method's effective Case batch size. The simplified inner loop is:

```python
effectiveBatchSize = method.EffectiveBatchSize(batchSize)
for batch in batched(task.Cases(), effectiveBatchSize):
    workloads = [case.workload for case in batch]

    while unfinished(workloads):
        actions = collect_next_actions(workloads)
        assert one_action_kind(actions)

        if actions[0].kind is PREPARE:
            method.Prepare([action.data for action in actions])
            results = empty_action_results(actions)
        else:
            run_results = method.Run(
                [action.data for action in actions],
                [action.retainOutput for action in actions],
            )
            results = wrap_action_results(actions, run_results)
            for result in run_results:
                for metric in metrics:
                    metric.Update(result)

        deliver_results_to_workloads(results)

    for case in batch:
        task.Evaluate(final_run_result(case), case.metadata)
    method.Reset()
```

There can be multiple `RUN` Actions for one Case, as in the multi-agent
Workload. System and declared method metrics consume every RUN result, whereas
Task correctness is evaluated once from the Case's final RUN result. `Reset()`
is called after each batch.

``availableGpuIds="auto"`` uses NVML and selects a startup snapshot of GPUs
whose memory use is strictly below 30% and utilization is strictly below 5%.
An explicit list such as ``[0, 2, 3]`` can be used instead.
GPU discovery uses ``nvidia-ml-py`` (imported as ``pynvml``), and the dashboard
uses ``rich``.

``perfWeight`` is a positive relative estimate of per-task runtime. Extra
instances are assigned greedily by estimated marginal speedup per GPU. One
method can therefore have multiple instances, while each individual instance
keeps its tasks sequential.

Pair exceptions are retried once in the same worker by default. A second
failure marks only that pair failed. A task timeout or process crash requires
replacing the worker because a blocked CUDA call cannot be safely interrupted;
the same pair retry budget still applies. Any method initialization failure
stops the complete benchmark.

After a worker exits, its GPUs enter a ``cooling`` state. Engine sweeps the
worker's complete process group and uses NVML to wait until memory returns to
the benchmark-start baseline (plus the configured tolerance), with no new
compute PID, for ``gpuReleaseStableSeconds`` continuously before those GPUs can
be scheduled again. Free GPUs are checked once more immediately before worker
dispatch. External GPU users are never killed; their PIDs are recorded in the
event log. Failure to obtain a stable GPU within ``gpuReleaseTimeout`` is
reported separately as ``resource_release_failed``.

Because workers use multiprocessing ``spawn``, Method constructor state, Task
objects, and Metric objects must be pickleable. GPU engines, subprocesses,
threads, and other runtime-only objects belong in ``Method.Initialize`` rather
than ``__init__``.

Each run is checkpointed under ``outputs/<timestamp>-<pid>/``:

```text
manifest.json              run configuration and lifecycle
events.jsonl               scheduler/process event stream
logs/<method>/             instance/runtime and per-pair attempt stdout/stderr
pairs/<method>/<task>.json incremental pair checkpoints
results/full.json          complete aggregate report
results/core.json          flattened core metrics
results/timing.json        initialization/task/close/total wall times
results/failures.json      isolated failure diagnostics
```

The Rich TUI defaults to the core view and supports Core, Full, Timing,
Schedule/GPU, Logs, and Failures views. Pressing ``q`` while a benchmark is
running requests a clean cancellation. After a benchmark finishes, the final
dashboard remains interactive until ``q`` is pressed. Non-interactive runs
still exit automatically, and the TUI falls back to plain coordinator output
when stdout or keyboard input is unavailable.

---

# Extension Principles

## No forced KV abstraction

KVBench intentionally does not define a universal KVState or KVBackend in the core API.

Reason:

KV cache implementations differ significantly:

- HuggingFace DynamicCache
    
- vLLM PagedAttention
    
- SGLang Radix Cache
    
- TensorRT-LLM KV manager
    
- custom accelerators
    

A benchmark framework should not become a KV runtime.

If future analysis requires KV introspection, optional extensions may be introduced.

---

# Project Structure

```
KVBench/

├── Main.py
├── GenerateRuler.py
├── config.yaml
├── readme.md
├── core/
│   ├── Config.py
│   ├── Engine.py
│   ├── Worker.py
│   ├── tui.py
│   ├── Method.py
│   ├── Task.py
│   ├── Workload.py
│   ├── Result.py
│   └── Metrics.py
│
├── methods/
│   ├── CacheblendLmcache.py
│   ├── CacheblendRepo.py
│   ├── FullPrefill.py
│   ├── Hypic.py
│   └── Naive.py
│
├── tasks/
│   ├── Niah.py
│   ├── Vt.py
│   ├── Cwe.py
│   ├── Musique.py
│   ├── WikimQA.py
│   ├── Samsum.py
│   ├── FreshGap.py
│   ├── KVCommTasks.py
│   ├── TemplateHelper.py
│   └── bases/
│
├── workload/
│   ├── RAGWorkload.py
│   └── MultiAgentFullConnectionWorkload.py
│
├── metrics/
│   ├── Ttft.py
│   └── Throughput.py
│
├── helpers/
│   ├── Gpu.py
│   ├── Prompt.py
│   ├── VllmHelper.py
│   ├── TransformersHelper.py
│   ├── VllmCacheblendPatches.py
│   ├── CacheblendRepoHelper.py
│   └── Qwen3ForCacheBlendRepo.py
│
└── tests/
```

---

# Testing

The default suite is CPU-only and does not load a model or require a GPU:

```bash
pytest -q
```

Regression tests cover the core contracts, batch/Workload execution loop,
metrics, prompt-reuse matching, task helpers, and scheduler failure paths.

---

# Future Directions

Possible future extensions:

- KV cache visualization
    
- KV reuse profiling
    
- automatic benchmark reports
    
- leaderboard support
    
- distributed evaluation
    
- optional KV introspection APIs
    

---

# Summary

KVBench provides a unified protocol for evaluating KV cache optimization methods.

The key abstraction is:

```
Task defines what to test.

Method defines how to optimize.

Engine defines how to execute.

Metrics defines how to measure.
```

KVBench aims to make KV optimization research easier to reproduce, compare, and extend.
