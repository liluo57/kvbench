
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
              Task                 Method
              |                      |
       benchmark workload      KV optimization logic
                       |
                     Result
                       |
                    Metrics
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

Interface:

```python
class Method:

    def __init__(self, gpuNums=1, perfWeight=1.0, ...):
        # Lightweight configuration only; no CUDA/model loading here.
        pass

    def Initialize(self, gpuIds):
        # Called by Engine in the spawned worker with exactly gpuNums ids.
        pass

    def Prepare(self, data):
        """
        Prepare reusable states.

        Examples:
        - prefill KV cache
        - build index
        - initialize metadata
        """

    def Run(self, data) -> Result:
        """
        Run inference.
        """

    def Reset(self):
        """
        Clear internal states.
        """

    def Close(self):
        """Release backend resources before process exit."""
```

A Method is responsible for:

- model interaction
- KV cache management
- optimization algorithm logic
- runtime-specific implementation

KVBench does not impose restrictions on how a Method manages KV cache internally.

---

# 2. Task

A `Task` defines what should be evaluated.

Examples:

- RULER
- SCBench
- LongBench
- custom long-context workloads

A task produces evaluation cases.

---

## Case

Each sample is represented as:

```python
@dataclass
class Case:
    prepare_input
    run_input
    metadata
```

Meaning:

|Field|Purpose|
|---|---|
|prepare_input|Data used for Method.prepare|
|run_input|Data used for Method.run|
|metadata|Information required for evaluation|

Example:

```python
Case(
    prepare_input=document,
    run_input=question,
    metadata={
        "answer": expected_answer,
        "needle_position": 0.8
    }
)
```

---

## Task Interface

```python
class Task:

    def cases(self):
        """
        Generate evaluation cases.
        """

        yield Case


    def evaluate(
        self,
        result,
        metadata
    ):
        """
        Evaluate correctness.

        Examples:
        - Exact Match
        - F1
        - ROUGE
        """
```

Task is responsible for:
- generating workloads
- checking correctness
    
Task is not responsible for:
- latency measurement
- memory measurement
    

---

# 3. Result

A Method returns a `Result`.

```python
@dataclass
class Result:

    output

    performance

    metadata
```

Example:

```python
Result(

    output="answer",

    performance={
        "ttft": 0.5,
        "latency": 2.0,
        "memory": "12GB"
    },

    metadata={
        "reuse_ratio": 0.8
    }
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

These belong to `Task.evaluate()`.

---

## Method Metrics

Method-specific measurements that describe *how the method ran*, e.g. the KV
reuse rate (share of the run stream served from cached KV). Unlike system
metrics, these are owned by the method itself: the method records the per-case
value in `Result.metadata` and declares which keys are method metrics via the
`Method.method_metrics` class attribute. The engine aggregates them (mean /
min / max / percentiles) into the report's `method_metrics` section, per
(method, task) pair.

A method with no method metrics (empty `method_metrics`) gets no
`method_metrics` section in the report.

---

## System Metrics

Examples:

- TTFT
- latency
- throughput
- GPU memory usage
- KV memory usage
    
These belong to `Metrics`.

Interface:

```python
class Metrics:

    def update(result):
        pass


    def summary():
        pass
```

---

# 5. Engine

Engine is a GPU scheduler and process supervisor. The coordinator never loads
a model. Every method instance is a spawned worker that owns exactly
``gpuNums`` GPUs, initializes once, and executes assigned tasks sequentially
with ``Reset`` at task boundaries.

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

├── core/
│   ├── Engine.py
│   ├── Worker.py
│   ├── Gpu.py
│   ├── tui.py
│   ├── Method.py
│   ├── Task.py
│   ├── Result.py
│   └── Metrics.py
│
├── methods/
│   ├── prefix_cache.py
│   ├── cacheblend.py
│   └── full_prefill.py
│
├── tasks/
│   ├── niah.py
│
├── metrics/
│
└── configs/
```

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
