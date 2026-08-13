
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

    def __init__(
        self,
        model_config,
        runtime_config
    ):
        """
        Initialize method.
        """

    def prepare(self, data):
        """
        Prepare reusable states.

        Examples:
        - prefill KV cache
        - build index
        - initialize metadata
        """

    def run(self, data) -> Result:
        """
        Run inference.
        """

    def reset(self):
        """
        Clear internal states.
        """
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

Engine controls the evaluation process.

```python
class Engine:

    def evaluate(
        self,
        method,
        task,
        metrics
    ):

        for case in task.cases():

            method.prepare(
                case.prepare_input
            )

            result = method.run(
                case.run_input
            )

            score = task.evaluate(
                result,
                case.metadata
            )

            metrics.update(
                result
            )

            method.reset()
```

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
│   ├── engine.py
│   ├── method.py
│   ├── task.py
│   ├── result.py
│   └── metrics.py
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