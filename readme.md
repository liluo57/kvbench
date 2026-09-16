
**KVBench: A Unified Evaluation Framework for KV Cache Optimization Methods**

KVBench is a lightweight and extensible evaluation framework for KV cache optimization research.

The goal of KVBench is to provide a unified experimental protocol for evaluating different KV optimization methods.
    
KVBench focuses on **benchmarking and fair comparison**.

---

## Quick Start

1. Clone
```bash
git clone https://github.com/liluo57/kvbench.git
```

2. Environment Setup
```bash
pip install -r requirement.txt
```
> If you need FullPrefillVllm method, `vllm` is acquired.

> If you need CacheblendLmcache method, `vllm` and `lmcache` are acquired.

> If you need CacheblendRepo method, 
> 1. Clone [Cacheblend Repo](https://github.com/YaoJiayi/CacheBlend)
> 2. Setup `venv` in that repo.
> 3. Setup Cacheblend Repo according to their instructions.
> 4. Write the repo path in `config.yaml`.

> If you need Hypic method,
> 1. Clone [Hypic Repo](https://github.com/redai-studio/HYPIC)
> 2. Setup that repo according to their instructions.
> 3. Write the repo path in `config.yaml`.


3. Edit config.yaml and Main.py

4. Just do it!
```bash
python Main.py
```

## Design Philosophy

KV optimization methods are highly diverse.

KVBench abstracts the **evaluation workflow**:

```
                 KVBench Engine
                       |
        +--------------+--------------+
      Task / Case                  Method
          |                           |
       Workflow                 KV optimization
      (Actions)                     logic
          +-------------+-------------+
                        |
                      Result
                        |
              Task + system metrics
```

The framework only defines how experiments are executed.
The method implementation remains fully customizable.

## Repeated rollout evaluation

`RolloutMethod` decorates an existing Method and repeats each `Run` action
within its original Case. It returns one aggregate Result per input, so a task
with 64 Cases and 10 rollouts still reports 64 Cases:

```python
from methods import FullPrefillTransformer, RolloutMethod

method = RolloutMethod(
    base_method=FullPrefillTransformer(gpuNums=1),
    num_rollouts=10,
    keep_individual_results=False,
    keep_optional_metadata=False,
)
```

Numeric Result fields use population variance (`ddof=0`). The existing task
scorer is applied to each raw rollout and its scalar task metrics are retained
with `mean`, `variance`, `std`, and raw `samples`, then averaged once per
original sample. Rollout-aware runs add one record per Case under
`sample_results` in the pair report, with `sample_id`, `rollout_mean`,
`rollout_variance`, `rollout_std`, and a nested `rollout` summary. The
report-level task metrics also expose `mean`, `variance`, and `std`; the compact
`core.json` adds suffixed keys such as `accuracy_mean` and `accuracy_std`.

Performance metrics are always retained. Set `keep_individual_results=False`
to omit generated text and nested raw Results while retaining per-rollout
performance metric samples and task-metric samples. All raw `Result.metadata`
is optional; set `keep_optional_metadata=True` when diagnostic metadata such as
vLLM stop reasons is needed in nested raw Results. The default entry point
disables both optional payloads.

---

# Testing

The default suite is CPU-only and does not load a model or require a GPU:

```bash
pytest -q
```

Regression tests cover the core contracts, batch/Workflow execution loop,
metrics, prompt-reuse matching, task helpers, and scheduler failure paths.

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
