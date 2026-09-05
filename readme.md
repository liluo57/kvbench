
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

# Testing

The default suite is CPU-only and does not load a model or require a GPU:

```bash
pytest -q
```

Regression tests cover the core contracts, batch/Workload execution loop,
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
