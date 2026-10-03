
**KVBench: A Unified Evaluation Framework for KV Cache Optimization Methods**

KVBench is a lightweight and extensible evaluation framework for KV cache optimization research.

The goal of KVBench is to provide a unified experimental protocol for evaluating different KV optimization methods.
    
KVBench focuses on **benchmarking and fair comparison**.

---

## Quick Start

1. Clone

2. Environment Setup
```bash
pip install -r requirement.txt
```
> If you need FullPrefillVllm method, `vllm` is acquired.

> If you need CacheblendLmcache method, `vllm` and `lmcache` are acquired.

> If you need CacheblendRepo or NaiveCacheblendRepo, install `uv` and a CUDA
> toolkit, then run `./setup.py --cacheblend` (or `python setup.py --cacheblend`).
> This installs the project and optional CacheBlend dependencies into the current
> Python environment, then builds the CUDA extensions. Use Python 3.12 or 3.13;
> the shared stack uses Torch 2.13.0, Transformers 4.57.6 and xFormers 0.0.35.
> Running it again checks the installation and skips a ready environment.
> Use `--rebuild` after changing native CUDA/C++ sources. On older Linux x86-64
> hosts, setup downloads GCC 12 build tools into `.cache/cacheblend`; you can
> also supply `CC`, `CXX` and `CUDAHOSTCXX` yourself.
> The bundled backend supports Llama/Mistral and dense Qwen3, with the private
> package name `kvbench_cacheblend` to coexist with installed vLLM. The worker
> uses the same Python as KVBench. No additional CacheBlend checkout is required.
> `--python` selects an existing project interpreter for installation;
> `Cacheblend.Python` optionally overrides the runtime worker interpreter.
> The upgraded stack preserves the worker protocol and reuse metrics, but
> floating-point differences can change recomputation selections and generated
> text. Timings from the former Torch 2.2 environment need to be measured again.

> KVPacket's core runtime is vendored in this repository and uses the shared
> Python 3.12 or 3.13, Torch 2.13.0, Transformers 4.57.6 and PyYAML 6.0.3 environment.
> Run `./setup.py --kvpacket` to check it offline; this does not install or
> replace packages. Before running KVPacket, set `KVPacket.WrapperCheckpointPath`
> to a trained wrapper checkpoint compatible with `ModelPath`. Random wrappers
> are disabled by default; `AllowUntrainedWrapper: true` is for smoke tests only.
> Legacy tensor-only wrappers also require `KVPacket.WrapperModelId` to equal the
> exact runtime Hugging Face `config._name_or_path` value (Hub ID or local directory).
> This is an explicit source assertion: legacy files contain no revision or weight
> digest, so KVBench also checks tensor shapes but cannot prove the source weights.

> A3Repo's [ragkv implementation](https://github.com/yczhounlp/ragkv) is bundled
> under `methods/A3Repo`. Run `./setup.py --a3` to install its optional dependencies
> in the project Python environment. The isolated worker supports Llama, Mistral,
> Qwen2 and dense Qwen3; it defaults to the current interpreter, with an optional
> `A3.Python` override. No external repository path or separate venv is required.
> Prepared chunks must form a contiguous prompt prefix followed by a fresh query;
> other prompts use full prefill. Decoding follows upstream's greedy policy.
> The integrated Qwen3 bridge requires `head_dim == hidden_size / num_attention_heads`.

> HypicMethod and LinearKVMethod are bundled under `methods/hypic`.
> Run `./setup.py --hypic` to install their dependencies and build CUDA ops
> in the project environment.

3. Prepare datasets
```bash
python scripts/PrepareDataset.py
```

The benchmark data is intentionally not stored in git.  The command reads all
dataset sources, splits, revisions, and RULER generation parameters from
`config.yaml`, prepares complete declared splits, and installs them under
`DatasetPath`.


4. Set `ModelPath` and any optional third-party repository paths in `config.yaml`.
   The default model value is a Hugging Face model id; local checkpoint paths
   are also supported.

5. Run the benchmark:
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
