
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

3. Prepare datasets
```bash
python scripts/PrepareDataset.py
```

The benchmark data is intentionally not stored in git.  The command reads all
dataset sources, splits, revisions, and RULER generation parameters from
`config.yaml`, prepares complete declared splits, and installs them under
`DatasetPath`. Sources use `test` where labeled test data is public. MuSiQue,
2WikiMultiHopQA, HotpotQA, and TriviaQA retain their labeled dev/validation
splits because their official test answers are private; each exception is
documented in `SplitReason`.

RULER uses NVIDIA's pinned official generators for `niah_single_2`, `vt`, and
`cwe`, explicitly passing `--subset test`. Source code and the LFS word list
are checksum-verified and cached under `.cache/ruler/`. Every official essay
URL is fetched with retries and cached before the unchanged upstream parser
builds the NIAH corpus. Missing sources or parser failures stop preparation;
partial corpora are never installed. HTTP is upgraded to HTTPS, and GitHub
raw links use the equivalent `raw.githubusercontent.com` transport. Install
the RULER dependencies in `requirement.txt` before preparation; the first run
also downloads NLTK tokenizer resources. Original generator output is kept
unchanged in the local `*_len*.jsonl` files, with revision, split, parameters,
asset hashes, and output hashes in `data/ruler/provenance.json`. Real dataset
sources are recorded in each dataset's `_metadata/source.json`.
Download counts, raw source hashes, and the complete-corpus status are recorded
in RULER provenance, with the parser log in the cache.

Run `python scripts/PrepareDataset.py --dry-run` to inspect the source/split
plan. Rerun preparation after changing it; existing data is not regenerated
by running `Main.py`. Preparation records every output hash and the producing
configuration/code in `data/preparation.json`. Check the installed data without
network access with `python scripts/PrepareDataset.py --verify`. Previous
datasets are retained under `.data.backups/`; installation failures roll back
all replaced directories.


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

---

# Testing

The default suite is CPU-only and does not download models or datasets. Numerical
tests construct tiny models with random weights and tokenizers in memory:

```bash
pytest -q
```

Tests are organized by responsibility under `tests/`: `methods`, `backends`,
`core`, `workflows`, `tasks`, `helpers`, and `scripts`. Each method's suite includes
its output-retention, reuse, sampling, timing, and lifecycle regressions.
Shared fixtures live in `tests/conftest.py` and reusable doubles in `tests/support`.

Optional compiled-backend and CUDA tests require explicit opt-in:

```bash
python -m pytest -q --run-native -m native
python -m pytest -q --run-gpu -m gpu
```

See [tests/README.md](tests/README.md) for focused runs and test conventions.

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
