# Run LinearKV with different selectors

This integration runs LinearKV on KVBench's bundled hybrid-attention backend.
Use `epic`, `cacheblend`, or `prophetkv` to choose which cached document tokens
are recomputed. All three use the same last-block (LB) initializer and replay
path. The independent `EPIC`, `CacheblendRepo`, and `ProphetKV` classes are
different methods; use `LinearKVMethod(selector=...)` for this comparison.

The implementation targets compatible `Qwen3_5` GDN architectures, including
Qwen3.6 checkpoints. It accepts unquantized FP16/BF16 weights on NVIDIA GPUs.
OLMo, EX initialization, and a separate Transformers runtime are outside this
integration. Source and license information is retained in
[methods/hypic/UPSTREAM](../methods/hypic/UPSTREAM).

## Install the bundled backend

Run these commands from the repository root, with a CUDA 12.9 or newer
development toolkit installed. Set `CUDA_HOME` to that toolkit's directory;
the CUDA runtime bundled with a PyTorch wheel does not include the compiler.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install uv
python setup.py --hypic
```

Setup installs the repository's common and HYPIC requirements and builds the
retained CUDA operations for the active PyTorch ABI. The common requirements
pin PyTorch 2.13.0 and Transformers 4.57.6. Use the PyTorch CUDA wheel build
matching the target machine. This path needs no external HYPIC checkout or
separately installed SGLang, vLLM, or LMCache.

## Set the model and GPU pool

Edit the existing sections in `config.yaml`; keep the other settings:

```yaml
ModelPath: /path/to/Qwen3.6-27B
MaxModelLen: 32768
ModelConfig:
  mode: greedy
Engine:
  AvailableGpuIds: [0]
  BatchSize: 1
  Tui: false
  TuiWaitForQuit: false
```

Replace the checkpoint path with a complete local model directory or supported
Hugging Face model ID. The repository's default Mistral checkpoint is not
supported by LinearKV. The chosen GPUs must hold the weights, KV, recurrent
cache and prefill workspace. To use tensor parallelism, list the available GPU
IDs, such as `[0, 1]`, and pass `--gpu-nums 2` to the runner. The default is
one GPU per method; the Engine schedules methods over the configured pool.

## Start with a small smoke run

```bash
python scripts/RunLinearKV.py --dry-run
python scripts/RunLinearKV.py --task fresh-gap --samples 1
```

Dry-run prints the model, GPU pool and method matrix without loading a dataset,
model or CUDA backend. The fresh-gap task needs no downloaded benchmark data:
it reuses two documents around a fresh intermediate span, followed by a fresh
question. The default run compares all three selectors at ratio 0.2 with
naive reuse and Full controls. Each configuration loads a model, so start with
one case. A smoke run checks execution; its score or first-use JIT timing is
not a benchmark-quality or steady-state latency result.

To select one strategy or sweep ratios:

```bash
python scripts/RunLinearKV.py --selectors prophetkv --ratios 0.2
python scripts/RunLinearKV.py \
  --selectors epic cacheblend prophetkv --ratios 0.05 0.1 0.2 --samples 5
```

Naive reuse and Full are always included once. Configuration-wide model paths,
GPU IDs, context limits and sampling remain in `config.yaml`, which workers
also read. `Main.py` and its broader benchmark matrix do not need to change.

## Use the existing QA tasks

Prepare the declared datasets with `python scripts/PrepareDataset.py`, following
[Reproduction.md](Reproduction.md#2-environment-and-dataset-preparation).
The same runner then accepts `--task hotpotqa`, `--task musique`, or `--task 2wiki`:

```bash
python scripts/RunLinearKV.py --task hotpotqa --samples 5 --ratios 0.2
```

The runner keeps each Task's existing tokenization, chunking, sample-length
filter, generation budget and scorer. Use the same accepted cases and decoding
settings when comparing methods. `--samples` is an upper limit; inspect the
recorded case counts before calculating relative quality or TTFT against Full.

## Interpret the results

The command prints the result directory under `Engine.OutputRoot`. Its
`results/full.json` contains full reports, `results/core.json` contains compact
reports, and `results/failures.json` lists failed configurations. The command
exits nonzero if any configuration fails or the run does not complete.

| Setting or metric | Meaning |
| --- | --- |
| LB initialization | Restore the last matched document's recurrent state and paired convolution history, assemble and relocate FA KV, then replay selected positions in original order. |
| `epic` | Per-chunk prefix repair. Exclude the chunk beginning at prompt position zero at every ratio, including 1, without reallocating its budget. A first hit after a fresh prefix remains eligible. |
| `cacheblend` | Probe the full prompt through one FA layer and rank squared current-versus-cached V differences. `selectorLayer` optionally selects an absolute FA layer index through the Method API. |
| `prophetkv` | Probe fresh spans against assembled document KV and LB state. Rank frozen context keys using terminal-query attention averaged over heads and FA layers. |
| `recompRatio=0` / naive | Skip cached-token repair and dynamic probes; fresh spans and Query still execute. |
| `recompRatio=1` | Replay eligible cached rows while retaining LB initialization. This is not Full; EPIC still excludes the position-zero chunk. |
| `fullPrefill=True` | Same-backend no-PIC control. Ordinary prefix caching remains enabled; reset between cold comparison cases. |
| `reuse_ratio` | Matched input coverage, including rows selected for repair. |
| `num_matched_tokens`, `num_recomputed_tokens`, `recompute_ratio` | Matched document tokens, final selected repair count, and their ratio. These exclude mandatory fresh/query rows and probe work. |

Dynamic selectors use a global budget; EPIC uses each chunk's budget, so equal
requested ratios can give different actual ratios. Probing counts toward online
TTFT and total time. Keep enough `Hypic.MaxMambaCacheSize` slots for the prepared
chunks and active request, and check observed cache coverage before interpreting
a run as reuse of the intended context.

Dynamic probes support one request with TP; PP, DP, split prefill, embedding
overrides, and speculative decoding are rejected. CacheBlend uses a separate
probe and fixed-set replay, rather than its author runtime's fused layer-wise
pruning. This integration enables controlled selector comparisons within
KVBench; it does not establish complete paper reproduction.

## Check the integration

```bash
python -m pytest -q tests/methods/test_hypic.py \
  tests/backends/test_prompt.py tests/scripts/test_run_linearkv.py
```

CPU tests exercise selection, state restoration and routing with isolated
runtime definitions and test doubles. Enable `--run-native` after installing
the compiled backend for the additional native checks. Real-checkpoint and
GPU validation remain separate from both the dry-run and CPU test results.
