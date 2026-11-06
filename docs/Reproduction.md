# Paper Reproduction Guide

This guide corresponds to the KVBench paper. It describes how to reproduce the paper's experimental setup and reported metrics with this checkout. No reproduction runs were performed while writing this document; GPU backends and external method repositories must be installed on the target machines.

## 1. Evaluation protocol

For every model–method–task configuration, the paper evaluates a Full Prefill baseline with the same model, hardware, examples, and decoding settings. Unless otherwise stated, each task uses 64 samples and all latency experiments use `BatchSize=1`.

Task quality uses each task's native metric; inference efficiency uses end-to-end TTFT. For each reuse configuration, calculate the following against its paired Full Prefill run:

```text
RQ = Q_reuse / Q_full       # higher is better
RT = TTFT_reuse / TTFT_full # lower is better
```

TTFT includes request-dependent cache lookup/loading, assembly, transformation, selection, repair, recomputation, and transfers during `Run`. Model initialization and reusable state built in `Prepare` are excluded. The paper primarily reports medians across the RAG-style tasks in the main text and per-task results in the appendix. Use the Full Prefill baseline for the same model, examples, and configuration as the denominator.

## 2. Environment and dataset preparation

1. Install the common dependencies from the repository root:

   ```bash
   pip install -r requirement.txt
   ```

2. Set the model checkpoint, `MaxModelLen`, sampling mode, GPU pool, and dataset path in `config.yaml`. The paper uses 64 samples per configuration and batch size 1 for the standard RAG and multi-agent experiments. Skill reuse has separate context-length and decoding settings; see Section 4.

3. Download or generate the datasets declared in the configuration:

   ```bash
   python scripts/PrepareDataset.py
   ```

   The current configuration pins source revisions, splits, and expected row counts. SAMSum, GovReport, Multi-News, MMLU, GSM8K, and HumanEval use `test`. MuSiQue and 2WikiMultiHopQA use `dev`; HotpotQA (`distractor`) and TriviaQA (`rc`) use `validation`. Those four exceptions retain public reference answers for local F1/EM; their official test answers are private. Each exception has a `SplitReason` in `config.yaml`. The preparer rejects non-test sources without a documented exception and records each real source in `<dataset>/_metadata/source.json`.

   RULER runs the unmodified official NVIDIA generators at the commit and source-archive checksum in `DatasetPreparation.Ruler`. The local NIAH task maps to official `niah_single_2` (essay haystack), VT to `vt`, and CWE to `cwe`. Each generator receives `--subset test`, the configured HF tokenizer, random seed 42, lengths 4096 and 8192, and 500 samples per length. KVBench loads the official task definitions/templates and preserves the generated JSONL bytes. The source archive, LFS word list, essay corpus, and NLTK resources are cached under `.cache/ruler/`. The official essay downloader is used with HTTPS transport. `ruler/provenance.json` records the source revision, split, tokenizer, parameters, asset hashes, and output hashes.

   This replaces KVBench's previous local implementations of the three synthetic generators, so existing RULER files must be regenerated. Install the updated `requirement.txt`, inspect `python scripts/PrepareDataset.py --dry-run`, then rerun preparation. Do not mix old and new generated data within an experiment. `maxSamples=64` in `Main.py` caps the number of examples used. The RULER loader combines files in sorted filename order before truncation, so for an exact match to a length-wise sample composition, pin the sample indices explicitly.

   Every URL in the official essay list must be cached and hash-verified before the unchanged upstream essay parser runs. Fetches use bounded retries and equivalent HTTPS/GitHub raw transports. Missing sources or parser skips stop preparation; partial corpora are not installed. Raw source hashes and the complete-corpus status are recorded in RULER metadata; `essay_download.log` keeps the parser output. Compare corpus hashes when comparing runs from different preparations, or reuse the same cached corpus.

   `data/preparation.json` records hashes of every installed file, the producing scripts, and the source configuration. Run `python scripts/PrepareDataset.py --verify` to check those bytes against the current code/configuration without downloading anything. Direct source archives are cached with hashes under `.cache/dataset_sources/`; pinned Hugging Face cache-file hashes and dataset fingerprints are recorded in each dataset's metadata. Installation retains previous datasets under `.data.backups/` and rolls back all replacements if any installation step fails.

4. Install the runtime required by each method. Repository-specific setup and paths are described in Section 3.

## 3. Bundled backends

The following integrations are included in this checkout and use the project's
Python environment. Run the relevant setup command from the repository root:

| Backend | Setup command | Additional configuration |
| --- | --- | --- |
| CacheBlend | `./setup.py --cacheblend` | Configure GPU memory use under `Cacheblend`. |
| HYPIC / LinearKV | `./setup.py --hypic` | Use a supported hybrid-model checkpoint; configure the Mamba pool under `Hypic`. |
| A3 | `./setup.py --a3` | Use a model supported by the bundled A3 worker. |
| KVPacket | `./setup.py --kvpacket` | This checks the shared environment. Set a trained, model-matching `KVPacket.WrapperCheckpointPath` for benchmark runs. |

The setup commands install or validate optional dependencies; HYPIC also builds
its retained native operations. Source revisions and local adaptations are
recorded in the backends' `UPSTREAM` files. Other runtime requirements remain
method-specific; consult `docs/Method.md` and the selected class before running
an evaluation. This guide does not establish that every method supports the
same model or sampling configuration.

## 4. Skill reuse

The paper evaluates one agent rollout per task/configuration on all 87 SkillsBench 1.1 tasks. It uses Qwen3.8-27B with pi 0.73.1 through pi-acp 0.0.32 on two RTX PRO 6000 GPUs with 96 GB each. Sampling uses temperature 1.0, top-p 0.95, top-k 20, and minimum-p 0. The maximum generation length is 40,960 tokens, the maximum model length is 256,000 tokens, and each agent execution has a five-hour timeout.

The current `AgentBenchFlowTask`/`AgentBenchFlowWorkflow` supports BenchFlow/SkillsBench sources and skill reuse. `config.yaml` contains the agent, sandbox, timeout, and model-serving settings.

We recommend prebuilding SkillsBench Docker images with `scripts/PrepareSkillsbench.py` to reduce network-related variability during the experiment.

If the KVBench machine cannot run Docker, run the BenchFlow CLI and Docker sandbox on another machine and set `AgentBenchFlow.Sandbox: remote-docker` in `config.yaml`. See the [Remote Docker runtime section in AgentBenchFlow.md](AgentBenchFlow.md#remote-docker-runtime) for the full procedure, including A/B machine roles, image preparation, authentication, dynamic provider ports, and SSH forwarding. Remote Docker moves only the BenchFlow/Docker task runtime; model inference and Method dependencies such as CacheBlend, HYPIC, and A3 stay on the machine running KVBench.
