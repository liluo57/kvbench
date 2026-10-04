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

   The current configuration pins several Hugging Face revisions, dataset splits, and expected row counts. It generates RULER NIAH, VT, and CWE data with the configured tokenizer and random seed 42, using lengths 4096 and 8192 with 500 samples at each length. The generation limits are also specified in the configuration. Do not change the split or tokenizer during an experiment. `maxSamples=64` in `Main.py` caps the number of examples used. The RULER loader combines files in sorted filename order before truncation, so for an exact match to the paper's length-wise sample composition, verify the sample list or supplementary configuration and pin the sample indices explicitly.

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
