"""KVBench entry point / configuration.

This file doubles as the configuration: edit the ``tasks`` / ``methods``
lists below and run from the project root::

    python Main.py

The model path comes from ``config.yaml`` (``ModelPath``);
datasets resolve by name against ``DatasetPath``.
"""

import json
import sys

from core import ModelPath
from core.engine import Engine
from metrics import ThroughputMetric, TTFTMetric

from methods import (
    CacheblendLmcache,
    CacheblendRepo,
    FullPrefillVllm,
    FullPrefillTransformer,
    NaiveTransformer,
)
from tasks import (
    AgentBenchFlowTask,
    CWEShuffleTask,
    NIAHShuffleTask,
    VTShuffleTask,
    MusiqueTask,
    SamsumTask,
    WikimQATask,
    KVCommMMLUTask,
    KVCommGSM8KTask,
    KVCommHumanEvalTask,
    KVCommCopyTask,
)
from tasks.FreshGap import FreshGapTask

MAX_SAMPLES=64
MAX_NEW_TOKENS=64

def Main() -> None:
    tasks = [
        AgentBenchFlowTask(
            image_override="docker://python:3.13-slim",
            task_ids=["azure-bgp-oscillation-route-leak"],
            thinking=False,
            agent_extra_args=[
                "--config", "/host_lib/minisweagent/config/mini.yaml",
                "--config", "agent.step_limit=10",
            ],
        ),
    ]

    methods = [
        # CacheblendRepo(gpuNums=1, perfWeight=4, maxNewTokens=MAX_NEW_TOKENS),
        # CacheblendRepo(gpuNums=1, perfWeight=4, maxNewTokens=MAX_NEW_TOKENS, fullPrefill=True, tag="full_prefill"),
        # FullPrefillVllm(gpuNums=2, perfWeight=2, maxNewTokens=MAX_NEW_TOKENS),
        # NaiveTransformer(gpuNums=1, perfWeight=1, maxNewTokens=MAX_NEW_TOKENS),
        FullPrefillVllm(
            gpuNums=2, perfWeight=2, maxNewTokens=4096,
            gpuMemoryUtilization=0.85,
            maxModelLen=32768,
            enforceEager=True,
            languageModelOnly=True,
        ),
    ]

    metrics = [TTFTMetric(), ThroughputMetric()]

    batchSize = 1

    print(
        f"[main] model={ModelPath()}\n"
        f"[main] tasks={[t.name for t in tasks]} "
        f"methods={[(m.Label, m.gpuNums, m.perfWeight) for m in methods]} "
        f"batchSize={batchSize}"
    )
    sys.stdout.flush()

    engine = Engine(
        availableGpuIds=[1, 3],
        batchSize=batchSize,
        initializeTimeout=1800,
        taskTimeout=3600,
        shutdownGracePeriod=30,
        gpuReleaseTimeout=30,
        gpuReleaseStableSeconds=1,
        gpuReleaseMemoryToleranceMiB=256,
        pairRetries=1,
        tui=True,
        verbose=True,
    )
    report = engine.Evaluate(tasks=tasks, methods=methods, metrics=metrics)

    print("\n=== KVBench report ===")
    print(json.dumps(report["cores"], indent=2, ensure_ascii=False))
    print(f"full outputs: {report['output_dir']}")


if __name__ == "__main__":
    Main()
