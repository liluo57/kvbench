"""KVBench entry point / configuration.

This file doubles as the configuration: edit the ``tasks`` / ``methods``
lists below and run from the project root::

    python Main.py

The model path comes from ``config.yaml`` (``ModelPath``);
datasets resolve by name against ``DatasetPath``.
"""

import json
import sys
from pathlib import Path

from core import ModelPath
from core.Config import Get
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
MAX_NEW_TOKENS=512


def Main() -> None:

    tasks = [
        NIAHShuffleTask(maxSamples=MAX_SAMPLES),
        CWEShuffleTask(maxSamples=MAX_SAMPLES),
        VTShuffleTask(maxSamples=MAX_SAMPLES),
        MusiqueTask(maxSamples=MAX_SAMPLES),
        SamsumTask(maxSamples=MAX_SAMPLES),
        WikimQATask(maxSamples=MAX_SAMPLES),
        FreshGapTask(nCases=MAX_SAMPLES),
        KVCommMMLUTask(maxSamples=MAX_SAMPLES, agentCount=5),
        KVCommGSM8KTask(maxSamples=MAX_SAMPLES, agentCount=3),
        KVCommHumanEvalTask(maxSamples=MAX_SAMPLES, agentCount=5),
        KVCommCopyTask(nCases=MAX_SAMPLES, agentCount=5),
    ]

    methods = [
        CacheblendRepo(gpuNums=1, perfWeight=4, maxNewTokens=MAX_NEW_TOKENS),
        CacheblendRepo(gpuNums=1, perfWeight=4, maxNewTokens=MAX_NEW_TOKENS, fullPrefill=True, tag="full_prefill"),
        FullPrefillVllm(gpuNums=2, perfWeight=2, maxNewTokens=MAX_NEW_TOKENS),
        NaiveTransformer(gpuNums=1, perfWeight=1, maxNewTokens=MAX_NEW_TOKENS),
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
        availableGpuIds='auto',
        batchSize=batchSize,
        initializeTimeout=600,
        taskTimeout=18000,
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
    try:
        Main()
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[main] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
