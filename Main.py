"""KVBench entry point / configuration.

This file doubles as the configuration: edit the ``tasks`` / ``methods``
lists below and run from the project root::

    python Main.py

The model path comes from ``config.yaml`` (``ModelPath``);
datasets resolve by name against ``DatasetPath``.

Methods declare only ``gpuNums`` / ``perfWeight``.  Engine assigns concrete
GPU ids and initializes every instance in its own spawned process:
    cacheblend_lmcache  CacheBlend via vLLM 0.25 + LMCache in-process blending
    cacheblend_repo     CacheBlend via the original repo's patched vLLM worker
    full_prefill        FullPrefillTransformer (transformers, recompute all)
    full_prefill_vllm   FullPrefillVllm (system vLLM, recompute all)
    naive               NaiveTransformer (transformers, reuse context KV)
"""

import json
import sys

from core import Engine, ModelPath
from metrics import ThroughputMetric, TTFTMetric

from methods import (
    CacheblendLmcache,
    CacheblendRepo,
    FullPrefillVllm,
    FullPrefillTransformer,
    NaiveTransformer,
)
from tasks import (
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
    #: Tasks to evaluate (edit to taste). The RULER shuffle tasks read from
    #: ``<DatasetPath>/ruler/<name>_len*.jsonl``; ``maxSamples`` caps the count
    #: (omit for all samples).
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

    #: Methods to run (edit to taste). Constructors are lightweight: concrete
    #: GPU ids are assigned later by Engine. perfWeight is relative expected
    #: per-task runtime and controls how additional instances are distributed.
    methods = [
        # COPY benchmark expects Method(maxNewTokens=512).
        CacheblendRepo(gpuNums=1, perfWeight=4, maxNewTokens=MAX_NEW_TOKENS),
        CacheblendRepo(gpuNums=1, perfWeight=4, maxNewTokens=MAX_NEW_TOKENS, fullPrefill=True, tag="full_prefill"),
        FullPrefillVllm(gpuNums=2, perfWeight=2, maxNewTokens=MAX_NEW_TOKENS),
        NaiveTransformer(gpuNums=1, perfWeight=1, maxNewTokens=MAX_NEW_TOKENS),
    ]

    metrics = [TTFTMetric(), ThroughputMetric()]

    #: Cases per batch handed to each method (see Engine). Edit to taste;
    #: 1 keeps one Case per batch. No CLI args by design.
    batchSize = 4

    print(
        f"[main] model={ModelPath()}\n"
        f"[main] tasks={[t.name for t in tasks]} "
        f"methods={[(m.Label, m.gpuNums, m.perfWeight) for m in methods]} "
        f"batchSize={batchSize}"
    )
    sys.stdout.flush()

    engine = Engine(
        availableGpuIds="auto",
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
