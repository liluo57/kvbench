"""KVBench entry point / configuration.

This file doubles as the configuration: edit the ``tasks`` / ``methods``
lists below and run from the project root::

    python Main.py

The model path comes from ``config.yaml`` (``ModelPath``);
datasets resolve by name against ``DatasetPath``.

Methods (each loads its own copy of the model on its ``gpuIds``):
    cacheblend          CacheBlend via vLLM 0.25 + LMCache in-process blending
    full_prefill        FullPrefillTransformer (transformers, recompute all)
    full_prefill_vllm   FullPrefillVllm (system vLLM, recompute all)
    naive               NaiveTransformer (transformers, reuse context KV)
"""

import json
import sys

from core import Engine, ModelPath
from metrics import ThroughputMetric, TTFTMetric

from methods import (
    CacheBlendMethod,
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
)

MAX_SAMPLES=64


def Main() -> None:
    #: Tasks to evaluate (edit to taste). The RULER shuffle tasks read from
    #: ``<DatasetPath>/ruler/<name>_len*.jsonl``; ``maxSamples`` caps the count
    #: (omit for all samples).
    tasks = [
        # NIAHShuffleTask(maxSamples=MAX_SAMPLES),
        # CWEShuffleTask(maxSamples=MAX_SAMPLES),
        # VTShuffleTask(maxSamples=MAX_SAMPLES),
        # MusiqueTask(maxSamples=MAX_SAMPLES),
        # SamsumTask(maxSamples=MAX_SAMPLES),
        WikimQATask(maxSamples=MAX_SAMPLES),
    ]

    #: Methods to run (edit to taste). Each loads its own copy of the model on
    #: its ``gpuIds``; adjust the GPU ids to match your GPU count / memory.
    #: Constructed *inside* ``Main()`` (never at module top level): the method
    #: constructors load the models, and CacheBlendMethod spawns a vLLM
    #: EngineCore — starting a spawn'd process while the main module is still
    #: being imported raises multiprocessing's "bootstrapping phase" error.
    methods = [
        CacheblendRepo(gpuIds="0", maxNewTokens=64),
        CacheblendRepo(gpuIds="5", maxNewTokens=64,fullPrefill=True,tag='full_prefill'),
        FullPrefillVllm(gpuIds="1", maxNewTokens=64),
        # NaiveTransformer(gpuIds="2", maxNewTokens=64),
    ]

    metrics = [TTFTMetric(), ThroughputMetric()]

    #: Cases per batch handed to each method (see Engine). Edit to taste;
    #: 1 keeps the per-case behavior. No CLI args by design.
    batchSize = 32

    print(
        f"[main] model={ModelPath()}\n"
        f"[main] tasks={[t.name for t in tasks]} "
        f"methods={[m.Label for m in methods]} batchSize={batchSize} "
    )
    sys.stdout.flush()

    engine = Engine(verbose=True, batchSize=batchSize)
    report = engine.Evaluate(tasks=tasks, methods=methods, metrics=metrics)

    print("\n=== KVBench report ===")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    Main()
