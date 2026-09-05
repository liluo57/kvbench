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
    HypicMethod,
    NaiveTransformer,
    Qwen38TestMethod
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
BENCHFLOW_TIMEOUT_SEC = 18000


def Main() -> None:
    skillsbench_root = Get("AgentBenchFlow", {}).get("SkillsBenchRepo")
    task_ids =['financial-modeling-qa', 'fix-build-agentops', 'fix-erlang-ssh-cve', 'flink-query', 'flood-risk-analysis', 'glm-lake-mendota', 'gravitational-wave-detection', 'grid-dispatch-operator', 'hvac-control', 'invoice-fraud-detection', 'jax-computing-basics', 'jpg-ocr-stat', 'lab-unit-harmonization', 'lake-warming-attribution', 'latex-formula-extraction', 'lean4-proof', 'llm-prefix-cache-replay', 'manufacturing-codebook-normalization', 'manufacturing-equipment-maintenance', 'manufacturing-fjsp-optimization', 'mario-coin-counting', 'mars-clouds-clustering', 'offer-letter-generator', 'organize-messy-files', 'paper-anonymizer', 'parallel-tfidf-search', 'paratransit-routing', 'pddl-airport-planning', 'pddl-tpp-planning', 'pdf-excel-diff', 'powerlifting-coef-calc', 'pptx-reference-formatting', 'protein-expression-analysis', 'python-scala-translation', 'quantum-numerical-simulation', 'r2r-mpc-control', 'radar-vital-signs', 'react-performance-debugging', 'reserves-at-risk-calc', 'sales-pivot-analysis', 'sec-financial-report', 'seismic-phase-picking', 'setup-fuzzing-py', 'shock-analysis-demand', 'shock-analysis-supply', 'simpo-code-reproduction', 'software-dependency-audit', 'syzkaller-ppdev-syzlang', 'threejs-structure-parser', 'threejs-to-obj', 'tictoc-unnecessary-abort-detection', 'travel-planning', 'video-silence-remover', 'weighted-gdp-calc', 'xlsx-recover-data', 'fix-build-google-auto', 'fix-visual-stability']

    tasks = [
        AgentBenchFlowTask(
            source_mode="local",
            skillsbench_dir=skillsbench_root,
            task_ids=[task_id],
            agent="pi-acp",
            skill_mode="with-skill",
            thinking=True,
            result_json_timeout=BENCHFLOW_TIMEOUT_SEC,
            bench_extra_args=[
                "--agent-idle-timeout", str(BENCHFLOW_TIMEOUT_SEC),
                "--config-override",
                '{"agent":{"timeout_sec":18000}}',
                "--agent-env", "REQUEST_TIMEOUT=18000",
            ],
        )
        for task_id in task_ids
    ]

    methods = [
        HypicMethod(
            maxNewTokens=40960,
            maxModelLen=256000,
            memFractionStatic=0.80,
            picMode="addition",
        ),
        HypicMethod(
            maxNewTokens=40960,
            maxModelLen=256000,
            memFractionStatic=0.80,
            fullPrefill=True,
            tag="full_prefill",
        )
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
        availableGpuIds=[0,1,2,3],
        batchSize=batchSize,
        initializeTimeout=1800,
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
