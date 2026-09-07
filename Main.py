"""KVBench entry point.

Task selection and method parameters live here. Shared runtime settings live in
``config.yaml``.
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
    HypicMethod,
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


def Main() -> None:
    taskIds = [
        "ada-bathroom-plan-repair",
        "adaptive-cruise-control",
        "data-to-d3",
        "dynamic-object-aware-egomotion",
        "enterprise-information-search",
        "exoplanet-detection-period",
        "lab-unit-harmonization",
        "manufacturing-codebook-normalization",
        "sec-financial-report",
        "setup-fuzzing-py",
        "travel-planning",
        "video-silence-remover",
        "weighted-gdp-calc",
        "xlsx-recover-data",
    ]
    tasks = [AgentBenchFlowTask(taskId) for taskId in taskIds]

    methods = [
        # HypicMethod(
        #     maxNewTokens=40960,
        #     maxModelLen=256000,
        #     memFractionStatic=0.80,
        #     picMode="addition",
        # ),
        HypicMethod(
            maxNewTokens=40960,
            maxModelLen=256000,
            memFractionStatic=0.80,
            fullPrefill=True,
            tag="full_prefill",
        )
    ]

    metrics = [TTFTMetric(), ThroughputMetric()]
    print(
        f"[main] model={ModelPath()}\n"
        f"[main] tasks={[task.Label for task in tasks]} "
        f"methods={[(method.Label, method.gpuNums, method.perfWeight) for method in methods]}"
    )
    sys.stdout.flush()

    engine = Engine()
    report = engine.Evaluate(tasks=tasks, methods=methods, metrics=metrics)

    print("\n=== KVBench report ===")
    print(json.dumps(report["cores"], indent=2, ensure_ascii=False))
    print(f"full outputs: {report['output_dir']}")


if __name__ == "__main__":
    try:
        Main()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[main] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
