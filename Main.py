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
    GovReportTask,
    HotpotQATask,
    MultiNewsTask,
    NIAHShuffleTask,
    VTShuffleTask,
    MusiqueTask,
    SamsumTask,
    TriviaQATask,
    WikimQATask,
    KVCommMMLUTask,
    KVCommGSM8KTask,
    KVCommHumanEvalTask,
    KVCommCopyTask,
)


def Main() -> None:
    taskIds = [
        'debug-trl-grpo',
        'earthquake-phase-association',
        'fix-build-google-auto',
        'fix-visual-stability',
        'flink-query',
        'glm-lake-mendota',
        'latex-formula-extraction',
        'lean4-proof',
        'multilingual-video-dubbing',
        'seismic-phase-picking',
        'software-dependency-audit',
        'spring-boot-jakarta-migration',
        'suricata-custom-exfil',
    ]

    tasks = [AgentBenchFlowTask(taskId, firstRunOnly=False) for taskId in taskIds]

    MAX_NEW_TOKENS = 40960
    methods = [
        HypicMethod(
            maxNewTokens=MAX_NEW_TOKENS,
            gpuNums=2,
            maxModelLen=256000,
            memFractionStatic=0.90,
            picMode="addition",
            tag="addition",
        ),
        HypicMethod(
            maxNewTokens=MAX_NEW_TOKENS,
            gpuNums=2,
            maxModelLen=256000,
            memFractionStatic=0.80,
            fullPrefill=True,
            tag="full_prefill",
        ),
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
