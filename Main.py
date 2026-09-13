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
    DependencyAnalysisMethod,
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
    # taskIds = ['paper-anonymizer','energy-market-pricing','tictoc-unnecessary-abort-detection','fix-visual-stability','lab-unit-harmonization']

    # tasks = [AgentBenchFlowTask(taskId) for taskId in taskIds]

    MAX_SAMPLES = 4
    tasks = [
        NIAHShuffleTask(maxSamples=MAX_SAMPLES),
        CWEShuffleTask(maxSamples=MAX_SAMPLES),
        VTShuffleTask(maxSamples=MAX_SAMPLES),
        MusiqueTask(maxSamples=MAX_SAMPLES),
        SamsumTask(maxSamples=MAX_SAMPLES),
        WikimQATask(maxSamples=MAX_SAMPLES),
        # FreshGapTask(nCases=MAX_SAMPLES),
        # GovReportTask(maxSamples=MAX_SAMPLES, nChunks=1, maxSampleLength=32768, tag="1"),
        # GovReportTask(maxSamples=MAX_SAMPLES, nChunks=4, maxSampleLength=32768, tag="4"),
        # GovReportTask(maxSamples=MAX_SAMPLES, nChunks=8, maxSampleLength=32768, tag="8"),
        # GovReportTask(maxSamples=MAX_SAMPLES, nChunks=16, maxSampleLength=32768, tag="16"),
        # HotpotQATask(maxSamples=MAX_SAMPLES),
        # MultiNewsTask(maxSamples=MAX_SAMPLES),
        # TriviaQATask(maxSamples=MAX_SAMPLES),
        # KVCommMMLUTask(maxSamples=MAX_SAMPLES, agentCount=5),
        # KVCommGSM8KTask(maxSamples=MAX_SAMPLES, agentCount=3),
        # KVCommHumanEvalTask(maxSamples=MAX_SAMPLES, agentCount=5),
        # KVCommCopyTask(nCases=MAX_SAMPLES, agentCount=5),
    ]

    MAX_NEW_TOKENS = 64
    methods = [
        DependencyAnalysisMethod(gpuNums=1, perfWeight=1, maxNewTokens=MAX_NEW_TOKENS),
        # HypicMethod(
        #     maxNewTokens=MAX_NEW_TOKENS,
        #     gpuNums=2,
        #     maxModelLen=256000,
        #     memFractionStatic=0.90,
        #     picMode="addition",
        #     tag="addition",
        # ),
        # HypicMethod(
        #     maxNewTokens=MAX_NEW_TOKENS,
        #     gpuNums=2,
        #     maxModelLen=256000,
        #     memFractionStatic=0.90,
        #     picMode="transition",
        #     tag="transition",
        # ),
        # HypicMethod(
        #     maxNewTokens=MAX_NEW_TOKENS,
        #     gpuNums=2,
        #     maxModelLen=256000,
        #     memFractionStatic=0.90,
        #     picMode="transition_rope",
        #     tag="transition_rope",
        # ),
        # HypicMethod(
        #     maxNewTokens=MAX_NEW_TOKENS,
        #     gpuNums=2,
        #     maxModelLen=256000,
        #     memFractionStatic=0.90,
        #     picMode="transition_rope_recompute",
        #     tag="transition_rope_recompute",
        # ),
        # HypicMethod(
        #     maxNewTokens=MAX_NEW_TOKENS,
        #     gpuNums=2,
        #     maxModelLen=256000,
        #     memFractionStatic=0.80,
        #     fullPrefill=True,
        #     tag="full_prefill",
        # ),
        # FullPrefillVllm(
        #     gpuNums=1, perfWeight=2, maxNewTokens=40960,
        #     gpuMemoryUtilization=0.85,
        #     maxModelLen=256000,
        #     languageModelOnly=True,
        # ),
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
