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

    # tasks = [
    #     NIAHShuffleTask(maxSamples=MAX_SAMPLES),
    #     CWEShuffleTask(maxSamples=MAX_SAMPLES),
    #     VTShuffleTask(maxSamples=MAX_SAMPLES),
    #     MusiqueTask(maxSamples=MAX_SAMPLES),
    #     SamsumTask(maxSamples=MAX_SAMPLES),
    #     WikimQATask(maxSamples=MAX_SAMPLES),
    #     FreshGapTask(nCases=MAX_SAMPLES),
    #     KVCommMMLUTask(maxSamples=MAX_SAMPLES, agentCount=5),
    #     KVCommGSM8KTask(maxSamples=MAX_SAMPLES, agentCount=3),
    #     KVCommHumanEvalTask(maxSamples=MAX_SAMPLES, agentCount=5),
    #     KVCommCopyTask(nCases=MAX_SAMPLES, agentCount=5),
    # ]

    skillsbench_root = Get("AgentBenchFlow", {}).get("SkillsBenchRepo")
    task_ids =['manufacturing-codebook-normalization', 'manufacturing-equipment-maintenance', 'manufacturing-fjsp-optimization', 'mario-coin-counting', 'mars-clouds-clustering', 'offer-letter-generator', 'organize-messy-files', 'paper-anonymizer', 'parallel-tfidf-search', 'paratransit-routing', 'pddl-airport-planning', 'pddl-tpp-planning', 'pdf-excel-diff', 'powerlifting-coef-calc', 'pptx-reference-formatting', 'protein-expression-analysis', 'python-scala-translation', 'quantum-numerical-simulation', 'r2r-mpc-control', 'radar-vital-signs', 'react-performance-debugging', 'reserves-at-risk-calc', 'sales-pivot-analysis', 'sec-financial-report', 'seismic-phase-picking', 'setup-fuzzing-py', 'shock-analysis-demand', 'shock-analysis-supply', 'simpo-code-reproduction', 'software-dependency-audit', 'syzkaller-ppdev-syzlang', 'threejs-structure-parser', 'threejs-to-obj', 'tictoc-unnecessary-abort-detection', 'travel-planning', 'video-silence-remover', 'weighted-gdp-calc', 'xlsx-recover-data', 'fix-build-google-auto', 'fix-visual-stability']
    
    tasks = [
        AgentBenchFlowTask(
            source_mode="local",
            skillsbench_dir=skillsbench_root,
            task_ids=[task_id],
            agent="pi-acp",
            skill_mode="with-skill",
            thinking=True,
            result_json_timeout=18000,
            bench_extra_args=[
                "--agent-idle-timeout", "18000",
                "--config-override",
                '{"agent":{"timeout_sec":18000}}',
                # LiteLLM's built-in completion fallback is 600s unless the
                # proxy receives an explicit REQUEST_TIMEOUT. Agent turns in
                # this benchmark can legitimately take longer than that.
                "--agent-env", "REQUEST_TIMEOUT=18000",
            ],
        )
        for task_id in task_ids
    ]
    tasks = [AgentBenchFlowTask(taskId) for taskId in taskIds]

    methods = [
        # CacheblendRepo(gpuNums=1, perfWeight=4, maxNewTokens=MAX_NEW_TOKENS),
        # CacheblendRepo(gpuNums=1, perfWeight=4, maxNewTokens=MAX_NEW_TOKENS, fullPrefill=True, tag="full_prefill"),
        # FullPrefillVllm(gpuNums=2, perfWeight=2, maxNewTokens=MAX_NEW_TOKENS),
        # NaiveTransformer(gpuNums=1, perfWeight=1, maxNewTokens=MAX_NEW_TOKENS),
        FullPrefillVllm(
            gpuNums=2, perfWeight=2, maxNewTokens=40960,
            gpuMemoryUtilization=0.85,
            maxModelLen=256000,
            enforceEager=True,
            languageModelOnly=True,
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
