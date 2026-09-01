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
from core.Config import AgentBenchFlowDefaults
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
    agentBenchFlowConfig = AgentBenchFlowDefaults()
    skillsbench_root = Path(
        agentBenchFlowConfig.get("SkillsBenchRepo", "/data/lyh/skillsbench")
    )
    skillsbench_tasks_root = skillsbench_root / "tasks"
    task_ids = sorted(
        entry.name
        for entry in skillsbench_tasks_root.iterdir()
        if (entry / "task.md").is_file()
    )
    SKIP_TASKS = {
        # No sandbox image built (pre-existing gaps)
        "3d-scan-calc",
        "crystallographic-wyckoff-position-analysis",
        # Dockerfile builds still failing under --use-mirror; the SkillsBench
        # tasks below need downloads (Maven / sdkman / nodesource setup /
        # playwright bundles / kokoro weights) that the wrap pipeline cannot
        # cover from this host's proxy. Re-enable once their base images exist.
        "fix-build-google-auto",
        "fix-druid-loophole-cve",
        "fix-visual-stability",
        "multilingual-video-dubbing",
        "python-scala-translation",
        "spring-boot-jakarta-migration",
        "suricata-custom-exfil",
        "threejs-structure-parser",
        "threejs-to-obj",
    }
    task_ids = [t for t in task_ids if t not in SKIP_TASKS]
    # SMOKE: scope down to a single prebuilt, fast verification task so the
    # first end-to-end run completes within minutes rather than days.
    SMOKE_TASK = "reserves-at-risk-calc"
    if SMOKE_TASK:
        task_ids = [t for t in task_ids if t == SMOKE_TASK]
    print(f"[main] running {len(task_ids)} tasks (skipped {len(SKIP_TASKS)} known-good)")

    tasks = [
        AgentBenchFlowTask(
            source_mode="local",
            skillsbench_dir=skillsbench_root,
            task_ids=[task_id],
            agent="pi-acp",
            skill_mode="with-skill",
            thinking=True,
            bench_extra_args=[
                "--agent-idle-timeout", "3600",
                "--config-override", '{"agent":{"timeout_sec":7200}}',
            ],
        )
        for task_id in task_ids
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
        availableGpuIds='auto',
        batchSize=batchSize,
        initializeTimeout=1800,
        taskTimeout=10800,
        shutdownGracePeriod=30,
        gpuReleaseTimeout=30,
        gpuReleaseStableSeconds=1,
        gpuReleaseMemoryToleranceMiB=256,
        pairRetries=1,
        tui=False,
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
