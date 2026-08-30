"""Shared state, exception types, and pure helpers for the Engine package.

``core/engine/`` is split into four collaborating classes (``Scheduler``,
``GpuGovernor``, ``Reporter``, and the ``Engine`` orchestrator) that all
read and mutate a single :class:`RunContext`. This module owns that
context type, the worker-state dataclass, and the small helpers / exception
classes that don't belong on any one of those four classes.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Deque,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)


_GPU_SNAPSHOT_INTERVAL = 1.0


# --------------------------------------------------------------------- workers


@dataclass
class _WorkerState:
    """Per-worker process state held by the coordinator.

    Mirrors the fields the closures inside the original ``Engine.Evaluate``
    used to mutate by ``nonlocal``. Now each field is a normal dataclass
    attribute, so ``Scheduler`` / ``GpuGovernor`` / ``Reporter`` can read
    and write them without capturing closures.
    """

    workerId: str
    methodIndex: int
    methodLabel: str
    gpuIds: List[int]
    process: Any
    connection: Any
    instanceLog: str
    state: str = "initializing"
    deadline: Optional[float] = None
    taskIndex: Optional[int] = None
    taskName: str = ""
    attempt: int = 0
    logPath: str = ""
    initDuration: Optional[float] = None
    startedAt: float = field(default_factory=time.time)

    def Snapshot(self) -> Dict[str, Any]:
        return {
            "worker_id": self.workerId,
            "method_index": self.methodIndex,
            "method": self.methodLabel,
            "gpu_ids": self.gpuIds,
            "state": self.state,
            "task": self.taskName,
            "attempt": self.attempt,
            "log_path": self.logPath or self.instanceLog,
        }


# --------------------------------------------------------------------- errors


class BenchmarkInitializationError(RuntimeError):
    """Raised after an instance cannot initialize within the configured limit."""


class BenchmarkResourceReleaseError(RuntimeError):
    """Raised when a worker exits but its assigned GPU stays occupied."""


# --------------------------------------------------------------------- helpers


def MeanValue(stats: Dict[str, Any]) -> Any:
    """Reduce a per-metric stat dict to the value the report should show.

    Prefers an explicit ``"mean"`` key (system metrics emit it) and falls back
    to any ``*_mean`` suffix (method metrics use those). Returns ``None``
    when ``stats`` is empty.
    """
    if not stats:
        return None
    if "mean" in stats:
        return stats["mean"]
    for key, value in stats.items():
        if key.endswith("_mean"):
            return value
    return next(iter(stats.values()), None)


def CoreReport(run: Dict[str, Any]) -> Dict[str, Any]:
    """Project a per-pair run dict down to its per-task / system / method means."""
    core: Dict[str, Any] = {"method": run["method"], "task": run["task"]}
    for group in ("task_metrics", "system_metrics", "method_metrics"):
        for name, stats in run.get(group, {}).items():
            core[name] = MeanValue(stats)
    return core


def Slug(value: str) -> str:
    """Filesystem-safe slug for log path / pair-report path components."""
    cleaned = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in value
    ).strip("._")
    return cleaned or "unnamed"


def AtomicJson(path: Path, value: Any) -> None:
    """Atomically write ``value`` as JSON to ``path`` (write+rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False, default=str)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


# --------------------------------------------------------------------- context


@dataclass
class RunContext:
    """All mutable state shared between ``Engine`` / ``Scheduler`` /
    ``GpuGovernor`` / ``Reporter`` for one ``Evaluate`` invocation.

    The Engine builds one ``RunContext`` at the top of ``Evaluate``, hands
    it to the three support classes, and reads ``status`` /
    ``fatalError`` / ``fatalStatus`` out of it during the main loop. The
    support classes own most field-level mutation.
    """

    pending: Dict[int, Deque[int]]
    pairStatus: Dict[Tuple[int, int], str]
    attempts: Dict[Tuple[int, int], int]
    results: Dict[Tuple[int, int], Dict[str, Any]]
    failures: List[Dict[str, Any]]
    timings: List[Dict[str, Any]]
    logs: List[Dict[str, str]]
    knownLogs: Set[str]
    workers: Dict[str, _WorkerState]
    workerHistory: List[Dict[str, Any]]
    freeGpus: List[int]
    coolingGpus: Dict[int, Dict[str, Any]]
    gpuBaseline: Dict[int, int]
    gpuBaselinePids: Dict[int, Set[int]]
    lastGpuReleasePoll: float
    lastGpuSnapshotPoll: float
    gpuSnapshotAt: float
    gpuSnapshotError: str
    workerSerial: int
    fatalError: Optional[str]
    fatalStatus: Optional[str]
    cancelled: bool
    status: str
    mpContext: Any
    eventQueue: Any
    eventsFile: Any
    eventsPath: Path
    startedWall: float
    startedMono: float
    outputDir: Path
    manifest: Dict[str, Any]
    methods: Any  # List[Method] — annotated at use site to avoid a circular import
    tasks: Any  # List[Task]
    effectiveBatchSizes: List[int]
    maxAttempts: int


def BuildRunContext(
    *,
    methods: List[Any],
    tasks: List[Any],
    effectiveBatchSizes: List[int],
    gpuPool: List[int],
    gpuSnapshot: List[Any],
    outputDir: Path,
    maxAttempts: int,
) -> RunContext:
    """Initialize the mutable state ``Engine.Evaluate`` starts with.

    Pure factory — keeps ``Engine`` short. The first ``BenchmarkTui``,
    ``mp`` context, and ``eventQueue`` are constructed by ``Engine``
    directly because they belong to that class, not the context.
    """
    startedWall = time.time()
    startedMono = time.monotonic()
    return RunContext(
        pending={
            methodIndex: deque(range(len(tasks)))
            for methodIndex in range(len(methods))
        },
        pairStatus={
            (methodIndex, taskIndex): "pending"
            for methodIndex in range(len(methods))
            for taskIndex in range(len(tasks))
        },
        attempts={pair: 0 for pair in {
            (methodIndex, taskIndex)
            for methodIndex in range(len(methods))
            for taskIndex in range(len(tasks))
        }},
        results={},
        failures=[],
        timings=[],
        logs=[],
        knownLogs=set(),
        workers={},
        workerHistory=[],
        freeGpus=list(gpuPool),
        coolingGpus={},
        gpuBaseline={gpu.id: gpu.memoryUsed for gpu in gpuSnapshot},
        gpuBaselinePids={gpu.id: set(gpu.computePids) for gpu in gpuSnapshot},
        lastGpuReleasePoll=0.0,
        lastGpuSnapshotPoll=startedMono,
        gpuSnapshotAt=startedWall,
        gpuSnapshotError="",
        workerSerial=0,
        fatalError=None,
        fatalStatus=None,
        cancelled=False,
        status="running",
        mpContext=mp.get_context("spawn"),
        eventQueue=None,  # set by Engine right after construction
        eventsFile=None,
        eventsPath=outputDir / "events.jsonl",
        startedWall=startedWall,
        startedMono=startedMono,
        outputDir=outputDir,
        manifest={},
        methods=methods,
        tasks=tasks,
        effectiveBatchSizes=effectiveBatchSizes,
        maxAttempts=maxAttempts,
    )