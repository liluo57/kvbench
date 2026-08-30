"""Concurrent GPU scheduler, process supervisor, and report coordinator.

The :class:`Engine` is the top-level orchestrator. It builds a
:class:`~core.engine.State.RunContext` of mutable per-run state and three
support classes (``Scheduler``, ``GpuGovernor``, ``Reporter``), then runs
a single main loop that drains events, refreshes GPU cooling, reaps dead
workers, and dispatches new tasks until every (method, task) pair is
terminal.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import signal
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from ..Method import Method
from ..Metrics import Metric
from ..Task import Task
from ..Worker import WorkerMain
from ..tui import BenchmarkTui
from .Gpu import ResolveGpuIds
from .GpuGovernor import GpuGovernor
from .Reporter import Reporter
from .Scheduler import Scheduler
from .State import (
    AtomicJson,
    BenchmarkInitializationError,
    BenchmarkResourceReleaseError,
    BuildRunContext,
    RunContext,
)


class Engine:
    """Schedule method instances over a strict GPU pool and aggregate results."""

    def __init__(
        self,
        *,
        availableGpuIds: Union[str, Iterable[int]] = "auto",
        batchSize: int = 1,
        initializeTimeout: float = 1800.0,
        taskTimeout: float = 3600.0,
        shutdownGracePeriod: float = 30.0,
        gpuReleaseTimeout: float = 30.0,
        gpuReleaseStableSeconds: float = 1.0,
        gpuReleaseMemoryToleranceMiB: int = 256,
        pairRetries: int = 1,
        outputRoot: Union[str, Path] = "outputs",
        tui: bool = True,
        verbose: bool = True,
    ):
        if batchSize < 1:
            raise ValueError("batchSize must be at least 1")
        if (
            initializeTimeout <= 0
            or taskTimeout <= 0
            or shutdownGracePeriod <= 0
            or gpuReleaseTimeout <= 0
        ):
            raise ValueError("all timeouts must be greater than 0")
        if gpuReleaseStableSeconds < 0:
            raise ValueError("gpuReleaseStableSeconds must not be negative")
        if gpuReleaseMemoryToleranceMiB < 0:
            raise ValueError("gpuReleaseMemoryToleranceMiB must not be negative")
        if pairRetries < 0:
            raise ValueError("pairRetries must not be negative")
        self.availableGpuIds = availableGpuIds
        self.batchSize = int(batchSize)
        self.initializeTimeout = float(initializeTimeout)
        self.taskTimeout = float(taskTimeout)
        self.shutdownGracePeriod = float(shutdownGracePeriod)
        self.gpuReleaseTimeout = float(gpuReleaseTimeout)
        self.gpuReleaseStableSeconds = float(gpuReleaseStableSeconds)
        self.gpuReleaseMemoryTolerance = int(gpuReleaseMemoryToleranceMiB) * 1024 * 1024
        self.pairRetries = int(pairRetries)
        self.outputRoot = Path(outputRoot)
        self.tuiEnabled = tui
        self.verbose = verbose
        self.outputDir: Optional[Path] = None

    def Evaluate(
        self,
        tasks: Iterable[Task],
        methods: Iterable[Method],
        metrics: Iterable[Metric],
    ) -> Dict[str, Any]:
        tasks = list(tasks)
        methods = list(methods)
        metrics = list(metrics)
        self._metrics = metrics
        effectiveBatchSizes = [
            method.EffectiveBatchSize(self.batchSize) for method in methods
        ]
        self._gpuPool, self._gpuSnapshot = ResolveGpuIds(self.availableGpuIds)
        self.outputDir = self._CreateOutputDir()
        maxAttempts = self.pairRetries + 1

        ctx = BuildRunContext(
            methods=methods,
            tasks=tasks,
            effectiveBatchSizes=effectiveBatchSizes,
            gpuPool=self._gpuPool,
            gpuSnapshot=self._gpuSnapshot,
            outputDir=self.outputDir,
            maxAttempts=maxAttempts,
        )
        ctx.eventQueue = ctx.mpContext.Queue()
        ctx.eventsFile = ctx.eventsPath.open("a", encoding="utf-8", buffering=1)
        self._tui = BenchmarkTui(enabled=self.tuiEnabled)
        self.reporter = Reporter(ctx, self)
        self.gpuGovernor = GpuGovernor(ctx, self)
        self.scheduler = Scheduler(ctx, self.reporter, self)

        ctx.manifest = {
            "status": ctx.status,
            "started_at": datetime.fromtimestamp(ctx.startedWall).astimezone().isoformat(),
            "output_dir": str(self.outputDir.resolve()),
            "batch_size": self.batchSize,
            "effective_batch_sizes": [
                {
                    "method_index": index,
                    "method": method.Label,
                    "batch_size": effectiveBatchSizes[index],
                }
                for index, method in enumerate(methods)
            ],
            "timeouts": {
                "initialize": self.initializeTimeout,
                "task": self.taskTimeout,
                "shutdown_grace_period": self.shutdownGracePeriod,
                "gpu_release": self.gpuReleaseTimeout,
                "gpu_release_stable": self.gpuReleaseStableSeconds,
            },
            "gpu_release_memory_tolerance_mib": (
                self.gpuReleaseMemoryTolerance // (1024 * 1024)
            ),
            "pair_retries": self.pairRetries,
            "gpu_pool": self._gpuPool,
            "gpu_snapshot": [gpu.AsDict() for gpu in self._gpuSnapshot],
            "methods": [
                {
                    "index": index,
                    "class": f"{type(method).__module__}.{type(method).__qualname__}",
                    "label": method.Label,
                    "gpu_nums": method.gpuNums,
                    "perf_weight": method.perfWeight,
                }
                for index, method in enumerate(methods)
            ],
            "tasks": [
                {
                    "index": index,
                    "class": f"{type(task).__module__}.{type(task).__qualname__}",
                    "name": task.name,
                }
                for index, task in enumerate(tasks)
            ],
        }
        AtomicJson(self.outputDir / "manifest.json", ctx.manifest)

        self.reporter.writeReports()
        for methodIndex, method in enumerate(methods):
            if method.gpuNums <= len(self._gpuPool):
                continue
            ctx.pending[methodIndex].clear()
            for taskIndex in range(len(tasks)):
                self.reporter.pairFailure(
                    methodIndex, taskIndex,
                    error=(
                        f"requires {method.gpuNums} GPU(s), but the Engine "
                        f"pool contains {len(self._gpuPool)}: {self._gpuPool}"
                    ),
                    kind="unschedulable",
                )

        self._tui.Start()
        lastTuiUpdate = 0.0
        try:
            while True:
                try:
                    event = ctx.eventQueue.get(timeout=0.1)
                    self.scheduler.handleEvent(event)
                    while True:
                        self.scheduler.handleEvent(ctx.eventQueue.get_nowait())
                except queue.Empty:
                    pass

                now = time.monotonic()
                self.gpuGovernor.refreshCoolingGpus(now)
                for workerId, worker in list(ctx.workers.items()):
                    if worker.deadline is not None and now >= worker.deadline:
                        if worker.state == "initializing":
                            ctx.fatalError = (
                                f"{worker.methodLabel} initialization timed out "
                                f"after {self.initializeTimeout}s on GPUs "
                                f"{worker.gpuIds}"
                            )
                            ctx.fatalStatus = "initialization_failed"
                            self.scheduler.terminateWorker(worker)
                            worker.state = "failed"
                        elif worker.state == "busy":
                            reason = (
                                f"task timed out after {self.taskTimeout}s: "
                                f"{worker.methodLabel}/{worker.taskName} "
                                f"attempt {worker.attempt}"
                            )
                            self.reporter.recordEvent({
                                "type": "task_timeout", "time": time.time(),
                                "worker_id": worker.workerId,
                                "method": worker.methodLabel,
                                "task": worker.taskName, "attempt": worker.attempt,
                                "error": reason, "log_path": worker.logPath,
                            })
                            self.scheduler.recoverCurrentPair(worker, reason, "task_timeout")
                            self.scheduler.terminateWorker(worker)
                            worker.state = "failed"
                        elif worker.state == "stopping":
                            self.scheduler.terminateWorker(worker)
                            worker.deadline = now + min(5.0, self.shutdownGracePeriod)

                    if worker.process.is_alive():
                        continue
                    exitCode = worker.process.exitcode
                    if worker.state == "initializing":
                        ctx.fatalError = (
                            f"{worker.methodLabel} initialization process exited "
                            f"with code {exitCode}; log: {worker.instanceLog}"
                        )
                    elif worker.state in ("busy", "failed"):
                        self.scheduler.recoverCurrentPair(
                            worker, f"worker exited with code {exitCode}", "worker_exit"
                        )
                        self.scheduler.terminateWorker(worker, force=True)
                    self.scheduler.releaseWorker(workerId)

                if self._tui.cancelRequested:
                    ctx.cancelled = True
                    ctx.status = "cancelled"
                if ctx.fatalError:
                    ctx.status = ctx.fatalStatus or "failed"
                if ctx.fatalError or ctx.cancelled:
                    for worker in ctx.workers.values():
                        self.scheduler.stopWorker(worker)
                    break

                for worker in list(ctx.workers.values()):
                    if worker.state == "idle":
                        self.scheduler.dispatch(worker)

                while True:
                    methodIndex = self.scheduler.chooseMethod()
                    if methodIndex is None:
                        break
                    self.gpuGovernor.validateFreeGpus(now)
                    methodIndex = self.scheduler.chooseMethod()
                    if methodIndex is None:
                        break
                    self.scheduler.spawnWorker(methodIndex)

                terminal = all(
                    value in ("done", "failed", "unschedulable")
                    for value in ctx.pairStatus.values()
                )
                if terminal:
                    for worker in ctx.workers.values():
                        if worker.state == "idle":
                            self.scheduler.stopWorker(worker)
                    if not ctx.workers and not ctx.coolingGpus:
                        break

                self.gpuGovernor.refreshGpuSnapshot(now)
                if now - lastTuiUpdate >= 0.2:
                    self._tui.Update(self.reporter.snapshot())
                    lastTuiUpdate = now
        except KeyboardInterrupt:
            ctx.cancelled = True
            ctx.status = "cancelled"
            for worker in ctx.workers.values():
                self.scheduler.stopWorker(worker)
        finally:
            # A second Ctrl-C must not interrupt process-group cleanup and
            # leave GPU-owning descendants behind.
            previousSigintHandler = None
            try:
                previousSigintHandler = signal.signal(signal.SIGINT, signal.SIG_IGN)
            except ValueError:  # Evaluate called outside the main thread
                pass
            shutdownDeadline = time.monotonic() + self.shutdownGracePeriod
            for worker in ctx.workers.values():
                self.scheduler.stopWorker(worker)
            while ctx.workers and time.monotonic() < shutdownDeadline:
                try:
                    self.scheduler.handleEvent(ctx.eventQueue.get(timeout=0.1))
                except queue.Empty:
                    pass
                for workerId, worker in list(ctx.workers.items()):
                    if not worker.process.is_alive():
                        self.scheduler.releaseWorker(workerId)
            for worker in list(ctx.workers.values()):
                self.scheduler.terminateWorker(worker)
            forceDeadline = time.monotonic() + 5.0
            while ctx.workers and time.monotonic() < forceDeadline:
                for workerId, worker in list(ctx.workers.items()):
                    if not worker.process.is_alive():
                        self.scheduler.releaseWorker(workerId)
                time.sleep(0.05)
            for workerId, worker in list(ctx.workers.items()):
                self.scheduler.terminateWorker(worker, force=True)
                worker.process.join(timeout=1.0)
                self.scheduler.releaseWorker(workerId)

            while ctx.coolingGpus:
                now = time.monotonic()
                self.gpuGovernor.refreshCoolingGpus(now, forcePoll=True)
                if not ctx.coolingGpus or all(
                    item.get("timed_out") for item in ctx.coolingGpus.values()
                ):
                    break
                time.sleep(0.1)

            if ctx.fatalError and ctx.status == "running":
                ctx.status = ctx.fatalStatus or "failed"

            totalDuration = time.monotonic() - ctx.startedMono
            ctx.timings.append({
                "kind": "benchmark_total", "method": "", "task": "",
                "worker_id": "", "attempt": "", "duration": totalDuration,
            })
            if ctx.status == "running":
                ctx.status = "completed"
            ctx.manifest.update({
                "status": ctx.status,
                "finished_at": datetime.now().astimezone().isoformat(),
                "total_duration": totalDuration,
                "fatal_error": ctx.fatalError,
                "unreleased_gpus": {
                    str(gpuId): details
                    for gpuId, details in ctx.coolingGpus.items()
                },
                "worker_history": ctx.workerHistory,
            })
            AtomicJson(self.outputDir / "manifest.json", ctx.manifest)
            finalReport = self.reporter.writeReports(ctx.status)
            self._tui.Update(self.reporter.snapshot())
            self._tui.FinishAndWait()
            self._tui.Stop()
            ctx.eventsFile.close()
            ctx.eventQueue.close()
            ctx.eventQueue.join_thread()
            if previousSigintHandler is not None:
                signal.signal(signal.SIGINT, previousSigintHandler)

        if self.verbose and not self._tui.enabled:
            print(
                f"[engine] {ctx.status}: {len(ctx.results)} succeeded, "
                f"{len(ctx.failures)} failed; outputs={self.outputDir.resolve()}"
            )
        if ctx.fatalError and ctx.status == "resource_release_failed":
            raise BenchmarkResourceReleaseError(
                f"{ctx.fatalError}; partial outputs: {self.outputDir.resolve()}"
            )
        if ctx.fatalError:
            raise BenchmarkInitializationError(
                f"{ctx.fatalError}; partial outputs: {self.outputDir.resolve()}"
            )
        return finalReport

    def _CreateOutputDir(self) -> Path:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        candidate = self.outputRoot / f"{stamp}-{os.getpid()}"
        serial = 1
        while candidate.exists():
            candidate = self.outputRoot / f"{stamp}-{os.getpid()}-{serial}"
            serial += 1
        candidate.mkdir(parents=True)
        return candidate