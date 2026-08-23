"""Concurrent GPU scheduler, process supervisor, and report coordinator."""

import json
import multiprocessing as mp
import os
import queue
import signal
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple, Union

from helpers.Gpu import QueryGpus, ResolveGpuIds
from .Method import Method
from .Metrics import Metric
from .Task import Task
from .Worker import WorkerMain
from .tui import BenchmarkTui


def _MeanValue(stats: Dict[str, Any]) -> Any:
    if not stats:
        return None
    if "mean" in stats:
        return stats["mean"]
    for key, value in stats.items():
        if key.endswith("_mean"):
            return value
    return next(iter(stats.values()), None)


def _CoreReport(run: Dict[str, Any]) -> Dict[str, Any]:
    core: Dict[str, Any] = {"method": run["method"], "task": run["task"]}
    for group in ("task_metrics", "system_metrics", "method_metrics"):
        for name, stats in run.get(group, {}).items():
            core[name] = _MeanValue(stats)
    return core


def _Slug(value: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in value
    ).strip("._")
    return cleaned or "unnamed"


def _AtomicJson(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False, default=str)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


@dataclass
class _WorkerState:
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


class BenchmarkInitializationError(RuntimeError):
    """Raised after an instance cannot initialize within the configured limit."""


class BenchmarkResourceReleaseError(RuntimeError):
    """Raised when a worker exits but its assigned GPU stays occupied."""


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
        gpuPool, gpuSnapshot = ResolveGpuIds(self.availableGpuIds)
        self.outputDir = self._CreateOutputDir()
        startedWall = time.time()
        startedMono = time.monotonic()
        maxAttempts = self.pairRetries + 1

        pending: Dict[int, Deque[int]] = {
            methodIndex: deque(range(len(tasks)))
            for methodIndex in range(len(methods))
        }
        pairStatus: Dict[Tuple[int, int], str] = {
            (methodIndex, taskIndex): "pending"
            for methodIndex in range(len(methods))
            for taskIndex in range(len(tasks))
        }
        attempts: Dict[Tuple[int, int], int] = {pair: 0 for pair in pairStatus}
        results: Dict[Tuple[int, int], Dict[str, Any]] = {}
        failures: List[Dict[str, Any]] = []
        timings: List[Dict[str, Any]] = []
        logs: List[Dict[str, str]] = []
        knownLogs = set()
        workers: Dict[str, _WorkerState] = {}
        workerHistory: List[Dict[str, Any]] = []
        freeGpus = list(gpuPool)
        coolingGpus: Dict[int, Dict[str, Any]] = {}
        gpuBaseline = {gpu.id: gpu.memoryUsed for gpu in gpuSnapshot}
        gpuBaselinePids = {gpu.id: set(gpu.computePids) for gpu in gpuSnapshot}
        lastGpuReleasePoll = 0.0
        workerSerial = 0
        fatalError: Optional[str] = None
        fatalStatus: Optional[str] = None
        cancelled = False
        status = "running"

        context = mp.get_context("spawn")
        eventQueue = context.Queue()
        eventsPath = self.outputDir / "events.jsonl"
        eventsFile = eventsPath.open("a", encoding="utf-8", buffering=1)
        tui = BenchmarkTui(enabled=self.tuiEnabled)

        manifest = {
            "status": status,
            "started_at": datetime.fromtimestamp(startedWall).astimezone().isoformat(),
            "output_dir": str(self.outputDir.resolve()),
            "batch_size": self.batchSize,
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
            "gpu_pool": gpuPool,
            "gpu_snapshot": [gpu.AsDict() for gpu in gpuSnapshot],
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
        _AtomicJson(self.outputDir / "manifest.json", manifest)

        def addLog(path: str, label: str) -> None:
            if path not in knownLogs:
                knownLogs.add(path)
                logs.append({"path": path, "label": label})

        def recordEvent(event: Dict[str, Any]) -> None:
            persisted = {key: value for key, value in event.items() if key != "report"}
            eventsFile.write(json.dumps(persisted, ensure_ascii=False, default=str) + "\n")

        def writeReports(finalStatus: Optional[str] = None) -> Dict[str, Any]:
            orderedRuns = [results[pair] for pair in sorted(results)]
            report = {
                "status": finalStatus or status,
                "output_dir": str(self.outputDir.resolve()),
                "runs": orderedRuns,
                "cores": [_CoreReport(run) for run in orderedRuns],
                "failures": sorted(
                    failures,
                    key=lambda item: (item["method_index"], item["task_index"]),
                ),
            }
            _AtomicJson(self.outputDir / "results" / "full.json", report)
            _AtomicJson(self.outputDir / "results" / "core.json", report["cores"])
            _AtomicJson(self.outputDir / "results" / "timing.json", timings)
            _AtomicJson(self.outputDir / "results" / "failures.json", report["failures"])
            return report

        def pairFailure(
            methodIndex: int,
            taskIndex: int,
            *,
            error: str,
            kind: str,
            logPath: str = "",
            tracebackText: str = "",
        ) -> None:
            pair = (methodIndex, taskIndex)
            if pairStatus.get(pair) in ("done", "failed", "unschedulable"):
                return
            pairStatus[pair] = "unschedulable" if kind == "unschedulable" else "failed"
            failure = {
                "method_index": methodIndex,
                "task_index": taskIndex,
                "method": methods[methodIndex].Label,
                "task": tasks[taskIndex].name,
                "kind": kind,
                "attempts": attempts[pair],
                "error": error,
                "traceback": tracebackText,
                "log_path": logPath,
            }
            failures.append(failure)
            self._WritePair(methodIndex, taskIndex, methods, tasks, failure=failure)
            writeReports()

        def activeCount(methodIndex: int) -> int:
            return sum(
                worker.methodIndex == methodIndex and worker.state != "stopping"
                for worker in workers.values()
            )

        def chooseMethod() -> Optional[int]:
            candidates = []
            for methodIndex, method in enumerate(methods):
                if not pending[methodIndex] or method.gpuNums > len(freeGpus):
                    continue
                active = activeCount(methodIndex)
                nonterminal = sum(
                    pairStatus[(methodIndex, taskIndex)] in ("pending", "running")
                    for taskIndex in range(len(tasks))
                )
                if active >= nonterminal:
                    continue
                work = method.perfWeight * nonterminal
                score = (
                    work / method.gpuNums
                    if active == 0
                    else work / (active * (active + 1) * method.gpuNums)
                )
                candidates.append((score, work, -methodIndex, methodIndex))
            return max(candidates)[-1] if candidates else None

        def spawnWorker(methodIndex: int) -> None:
            nonlocal workerSerial
            method = methods[methodIndex]
            gpuIds = [freeGpus.pop(0) for _ in range(method.gpuNums)]
            workerId = f"m{methodIndex:02d}-i{workerSerial:03d}"
            workerSerial += 1
            methodDir = self.outputDir / "logs" / f"{methodIndex:02d}-{_Slug(method.Label)}"
            instanceLog = str((methodDir / f"{workerId}-instance.log").resolve())
            addLog(instanceLog, f"{method.Label} {workerId} instance/runtime")
            parentConnection, childConnection = context.Pipe()
            process = context.Process(
                target=WorkerMain,
                name=f"kvbench-{workerId}",
                args=(workerId, methodIndex, method, metrics, gpuIds, self.batchSize,
                      childConnection, eventQueue, instanceLog),
            )
            process.start()
            childConnection.close()
            workers[workerId] = _WorkerState(
                workerId=workerId,
                methodIndex=methodIndex,
                methodLabel=method.Label,
                gpuIds=gpuIds,
                process=process,
                connection=parentConnection,
                instanceLog=instanceLog,
                deadline=time.monotonic() + self.initializeTimeout,
            )
            recordEvent({
                "type": "worker_spawned", "time": time.time(), "worker_id": workerId,
                "method_index": methodIndex, "method": method.Label,
                "gpu_ids": gpuIds, "pid": process.pid, "log_path": instanceLog,
            })

        def dispatch(worker: _WorkerState) -> None:
            if not pending[worker.methodIndex]:
                stopWorker(worker)
                return
            taskIndex = pending[worker.methodIndex].popleft()
            pair = (worker.methodIndex, taskIndex)
            if pairStatus[pair] != "pending":
                return
            startAttempt = attempts[pair] + 1
            methodDir = Path(worker.instanceLog).parent
            logPaths = {
                attempt: str((methodDir / (
                    f"{worker.workerId}-t{taskIndex:03d}-{_Slug(tasks[taskIndex].name)}-"
                    f"attempt{attempt}.log"
                )).resolve())
                for attempt in range(startAttempt, maxAttempts + 1)
            }
            for attempt, path in logPaths.items():
                addLog(path, f"{methods[worker.methodIndex].Label} / "
                             f"{tasks[taskIndex].name} attempt {attempt}")
            worker.connection.send({
                "op": "task", "task_index": taskIndex, "task": tasks[taskIndex],
                "start_attempt": startAttempt, "max_attempts": maxAttempts,
                "log_paths": logPaths,
            })
            pairStatus[pair] = "running"
            worker.state = "busy"
            worker.taskIndex = taskIndex
            worker.taskName = tasks[taskIndex].name
            worker.attempt = startAttempt
            worker.logPath = logPaths[startAttempt]
            worker.deadline = time.monotonic() + self.taskTimeout
            recordEvent({
                "type": "task_dispatched",
                "time": time.time(),
                "worker_id": worker.workerId,
                "method_index": worker.methodIndex,
                "task_index": taskIndex,
                "method": methods[worker.methodIndex].Label,
                "task": tasks[taskIndex].name,
                "attempt": startAttempt,
                "log_path": logPaths[startAttempt],
            })

        def stopWorker(worker: _WorkerState) -> None:
            if worker.state == "stopping":
                return
            try:
                worker.connection.send({"op": "shutdown"})
            except (BrokenPipeError, EOFError, OSError):
                pass
            worker.state = "stopping"
            worker.taskName = ""
            worker.taskIndex = None
            worker.deadline = time.monotonic() + self.shutdownGracePeriod

        def terminateWorker(worker: _WorkerState, force: bool = False) -> None:
            process = worker.process
            sig = signal.SIGKILL if force else signal.SIGTERM
            try:
                processGroup = os.getpgid(process.pid)
                if processGroup == process.pid:
                    os.killpg(processGroup, sig)
                elif force:
                    process.kill()
                else:
                    process.terminate()
            except ProcessLookupError:
                # The group leader may already be dead while a nested helper
                # still owns its process group. The PID is not reused before
                # multiprocessing joins/reaps it.
                try:
                    os.killpg(process.pid, sig)
                except (ProcessLookupError, PermissionError, OSError):
                    pass
            except (PermissionError, OSError):
                try:
                    process.kill() if force else process.terminate()
                except Exception:  # noqa: BLE001
                    pass

        def beginGpuCooling(
            gpuId: int,
            *,
            workerId: str,
            method: str,
            reason: str,
            now: float,
        ) -> None:
            coolingGpus[gpuId] = {
                "worker_id": workerId,
                "method": method,
                "reason": reason,
                "started": now,
                "deadline": now + self.gpuReleaseTimeout,
                "baseline_memory": gpuBaseline[gpuId],
                "baseline_compute_pids": sorted(gpuBaselinePids[gpuId]),
                "last_memory": None,
                "last_compute_pids": [],
                "external_compute_pids": [],
                "clean_since": None,
            }

        def releaseWorker(workerId: str) -> None:
            worker = workers.pop(workerId)
            # Even an orderly Method.Close may leave a backend descendant
            # alive. The outer worker is the process-group leader, so sweep
            # the complete group before its PID can be reaped/reused.
            terminateWorker(worker, force=True)
            worker.process.join(timeout=0.2)
            try:
                worker.connection.close()
            except Exception:
                pass
            now = time.monotonic()
            for gpuId in worker.gpuIds:
                beginGpuCooling(
                    gpuId,
                    workerId=worker.workerId,
                    method=worker.methodLabel,
                    reason="worker_exit",
                    now=now,
                )
            recordEvent({
                "type": "gpu_cooling_started",
                "time": time.time(),
                "worker_id": worker.workerId,
                "method": worker.methodLabel,
                "gpu_ids": worker.gpuIds,
            })
            workerHistory.append(worker.Snapshot())

        def gpuIsClean(gpuId: int, info: Any) -> Tuple[bool, List[int]]:
            externalPids = sorted(set(info.computePids) - gpuBaselinePids[gpuId])
            releaseLimit = gpuBaseline[gpuId] + self.gpuReleaseMemoryTolerance
            return info.memoryUsed <= releaseLimit and not externalPids, externalPids

        def validateFreeGpus(now: float) -> None:
            """Quarantine pool GPUs taken by another process before dispatch."""
            if not freeGpus:
                return
            try:
                current = {gpu.id: gpu for gpu in QueryGpus()}
            except RuntimeError as exc:
                for gpuId in list(freeGpus):
                    freeGpus.remove(gpuId)
                    beginGpuCooling(
                        gpuId,
                        workerId="",
                        method="",
                        reason="gpu_validation_error",
                        now=now,
                    )
                recordEvent({
                    "type": "gpu_validation_failed",
                    "time": time.time(),
                    "error": str(exc),
                    "gpu_ids": sorted(coolingGpus),
                })
                return
            for gpuId in list(freeGpus):
                info = current.get(gpuId)
                clean, externalPids = (
                    gpuIsClean(gpuId, info) if info is not None else (False, [])
                )
                if clean:
                    continue
                freeGpus.remove(gpuId)
                beginGpuCooling(
                    gpuId,
                    workerId="",
                    method="",
                    reason="external_contention",
                    now=now,
                )
                coolingGpus[gpuId]["last_memory"] = (
                    info.memoryUsed if info is not None else None
                )
                coolingGpus[gpuId]["last_compute_pids"] = (
                    list(info.computePids) if info is not None else []
                )
                coolingGpus[gpuId]["external_compute_pids"] = externalPids
                recordEvent({
                    "type": "gpu_contention_detected",
                    "time": time.time(),
                    "gpu_id": gpuId,
                    "memory_used": info.memoryUsed if info is not None else None,
                    "baseline_memory": gpuBaseline[gpuId],
                    "compute_pids": list(info.computePids) if info is not None else [],
                    "external_compute_pids": externalPids,
                })

        def refreshCoolingGpus(now: float, *, forcePoll: bool = False) -> None:
            nonlocal lastGpuReleasePoll, fatalError, fatalStatus
            if not coolingGpus:
                return
            if not forcePoll and now - lastGpuReleasePoll < 0.2:
                return
            lastGpuReleasePoll = now
            try:
                current = {gpu.id: gpu for gpu in QueryGpus()}
            except RuntimeError as exc:
                current = {}
                queryError = str(exc)
            else:
                queryError = ""

            released = []
            for gpuId, cooling in list(coolingGpus.items()):
                info = current.get(gpuId)
                if info is not None:
                    cooling["last_memory"] = info.memoryUsed
                    cooling["last_compute_pids"] = list(info.computePids)
                    clean, externalPids = gpuIsClean(gpuId, info)
                    cooling["external_compute_pids"] = externalPids
                    if clean and cooling["clean_since"] is None:
                        cooling["clean_since"] = now
                    elif not clean:
                        cooling["clean_since"] = None
                    stableFor = (
                        now - cooling["clean_since"]
                        if cooling["clean_since"] is not None else 0.0
                    )
                    if clean and stableFor >= self.gpuReleaseStableSeconds:
                        released.append(gpuId)
                        timings.append({
                            "kind": "gpu_release",
                            "worker_id": cooling["worker_id"],
                            "method": cooling["method"],
                            "task": "",
                            "attempt": "",
                            "gpu_id": gpuId,
                            "duration": now - cooling["started"],
                        })
                        recordEvent({
                            "type": "gpu_released",
                            "time": time.time(),
                            "worker_id": cooling["worker_id"],
                            "method": cooling["method"],
                            "gpu_id": gpuId,
                            "memory_used": info.memoryUsed,
                            "baseline_memory": cooling["baseline_memory"],
                            "compute_pids": list(info.computePids),
                            "stable_duration": stableFor,
                            "cooling_duration": now - cooling["started"],
                        })
                        continue

                if now < cooling["deadline"] or cooling.get("timed_out"):
                    continue
                cooling["timed_out"] = True
                detail = (
                    f"last memory={cooling['last_memory']} bytes, compute PIDs="
                    f"{cooling['last_compute_pids']}, external compute PIDs="
                    f"{cooling['external_compute_pids']}"
                    if info is not None
                    else f"NVML query failed: {queryError or 'GPU disappeared'}"
                )
                releaseError = (
                    f"GPU {gpuId} did not return to its startup memory baseline "
                    f"within {self.gpuReleaseTimeout}s ({detail}, baseline="
                    f"{cooling['baseline_memory']} bytes, tolerance="
                    f"{self.gpuReleaseMemoryTolerance} bytes, stable window="
                    f"{self.gpuReleaseStableSeconds}s)"
                )
                if fatalError:
                    fatalError = f"{fatalError}; additionally: {releaseError}"
                else:
                    fatalStatus = "resource_release_failed"
                    fatalError = releaseError
                recordEvent({
                    "type": "gpu_release_failed",
                    "time": time.time(),
                    "worker_id": cooling["worker_id"],
                    "method": cooling["method"],
                    "gpu_id": gpuId,
                    "error": releaseError,
                })

            for gpuId in released:
                coolingGpus.pop(gpuId, None)
                freeGpus.append(gpuId)
            freeGpus.sort(key=gpuPool.index)

        def recoverCurrentPair(worker: _WorkerState, reason: str, kind: str) -> None:
            if worker.taskIndex is None:
                return
            pair = (worker.methodIndex, worker.taskIndex)
            if pairStatus.get(pair) != "running":
                return
            attempt = max(attempts[pair], worker.attempt, 1)
            attempts[pair] = attempt
            timings.append({
                "kind": kind, "worker_id": worker.workerId, "method": worker.methodLabel,
                "task": worker.taskName, "attempt": attempt,
                "duration": self.taskTimeout if kind == "task_timeout" else 0.0,
            })
            if attempt < maxAttempts:
                pairStatus[pair] = "pending"
                pending[worker.methodIndex].appendleft(worker.taskIndex)
            else:
                pairFailure(worker.methodIndex, worker.taskIndex, error=reason,
                            kind=kind, logPath=worker.logPath)

        def handleEvent(event: Dict[str, Any]) -> None:
            nonlocal fatalError, fatalStatus
            recordEvent(event)
            worker = workers.get(event.get("worker_id"))
            kind = event.get("type")
            if worker is None:
                return
            if event.get("log_path"):
                worker.logPath = event["log_path"]
            if kind == "initialize_done":
                worker.state = "idle"
                worker.deadline = None
                worker.initDuration = float(event["duration"])
                timings.append({
                    "kind": "initialize", "worker_id": worker.workerId,
                    "method": worker.methodLabel, "task": "", "attempt": "",
                    "duration": worker.initDuration,
                })
            elif kind == "initialize_failed":
                fatalError = (f"{worker.methodLabel} initialization failed on GPUs "
                              f"{worker.gpuIds}: {event.get('error')}")
                fatalStatus = "initialization_failed"
                worker.state = "failed"
            elif kind == "task_started":
                taskIndex = int(event["task_index"])
                pair = (worker.methodIndex, taskIndex)
                attempt = int(event["attempt"])
                attempts[pair] = max(attempts[pair], attempt)
                worker.attempt = attempt
                worker.state = "busy"
                worker.taskIndex = taskIndex
                worker.taskName = event["task"]
                worker.deadline = time.monotonic() + self.taskTimeout
            elif kind == "task_attempt_failed":
                timings.append({
                    "kind": "task_attempt_failed", "worker_id": worker.workerId,
                    "method": worker.methodLabel, "task": event["task"],
                    "attempt": event["attempt"], "duration": event["duration"],
                })
            elif kind == "task_done":
                pair = (worker.methodIndex, int(event["task_index"]))
                pairStatus[pair] = "done"
                results[pair] = event["report"]
                timings.append({
                    "kind": "task", "worker_id": worker.workerId,
                    "method": worker.methodLabel, "task": event["task"],
                    "attempt": event["attempt"], "duration": event["duration"],
                })
                self._WritePair(pair[0], pair[1], methods, tasks, report=event["report"])
                writeReports()
            elif kind == "task_failed":
                pairFailure(worker.methodIndex, int(event["task_index"]),
                            error=event.get("error", "pair failed"), kind="exception",
                            logPath=event.get("log_path", ""),
                            tracebackText=event.get("traceback", ""))
            elif kind == "worker_idle":
                worker.state = "idle"
                worker.taskIndex = None
                worker.taskName = ""
                worker.attempt = 0
                worker.logPath = worker.instanceLog
                worker.deadline = None
            elif kind == "worker_closed":
                worker.state = "closed"
                worker.deadline = None
                timings.append({
                    "kind": "close", "worker_id": worker.workerId,
                    "method": worker.methodLabel, "task": "", "attempt": "",
                    "duration": event.get("duration", 0.0),
                })

        def snapshot() -> Dict[str, Any]:
            return {
                "status": status,
                "elapsed": time.monotonic() - startedMono,
                "output_dir": str(self.outputDir),
                "progress": {
                    "done": sum(value == "done" for value in pairStatus.values()),
                    "failed": sum(value in ("failed", "unschedulable")
                                  for value in pairStatus.values()),
                    "total": len(pairStatus),
                },
                "gpu_pool": gpuPool,
                "free_gpus": freeGpus,
                "cooling_gpus": sorted(coolingGpus),
                "gpu_snapshot": [gpu.AsDict() for gpu in gpuSnapshot],
                "workers": [worker.Snapshot() for worker in workers.values()],
                "runs": [results[pair] for pair in sorted(results)],
                "cores": [_CoreReport(results[pair]) for pair in sorted(results)],
                "timings": timings, "failures": failures, "logs": logs,
            }

        writeReports()
        for methodIndex, method in enumerate(methods):
            if method.gpuNums <= len(gpuPool):
                continue
            pending[methodIndex].clear()
            for taskIndex in range(len(tasks)):
                pairFailure(methodIndex, taskIndex,
                            error=(f"requires {method.gpuNums} GPU(s), but the Engine "
                                   f"pool contains {len(gpuPool)}: {gpuPool}"),
                            kind="unschedulable")

        tui.Start()
        lastTuiUpdate = 0.0
        try:
            while True:
                try:
                    event = eventQueue.get(timeout=0.1)
                    handleEvent(event)
                    while True:
                        handleEvent(eventQueue.get_nowait())
                except queue.Empty:
                    pass

                now = time.monotonic()
                refreshCoolingGpus(now)
                for workerId, worker in list(workers.items()):
                    if worker.deadline is not None and now >= worker.deadline:
                        if worker.state == "initializing":
                            fatalError = (f"{worker.methodLabel} initialization timed out "
                                          f"after {self.initializeTimeout}s on GPUs "
                                          f"{worker.gpuIds}")
                            fatalStatus = "initialization_failed"
                            terminateWorker(worker)
                            worker.state = "failed"
                        elif worker.state == "busy":
                            reason = (f"task timed out after {self.taskTimeout}s: "
                                      f"{worker.methodLabel}/{worker.taskName} "
                                      f"attempt {worker.attempt}")
                            recordEvent({
                                "type": "task_timeout", "time": time.time(),
                                "worker_id": worker.workerId, "method": worker.methodLabel,
                                "task": worker.taskName, "attempt": worker.attempt,
                                "error": reason, "log_path": worker.logPath,
                            })
                            recoverCurrentPair(worker, reason, "task_timeout")
                            terminateWorker(worker)
                            worker.state = "failed"
                        elif worker.state == "stopping":
                            terminateWorker(worker)
                            worker.deadline = now + min(5.0, self.shutdownGracePeriod)

                    if worker.process.is_alive():
                        continue
                    exitCode = worker.process.exitcode
                    if worker.state == "initializing":
                        fatalError = (f"{worker.methodLabel} initialization process exited "
                                      f"with code {exitCode}; log: {worker.instanceLog}")
                    elif worker.state in ("busy", "failed"):
                        recoverCurrentPair(worker, f"worker exited with code {exitCode}",
                                           "worker_exit")
                        terminateWorker(worker, force=True)
                    releaseWorker(workerId)

                if tui.cancelRequested:
                    cancelled = True
                    status = "cancelled"
                if fatalError:
                    status = fatalStatus or "failed"
                if fatalError or cancelled:
                    for worker in workers.values():
                        stopWorker(worker)
                    break

                for worker in list(workers.values()):
                    if worker.state == "idle":
                        dispatch(worker)

                while True:
                    methodIndex = chooseMethod()
                    if methodIndex is None:
                        break
                    validateFreeGpus(now)
                    methodIndex = chooseMethod()
                    if methodIndex is None:
                        break
                    spawnWorker(methodIndex)

                terminal = all(value in ("done", "failed", "unschedulable")
                               for value in pairStatus.values())
                if terminal:
                    for worker in workers.values():
                        if worker.state == "idle":
                            stopWorker(worker)
                    if not workers and not coolingGpus:
                        break

                if now - lastTuiUpdate >= 0.2:
                    tui.Update(snapshot())
                    lastTuiUpdate = now
        except KeyboardInterrupt:
            cancelled = True
            status = "cancelled"
            for worker in workers.values():
                stopWorker(worker)
        finally:
            # A second Ctrl-C must not interrupt process-group cleanup and
            # leave GPU-owning descendants behind.
            previousSigintHandler = None
            try:
                previousSigintHandler = signal.signal(signal.SIGINT, signal.SIG_IGN)
            except ValueError:  # Evaluate called outside the main thread
                pass
            shutdownDeadline = time.monotonic() + self.shutdownGracePeriod
            for worker in workers.values():
                stopWorker(worker)
            while workers and time.monotonic() < shutdownDeadline:
                try:
                    handleEvent(eventQueue.get(timeout=0.1))
                except queue.Empty:
                    pass
                for workerId, worker in list(workers.items()):
                    if not worker.process.is_alive():
                        releaseWorker(workerId)
            for worker in list(workers.values()):
                terminateWorker(worker)
            forceDeadline = time.monotonic() + 5.0
            while workers and time.monotonic() < forceDeadline:
                for workerId, worker in list(workers.items()):
                    if not worker.process.is_alive():
                        releaseWorker(workerId)
                time.sleep(0.05)
            for workerId, worker in list(workers.items()):
                terminateWorker(worker, force=True)
                worker.process.join(timeout=1.0)
                releaseWorker(workerId)

            while coolingGpus:
                now = time.monotonic()
                refreshCoolingGpus(now, forcePoll=True)
                if not coolingGpus or all(
                    item.get("timed_out") for item in coolingGpus.values()
                ):
                    break
                time.sleep(0.1)

            if fatalError and status == "running":
                status = fatalStatus or "failed"

            totalDuration = time.monotonic() - startedMono
            timings.append({
                "kind": "benchmark_total", "method": "", "task": "",
                "worker_id": "", "attempt": "", "duration": totalDuration,
            })
            if status == "running":
                status = "completed"
            manifest.update({
                "status": status,
                "finished_at": datetime.now().astimezone().isoformat(),
                "total_duration": totalDuration,
                "fatal_error": fatalError,
                "unreleased_gpus": {
                    str(gpuId): details
                    for gpuId, details in coolingGpus.items()
                },
                "worker_history": workerHistory,
            })
            _AtomicJson(self.outputDir / "manifest.json", manifest)
            finalReport = writeReports(status)
            tui.Update(snapshot())
            tui.FinishAndWait()
            tui.Stop()
            eventsFile.close()
            eventQueue.close()
            eventQueue.join_thread()
            if previousSigintHandler is not None:
                signal.signal(signal.SIGINT, previousSigintHandler)

        if self.verbose and not tui.enabled:
            print(f"[engine] {status}: {len(results)} succeeded, {len(failures)} "
                  f"failed; outputs={self.outputDir.resolve()}")
        if fatalError and status == "resource_release_failed":
            raise BenchmarkResourceReleaseError(
                f"{fatalError}; partial outputs: {self.outputDir.resolve()}"
            )
        if fatalError:
            raise BenchmarkInitializationError(
                f"{fatalError}; partial outputs: {self.outputDir.resolve()}"
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

    def _WritePair(
        self,
        methodIndex: int,
        taskIndex: int,
        methods: List[Method],
        tasks: List[Task],
        *,
        report: Optional[Dict[str, Any]] = None,
        failure: Optional[Dict[str, Any]] = None,
    ) -> None:
        assert self.outputDir is not None
        path = (self.outputDir / "pairs" /
                f"{methodIndex:02d}-{_Slug(methods[methodIndex].Label)}" /
                f"{taskIndex:03d}-{_Slug(tasks[taskIndex].name)}.json")
        payload = {
            "status": "success" if report is not None else "failed",
            "method_index": methodIndex,
            "task_index": taskIndex,
            "method": methods[methodIndex].Label,
            "task": tasks[taskIndex].name,
        }
        if report is not None:
            payload["report"] = report
        if failure is not None:
            payload["failure"] = failure
        _AtomicJson(path, payload)
