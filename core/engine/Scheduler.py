"""Worker / pair scheduler for the Engine.

The Scheduler owns the per-method work queue, the worker processes, and
the pair lifecycle. It reads from :class:`RunContext` and delegates every
side effect to :class:`Reporter` (event log + JSON reports) or
:class:`GpuGovernor` (GPU cooling on worker exit).
"""

from __future__ import annotations

import os
import signal
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from ..Worker import WorkerMain
from .State import Slug

if TYPE_CHECKING:
    from .Engine import Engine
    from .Reporter import Reporter
    from .State import RunContext, _WorkerState


class Scheduler:
    """Per-method work queue, worker lifecycle, and pair state machine."""

    def __init__(
        self, ctx: "RunContext", reporter: "Reporter", engine: "Engine"
    ) -> None:
        self.ctx = ctx
        self.reporter = reporter
        self.engine = engine

    # --------------------------------------------------------- scheduling
    def activeCount(self, methodIndex: int) -> int:
        return sum(
            worker.methodIndex == methodIndex and worker.state != "stopping"
            for worker in self.ctx.workers.values()
        )

    def chooseMethod(self) -> Optional[int]:
        candidates = []
        for methodIndex, method in enumerate(self.ctx.methods):
            if not self.ctx.pending[methodIndex] or method.gpuNums > len(self.ctx.freeGpus):
                continue
            active = self.activeCount(methodIndex)
            nonterminal = sum(
                self.ctx.pairStatus[(methodIndex, taskIndex)] in ("pending", "running")
                for taskIndex in range(len(self.ctx.tasks))
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

    def spawnWorker(self, methodIndex: int) -> None:
        method = self.ctx.methods[methodIndex]
        gpuIds = [self.ctx.freeGpus.pop(0) for _ in range(method.gpuNums)]
        workerSerial = self.ctx.workerSerial
        self.ctx.workerSerial += 1
        workerId = f"m{methodIndex:02d}-i{workerSerial:03d}"
        methodDir = (
            self.ctx.outputDir / "logs" / f"{methodIndex:02d}-{Slug(method.Label)}"
        )
        instanceLog = str((methodDir / f"{workerId}-instance.log").resolve())
        self.reporter.addLog(
            instanceLog, f"{method.Label} {workerId} instance/runtime"
        )
        parentConnection, childConnection = self.ctx.mpContext.Pipe()
        process = self.ctx.mpContext.Process(
            target=WorkerMain,
            name=f"kvbench-{workerId}",
            args=(
                workerId,
                methodIndex,
                method,
                self.engine._metrics,
                gpuIds,
                self.ctx.effectiveBatchSizes[methodIndex],
                childConnection,
                self.ctx.eventQueue,
                instanceLog,
            ),
        )
        process.start()
        childConnection.close()
        self.ctx.workers[workerId] = _WorkerState(
            workerId=workerId,
            methodIndex=methodIndex,
            methodLabel=method.Label,
            gpuIds=gpuIds,
            process=process,
            connection=parentConnection,
            instanceLog=instanceLog,
            deadline=time.monotonic() + self.engine.initializeTimeout,
        )
        self.reporter.recordEvent({
            "type": "worker_spawned", "time": time.time(), "worker_id": workerId,
            "method_index": methodIndex, "method": method.Label,
            "gpu_ids": gpuIds, "pid": process.pid, "log_path": instanceLog,
        })

    def dispatch(self, worker: "_WorkerState") -> None:
        if not self.ctx.pending[worker.methodIndex]:
            self.stopWorker(worker)
            return
        taskIndex = self.ctx.pending[worker.methodIndex].popleft()
        pair = (worker.methodIndex, taskIndex)
        if self.ctx.pairStatus[pair] != "pending":
            return
        startAttempt = self.ctx.attempts[pair] + 1
        methodDir = Path(worker.instanceLog).parent
        logPaths = {
            attempt: str((methodDir / (
                f"{worker.workerId}-t{taskIndex:03d}-{Slug(self.ctx.tasks[taskIndex].name)}-"
                f"attempt{attempt}.log"
            )).resolve())
            for attempt in range(startAttempt, self.ctx.maxAttempts + 1)
        }
        for attempt, path in logPaths.items():
            self.reporter.addLog(
                path,
                f"{self.ctx.methods[worker.methodIndex].Label} / "
                f"{self.ctx.tasks[taskIndex].name} attempt {attempt}",
            )
        worker.connection.send({
            "op": "task", "task_index": taskIndex, "task": self.ctx.tasks[taskIndex],
            "start_attempt": startAttempt, "max_attempts": self.ctx.maxAttempts,
            "log_paths": logPaths,
        })
        self.ctx.pairStatus[pair] = "running"
        worker.state = "busy"
        worker.taskIndex = taskIndex
        worker.taskName = self.ctx.tasks[taskIndex].name
        worker.attempt = startAttempt
        worker.logPath = logPaths[startAttempt]
        worker.deadline = time.monotonic() + self.engine.taskTimeout
        self.reporter.recordEvent({
            "type": "task_dispatched",
            "time": time.time(),
            "worker_id": worker.workerId,
            "method_index": worker.methodIndex,
            "task_index": taskIndex,
            "method": self.ctx.methods[worker.methodIndex].Label,
            "task": self.ctx.tasks[taskIndex].name,
            "attempt": startAttempt,
            "log_path": logPaths[startAttempt],
        })

    # --------------------------------------------------------- worker ctrl
    def stopWorker(self, worker: "_WorkerState") -> None:
        if worker.state == "stopping":
            return
        try:
            worker.connection.send({"op": "shutdown"})
        except (BrokenPipeError, EOFError, OSError):
            pass
        worker.state = "stopping"
        worker.taskName = ""
        worker.taskIndex = None
        worker.deadline = time.monotonic() + self.engine.shutdownGracePeriod

    def terminateWorker(self, worker: "_WorkerState", force: bool = False) -> None:
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

    def releaseWorker(self, workerId: str) -> None:
        worker = self.ctx.workers.pop(workerId)
        # Even an orderly Method.Close may leave a backend descendant
        # alive. The outer worker is the process-group leader, so sweep
        # the complete group before its PID can be reaped/reused.
        self.terminateWorker(worker, force=True)
        worker.process.join(timeout=0.2)
        try:
            worker.connection.close()
        except Exception:
            pass
        now = time.monotonic()
        for gpuId in worker.gpuIds:
            self.engine.gpuGovernor.beginGpuCooling(
                gpuId,
                workerId=worker.workerId,
                method=worker.methodLabel,
                reason="worker_exit",
                now=now,
            )
        self.reporter.recordEvent({
            "type": "gpu_cooling_started",
            "time": time.time(),
            "worker_id": worker.workerId,
            "method": worker.methodLabel,
            "gpu_ids": worker.gpuIds,
        })
        self.ctx.workerHistory.append(worker.Snapshot())

    # ------------------------------------------------------ recovery
    def recoverCurrentPair(
        self, worker: "_WorkerState", reason: str, kind: str
    ) -> None:
        if worker.taskIndex is None:
            return
        pair = (worker.methodIndex, worker.taskIndex)
        if self.ctx.pairStatus.get(pair) != "running":
            return
        attempt = max(
            self.ctx.attempts[pair], worker.attempt, 1
        )
        self.ctx.attempts[pair] = attempt
        self.ctx.timings.append({
            "kind": kind, "worker_id": worker.workerId,
            "method": worker.methodLabel, "task": worker.taskName,
            "attempt": attempt,
            "duration": self.engine.taskTimeout if kind == "task_timeout" else 0.0,
        })
        if attempt < self.ctx.maxAttempts:
            self.ctx.pairStatus[pair] = "pending"
            self.ctx.pending[worker.methodIndex].appendleft(worker.taskIndex)
        else:
            self.reporter.pairFailure(
                worker.methodIndex, worker.taskIndex,
                error=reason, kind=kind, logPath=worker.logPath,
            )

    # ------------------------------------------------------- event handler
    def handleEvent(self, event: Dict[str, Any]) -> None:
        # Persist the event before any early-return on missing worker:
        # the events.jsonl trail is the source of truth for what the workers
        # reported, even when the coordinator hadn't registered the worker
        # yet (rare race on initialize_started) or when a late event
        # arrives after the worker has already exited.
        self.reporter.recordEvent(event)
        worker = self.ctx.workers.get(event.get("worker_id"))
        kind = event.get("type")
        if worker is None:
            return
        if event.get("log_path"):
            worker.logPath = event["log_path"]
        if kind == "initialize_done":
            worker.state = "idle"
            worker.deadline = None
            worker.initDuration = float(event["duration"])
            self.ctx.timings.append({
                "kind": "initialize", "worker_id": worker.workerId,
                "method": worker.methodLabel, "task": "", "attempt": "",
                "duration": worker.initDuration,
            })
        elif kind == "initialize_failed":
            self.ctx.fatalError = (
                f"{worker.methodLabel} initialization failed on GPUs "
                f"{worker.gpuIds}: {event.get('error')}"
            )
            self.ctx.fatalStatus = "initialization_failed"
            worker.state = "failed"
        elif kind == "task_started":
            taskIndex = int(event["task_index"])
            pair = (worker.methodIndex, taskIndex)
            attempt = int(event["attempt"])
            self.ctx.attempts[pair] = max(self.ctx.attempts[pair], attempt)
            worker.attempt = attempt
            worker.state = "busy"
            worker.taskIndex = taskIndex
            worker.taskName = event["task"]
            worker.deadline = time.monotonic() + self.engine.taskTimeout
        elif kind == "task_attempt_failed":
            self.ctx.timings.append({
                "kind": "task_attempt_failed", "worker_id": worker.workerId,
                "method": worker.methodLabel, "task": event["task"],
                "attempt": event["attempt"], "duration": event["duration"],
            })
        elif kind == "task_done":
            pair = (worker.methodIndex, int(event["task_index"]))
            self.ctx.pairStatus[pair] = "done"
            self.ctx.results[pair] = event["report"]
            self.ctx.timings.append({
                "kind": "task", "worker_id": worker.workerId,
                "method": worker.methodLabel, "task": event["task"],
                "attempt": event["attempt"], "duration": event["duration"],
            })
            self.reporter._WritePair(
                pair[0], pair[1], report=event["report"]
            )
            self.reporter.writeReports()
        elif kind == "task_failed":
            self.reporter.pairFailure(
                worker.methodIndex, int(event["task_index"]),
                error=event.get("error", "pair failed"), kind="exception",
                logPath=event.get("log_path", ""),
                tracebackText=event.get("traceback", ""),
            )
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
            self.ctx.timings.append({
                "kind": "close", "worker_id": worker.workerId,
                "method": worker.methodLabel, "task": "", "attempt": "",
                "duration": event.get("duration", 0.0),
            })


# Local import to break the circular dependency: ``State`` defines
# ``_WorkerState``, ``Scheduler`` needs the type, and ``State`` does not
# reference ``Scheduler``. Doing the import here avoids leaking the
# underscore-prefixed type into the package's public surface.
from .State import _WorkerState  # noqa: E402  (intentional late import)