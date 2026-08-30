"""Event-log, manifest, and pair-report writer for the Engine.

The Reporter owns every JSON / JSONL side effect of the run:

- :meth:`recordEvent` writes one line to ``events.jsonl`` (worker-spawned,
  task-dispatched, task-done, …).
- :meth:`addLog` tracks which log files belong to the run so the TUI / final
  report can point at them.
- :meth:`writeReports` writes ``results/full.json``, ``results/core.json``,
  ``results/timing.json``, and ``results/failures.json`` atomically.
- :meth:`pairFailure` records a failed (method, task) pair and updates
  the report files.
- :meth:`_WritePair` writes one pair-report JSON file (per method/task).
- :meth:`snapshot` builds the TUI snapshot dict.

State lives on :class:`RunContext`; the Reporter reads / writes through it
and has no fields of its own.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from .State import AtomicJson, CoreReport, Slug

if TYPE_CHECKING:
    from .Engine import Engine
    from .State import RunContext


class Reporter:
    """Persist every per-run side effect (events, logs, pair reports)."""

    def __init__(self, ctx: RunContext, engine: "Engine") -> None:
        self.ctx = ctx
        self.engine = engine

    # ------------------------------------------------------------------ logs
    def addLog(self, path: str, label: str) -> None:
        if path not in self.ctx.knownLogs:
            self.ctx.knownLogs.add(path)
            self.ctx.logs.append({"path": path, "label": label})

    def recordEvent(self, event: Dict[str, Any]) -> None:
        persisted = {key: value for key, value in event.items() if key != "report"}
        self.ctx.eventsFile.write(
            json.dumps(persisted, ensure_ascii=False, default=str) + "\n"
        )

    # ----------------------------------------------------------------- reports
    def writeReports(self, finalStatus: Optional[str] = None) -> Dict[str, Any]:
        orderedRuns = [self.ctx.results[pair] for pair in sorted(self.ctx.results)]
        report = {
            "status": finalStatus or self.ctx.status,
            "output_dir": str(self.ctx.outputDir.resolve()),
            "runs": orderedRuns,
            "cores": [CoreReport(run) for run in orderedRuns],
            "failures": sorted(
                self.ctx.failures,
                key=lambda item: (item["method_index"], item["task_index"]),
            ),
        }
        AtomicJson(self.ctx.outputDir / "results" / "full.json", report)
        AtomicJson(self.ctx.outputDir / "results" / "core.json", report["cores"])
        AtomicJson(self.ctx.outputDir / "results" / "timing.json", self.ctx.timings)
        AtomicJson(
            self.ctx.outputDir / "results" / "failures.json", report["failures"]
        )
        return report

    def pairFailure(
        self,
        methodIndex: int,
        taskIndex: int,
        *,
        error: str,
        kind: str,
        logPath: str = "",
        tracebackText: str = "",
    ) -> None:
        pair = (methodIndex, taskIndex)
        if self.ctx.pairStatus.get(pair) in ("done", "failed", "unschedulable"):
            return
        self.ctx.pairStatus[pair] = (
            "unschedulable" if kind == "unschedulable" else "failed"
        )
        failure = {
            "method_index": methodIndex,
            "task_index": taskIndex,
            "method": self.ctx.methods[methodIndex].Label,
            "task": self.ctx.tasks[taskIndex].name,
            "kind": kind,
            "attempts": self.ctx.attempts[pair],
            "error": error,
            "traceback": tracebackText,
            "log_path": logPath,
        }
        self.ctx.failures.append(failure)
        self._WritePair(methodIndex, taskIndex, failure=failure)
        self.writeReports()

    def _WritePair(
        self,
        methodIndex: int,
        taskIndex: int,
        *,
        report: Optional[Dict[str, Any]] = None,
        failure: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write a single per-pair JSON under ``pairs/<method>/<task>.json``."""
        path = (
            self.ctx.outputDir / "pairs"
            / f"{methodIndex:02d}-{Slug(self.ctx.methods[methodIndex].Label)}"
            / f"{taskIndex:03d}-{Slug(self.ctx.tasks[taskIndex].name)}.json"
        )
        payload = {
            "status": "success" if report is not None else "failed",
            "method_index": methodIndex,
            "task_index": taskIndex,
            "method": self.ctx.methods[methodIndex].Label,
            "task": self.ctx.tasks[taskIndex].name,
        }
        if report is not None:
            payload["report"] = report
        if failure is not None:
            payload["failure"] = failure
        AtomicJson(path, payload)

    # ----------------------------------------------------------------- snapshot
    def snapshot(self) -> Dict[str, Any]:
        return {
            "status": self.ctx.status,
            "elapsed": time.monotonic() - self.ctx.startedMono,
            "output_dir": str(self.ctx.outputDir),
            "progress": {
                "done": sum(value == "done" for value in self.ctx.pairStatus.values()),
                "failed": sum(
                    value in ("failed", "unschedulable")
                    for value in self.ctx.pairStatus.values()
                ),
                "total": len(self.ctx.pairStatus),
            },
            "gpu_pool": self.engine._gpuPool,
            "free_gpus": self.ctx.freeGpus,
            "cooling_gpus": sorted(self.ctx.coolingGpus),
            "gpu_snapshot": [
                gpu.AsDict() for gpu in self.engine._gpuSnapshot
            ],
            "gpu_snapshot_at": self.ctx.gpuSnapshotAt,
            "gpu_snapshot_error": self.ctx.gpuSnapshotError,
            "workers": [
                worker.Snapshot() for worker in self.ctx.workers.values()
            ],
            "runs": [self.ctx.results[pair] for pair in sorted(self.ctx.results)],
            "cores": [
                CoreReport(self.ctx.results[pair]) for pair in sorted(self.ctx.results)
            ],
            "timings": self.ctx.timings,
            "failures": self.ctx.failures,
            "logs": self.ctx.logs,
        }