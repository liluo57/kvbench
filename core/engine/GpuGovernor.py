"""GPU pool, cooling, and snapshot policy for the Engine.

The GpuGovernor owns everything GPU-pool-related except the worker-process
side:

- :meth:`beginGpuCooling` marks a GPU as "cooling down" with a deadline,
  so a freshly-exited worker doesn't immediately re-host another.
- :meth:`refreshCoolingGpus` polls ``QueryGpus()`` (NVML) and returns GPUs
  to ``freeGpus`` once they drop back to the baseline memory and stop
  hosting external compute PIDs.
- :meth:`validateFreeGpus` quarantines any pool GPU that has external
  contention right now (someone else is using it).
- :meth:`refreshGpuSnapshot` updates the dashboard telemetry for the TUI.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, List, Tuple

from .Gpu import QueryGpus
from .State import _GPU_SNAPSHOT_INTERVAL

if TYPE_CHECKING:
    from .Engine import Engine
    from .State import RunContext


class GpuGovernor:
    """Owns the GPU pool, cooling, and dashboard telemetry."""

    def __init__(self, ctx: "RunContext", engine: "Engine") -> None:
        self.ctx = ctx
        self.engine = engine

    # ----------------------------------------------------------- cooling
    def beginGpuCooling(
        self,
        gpuId: int,
        *,
        workerId: str,
        method: str,
        reason: str,
        now: float,
    ) -> None:
        self.ctx.coolingGpus[gpuId] = {
            "worker_id": workerId,
            "method": method,
            "reason": reason,
            "started": now,
            "deadline": now + self.engine.gpuReleaseTimeout,
            "baseline_memory": self.ctx.gpuBaseline[gpuId],
            "baseline_compute_pids": sorted(self.ctx.gpuBaselinePids[gpuId]),
            "last_memory": None,
            "last_compute_pids": [],
            "external_compute_pids": [],
            "clean_since": None,
        }

    def gpuIsClean(self, gpuId: int, info: Any) -> Tuple[bool, List[int]]:
        externalPids = sorted(set(info.computePids) - self.ctx.gpuBaselinePids[gpuId])
        releaseLimit = self.ctx.gpuBaseline[gpuId] + self.engine.gpuReleaseMemoryTolerance
        return info.memoryUsed <= releaseLimit and not externalPids, externalPids

    def validateFreeGpus(self, now: float) -> None:
        """Quarantine pool GPUs taken by another process before dispatch."""
        if not self.ctx.freeGpus:
            return
        try:
            current = {gpu.id: gpu for gpu in QueryGpus()}
        except RuntimeError as exc:
            for gpuId in list(self.ctx.freeGpus):
                self.ctx.freeGpus.remove(gpuId)
                self.beginGpuCooling(
                    gpuId,
                    workerId="",
                    method="",
                    reason="gpu_validation_error",
                    now=now,
                )
            self.engine.reporter.recordEvent({
                "type": "gpu_validation_failed",
                "time": time.time(),
                "error": str(exc),
                "gpu_ids": sorted(self.ctx.coolingGpus),
            })
            return
        for gpuId in list(self.ctx.freeGpus):
            info = current.get(gpuId)
            clean, externalPids = (
                self.gpuIsClean(gpuId, info) if info is not None else (False, [])
            )
            if clean:
                continue
            self.ctx.freeGpus.remove(gpuId)
            self.beginGpuCooling(
                gpuId,
                workerId="",
                method="",
                reason="external_contention",
                now=now,
            )
            self.ctx.coolingGpus[gpuId]["last_memory"] = (
                info.memoryUsed if info is not None else None
            )
            self.ctx.coolingGpus[gpuId]["last_compute_pids"] = (
                list(info.computePids) if info is not None else []
            )
            self.ctx.coolingGpus[gpuId]["external_compute_pids"] = externalPids
            self.engine.reporter.recordEvent({
                "type": "gpu_contention_detected",
                "time": time.time(),
                "gpu_id": gpuId,
                "memory_used": info.memoryUsed if info is not None else None,
                "baseline_memory": self.ctx.gpuBaseline[gpuId],
                "compute_pids": list(info.computePids) if info is not None else [],
                "external_compute_pids": externalPids,
            })

    def refreshCoolingGpus(self, now: float, *, forcePoll: bool = False) -> None:
        if not self.ctx.coolingGpus:
            return
        if not forcePoll and now - self.ctx.lastGpuReleasePoll < 0.2:
            return
        self.ctx.lastGpuReleasePoll = now
        try:
            current = {gpu.id: gpu for gpu in QueryGpus()}
        except RuntimeError as exc:
            current = {}
            queryError = str(exc)
        else:
            queryError = ""

        released = []
        for gpuId, cooling in list(self.ctx.coolingGpus.items()):
            info = current.get(gpuId)
            if info is not None:
                cooling["last_memory"] = info.memoryUsed
                cooling["last_compute_pids"] = list(info.computePids)
                clean, externalPids = self.gpuIsClean(gpuId, info)
                cooling["external_compute_pids"] = externalPids
                if clean and cooling["clean_since"] is None:
                    cooling["clean_since"] = now
                elif not clean:
                    cooling["clean_since"] = None
                stableFor = (
                    now - cooling["clean_since"]
                    if cooling["clean_since"] is not None else 0.0
                )
                if clean and stableFor >= self.engine.gpuReleaseStableSeconds:
                    released.append(gpuId)
                    self.ctx.timings.append({
                        "kind": "gpu_release",
                        "worker_id": cooling["worker_id"],
                        "method": cooling["method"],
                        "task": "",
                        "attempt": "",
                        "gpu_id": gpuId,
                        "duration": now - cooling["started"],
                    })
                    self.engine.reporter.recordEvent({
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
                f"within {self.engine.gpuReleaseTimeout}s ({detail}, baseline="
                f"{cooling['baseline_memory']} bytes, tolerance="
                f"{self.engine.gpuReleaseMemoryTolerance} bytes, stable window="
                f"{self.engine.gpuReleaseStableSeconds}s)"
            )
            if self.ctx.fatalError:
                self.ctx.fatalError = f"{self.ctx.fatalError}; additionally: {releaseError}"
            else:
                self.ctx.fatalStatus = "resource_release_failed"
                self.ctx.fatalError = releaseError
            self.engine.reporter.recordEvent({
                "type": "gpu_release_failed",
                "time": time.time(),
                "worker_id": cooling["worker_id"],
                "method": cooling["method"],
                "gpu_id": gpuId,
                "error": releaseError,
            })

        for gpuId in released:
            self.ctx.coolingGpus.pop(gpuId, None)
            self.ctx.freeGpus.append(gpuId)
        self.ctx.freeGpus.sort(key=self.engine._gpuPool.index)

    def refreshGpuSnapshot(self, now: float) -> None:
        """Refresh dashboard GPU telemetry once per second."""
        if (
            not self.engine._tui.enabled
            or now - self.ctx.lastGpuSnapshotPoll < _GPU_SNAPSHOT_INTERVAL
        ):
            return
        self.ctx.lastGpuSnapshotPoll = now
        try:
            self.engine._gpuSnapshot = QueryGpus()
        except RuntimeError as exc:
            # Telemetry is best-effort: retain the last good sample instead
            # of interrupting a running benchmark for a transient NVML error.
            self.ctx.gpuSnapshotError = str(exc)
            return
        self.ctx.gpuSnapshotAt = time.time()
        self.ctx.gpuSnapshotError = ""