"""Dashboard rendering + view dispatch.

``DashboardView`` is a pure render — it owns the snapshot dict, the current
view name, the log-pagination index, and nothing else. :class:`BenchmarkTui`
holds the Live handle and the threading concerns.

The view-dispatch table is **module-level** on purpose: adding a new
dashboard view is one ``@RegisterView("name")`` decorator on a function
that takes ``(view, snapshot)`` and returns a Rich renderable. No edits
to :class:`BenchmarkTui`, no edits to this module's registry itself — the
new view appears in the footer and the key-binding map automatically.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


#: View name → ``(key, label)`` for the help footer. Ordered.
KEY_MAP: Dict[str, str] = {
    "c": "core",
    "f": "full",
    "t": "timing",
    "s": "schedule",
    "l": "logs",
    "e": "failures",
}


#: Module-level view registry. Each entry takes ``(view, snapshot)`` and
#: returns a Rich renderable. Adding a view is one decorator line.
_VIEW_REGISTRY: Dict[str, Callable[["DashboardView", Dict[str, Any]], Any]] = {}


def RegisterView(name: str):
    """Decorator that registers a function as a named dashboard view.

    The wrapped function receives ``(view, snapshot)`` so it can read
    pagination state from ``view`` (``view.logIndex`` etc.) without
    coupling the renderer to :class:`BenchmarkTui`.
    """
    def decorator(fn):
        _VIEW_REGISTRY[name] = fn
        return fn
    return decorator


class DashboardView:
    """Snapshot-backed Rich dashboard.

    Holds ``view`` (current view name), ``logIndex`` (paginated within the
    logs view), and ``_snapshot`` (the latest coordinator-supplied dict).
    Everything else is a render computed from these three plus the
    module-level :data:`_VIEW_REGISTRY`.
    """

    def __init__(self) -> None:
        self.view: str = "core"
        self.logIndex: int = 0
        self._snapshot: Dict[str, Any] = {}

    def Update(self, snapshot: Dict[str, Any]) -> None:
        """Replace the rendered snapshot; clamp ``logIndex`` to the new logs list."""
        self._snapshot = snapshot
        logs = snapshot.get("logs", [])
        if logs:
            self.logIndex %= len(logs)
        else:
            self.logIndex = 0

    def OnKey(self, key: str) -> None:
        """Translate a keypress into view / pagination state.

        Returns ``True`` iff the key was a quit (``q``) — callers use this
        to flip their own cancel / finished state.
        """
        if key in KEY_MAP:
            self.view = KEY_MAP[key]
        elif key in ("n", "]"):
            self.logIndex += 1
        elif key in ("p", "["):
            self.logIndex = max(0, self.logIndex - 1)
        else:
            return None
        return None

    def Render(self, *, finished: bool = False):
        snapshot = self._snapshot
        progress = snapshot.get("progress", {})
        header = Table.grid(expand=True)
        header.add_column(ratio=1)
        header.add_column(justify="right")
        header.add_row(
            Text(
                f"KVBench  {snapshot.get('status', 'starting')}  "
                f"{snapshot.get('elapsed', 0.0):.1f}s",
                style="bold cyan",
            ),
            Text(
                f"pairs {progress.get('done', 0)}/{progress.get('total', 0)}  "
                f"failed {progress.get('failed', 0)}",
                style="bold",
            ),
        )
        renderer = _VIEW_REGISTRY.get(self.view)
        body = renderer(self, snapshot) if renderer is not None else Text(
            f"unknown view: {self.view!r}", style="red",
        )
        quitAction = "quit" if finished else "stop"
        footer = Text(
            "views: [c]ore [f]ull [t]iming [s]chedule [l]ogs [e]rrors  "
            f"logs: [n]/[p]  [q] {quitAction}",
            style="dim",
        )
        return Group(
            Panel(header, border_style="cyan"),
            Panel(body, title=self.view, border_style="blue"),
            footer,
        )


# ---------------------------------------------------------- registered views

@RegisterView("core")
def _render_core(view: DashboardView, snapshot: Dict[str, Any]):
    table = Table(expand=True)
    cores = snapshot.get("cores", [])
    keys: List[str] = ["method", "task"]
    for core in cores:
        for key in core:
            if key not in keys:
                keys.append(key)
    for key in keys:
        table.add_column(key)
    for core in cores[-20:]:
        table.add_row(*[DashboardView._Value(core.get(key)) for key in keys])
    if not cores:
        table.add_column("status")
        table.add_row("waiting for the first completed pair")
    return table


@RegisterView("full")
def _render_full(view: DashboardView, snapshot: Dict[str, Any]):
    text = json.dumps(
        {"runs": snapshot.get("runs", [])},
        indent=2,
        ensure_ascii=False,
        default=str,
    )
    return Text("\n".join(text.splitlines()[-45:]), overflow="ellipsis")


@RegisterView("timing")
def _render_timing(view: DashboardView, snapshot: Dict[str, Any]):
    table = Table(expand=True)
    for column in ("kind", "method", "task", "worker", "attempt", "seconds"):
        table.add_column(column)
    for item in snapshot.get("timings", [])[-25:]:
        table.add_row(
            str(item.get("kind", "")),
            str(item.get("method", "")),
            str(item.get("task", "")),
            str(item.get("worker_id", "")),
            str(item.get("attempt", "")),
            f"{float(item.get('duration', 0.0)):.3f}",
        )
    return table


@RegisterView("schedule")
def _render_schedule(view: DashboardView, snapshot: Dict[str, Any]):
    sampledAt = snapshot.get("gpu_snapshot_at")
    sampleLabel = (
        time.strftime("%H:%M:%S", time.localtime(sampledAt))
        if isinstance(sampledAt, (int, float))
        else "waiting"
    )
    error = snapshot.get("gpu_snapshot_error", "")
    title = f"GPU pool (updated {sampleLabel}, every 1s)"
    if error:
        title += " [NVML error; showing last sample]"
    gpu = Table(title=title, expand=True)
    for column in ("id", "state", "memory", "util", "worker"):
        gpu.add_column(column)
    assignments = {}
    for worker in snapshot.get("workers", []):
        for gpuId in worker.get("gpu_ids", []):
            assignments[gpuId] = worker
    selected = set(snapshot.get("gpu_pool", []))
    cooling = set(snapshot.get("cooling_gpus", []))
    for item in snapshot.get("gpu_snapshot", []):
        gpuId = item["id"]
        worker = assignments.get(gpuId)
        state = "outside pool"
        if gpuId in selected:
            if worker:
                state = worker.get("state", "busy")
            elif gpuId in cooling:
                state = "cooling"
            else:
                state = "free"
        gpu.add_row(
            str(gpuId),
            state,
            f"{item['memoryRatio'] * 100:.1f}%",
            f"{item['utilization']}%",
            worker.get("worker_id", "") if worker else "",
        )

    workers = Table(title="Instances", expand=True)
    for column in ("instance", "method", "GPUs", "state", "task"):
        workers.add_column(column)
    for worker in snapshot.get("workers", []):
        workers.add_row(
            worker["worker_id"],
            worker["method"],
            ",".join(map(str, worker["gpu_ids"])),
            worker["state"],
            worker.get("task", ""),
        )
    return Group(gpu, workers)


@RegisterView("logs")
def _render_logs(view: DashboardView, snapshot: Dict[str, Any]):
    logs = snapshot.get("logs", [])
    if not logs:
        return Text("no logs yet")
    view.logIndex %= len(logs)
    item = logs[view.logIndex]
    path = Path(item["path"])
    try:
        with path.open("rb") as file:
            file.seek(0, os.SEEK_END)
            size = file.tell()
            file.seek(max(0, size - 32_000))
            content = file.read().decode(errors="replace")
    except OSError as exc:
        content = f"unable to read log: {exc}"
    title = f"{view.logIndex + 1}/{len(logs)} {item.get('label', path.name)}\n{path}"
    return Group(Text(title, style="bold"), Text("\n".join(content.splitlines()[-40:])))


@RegisterView("failures")
def _render_failures(view: DashboardView, snapshot: Dict[str, Any]):
    failures = snapshot.get("failures", [])
    if not failures:
        return Text("no failures", style="green")
    table = Table(expand=True)
    for column in ("method", "task", "attempts", "error", "log"):
        table.add_column(column)
    for failure in failures[-20:]:
        table.add_row(
            str(failure.get("method", "")),
            str(failure.get("task", "")),
            str(failure.get("attempts", "")),
            str(failure.get("error", "")),
            str(failure.get("log_path", "")),
        )
    return table


# Static helper used by the views above; exposed on the class for backward
# compat with the pre-split BenchmarkTui.
DashboardView._Value = staticmethod(  # type: ignore[attr-defined]
    lambda value: (
        f"{value:.6g}" if isinstance(value, float) else
        ("" if value is None else str(value))
    )
)