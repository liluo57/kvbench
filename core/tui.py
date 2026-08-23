"""Small Rich-based interactive dashboard for the coordinator process."""

import json
import os
import select
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class BenchmarkTui:
    """Render scheduler state; GPU/model work always remains in children."""

    _views = ["core", "full", "timing", "schedule", "logs", "failures"]
    _keys = {
        "c": "core",
        "f": "full",
        "t": "timing",
        "s": "schedule",
        "l": "logs",
        "e": "failures",
    }

    def __init__(self, enabled: bool = True):
        self.console = Console()
        self.enabled = bool(enabled and self.console.is_terminal)
        self.view = "core"
        self.logIndex = 0
        self.cancelRequested = False
        self.quitRequested = False
        self.finished = False
        self._live = None
        self._stop = threading.Event()
        self._quit = threading.Event()
        self._thread = None
        self._termAttrs = None
        self._inputEnabled = False
        self._snapshot: Dict[str, Any] = {}

    def Start(self) -> None:
        if not self.enabled:
            return
        self._live = Live(
            self._Render(),
            console=self.console,
            refresh_per_second=4,
            transient=False,
        )
        self._live.start()
        try:
            import termios
            import tty

            self._termAttrs = termios.tcgetattr(sys.stdin.fileno())
            tty.setcbreak(sys.stdin.fileno())
            self._thread = threading.Thread(
                target=self._ReadKeys,
                daemon=True,
                name="kvbench-tui-input",
            )
            self._inputEnabled = True
            self._thread.start()
        except Exception:  # noqa: BLE001 - TUI input is optional
            self._termAttrs = None
            self._inputEnabled = False

    def Stop(self) -> None:
        self._stop.set()
        self._quit.set()
        if self._thread is not None:
            self._thread.join(timeout=0.2)
        if self._termAttrs is not None:
            try:
                import termios

                termios.tcsetattr(
                    sys.stdin.fileno(), termios.TCSADRAIN, self._termAttrs
                )
            except Exception:  # noqa: BLE001
                pass
        if self._live is not None:
            self._live.update(self._Render(), refresh=True)
            self._live.stop()
            self._live = None

    def FinishAndWait(self) -> None:
        """Keep the final dashboard available until the user presses q."""
        if not self.enabled or not self._inputEnabled:
            return
        self.finished = True
        if self._live is not None:
            self._live.update(self._Render(), refresh=True)
        if not self.quitRequested:
            self._quit.wait()

    def Update(self, snapshot: Dict[str, Any]) -> None:
        self._snapshot = snapshot
        logs = snapshot.get("logs", [])
        if logs:
            self.logIndex %= len(logs)
        else:
            self.logIndex = 0
        if self._live is not None:
            self._live.update(self._Render(), refresh=True)

    def _ReadKeys(self) -> None:
        fd = sys.stdin.fileno()
        while not self._stop.is_set():
            try:
                ready, _, _ = select.select([fd], [], [], 0.1)
                if not ready:
                    continue
                key = os.read(fd, 1).decode(errors="ignore").lower()
            except OSError:
                return
            self._HandleKey(key)

    def _HandleKey(self, key: str) -> None:
        if key in self._keys:
            self.view = self._keys[key]
        elif key in ("n", "]"):
            self.logIndex += 1
        elif key in ("p", "["):
            self.logIndex = max(0, self.logIndex - 1)
        elif key == "q":
            self.quitRequested = True
            if not self.finished:
                self.cancelRequested = True
            self._quit.set()
        if self._live is not None:
            self._live.update(self._Render(), refresh=True)

    def _Render(self):
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
        body = {
            "core": self._Core,
            "full": self._Full,
            "timing": self._Timing,
            "schedule": self._Schedule,
            "logs": self._Logs,
            "failures": self._Failures,
        }[self.view](snapshot)
        quitAction = "quit" if self.finished else "stop"
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

    def _Core(self, snapshot: Dict[str, Any]):
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
            table.add_row(*[self._Value(core.get(key)) for key in keys])
        if not cores:
            table.add_column("status")
            table.add_row("waiting for the first completed pair")
        return table

    def _Full(self, snapshot: Dict[str, Any]):
        text = json.dumps(
            {"runs": snapshot.get("runs", [])},
            indent=2,
            ensure_ascii=False,
            default=str,
        )
        return Text("\n".join(text.splitlines()[-45:]), overflow="ellipsis")

    def _Timing(self, snapshot: Dict[str, Any]):
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

    def _Schedule(self, snapshot: Dict[str, Any]):
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

    def _Logs(self, snapshot: Dict[str, Any]):
        logs = snapshot.get("logs", [])
        if not logs:
            return Text("no logs yet")
        self.logIndex %= len(logs)
        item = logs[self.logIndex]
        path = Path(item["path"])
        try:
            with path.open("rb") as file:
                file.seek(0, os.SEEK_END)
                size = file.tell()
                file.seek(max(0, size - 32_000))
                content = file.read().decode(errors="replace")
        except OSError as exc:
            content = f"unable to read log: {exc}"
        title = f"{self.logIndex + 1}/{len(logs)} {item.get('label', path.name)}\n{path}"
        return Group(Text(title, style="bold"), Text("\n".join(content.splitlines()[-40:])))

    def _Failures(self, snapshot: Dict[str, Any]):
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

    @staticmethod
    def _Value(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.6g}"
        return "" if value is None else str(value)
