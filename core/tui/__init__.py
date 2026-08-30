"""KVBench coordinator-process TUI.

The public surface is :class:`BenchmarkTui` — a thin facade that owns the
Rich ``Live`` handle and the two lifecycle helpers (:func:`TerminalMode`,
:class:`KeyReader`), and delegates every render to
:class:`~core.tui.dashboard_view.DashboardView`. The split exists so each
piece has one reason to change:

  - **TerminalMode** changes when the way we ask the OS for raw input
    changes (e.g. a Windows port).
  - **KeyReader** changes when the threading / ``select`` polling model
    changes (e.g. swapping to asyncio on Windows).
  - **DashboardView** changes when the rendering changes (new view, new
    layout, snapshot schema). New views are added by decorating a
    function with :func:`~core.tui.dashboard_view.RegisterView`; the
    key-binding map and the help footer pick it up automatically.

Backward-compat shims:
  - :meth:`BenchmarkTui._HandleKey` delegates to ``view.OnKey`` plus
    the cancel / quit state — the test suite reaches into it.
  - :attr:`BenchmarkTui._inputEnabled` tracks whether the input thread
    is live (used by the ``FinishAndWait`` interaction tests).
"""

from __future__ import annotations

import sys
import threading
from typing import Any, Dict

from rich.console import Console
from rich.live import Live

from .DashboardView import DashboardView, KEY_MAP
from .KeyReader import KeyReader
from .TerminalMode import TerminalMode


class BenchmarkTui:
    """Coordinator-process Rich dashboard facade.

    Owns the :class:`rich.live.Live` handle, the :class:`KeyReader`
    thread, and the :class:`TerminalMode` context. Snapshot state lives on
    the :class:`DashboardView` instance so render-only changes don't
    touch this file.
    """

    def __init__(self, enabled: bool = True):
        self.console = Console()
        self.enabled = bool(enabled and self.console.is_terminal)
        self.view = DashboardView()
        self.cancelRequested = False
        self.quitRequested = False
        self.finished = False
        self._live: Live | None = None
        self._quit = threading.Event()
        self._keyReader: KeyReader | None = None
        self._terminalGuard: Any = None
        self._inputEnabled = False

    def Start(self) -> None:
        if not self.enabled:
            return
        self._live = Live(
            self.view.Render(),
            console=self.console,
            refresh_per_second=4,
            transient=False,
        )
        self._live.start()
        try:
            self._terminalGuard = TerminalMode()
            self._terminalGuard.__enter__()
        except Exception:  # noqa: BLE001 - TUI input is optional
            self._terminalGuard = None
            return
        self._keyReader = KeyReader(self._onKey)
        self._keyReader.Start()
        self._inputEnabled = True

    def Stop(self) -> None:
        if self._keyReader is not None:
            self._keyReader.Stop()
            self._keyReader = None
        if self._terminalGuard is not None:
            try:
                self._terminalGuard.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass
            self._terminalGuard = None
        if self._live is not None:
            self._live.update(self.view.Render(finished=self.finished), refresh=True)
            self._live.stop()
            self._live = None
        self._inputEnabled = False

    def FinishAndWait(self) -> None:
        """Keep the final dashboard available until the user presses ``q``."""
        if not self.enabled or not self._inputEnabled:
            return
        self.finished = True
        if self._live is not None:
            self._live.update(self.view.Render(finished=True), refresh=True)
        if not self.quitRequested:
            self._quit.wait()

    def Update(self, snapshot: Dict[str, Any]) -> None:
        self.view.Update(snapshot)
        if self._live is not None:
            self._live.update(self.view.Render(finished=self.finished), refresh=True)

    def _onKey(self, key: str) -> None:
        self.view.OnKey(key)
        if key == "q":
            self.quitRequested = True
            if not self.finished:
                self.cancelRequested = True
            self._quit.set()
        if self._live is not None:
            self._live.update(self.view.Render(finished=self.finished), refresh=True)

    # -------------------------------------------------- backward-compat shim
    def _HandleKey(self, key: str) -> None:
        """Backwards-compat entry point — older tests call this directly.

        Delegates to ``view.OnKey`` (which is the new canonical home for
        view / pagination state) then handles the cancel / quit flip on
        the facade itself (state that doesn't belong on the renderer).
        """
        self.view.OnKey(key)
        if key == "q":
            self.quitRequested = True
            if not self.finished:
                self.cancelRequested = True
            self._quit.set()
        if self._live is not None:
            self._live.update(self.view.Render(finished=self.finished), refresh=True)


__all__ = ["BenchmarkTui", "KEY_MAP"]