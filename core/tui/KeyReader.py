"""Background key-reading thread.

Polls stdin with :func:`select.select` and hands each lowercase byte to the
caller-supplied ``on_key`` callback. :meth:`Stop` is idempotent and bounded
by a 200ms join — the input thread is a daemon, so it won't block process
exit longer than that.
"""

from __future__ import annotations

import os
import select
import sys
import threading
from typing import Callable


class KeyReader:
    """Read single-byte keypresses from stdin in a background thread.

    ``on_key`` is called once per character (already lowercased and
    decoded with ``errors="ignore"``). The callback runs on the reader
    thread — any state it touches must be thread-safe (the dashboard uses
    a :class:`threading.Event` for shutdown and the ``_Render`` result is
    rebuilt on each :meth:`Live.update`, so a stale snapshot mid-frame is
    acceptable).
    """

    def __init__(self, on_key: Callable[[str], None]) -> None:
        self._on_key = on_key
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def Start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="kvbench-tui-input",
        )
        self._thread.start()

    def Stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.2)
            self._thread = None

    def _loop(self) -> None:
        fd = sys.stdin.fileno()
        while not self._stop.is_set():
            try:
                ready, _, _ = select.select([fd], [], [], 0.1)
                if not ready:
                    continue
                key = os.read(fd, 1).decode(errors="ignore").lower()
            except OSError:
                return
            self._on_key(key)