"""Terminal-mode context manager.

Putting stdin into cbreak mode lets the dashboard read single keystrokes
without buffering the user's line. The previous attrs are saved on enter
and restored on exit so a Ctrl-C / abnormal exit doesn't leave the
coordinator process in a state where the user's shell is unusable.

Failure to enter cbreak is non-fatal — the dashboard falls back to a
non-interactive render that the coordinator drives from ``Stop``. The
``enabled`` flag is the caller's signal; :class:`TerminalMode` itself
never raises out of its ``__enter__``.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Iterator, Optional


@contextmanager
def TerminalMode() -> Iterator[None]:
    """Run the body with stdin in cbreak mode; restore attrs on exit."""
    import termios
    import tty

    attrs: Optional[list] = None
    try:
        attrs = termios.tcgetattr(sys.stdin.fileno())
        tty.setcbreak(sys.stdin.fileno())
    except Exception:  # noqa: BLE001 - TUI input is optional
        # No terminal attached (CI / non-interactive shell / piped stdin).
        # Yielding without entering cbreak is fine — the body just won't
        # see raw key events, which is what the caller's fallback handles.
        yield
        return
    try:
        yield
    finally:
        try:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, attrs)
        except Exception:  # noqa: BLE001
            pass