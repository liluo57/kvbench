"""Small helpers shared by the sandbox and HTTP-handler modules."""

from __future__ import annotations

import socket
import sysconfig
from pathlib import Path
from typing import Optional


def FindFreePort(host: str) -> int:
    """Bind ``host:0`` and return the OS-assigned port number."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def ResolveHostSitePackages() -> Optional[Path]:
    """Locate the host's site-packages dir (so we can mount it into the SIF).

    mini-swe-agent is installed on the host (e.g.
    ``/data/lyh/miniconda3/lib/python3.13/site-packages``). Inside the SIF
    the only Python is whatever the base image ships (often empty), so we
    stage the entire host site-packages tree at ``/host_tools/_lib`` and
    point ``PYTHONPATH`` at it.
    """
    # Strategy 1: ask the active interpreter where its ``purelib`` is.
    try:
        purelib = sysconfig.get_path("purelib")
        if purelib and Path(purelib).is_dir():
            return Path(purelib)
    except Exception:  # noqa: BLE001
        pass
    # Strategy 2: find an installed ``minisweagent`` package and walk up
    # to its parent (the site-packages root).
    try:
        import minisweagent  # type: ignore[import-not-found]
        pkgFile = Path(minisweagent.__file__).resolve()
        # minisweagent/__init__.py -> minisweagent/ -> site-packages/
        return pkgFile.parent.parent
    except Exception:  # noqa: BLE001
        pass
    return None