"""Runtime configuration: dataset / model paths, declared in ``config.yaml``.

KVBench keeps global path settings out of code — they live in
``config.yaml`` at the project root. This module loads that file and resolves a
dataset *name* to its directory, e.g. ``DatasetDir("ruler")`` ->
``<DatasetPath>/ruler``.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

#: Default config file (project-root relative).
DefaultConfigPath = Path(__file__).resolve().parent.parent / "config.yaml"

_ConfigCache: Dict[str, Any] = {}


def LoadConfig(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load and cache a YAML config file as a dict.

    Falls back to defaults for missing keys. The result is cached so repeated
    lookups (one per task construction) do not re-read the file.
    """
    path = Path(path or DefaultConfigPath)
    if path not in _ConfigCache:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        else:
            cfg = {}
        _ConfigCache[path] = cfg
    return _ConfigCache[path]


def Get(key: str, default: Any = None, config: Optional[Dict[str, Any]] = None) -> Any:
    """Read a top-level config value, e.g. ``Get("DatasetPath")``."""
    source = LoadConfig() if config is None else config
    return source.get(key, default)


def DatasetDir(dataset: str, config: Optional[Dict[str, Any]] = None) -> Path:
    """Resolve a dataset *name* to its directory: ``<DatasetPath>/<dataset>``.

    Raises ``FileNotFoundError`` when the directory does not exist so a typo or
    a missing dataset is reported loudly instead of silently yielding no cases.
    """
    root = Path(Get("DatasetPath", "data", config=config))
    path = root / dataset
    if not path.is_dir():
        raise FileNotFoundError(
            f"dataset directory not found: {path} "
            f"(set DatasetPath in {DefaultConfigPath})"
        )
    return path


def ModelPath(config: Optional[Dict[str, Any]] = None) -> str:
    """Return the model path declared in ``config.yaml``.

    Single source of truth — there is no override parameter. Switch models
    by editing ``config.yaml``, not by passing paths at call sites.
    """
    return str(Get("ModelPath", None, config=config) or "")


