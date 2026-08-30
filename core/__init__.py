"""KVBench core abstractions.

Public surface (what tasks / methods / workloads may import):

- :class:`Case`, :class:`Task` (:mod:`core.Task`) — the task contract.
- :class:`Method` (:mod:`core.Method`) — the KV-cache method contract.
- :class:`Metric` (:mod:`core.Metrics`) — the metric contract.
- :class:`Result` + metric key constants (:mod:`core.Result`) — the result
  data currency shared across the framework.
- :class:`Workload`, :class:`Action`, :class:`ActionKind`,
  :class:`ActionResult` (:mod:`core.Workload`) — the workload contract.
- :func:`LoadConfig`, :func:`Get`, :func:`ModelPath`, :func:`DatasetDir`
  (:mod:`core.Config`) — config helpers.

Deliberately **not** re-exported from this barrel:

- :class:`core.engine.Engine` and the engine-package exception classes —
  import them via :mod:`core.engine` directly. The engine sub-package is
  treated as a separate, evolving module; routing it through this barrel
  turned a logical dependency-direction (engine → core abstractions) into a
  circular shape and made future engine sub-splits harder to land.
"""

from .Config import DatasetDir, Get, LoadConfig, ModelPath
from .Metrics import Metric
from .Method import Method
from .Result import NumOutputTokensKey, Result, TotalTimeKey, TtftKey
from .Task import Case, Task
from .Workload import Action, ActionKind, ActionResult, Workload

__all__ = [
    "Action",
    "ActionKind",
    "ActionResult",
    "Case",
    "DatasetDir",
    "Get",
    "LoadConfig",
    "Metric",
    "Method",
    "ModelPath",
    "NumOutputTokensKey",
    "Result",
    "Task",
    "TotalTimeKey",
    "TtftKey",
    "Workload",
]
