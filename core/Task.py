"""Task abstraction: defines *what* to evaluate."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List

from .Result import Result


@dataclass
class Case:
    """A single evaluation sample.

    Attributes:
        prepare_input: Text segments fed to ``Method.Prepare`` as warm-up
            (e.g. the document whose KV cache is prefilled). An empty list
            means no warm-up.
        run_input: The complete prompt fed to ``Method.Run`` — always the full
            text to generate from.
        metadata: Extra information needed for correctness evaluation
            (e.g. the expected answer).
    """

    prepare_input: List[str] = field(default_factory=list)
    run_input: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class Task(ABC):
    """A benchmark workload.

    A task owns the generation of evaluation cases and the correctness check.
    It is deliberately *not* responsible for latency / memory measurement —
    that belongs to system metrics (``core.Metrics.Metric``).

    Subclasses must implement :meth:`Cases` and :meth:`Evaluate`.
    """

    #: Short identifier used in reports. Override in subclasses.
    name: str = "task"

    @abstractmethod
    def Cases(self) -> Iterator[Case]:
        """Yield the evaluation cases, one :class:`Case` per sample."""

    @abstractmethod
    def Evaluate(self, result: Result, metadata: Dict[str, Any]) -> Dict[str, float]:
        """Score one :class:`Result` against the case ``metadata``.

        Returns a dict of *task metric name* -> score, e.g.::

            {"accuracy": 1.0, "exact_match": 1.0}

        These are the task's own metrics (accuracy, F1, EM, ...); the engine
        aggregates them into the report.
        """
