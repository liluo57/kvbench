"""Task abstraction: defines *what* to evaluate."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional

from .Result import Result

if TYPE_CHECKING:
    from .Workload import Workload


@dataclass
class Case:
    """A single evaluation sample.

    A Case combines:
    - input: The raw benchmark data (type defined by Workload, e.g. RAGInput)
    - workload: A stateful execution policy that maps input to Method calls
    - metadata: Extra info for Task.Evaluate (e.g. expected answer)

    The Workload produces Actions (Prepare/Run) that the Engine executes.
    This design supports both static RAG (prepare→run) and dynamic multi-agent
    scenarios where the execution graph emerges at runtime.

    Attributes:
        input: Input data for this workload. Type is defined by the Workload.
        workload: Stateful execution policy. Decides how to map input to
            a sequence of Method.Prepare/Run calls.
        metadata: Extra information needed for correctness evaluation
            (e.g. the expected answer).
    """

    input: Any = None
    workload: "Workload" = None  # type: ignore
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

    #: Optional distinguishing label appended to :attr:`name` in reports, e.g.
    #: a knob's value: ``name="niah"`` + ``tag="shuffle"`` renders as
    #: ``niah(shuffle)``. Set via the constructor's ``tag`` argument.
    tag: Optional[str] = None

    def __init__(self, *, tag: Optional[str] = None):
        """Create a lightweight task configuration.

        Subclasses must chain through ``super().__init__()`` (passing
        ``tag``) so :attr:`tag` is set uniformly on every subclass.
        """
        self.tag = tag

    @property
    def Label(self) -> str:
        """Report name: :attr:`name`, or ``name(tag)`` when a tag is set."""
        return f"{self.name}({self.tag})" if self.tag else self.name

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
