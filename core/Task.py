"""Task abstraction: defines *what* to evaluate."""

import sys
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import wraps
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional

from .Config import MaxSampleLen as ConfiguredMaxSampleLen
from .Result import Result

if TYPE_CHECKING:
    from .Workflow import Workflow


@dataclass
class Case:
    """A single evaluation sample.

    A Case combines:
    - input: The raw benchmark data (type defined by Workflow, e.g. RAGInput)
    - workflow: A stateful execution policy that maps input to Method calls
    - metadata: Extra info for Task.Evaluate (e.g. expected answer)

    The Workflow produces Actions (Prepare/Run) that the Engine executes.
    This design supports both static RAG (prepare→run) and dynamic multi-agent
    scenarios where the execution graph emerges at runtime.

    Attributes:
        input: Input data for this workflow. Type is defined by the Workflow.
        workflow: Stateful execution policy. Decides how to map input to
            a sequence of Method.Prepare/Run calls.
        metadata: Extra information needed for correctness evaluation
            (e.g. the expected answer).
    """

    input: Any = None
    workflow: "Workflow" = None  # type: ignore
    metadata: Dict[str, Any] = field(default_factory=dict)


class Task(ABC):
    """A benchmark workflow.

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
    #: Default generation budget for this task family. Subclasses override it
    #: when their benchmark needs longer answers.
    defaultMaxNewTokens: int = 64

    def __init_subclass__(cls, **kwargs):
        """Apply the shared sample-length filter to every task implementation."""
        super().__init_subclass__(**kwargs)
        implementation = cls.__dict__.get("Cases")
        if implementation is None or getattr(
            implementation, "_max_sample_filter_applied", False
        ):
            return

        @wraps(implementation)
        def filtered_cases(self, *args, **kwargs):
            limit = ConfiguredMaxSampleLen()
            requested = self._RequestedCaseCount()
            if requested == 0:
                return

            # Existing dataset loaders cap maxSamples before yielding. Raise
            # that source cap while filtering so skipped examples can be
            # replaced, then restore the caller's setting on every exit path.
            source_limit = getattr(self, "maxSamples", None)
            expand_source = (
                isinstance(source_limit, int)
                and not isinstance(source_limit, bool)
                and source_limit >= 0
            )
            if expand_source:
                self.maxSamples = sys.maxsize

            accepted = 0
            try:
                for case in implementation(self, *args, **kwargs):
                    if self._CaseSampleLength(case) >= limit:
                        continue
                    yield case
                    accepted += 1
                    if requested is not None and accepted >= requested:
                        break
            finally:
                if expand_source:
                    self.maxSamples = source_limit

        filtered_cases._max_sample_filter_applied = True
        cls.Cases = filtered_cases

    def _RequestedCaseCount(self) -> Optional[int]:
        """Return the task's requested case count, if it declares one."""
        for attribute in ("maxSamples", "nCases"):
            value = getattr(self, attribute, None)
            if value is None or value == -1:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{attribute} must be an integer or None")
            if value < 0:
                raise ValueError(f"{attribute} must be non-negative")
            return value
        return None

    @staticmethod
    def _CaseSampleLength(case: Case) -> int:
        """Measure task payloads in model tokens before returning a Case."""
        length = Task._SampleValueLength(getattr(case, "input", None))
        if length is None:
            workflowData = getattr(getattr(case, "workflow", None), "_data", None)
            length = Task._SampleValueLength(workflowData)
        if length is not None:
            return length

        # External-task adapters may not expose their prompt before execution;
        # use any dataset-provided token length when available.
        metadata = getattr(case, "metadata", {})
        if isinstance(metadata, Mapping):
            for key in ("sample_length", "length"):
                length = metadata.get(key)
                if (
                    isinstance(length, int)
                    and not isinstance(length, bool)
                    and length >= 0
                ):
                    return length
        return 0

    @staticmethod
    def _SampleValueLength(value: Any) -> Optional[int]:
        """Count a nested text or token sequence using the configured model."""
        # RAG inputs carry the complete rendered prompt as native token IDs;
        # multi-agent inputs carry the source user task as text.
        for attribute in (
            "run_input",
            "task",
            "prompt",
            "text",
            "context",
            "question",
            "input",
            "content",
            "messages",
        ):
            if isinstance(value, Mapping):
                candidate = value.get(attribute)
            else:
                candidate = getattr(value, attribute, None)
            if candidate is not None:
                return Task._SampleValueLength(candidate)

        if isinstance(value, str):
            from .Config import ModelPath
            from helpers.backends.Tokenization import EncodeText

            return len(EncodeText(value, ModelPath()))
        if isinstance(value, (list, tuple)):
            if all(
                isinstance(token, int) and not isinstance(token, bool)
                for token in value
            ):
                return len(value)
            return sum(
                length
                for item in value
                if (length := Task._SampleValueLength(item)) is not None
            )
        return None

    def __init__(
        self,
        *,
        tag: Optional[str] = None,
        maxNewTokens: Optional[int] = None,
    ):
        """Create a lightweight task configuration.

        Subclasses must chain through ``super().__init__()`` (passing
        ``tag``) so :attr:`tag` is set uniformly on every subclass.
        """
        self.tag = tag
        if maxNewTokens is None:
            maxNewTokens = self.defaultMaxNewTokens
        if isinstance(maxNewTokens, bool) or not isinstance(maxNewTokens, int):
            raise TypeError("maxNewTokens must be an integer")
        if maxNewTokens < 1:
            raise ValueError("maxNewTokens must be at least 1")
        self.maxNewTokens = maxNewTokens

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
