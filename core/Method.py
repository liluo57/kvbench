"""Method abstraction: defines *how* to optimize and run."""

from abc import ABC, abstractmethod
from typing import List, Optional

from .Result import Result


class Method(ABC):
    """A KV cache optimization method, treated as a black box.

    A method owns model interaction, KV cache management, and the optimization
    algorithm. KVBench deliberately does not impose any KV cache abstraction —
    see the "No forced KV abstraction" section of the readme.

    Subclasses must implement :meth:`Prepare` and :meth:`Run`.
    """

    #: Short identifier used in reports. Override in subclasses.
    name: str = "method"

    #: Optional distinguishing label appended to :attr:`name` in reports, e.g.
    #: a knob's value: ``name="cacheblend"`` + ``tag="0.15"`` renders as
    #: ``cacheblend(0.15)``. Set via the constructor's ``tag`` argument.
    tag: Optional[str] = None

    #: Metadata keys that are *method metrics*: per-case values the method
    #: records in ``Result.metadata``, which the engine aggregates (with
    #: :func:`~core.Metrics.AggregateStats`) into the report's
    #: ``method_metrics`` section per (method, task) pair. Empty means the
    #: method has no method metrics and the report omits that section entirely.
    method_metrics: tuple[str, ...] = ()

    def __init__(self, tag: Optional[str] = None):
        """Initialize with an optional distinguishing :attr:`tag`."""
        self.tag = tag

    @property
    def Label(self) -> str:
        """Report name: :attr:`name`, or ``name(tag)`` when a tag is set."""
        return f"{self.name}({self.tag})" if self.tag else self.name

    @abstractmethod
    def Prepare(self, data: List[List[str]]) -> None:
        """Build reusable state for a batch of cases.

        ``data`` is a list of per-case warm-up segments: ``data[i]`` is the
        list of text segments for case ``i`` (KV prefill, index build, ...).
        An empty inner list means there is nothing to warm up for that case.
        Called once per batch, before :meth:`Run`.
        """

    @abstractmethod
    def Run(self, data: List[str], retainOutput: Optional[List[bool]] = None) -> List[Result]:
        """Run inference on a batch of complete prompts.

        ``data[i]`` is the complete prompt for case ``i``. ``retainOutput[i]``
        is a future-reuse/lifetime hint for that generated output; methods may
        preserve backend-specific reusable state or ignore it. Returns a list of
        :class:`Result` objects in the same order as the input.

        The method is responsible for recording raw system timings into each
        ``Result.performance`` so system metrics can be computed, e.g.::

            result.performance[TtftKey] = ttftSeconds
            result.performance[NumOutputTokensKey] = numOutputTokens
            result.performance[TotalTimeKey] = totalSeconds
        """

    def Reset(self) -> None:
        """Clear internal state. Called after each batch.

        The default is a no-op; override only if the method is stateful.
        """
