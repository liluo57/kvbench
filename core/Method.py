"""Method abstraction: defines *how* to optimize and run."""

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

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

    #: Metadata keys that are *method metrics*: per-inference-RUN values the
    #: method records in ``Result.metadata``, which the engine aggregates (with
    #: :func:`~core.Metrics.AggregateStats`) into the report's
    #: ``method_metrics`` section per (method, task) pair. Empty means the
    #: method has no method metrics and the report omits that section entirely.
    method_metrics: tuple[str, ...] = ()

    def __init__(
        self,
        *,
        gpuNums: int = 1,
        perfWeight: float = 1.0,
        maxGpuNums: Optional[int] = None,
        tag: Optional[str] = None,
    ):
        """Create a lightweight method configuration.

        Constructors must not initialize CUDA or load a model.  The Engine
        constructs methods in the coordinator process, assigns physical GPUs,
        then calls :meth:`Initialize` inside a dedicated worker process.
        """
        if isinstance(gpuNums, bool) or not isinstance(gpuNums, int):
            raise TypeError("gpuNums must be an integer")
        if gpuNums < 1:
            raise ValueError("gpuNums must be at least 1")
        if maxGpuNums is not None and gpuNums > maxGpuNums:
            raise ValueError(
                f"{type(self).__name__} supports at most {maxGpuNums} GPU(s), "
                f"got gpuNums={gpuNums}"
            )
        if isinstance(perfWeight, bool) or not isinstance(perfWeight, (int, float)):
            raise TypeError("perfWeight must be a number")
        if float(perfWeight) <= 0:
            raise ValueError("perfWeight must be greater than 0")

        self.tag = tag
        self.gpuNums = gpuNums
        self.perfWeight = float(perfWeight)
        self.gpuIds: List[int] = []

    @property
    def Label(self) -> str:
        """Report name: :attr:`name`, or ``name(tag)`` when a tag is set."""
        return f"{self.name}({self.tag})" if self.tag else self.name

    def Initialize(self, gpuIds: Sequence[int]) -> None:
        """Bind the exact physical GPUs assigned by the Engine.

        Subclasses should call ``super().Initialize(gpuIds)`` before loading
        their backend.  This method intentionally runs only in the worker.
        """
        ids = [int(gpu) for gpu in gpuIds]
        if len(ids) != self.gpuNums:
            raise ValueError(
                f"{self.Label} requires exactly {self.gpuNums} GPU(s), got {ids}"
            )
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate GPU ids assigned to {self.Label}: {ids}")
        self.gpuIds = ids

    @abstractmethod
    def Prepare(self, data: List[List[str]]) -> None:
        """Build reusable state for a batch of PREPARE actions.

        ``data[i]`` is the list of text segments for PREPARE action ``i`` (KV
        prefill, index build, ...). An empty inner list means there is nothing
        to warm up for that action. Called once for each PREPARE action step.
        """

    @abstractmethod
    def Run(self, data: List[str], retainOutput: Optional[List[bool]] = None) -> List[Result]:
        """Run inference on a batch of complete prompts.

        ``data[i]`` is the complete prompt for RUN action ``i``.
        ``retainOutput[i]`` is a future-reuse/lifetime hint for that generated
        output; methods may preserve backend-specific reusable state or ignore
        it. Returns a list of :class:`Result` objects in input order.

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

    def Close(self) -> None:
        """Release backend resources before the worker exits.

        Process termination remains the final CUDA cleanup boundary, but the
        Engine always invokes this hook during an orderly shutdown.
        """
