"""System metrics: define *how to measure* system-level performance.

These are distinct from task metrics (accuracy, F1, EM, ...). Task metrics are
owned by :class:`~core.Task.Task.Evaluate`; system metrics (TTFT, throughput,
memory, ...) are owned by :class:`Metric` subclasses.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .Result import Result


def AggregateStats(
    samples: List[float], *, name: str, includeSamples: bool = False
) -> Dict[str, Any]:
    """Compute common summary statistics for numeric ``samples``.

    Returns a flat dict keyed as ``<name>_<stat>`` so metrics can be merged
    into one report without key collisions. When ``includeSamples`` is true,
    the input-order values are also returned under ``"samples"``. Returns
    ``{name: None}`` when ``samples`` is empty.
    """
    if not samples:
        stats: Dict[str, Any] = {name: None}
        if includeSamples:
            stats["samples"] = []
        return stats

    sortedSamples = sorted(samples)
    n = len(sortedSamples)
    mean = sum(sortedSamples) / n

    def _Percentile(q: float) -> float:
        # Linear-interpolated percentile, stdlib only (no numpy).
        pos = (n - 1) * q
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        return sortedSamples[lo] * (1.0 - frac) + sortedSamples[hi] * frac

    stats = {
        f"{name}_count": n,
        f"{name}_mean": mean,
        f"{name}_min": sortedSamples[0],
        f"{name}_max": sortedSamples[-1],
        f"{name}_p50": _Percentile(0.5),
        f"{name}_p90": _Percentile(0.9),
        f"{name}_p99": _Percentile(0.99),
    }
    if includeSamples:
        # Keep the input order here. Percentiles use sortedSamples above, but
        # callers may need to correlate each value with its benchmark Sample.
        stats["samples"] = list(samples)
    return stats


class Metric(ABC):
    """Aggregates system-level measurements across the results of one run.

    A metric instance is scoped to a single (method, task) pair: the engine
    resets it before each run and calls :meth:`Update` once per inference RUN.

    Subclasses must implement :meth:`Update` and :meth:`Summary`.
    """

    #: Short identifier used in reports. Override in subclasses.
    name: str = "metric"

    @abstractmethod
    def Update(self, result: Result) -> None:
        """Consume one :class:`Result` (called once per inference RUN)."""

    @abstractmethod
    def Summary(self) -> Dict[str, Any]:
        """Return aggregate statistics, e.g. from :func:`AggregateStats`."""

    def Samples(self) -> Optional[List[Any]]:
        """Return raw samples in update order when the metric exposes them.

        Metrics that do not retain raw values can leave the default ``None``;
        the report writer will then include only their summary statistics.
        """
        return None

    def Reset(self) -> None:
        """Clear accumulated state (default no-op)."""
