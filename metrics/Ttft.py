"""TTFT (Time To First Token) system metric."""

import math

from core.Metrics import AggregateStats, Metric
from core.Result import Result, TtftKey


class TTFTMetric(Metric):
    """Measures time to first output token, in seconds.

    Reads ``result.performance[TtftKey]``, which the method records. The
    summary reports count / mean / min / max / p50 / p90 / p99.
    """

    name = "ttft"

    def __init__(self, key: str = TtftKey):
        self.key = key
        self._samples: list[float] = []

    def Update(self, result: Result) -> None:
        value = result.performance.get(self.key)
        if value is not None:
            value = float(value)
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"{self.key} must be a finite non-negative number, got {value!r}"
                )
            self._samples.append(value)

    def Summary(self) -> dict:
        return AggregateStats(self._samples, name=self.name)

    def Reset(self) -> None:
        self._samples = []
