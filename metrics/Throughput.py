"""Throughput system metric."""

import math

from core.Metrics import AggregateStats, Metric
from core.Result import NumOutputTokensKey, Result, TotalTimeKey


class ThroughputMetric(Metric):
    """Measures generation throughput in output tokens per second.

    Per inference RUN: ``numOutputTokens / totalTime``. The summary
    additionally reports ``throughput_total_tokens_per_sec`` — the aggregate
    throughput over the whole run (sum tokens / sum time), the more meaningful
    number for an amortized / batched workload.
    """

    name = "throughput"

    def __init__(
        self,
        tokensKey: str = NumOutputTokensKey,
        timeKey: str = TotalTimeKey,
    ):
        self.tokensKey = tokensKey
        self.timeKey = timeKey
        self._samples: list[float] = []
        self._totalTokens = 0.0
        self._totalTime = 0.0

    def Update(self, result: Result) -> None:
        tokens = result.performance.get(self.tokensKey)
        time_ = result.performance.get(self.timeKey)
        if tokens is None or time_ is None:
            return
        tokens = float(tokens)
        time_ = float(time_)
        if not math.isfinite(tokens) or tokens < 0:
            raise ValueError(
                f"{self.tokensKey} must be a finite non-negative number, "
                f"got {tokens!r}"
            )
        if not math.isfinite(time_) or time_ <= 0:
            raise ValueError(
                f"{self.timeKey} must be a finite positive number, got {time_!r}"
            )
        self._samples.append(tokens / time_)
        self._totalTokens += tokens
        self._totalTime += time_

    def Summary(self) -> dict:
        stats = AggregateStats(self._samples, name=self.name)
        if self._totalTime > 0:
            stats[f"{self.name}_total_tokens_per_sec"] = self._totalTokens / self._totalTime
        else:
            stats[f"{self.name}_total_tokens_per_sec"] = None
        return stats

    def Reset(self) -> None:
        self._samples = []
        self._totalTokens = 0.0
        self._totalTime = 0.0
