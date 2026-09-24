"""Core data structures: :class:`Result` and the well-known performance keys.

The key constants define the contract between a :class:`Method` and the system
metrics. A method should fill the matching fields of ``Result.performance`` so
metrics (TTFT, throughput, ...) can be computed.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

#: End-to-end online time to the first output token, in seconds.
#:
#: The clock starts when the request enters ``Method.Run`` and ends when the
#: inference backend makes the first generated token available.  Request-
#: dependent lookup, composition, cache assembly/transformation, selection,
#: repair, fresh prefill, transfers, IPC, and backend queueing are included.
#: ``Method.Initialize`` and reusable state built by ``Method.Prepare`` are
#: outside this measurement.
TtftKey = "ttft"
#: Number of generated (output) tokens.
NumOutputTokensKey = "num_output_tokens"
#: Total wall-clock time of ``Method.Run``, in seconds.
TotalTimeKey = "total_time"


@dataclass
class Result:
    """A single inference outcome returned by ``Method.Run``.

    Attributes:
        output: The generated output. Consumed by ``Task.Evaluate``.
        performance: Raw system-level timings recorded by the method
            (see the key constants). Consumed by system metrics.
        metadata: Arbitrary extra information produced by the method
            (e.g. ``reuse_ratio``). Keys declared in
            ``Method.method_metrics`` are aggregated into the report's
            ``method_metrics`` section; the rest is kept for diagnostics.
    """

    output: Any = None
    performance: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Score aggregation helpers — used by ``core.Worker.EvaluatePair`` to roll
# per-case task / method scores into the report dict. They live here (rather
# than in ``core.Worker``) because they describe the shape of result
# aggregation, not the evaluation loop.
# ---------------------------------------------------------------------------


def NormalizeScores(scores: Any) -> Dict[str, float]:
    """Coerce a ``Task.Evaluate`` return value to a ``{name: float}`` dict.

    Accepts:

    - ``None`` → ``{}`` (no scoring signal — the task didn't score this case).
    - a single ``int``/``float`` → ``{"score": float(value)}``.
    - any mapping → ``dict(mapping)`` (the dict's own values are passed
      through unchanged; downstream callers handle type coercion per metric).
    """
    if scores is None:
        return {}
    if isinstance(scores, (int, float)):
        return {"score": float(scores)}
    return dict(scores)


def AggregateScores(
    perCase: Dict[str, List[float]],
    *,
    includeSamples: bool = False,
    includeStatistics: bool = False,
) -> Dict[str, Any]:
    """Roll per-case score lists into one stats dict per metric.

    An empty list for a name yields ``{"mean": None}`` so the field is still
    present in the report — callers can distinguish "no data" from "missing
    metric" without a key check. With ``includeSamples``, each score's input-
    order values are also returned under ``"samples"``. ``includeStatistics``
    adds population variance and standard deviation. It is opt-in so the
    historical report shape for non-rollout Methods remains unchanged.
    """
    result: Dict[str, Any] = {}
    for name, values in perCase.items():
        stats: Dict[str, Any] = {
            "mean": (sum(values) / len(values)) if values else None
        }
        if includeStatistics:
            if values:
                mean = stats["mean"]
                variance = sum((value - mean) ** 2 for value in values) / len(values)
                stats["variance"] = variance
                stats["std"] = variance ** 0.5
            else:
                stats["variance"] = None
                stats["std"] = None
        if includeSamples:
            stats["samples"] = list(values)
        result[name] = stats
    return result
