"""Core data structures: :class:`Result` and the well-known performance keys.

The key constants define the contract between a :class:`Method` and the system
metrics. A method should fill the matching fields of ``Result.performance`` so
metrics (TTFT, throughput, ...) can be computed.
"""

from dataclasses import dataclass, field
from typing import Any, Dict

#: Time to first output token, in seconds.
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
