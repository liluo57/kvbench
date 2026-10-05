"""Tests for metrics."""

import pytest

from core.Metrics import AggregateStats
from core.Result import NumOutputTokensKey, Result, TotalTimeKey, TtftKey
from metrics import ThroughputMetric, TTFTMetric


def test_aggregate_stats_empty_and_interpolated_percentiles():
    assert AggregateStats([], name="latency") == {"latency": None}

    stats = AggregateStats([4, 1, 3, 2], name="latency")
    assert stats == {
        "latency_count": 4,
        "latency_mean": 2.5,
        "latency_min": 1,
        "latency_max": 4,
        "latency_p50": 2.5,
        "latency_p90": pytest.approx(3.7),
        "latency_p99": pytest.approx(3.97),
    }


def test_builtin_metrics_ignore_incomplete_results_and_reset():
    ttft = TTFTMetric()
    throughput = ThroughputMetric()
    incomplete = Result(performance={TotalTimeKey: 1})
    zeroTime = Result(
        performance={TtftKey: 0.5, TotalTimeKey: 0, NumOutputTokensKey: 4}
    )
    complete = Result(
        performance={TtftKey: 1.5, TotalTimeKey: 2, NumOutputTokensKey: 6}
    )

    for result in (incomplete, zeroTime, complete):
        ttft.Update(result)
    for result in (incomplete, complete):
        throughput.Update(result)

    assert ttft.Summary()["ttft_count"] == 2
    assert ttft.Summary()["ttft_mean"] == 1.0
    assert throughput.Summary()["throughput_count"] == 1
    assert throughput.Summary()["throughput_mean"] == 3.0
    assert throughput.Summary()["throughput_total_tokens_per_sec"] == 3.0

    ttft.Reset()
    throughput.Reset()
    assert ttft.Summary() == {"ttft": None}
    assert throughput.Summary() == {
        "throughput": None,
        "throughput_total_tokens_per_sec": None,
    }


@pytest.mark.parametrize(
    ("performance", "message"),
    [
        ({NumOutputTokensKey: 4, TotalTimeKey: 0}, "total_time"),
        ({NumOutputTokensKey: 4, TotalTimeKey: -2}, "total_time"),
        ({NumOutputTokensKey: 4, TotalTimeKey: float("nan")}, "total_time"),
        ({NumOutputTokensKey: 4, TotalTimeKey: float("inf")}, "total_time"),
        ({NumOutputTokensKey: -1, TotalTimeKey: 2}, "num_output_tokens"),
        ({NumOutputTokensKey: float("nan"), TotalTimeKey: 2}, "num_output_tokens"),
        ({NumOutputTokensKey: float("inf"), TotalTimeKey: 2}, "num_output_tokens"),
    ],
)
def test_throughput_rejects_invalid_present_values(performance, message):
    metric = ThroughputMetric()

    with pytest.raises(ValueError, match=message):
        metric.Update(Result(performance=performance))


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_ttft_rejects_invalid_present_values(value):
    metric = TTFTMetric()

    with pytest.raises(ValueError, match="ttft"):
        metric.Update(Result(performance={TtftKey: value}))
