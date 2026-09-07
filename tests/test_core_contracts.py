import pytest

from core.Config import DatasetDir, Get, LoadConfig, ModelPath
from core.Method import Method
from core.Metrics import AggregateStats
from core.Result import NumOutputTokensKey, Result, TotalTimeKey, TtftKey
from core.Task import Task
from metrics import TTFTMetric, ThroughputMetric


class _Method(Method):
    name = "stub"

    def Prepare(self, data):
        pass

    def Run(self, data, retainOutput=None):
        return [Result(output=value) for value in data]


class _Task(Task):
    name = "stub"

    def Cases(self):
        return iter([])

    def Evaluate(self, result, metadata):
        return {}


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"gpuNums": True}, TypeError),
        ({"gpuNums": 1.5}, TypeError),
        ({"gpuNums": 0}, ValueError),
        ({"gpuNums": 2, "maxGpuNums": 1}, ValueError),
        ({"perfWeight": True}, TypeError),
        ({"perfWeight": "1"}, TypeError),
        ({"perfWeight": 0}, ValueError),
        ({"perfWeight": -1}, ValueError),
    ],
)
def test_method_rejects_invalid_resource_configuration(kwargs, error):
    with pytest.raises(error):
        _Method(**kwargs)


def test_method_label_and_gpu_binding_contract():
    method = _Method(gpuNums=2, perfWeight=3, tag="experiment")

    assert method.Label == "stub(experiment)"
    assert method.perfWeight == 3.0

    with pytest.raises(ValueError, match="requires exactly 2"):
        method.Initialize([0])
    with pytest.raises(ValueError, match="duplicate GPU ids"):
        method.Initialize([1, 1])

    method.Initialize(["2", 3])
    assert method.gpuIds == [2, 3]


def test_task_label_contract():
    task = _Task()

    assert task.tag is None
    assert task.Label == "stub"

    tagged = _Task(tag="experiment")
    assert tagged.tag == "experiment"
    assert tagged.Label == "stub(experiment)"


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


def test_config_loading_is_cached_per_path(tmp_path):
    configPath = tmp_path / "config.yaml"
    configPath.write_text("ModelPath: first\n", encoding="utf-8")

    first = LoadConfig(configPath)
    configPath.write_text("ModelPath: second\n", encoding="utf-8")

    assert first == {"ModelPath": "first"}
    assert LoadConfig(configPath) is first


def test_dataset_and_model_paths_use_explicit_config(tmp_path):
    datasetRoot = tmp_path / "datasets"
    expected = datasetRoot / "ruler"
    expected.mkdir(parents=True)
    config = {"DatasetPath": str(datasetRoot), "ModelPath": "/models/default"}

    assert DatasetDir("ruler", config=config) == expected
    assert ModelPath(config=config) == "/models/default"

    with pytest.raises(FileNotFoundError, match="missing"):
        DatasetDir("missing", config=config)


def test_get_honors_an_explicit_empty_config():
    assert Get("ModelPath", "fallback", config={}) == "fallback"


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
