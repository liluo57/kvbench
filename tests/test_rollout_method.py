import json
import math

import pytest

from core.Method import Method
from core.Result import Result
from core.Task import Case, Task
from core.Worker import EvaluatePair
from methods import RolloutMethod
from workflow.RAGWorkflow import RAGInput, RAGWorkflow


class SequenceMethod(Method):
    name = "sequence"

    def __init__(self, values_by_call):
        super().__init__(gpuNums=1)
        self.values_by_call = [list(values) for values in values_by_call]
        self.calls = []

    def Prepare(self, data):
        pass

    def Run(self, data, retainOutput=None, maxNewTokens=None):
        call = len(self.calls)
        self.calls.append(list(data))
        values = self.values_by_call[call]
        return [Result(output=value) for value in values]


class ScalarTask(Task):
    name = "scalar"

    def __init__(self, count):
        super().__init__()
        self.count = count

    def Cases(self):
        for sample_id in range(self.count):
            data = RAGInput(prepare_input=[], run_input=f"sample-{sample_id}")
            yield Case(
                input=data,
                workflow=RAGWorkflow(sample_id, data),
                metadata={"sample_id": sample_id},
            )

    def Evaluate(self, result, metadata):
        return {"quality": float(result.output)}


def test_rollout_uses_population_statistics_and_keeps_one_result_per_input():
    base = SequenceMethod([[1], [2], [3]])
    method = RolloutMethod(base, num_rollouts=3)

    results = method.Run(["sample"])

    assert len(results) == 1
    result = results[0]
    assert result.output == pytest.approx(2.0)
    assert result.metadata["num_rollouts"] == 3
    assert result.metadata["rollout_mean"] == pytest.approx(2.0)
    assert result.metadata["rollout_variance"] == pytest.approx(2 / 3)
    assert result.metadata["rollout_std"] == pytest.approx(math.sqrt(2 / 3))
    assert result.metadata["rollout"]["num_rollouts"] == 3
    assert [item.output for item in result.metadata["rollout"]["results"]] == [
        1,
        2,
        3,
    ]
    assert len(base.calls) == 3


def test_two_cases_are_not_flattened_into_six_samples_and_task_scores_are_means():
    base = SequenceMethod([[1, 10], [2, 20], [3, 30]])
    method = RolloutMethod(base, num_rollouts=3)

    report = EvaluatePair(
        ScalarTask(2),
        method,
        metrics=[],
        batchSize=2,
        recordAllSamples=True,
    )

    assert report["cases"] == 2
    assert report["task_metrics"]["quality"]["samples"] == [2.0, 20.0]
    assert len(report["sample_results"]) == 2
    assert [item["sample_id"] for item in report["sample_results"]] == [0, 1]
    assert [item["rollout_mean"] for item in report["sample_results"]] == [
        pytest.approx(2.0),
        pytest.approx(20.0),
    ]
    assert len(base.calls) == 3
    assert all(len(call) == 2 for call in base.calls)
    json.dumps(report)


def test_keep_individual_results_false_keeps_aggregate_and_serializes_without_raw_results():
    base = SequenceMethod([[1], [2], [3]])
    method = RolloutMethod(
        base,
        num_rollouts=3,
        keep_individual_results=False,
    )

    result = method.Run(["sample"])[0]

    assert result.output == pytest.approx(2.0)
    assert result.metadata["rollout"]["num_rollouts"] == 3
    assert "results" not in result.metadata["rollout"]
    json.dumps(result.metadata)

    report = EvaluatePair(
        ScalarTask(1),
        RolloutMethod(
            SequenceMethod([[1], [2], [3]]),
            num_rollouts=3,
            keep_individual_results=False,
        ),
        metrics=[],
        batchSize=1,
    )
    assert report["task_metrics"]["quality"]["mean"] == pytest.approx(2.0)
    assert "results" not in report["sample_results"][0]["rollout"]
    json.dumps(report)


def test_single_rollout_has_zero_variance_and_invalid_count_is_rejected():
    base = SequenceMethod([[7]])
    result = RolloutMethod(base, num_rollouts=1).Run(["sample"])[0]

    assert result.output == 7
    assert result.metadata["rollout"]["mean"] == pytest.approx(7)
    assert result.metadata["rollout"]["variance"] == 0
    assert result.metadata["rollout"]["std"] == 0

    with pytest.raises(ValueError, match="num_rollouts"):
        RolloutMethod(SequenceMethod([[1]]), num_rollouts=0)


def test_task_metrics_support_multiple_scalar_metrics():
    class MultiMetricTask(ScalarTask):
        def Evaluate(self, result, metadata):
            value = float(result.output)
            return {"quality": value, "cost_proxy": value * 2}

    report = EvaluatePair(
        MultiMetricTask(1),
        RolloutMethod(SequenceMethod([[1], [2], [3]]), num_rollouts=3),
        metrics=[],
        batchSize=1,
    )
    sample = report["sample_results"][0]
    assert sample["rollout_metrics"]["quality"]["variance"] == pytest.approx(2 / 3)
    assert sample["rollout_metrics"]["cost_proxy"]["mean"] == pytest.approx(4)
