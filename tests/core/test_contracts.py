"""Tests for contracts."""

import pytest

from core.Method import Method, ResolveMaxNewTokens
from core.Result import Result
from core.Task import Task
from methods import (
    EPIC,
    A3Repo,
    CacheblendLmcache,
    CacheblendRepo,
    CacheClip,
    DependencyAnalysisMethod,
    FullPrefillTransformer,
    FullPrefillVllm,
    FullRecomputeTransformer,
    HypicMethod,
    KVCommTransformer,
    KVPacket,
    LinearKVMethod,
    NaiveCacheblendRepo,
    NaiveTransformer,
    ProphetKV,
    RandRecomputeTransformer,
)


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


@pytest.mark.parametrize("budget,error", [(True, TypeError), (1.5, TypeError), ("2", TypeError), (0, ValueError), (-1, ValueError)])
def test_generation_budget_rejects_invalid_values(budget, error):
    with pytest.raises(error, match="maxNewTokens"):
        ResolveMaxNewTokens(budget)


@pytest.mark.parametrize("budget,expected", [(None, 64), (1, 1), (128, 128)])
def test_generation_budget_preserves_explicit_values_and_default(budget, expected):
    assert ResolveMaxNewTokens(budget) == expected


def test_effective_batch_size_preserves_request_or_applies_method_limit():
    class SerialMethod(_Method):
        maxCaseBatchSize = 1

    assert _Method().EffectiveBatchSize(8) == 8
    assert SerialMethod().EffectiveBatchSize(8) == 1
    assert SerialMethod().EffectiveBatchSize(1) == 1


@pytest.mark.parametrize("size,error", [(True, TypeError), (1.5, TypeError), (0, ValueError), (-1, ValueError)])
def test_invalid_requested_batch_size_is_rejected(size, error):
    with pytest.raises(error, match="batch size"):
        _Method().EffectiveBatchSize(size)


@pytest.mark.parametrize("method_class", [
    A3Repo, CacheblendRepo, CacheClip, DependencyAnalysisMethod, EPIC,
    FullPrefillTransformer, FullRecomputeTransformer, KVCommTransformer,
    KVPacket, NaiveCacheblendRepo, NaiveTransformer, ProphetKV,
    RandRecomputeTransformer,
])
def test_single_gpu_methods_reject_multiple_gpus_before_loading_backend(method_class):
    with pytest.raises(ValueError, match="at most 1 GPU"):
        method_class(gpuNums=2)


@pytest.mark.parametrize("method_class", [CacheblendLmcache, FullPrefillVllm, HypicMethod, LinearKVMethod])
def test_tensor_parallel_methods_accept_multiple_gpus_without_backend_loading(method_class):
    method = method_class(gpuNums=2)
    assert method.gpuNums == 2 and method.gpuIds == []
