import importlib
from types import SimpleNamespace

import pytest
import torch

from core import Config
from core.Worker import _RunMethod
from helpers.backends.TransformersHelper import TransformersGenerator
from methods import A3Repo, EPIC, FullRecomputeTransformer, ProphetKV


@pytest.fixture
def context_limit(monkeypatch):
    original = Config.Get

    def set_limit(limit):
        monkeypatch.setattr(
            Config, "Get",
            lambda key, *args, **kwargs: limit if key == "MaxModelLen" else original(key, *args, **kwargs),
        )

    set_limit(5)
    return set_limit


@pytest.mark.parametrize("length", [5, 6])
def test_worker_rejects_full_or_overlong_prompt_before_legacy_method(context_limit, length):
    method = SimpleNamespace(Run=lambda *_args: pytest.fail("overlong RUN reached method"))
    with pytest.raises(ValueError, match="MaxModelLen=5"):
        _RunMethod(method, [[2] * length], [False], 8)


@pytest.mark.parametrize("cls", [A3Repo, ProphetKV])
def test_native_methods_take_global_limit_and_reject_before_runtime(context_limit, cls):
    method = cls(maxModelLen=99)
    assert method.maxModelLen == 5
    if cls is ProphetKV:
        method._runtime = SimpleNamespace(
            full_generate=lambda *_args: pytest.fail("overlong prompt reached runtime"),
            prefill_isolated=lambda *_args: pytest.fail("overlong chunk reached runtime"),
        )
        with pytest.raises(ValueError, match="MaxModelLen=5"):
            method.Prepare([[[2] * 6]])
    else:
        method._request = lambda *_args: pytest.fail("overlong prompt reached worker")
    with pytest.raises(ValueError, match="MaxModelLen=5"):
        method.Run([[2] * 5], maxNewTokens=8)


def test_hf_generation_counts_cached_prefix_and_bounds_retained_cache(context_limit):
    from test_recompute_backend import runtime

    backend, sampler = runtime()
    sampler.prefillChunk = 2
    sampler._SampleNextToken = lambda *_args: 7
    prefix = sampler.Prefill([2, 3, 4]).past_key_values
    result = sampler.Generate([5], pastKeyValues=prefix, maxNewTokens=8, returnCache=True)
    assert result[3] == 1
    assert result[-2].get_seq_length() == 5
    for layer in result[-2].layers:
        assert not layer.keys.requires_grad
        assert not layer.values.requires_grad
    with pytest.raises(ValueError, match="MaxModelLen=5"):
        sampler.Forward([6], result[-2])
    with pytest.raises(ValueError, match="leave room"):
        sampler.Generate([2] * 5, maxNewTokens=8)
    with pytest.raises(ValueError, match="MaxModelLen=5"):
        sampler.Prefill([2] * 6)


@pytest.mark.parametrize("cls", [EPIC, FullRecomputeTransformer])
def test_recompute_bounds_each_prompt_independently_and_retains_only_reusable_cache(context_limit, cls):
    from test_recompute_backend import bind, runtime

    backend, sampler = runtime()
    sampler._SampleNextToken = lambda *_args: 7
    method = bind(cls(), backend, sampler)
    results = method.Run([[2, 3, 4], [2, 3, 4, 5]], retainOutput=[True, True], maxNewTokens=8)
    assert [r.performance["num_output_tokens"] for r in results] == [2, 1]
    if cls is FullRecomputeTransformer:
        assert method._states == [[], []]
    else:
        assert [state[-1].cache.seq_len for state in method._states] == [5, 5]
    with pytest.raises(ValueError, match="MaxModelLen=5"):
        method.Run([[2] * 5])


def test_a3_worker_bounds_decode_and_rejects_overlong_collection(context_limit, monkeypatch):
    from methods.A3Repo.Worker import A3Worker

    utils = importlib.import_module("methods.A3Repo.backend.utils")
    observed = []

    def decode(*args, **kwargs):
        budget = args[5]
        observed.append(budget)
        return "answer", 0.01, 0.02, [7] * budget

    monkeypatch.setattr(utils, "decode", decode)
    worker = object.__new__(A3Worker)
    worker.args = SimpleNamespace(max_model_len=5, max_new_tokens=8)
    worker._sampling_args = lambda: SimpleNamespace()
    worker.torch = SimpleNamespace(
        tensor=lambda ids, **kwargs: torch.tensor(ids),
        arange=lambda length, **kwargs: torch.arange(length), long=torch.long,
    )
    worker.model = None
    worker.tokenizer = SimpleNamespace(eos_token_id=1, bos_token_id=None)
    worker.chunk_kv = {}
    assert worker._decode([2, 3, 4], {}, 8)[3] == 2
    assert observed == [2]
    with pytest.raises(ValueError, match="max_model_len=5"):
        worker._decode([2] * 5, {})
    with pytest.raises(ValueError, match="max_model_len=5"):
        worker._collect_one("too long", [2] * 6)


def test_prophet_bounds_each_request_budget(context_limit):
    method = ProphetKV(maxNewTokens=8)
    runtime = SimpleNamespace()
    observed = []

    def generate(ids):
        observed.append(runtime.max_new_tokens)
        return "answer", 0.01, 0.02, runtime.max_new_tokens, [7] * runtime.max_new_tokens

    runtime.full_generate = generate
    method._runtime = runtime
    results = method.Run([[2, 3, 4], [2, 3, 4, 5]], maxNewTokens=8)
    assert observed == [2, 1]
    assert [r.performance["num_output_tokens"] for r in results] == [2, 1]
