"""Tests for naive."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from methods.Naive import NaiveTransformer
from tests.support.caches import _OutputCache
from tests.support.models import runtime


@pytest.mark.parametrize("length", [None, 4, 6])
def test_retention_rejects_incomplete_or_excess_native_cache_without_registering(length):
    method = _naive_cache_registrar()
    state = {"segments": [], "ids": [], "caches": []}
    cache = None if length is None else _OutputCache(length)
    with pytest.raises(AssertionError, match="native output KV length"):
        method._registerOutput(state, [7, 8], cache, promptLength=3)
    assert state == {"segments": [], "ids": [], "caches": []}


def test_native_generation_retains_internal_specials_and_excludes_terminal_eos():
    _, sampler = runtime()
    samples = iter([7, 2, 8, 1])
    sampler._SampleNextToken = lambda *args: next(samples)
    method = NaiveTransformer()
    method._gen = sampler
    result = method.Run([[3, 4, 5]], retainOutput=[True], maxNewTokens=5)[0]
    assert result.output_ids == [7, 2, 8]
    assert result.output == "7 8"
    assert result.performance["num_output_tokens"] == 4
    state = method._states[0]
    assert state["segments"] == state["ids"] == [[7, 2, 8]]
    assert all(key.shape[-2] == value.shape[-2] == 3 for key, value in state["caches"][0])


@pytest.mark.parametrize("retain", [None, [], [False], [False, True]])
def test_missing_false_or_out_of_range_retention_does_not_accumulate(retain):
    _, sampler = runtime()
    sampler._SampleNextToken = lambda *args: 7
    method = NaiveTransformer()
    method._gen = sampler
    method.Run([[3, 4]], retainOutput=retain, maxNewTokens=1)
    assert method._states[0]["ids"] == []


def test_no_match_ttft_includes_lookup_and_passes_native_ids(clock, monkeypatch):
    method = _naive_cache_registrar()
    from importlib import import_module
    module = import_module("methods.Naive")
    match = module.ComposeInterleavedReuse

    def lookup(*args):
        clock.advance(5)
        return match(*args)

    monkeypatch.setattr(module, "ComposeInterleavedReuse", lookup)
    result = method.Run([[41, 42]], maxNewTokens=1)[0]
    assert method._gen.generate_calls[0][0] == [41, 42]
    assert result.performance["ttft"] == pytest.approx(5.01)
    assert result.performance["total_time"] == pytest.approx(5.02)
    assert result.output_ids == [10]


def _naive_cache_registrar():
    method = object.__new__(NaiveTransformer)
    method._states = []

    class Generator:
        tokenizer = SimpleNamespace(all_special_ids=[99])
        _torch = SimpleNamespace(no_grad=nullcontext)

        def __init__(self):
            self.prefill_calls = []
            self.generate_calls = []

        @staticmethod
        def OutputTokenIds(token_ids):
            values = list(token_ids)
            return values[:-1] if values and values[-1] == 99 else values

        @staticmethod
        def Forward(token_ids, cache):
            assert token_ids == [8]
            key, value = cache.layers[0]
            cache.layers[0] = (
                torch.cat([key, torch.ones((1, 1, 1, 2))], dim=-2),
                torch.cat([value, torch.ones((1, 1, 1, 2))], dim=-2),
            )
            return SimpleNamespace(past_key_values=cache)

        def Prefill(self, token_ids, pastKeyValues=None):
            self.prefill_calls.append(list(token_ids))
            cache = _OutputCache(0) if pastKeyValues is None else pastKeyValues
            token_count = len(token_ids)
            key = torch.zeros((1, 1, token_count, 2))
            value = torch.zeros((1, 1, token_count, 2))
            cache.update(key, value, 0)
            return SimpleNamespace(past_key_values=cache)

        def Generate(self, token_ids, *, pastKeyValues=None, returnTokenIds=False, **_kwargs):
            self.generate_calls.append((list(token_ids), pastKeyValues))
            return ("answer", 0.01, 0.02, 1, [10]) if returnTokenIds else ("answer", 0.01, 0.02, 1)

    method._gen = Generator()
    return method


def test_naive_retains_internal_special_with_exact_native_cache_rows():
    method = _naive_cache_registrar()
    state = {"segments": [], "ids": [], "caches": []}

    method._registerOutput(
        state,
        [7, 99, 8],
        _OutputCache(6),
        promptLength=3,
    )

    assert state["ids"] == [[7, 99, 8]]
    assert state["caches"][0][0][0].shape[-2] == 3
    assert method._gen.prefill_calls == []


def test_naive_native_output_cache_hits_next_prompt_without_full_prefill():
    method = _naive_cache_registrar()
    state = {"segments": [], "ids": [], "caches": []}
    method._registerOutput(
        state,
        [7, 99, 8],
        _OutputCache(6),
        promptLength=3,
    )
    assert state["ids"] == [[7, 99, 8]]
    method._gen.prefill_calls.clear()
    method._states = [state]

    result = method.Run([[1, 7, 99, 8, 9]])[0]

    assert result.metadata["reuse_ratio"] == pytest.approx(0.6)
    assert method._gen.prefill_calls == [[1], [9]]
    assert method._gen.generate_calls[0][0] == [9]


def test_naive_counts_cached_final_prompt_token_even_though_it_is_replayed():
    method = _naive_cache_registrar()
    assert method.method_metrics == ("reuse_ratio",)
    state = {"segments": [], "ids": [], "caches": []}
    method._registerOutput(state, [7, 99, 8], _OutputCache(6), promptLength=3)
    method._states = [state]

    result = method.Run([[7, 99, 8]])[0]

    assert result.metadata["reuse_ratio"] == pytest.approx(1.0)
    assert method._gen.generate_calls[0][0] == [8]
    assert method._gen.generate_calls[0][1].get_seq_length() == 2


def test_naive_no_match_reports_zero_reuse_ratio():
    method = _naive_cache_registrar()
    method._states = [{"segments": [], "ids": [], "caches": []}]

    result = method.Run([[1, 2]])[0]

    assert result.metadata["reuse_ratio"] == 0.0
