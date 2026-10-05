"""Tests for full recompute."""

import pytest

from methods import FullRecomputeTransformer
from tests.support.models import bind, runtime


def test_full_recompute_does_not_accumulate_unused_kv_or_forward_the_final_sample(monkeypatch):
    backend, sampler = runtime()
    sampler._SampleNextToken = lambda *args: 7
    method = bind(FullRecomputeTransformer(), backend, sampler)
    forward = backend.forward
    forwarded = []

    def record_forward(**kwargs):
        forwarded.append(kwargs["input_ids"].tolist())
        return forward(**kwargs)

    monkeypatch.setattr(backend, "forward", record_forward)
    prompts = [[2, 3, 4], [2, 3, 4, 7, 5], [2, 3, 4, 7, 5, 7, 6]]
    for prompt in prompts:
        result = method.Run([prompt], retainOutput=[True], maxNewTokens=1)[0]
        assert result.output_ids == [7]
        assert result.metadata["cached_tokens"] == 0
        assert method._states == [[]]
    assert forwarded == [[prompt] for prompt in prompts]


def test_prepare_never_prefills_and_budget_is_capped_per_prompt(context_limit, monkeypatch):
    backend, sampler = runtime()
    sampler._SampleNextToken = lambda *args: 7
    method = bind(FullRecomputeTransformer(), backend, sampler)
    monkeypatch.setattr(backend, "prefill_caches", lambda **kwargs: pytest.fail("baseline warmed KV"))
    method.Prepare([[[2, 3]], [[4, 5]]])
    results = method.Run([[2, 3, 4], [2, 3, 4, 5]], retainOutput=[True, True], maxNewTokens=8)
    assert [r.performance["num_output_tokens"] for r in results] == [2, 1]
    assert all(r.metadata["reuse_ratio"] == 0 for r in results)
    assert method._states == [[], []]
