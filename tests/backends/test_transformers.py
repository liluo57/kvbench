"""Tests for transformers."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from core.Sampling import ResolveSamplingConfig
from helpers.backends.TransformersHelper import TransformersGenerator
from tests.support.models import runtime


def test_transformers_returncache_keeps_raw_ids_and_token_return_keeps_internal_special():
    generator = object.__new__(TransformersGenerator)
    generator._torch = SimpleNamespace(no_grad=nullcontext)
    generator._Generator = lambda: object()
    forwarded = []
    generated = iter([7, 99, 8])
    generator._SampleNextToken = lambda *_args: next(generated)
    generator.Prefill = lambda _ids: SimpleNamespace(
        logits=torch.zeros((1, 1, 1)), past_key_values=object()
    )
    def forward(ids, *_args, **_kwargs):
        forwarded.append(list(ids))
        return SimpleNamespace(logits=torch.zeros((1, 1, 1)), past_key_values=object())

    generator.Forward = forward
    generator.tokenizer = SimpleNamespace(
        all_special_ids=[99],
        decode=lambda ids, skip_special_tokens: "visible",
    )
    generator.eosIds = {88}

    cached = generator.Generate(
        [1], maxNewTokens=3, returnCache=True, returnTokenIds=True
    )
    assert cached[-1] == [7, 99, 8]
    assert forwarded[-1] == [8]  # final budget token receives its cache row

    forwarded.clear()
    generated = iter([7, 99, 8])
    visible = generator.Generate([1], maxNewTokens=3, returnTokenIds=True)
    assert visible[-1] == [7, 99, 8]
    assert forwarded[-1] == [99]  # no generated KV is retained on this call

    generator.eosIds = {88}
    generated = iter([7, 99, 88])
    stopped = generator.Generate([1], maxNewTokens=3, returnCache=True)
    assert stopped[-1] == [7, 99, 88]
    assert forwarded[-1] == [99]  # terminal EOS remains outside the KV fragment


@pytest.mark.parametrize("penalty", [
    {"repetition_penalty": 2.0},
    {"presence_penalty": 2.0},
    {"frequency_penalty": 1.0},
])
def test_hf_argmax_applies_resolved_history_penalties_without_mutating_logits(penalty):
    sampler = object.__new__(TransformersGenerator)
    sampler._torch = torch
    sampler.samplingConfig = ResolveSamplingConfig(config={
        "ModelConfig": {"mode": "override", "do_sample": False, **penalty},
    })
    logits = torch.tensor([7.0, 6.0, -1.0])
    original = logits.clone()

    assert sampler._SampleNextToken(logits, [0, 0], None) == 1
    torch.testing.assert_close(logits, original)


def test_canonical_greedy_policy_remains_unmodified_argmax():
    sampler = object.__new__(TransformersGenerator)
    sampler._torch = torch
    sampler.samplingConfig = ResolveSamplingConfig(config={
        "ModelConfig": {"mode": "greedy", "presence_penalty": 100.0},
    })

    assert sampler._SampleNextToken(torch.tensor([7.0, 6.0, -1.0]), [0, 0], None) == 0


def test_transformers_batch_helper_does_not_amortize_ttft():
    from helpers.backends.TransformersHelper import TransformersGenerator

    generator = object.__new__(TransformersGenerator)
    generator.Generate = lambda ids, **_kwargs: (
        f"answer-{ids[0]}",
        0.01 * ids[0],
        0.02 * ids[0],
        ids[0],
    )

    results = generator.GenerateBatch([[2], [5]])

    assert [item[1] for item in results] == [0.02, 0.05]


def test_hf_generation_counts_cached_prefix_and_bounds_retained_cache(context_limit):

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
