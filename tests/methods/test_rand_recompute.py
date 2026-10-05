"""Tests for rand recompute."""

import random

import pytest
import torch

from helpers.backends.CacheBackend.Recompute import CacheHit
from methods import RandRecomputeTransformer
from tests.support.models import (
    assert_reordered_workflow_matches_full_recompute,
    bind,
    runtime,
)


def _snapshots(cache):
    return [
        (layer.key.clone(), layer.value.clone(), layer.logical_positions.clone())
        for layer in cache.layers
    ]


def _assert_unchanged(cache, saved):
    for layer, (key, value, positions) in zip(cache.layers, saved, strict=True):
        torch.testing.assert_close(layer.key, key, rtol=0, atol=0)
        torch.testing.assert_close(layer.value, value, rtol=0, atol=0)
        torch.testing.assert_close(layer.logical_positions, positions, rtol=0, atol=0)


def test_rand_retained_output_is_native_slice_and_reused_after_moves_and_repeats(monkeypatch):
    backend, sampler = runtime()
    method = bind(RandRecomputeTransformer(recomputeRatio=0, seed=5), backend, sampler)
    method.Prepare([[[2, 3]]])
    prepared_snapshot = _snapshots(method._states[0][0].cache)
    # Retention must slice the active decode cache, never warm the generated
    # fragment with a second standalone prefill.
    monkeypatch.setattr(
        backend,
        "prefill_caches",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("output was re-prefilled")),
    )

    samples = iter([14, 15])
    sampler._SampleNextToken = lambda *args: next(samples)
    prompt = [7, 8, 9]
    native_caches = []
    forward = backend.forward

    def record_forward(**kwargs):
        result = forward(**kwargs)
        native_caches.append(result.cache.clone())
        return result

    monkeypatch.setattr(backend, "forward", record_forward)
    result = method.Run([prompt], retainOutput=[True], maxNewTokens=2)[0]
    retained = method._states[0][-1]
    assert retained.ids == result.output_ids == [14, 15]
    assert retained.cache.seq_len == retained.cache.doc_len == 2
    assert retained.cache.layers[0].logical_positions.tolist() == [[0, 1]]

    # The last captured cache is the completed native prompt+output cache. The
    # retained fragment must be its output slice, re-rotated to local positions.
    full = native_caches[-1]
    old_positions = full.layers[0].logical_positions[:, len(prompt):]
    new_positions = torch.arange(2, device=old_positions.device).unsqueeze(0)
    expected = full.__class__([
        type(layer)(
            layer.key[:, :, len(prompt):].clone(),
            layer.value[:, :, len(prompt):].clone(),
            layer.logical_positions[:, len(prompt):].clone(),
        )
        for layer in full.layers
    ]).rerotate(backend.rotary_emb, old_positions, new_positions, nope_dim=backend.nope_dim)
    for actual, native in zip(retained.cache.layers, expected.layers, strict=True):
        torch.testing.assert_close(actual.key, native.key)
        torch.testing.assert_close(actual.value, native.value, rtol=0, atol=0)
        torch.testing.assert_close(actual.logical_positions, native.logical_positions, rtol=0, atol=0)

    retained_snapshot = _snapshots(retained.cache)
    _assert_unchanged(method._states[0][0].cache, prepared_snapshot)
    _assert_unchanged(retained.cache, retained_snapshot)

    later = [20, *retained.ids, 21, *retained.ids, 22]
    sampler._SampleNextToken = lambda *args: 16
    reused = method.Run([later], maxNewTokens=1)[0]
    assert reused.metadata["cached_tokens"] == 2 * len(retained.ids)
    assert reused.output_ids == [16]
    _assert_unchanged(retained.cache, retained_snapshot)


def test_rand_retention_keeps_interior_special_id_and_excludes_only_terminal_eos():
    backend, sampler = runtime()
    method = bind(RandRecomputeTransformer(recomputeRatio=0), backend, sampler)
    method.Prepare([[]])
    samples = iter([2, 1])
    sampler._SampleNextToken = lambda *args: next(samples)

    result = method.Run([[7, 8, 9]], retainOutput=[True], maxNewTokens=4)[0]

    assert result.output_ids == [2]
    assert 2 in sampler.tokenizer.all_special_ids
    assert result.output == ""
    assert result.performance["num_output_tokens"] == 2
    assert method._states[0][-1].ids == [2]
    assert method._states[0][-1].cache.seq_len == 1


def test_rand_retention_eos_only_and_retain_false_do_not_accumulate():
    backend, sampler = runtime()
    method = bind(RandRecomputeTransformer(recomputeRatio=0), backend, sampler)
    method.Prepare([[[2, 3]]])
    initial = method._states[0][:]
    sampler._SampleNextToken = lambda *args: 1

    eos_only = method.Run([[7, 8, 9]], retainOutput=[True], maxNewTokens=2)[0]
    assert eos_only.output_ids == []
    assert method._states[0] == initial

    sampler._SampleNextToken = lambda *args: 14
    retained_false = method.Run([[7, 8, 9]], retainOutput=[False], maxNewTokens=1)[0]
    assert retained_false.output_ids == [14]
    assert method._states[0] == initial


def test_random_policy_is_reproducible_without_changing_global_rng():
    state = random.getstate()
    method = RandRecomputeTransformer(recomputeRatio=0.5, seed=42)
    hits = [CacheHit(0, 4, None), CacheHit(7, 15, None), CacheHit(20, 24, None)]
    first = method._select(hits)
    assert len(first) == 6 and all(7 <= i < 15 or 20 <= i < 24 for i in first)
    assert first == method._select(hits)
    assert random.getstate() == state


def test_reordering_repetition_and_fresh_spans_match_full_recompute():
    assert_reordered_workflow_matches_full_recompute(RandRecomputeTransformer, {"recomputeRatio": 1, "seed": 3})


@pytest.mark.parametrize("ratio", [float("nan"), float("inf"), -0.1, 1.1, True])
def test_invalid_recompute_ratio_is_rejected(ratio):
    with pytest.raises(ValueError, match="recomputeRatio"):
        RandRecomputeTransformer(recomputeRatio=ratio)
