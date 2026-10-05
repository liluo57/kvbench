"""Tests for kvcomm."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from methods.KVComm import KVCommTransformer, _NewCache, _StackCache
from tests.support.caches import _OutputCache
from tests.support.models import runtime


def test_kvcomm_retains_internal_special_and_reuses_native_cache(monkeypatch):
    method = object.__new__(KVCommTransformer)

    class Generator:
        tokenizer = SimpleNamespace(all_special_ids=[99])
        _torch = SimpleNamespace(no_grad=nullcontext)
        prefill_calls = []

        @staticmethod
        def OutputTokenIds(token_ids):
            values = list(token_ids)
            return values[:-1] if values and values[-1] == 99 else values

        @staticmethod
        def Forward(*_args):
            raise AssertionError("already aligned native cache needs no completion")

        @staticmethod
        def Prefill(token_ids, pastKeyValues=None):
            Generator.prefill_calls.append(list(token_ids))
            cache = _OutputCache(0) if pastKeyValues is None else pastKeyValues
            count = len(token_ids)
            cache.update(
                torch.zeros((1, 1, count, 2)),
                torch.zeros((1, 1, count, 2)),
                0,
            )
            return SimpleNamespace(past_key_values=cache)

    method._gen = Generator()
    method._runIndex = 2
    method._ShiftCache = lambda cache, _offset: cache
    state = {"segments": []}
    monkeypatch.setattr("methods.KVComm._SliceCache", lambda cache, _start, _end: cache)

    method._RegisterOutput(
        state,
        [7, 99, 8],
        fullCache=_OutputCache(6),
        promptLength=3,
    )

    assert state["segments"][0]["base_ids"] == (7, 99, 8)
    assert state["segments"][0]["base_cache"] is not None
    assert Generator.prefill_calls == []

    from helpers.backends.Prompt import ComposeInterleavedReuse

    segment = state["segments"][0]
    matches = ComposeInterleavedReuse([list(segment["base_ids"])], [1, 7, 99, 8, 9])
    assert matches == [(None, [1]), (0, [7, 99, 8]), (None, [9])]
    parts = method._MakeParts(state["segments"], matches)
    method._GetPrefixTemplate = lambda _parts: None
    method._GetIsolatedBase = lambda _ids: pytest.fail(
        "retained visible output should reuse its rebuilt KV"
    )

    method._BuildBase(parts, [[1], [7, 99, 8], [9]])

    assert parts[1]["raw_cache"] is segment["base_cache"]


def test_kvcomm_equal_answers_keep_their_contextual_cache_identity():
    from helpers.backends.Prompt import ComposeInterleavedReuse
    method = object.__new__(KVCommTransformer)
    segments = [
        {"base_ids": (1,), "base_cache": object(), "placeholder_id": "user_question"},
        {"base_ids": (2,), "base_cache": object(), "placeholder_id": "agent_0_current"},
        {"base_ids": (2,), "base_cache": object(), "placeholder_id": "agent_1_current"},
    ]
    matches = ComposeInterleavedReuse([list(s["base_ids"]) for s in segments], [1, 2, 2])
    parts = method._MakeParts(segments, matches)
    assert [part["segment"] for part in parts] == segments
    assert parts[1]["segment"]["base_cache"] is not parts[2]["segment"]["base_cache"]
    # A single immutable prepared chunk remains reusable more than once.
    repeated = method._MakeParts(segments[:1], [(0, [1]), (0, [1])])
    assert all(part["segment"] is segments[0] for part in repeated)


def test_prepare_registers_ids_without_prefill_and_reset_preserves_learned_state():
    method = KVCommTransformer()
    method._gen = SimpleNamespace(Prefill=lambda *args: pytest.fail("Prepare performed online work"))
    chunks = [[1, 2], [], [3]]
    method.Prepare([chunks])
    chunks[0].append(4)
    assert [s["base_ids"] for s in method._states[0]["segments"]] == [(1, 2), (3,)]
    anchor, template, isolated = object(), object(), object()
    method._anchors["pool"] = [anchor]
    method._anchorFlags["pool"] = {"message": False}
    method._prefixTemplates[("template",)] = template
    method._isolatedBases[(1, 2)] = isolated
    method._runIndex = 3
    method.Reset()
    assert method._states == [] and method._runIndex == 0
    assert method._anchors["pool"] == [anchor]
    assert method._anchorFlags["pool"] == {"message": False}
    assert method._prefixTemplates[("template",)] is template
    assert method._isolatedBases[(1, 2)] is isolated
    method.Close()
    method.Close()
    assert method._gen is None
    assert method._anchors == method._anchorFlags == method._prefixTemplates == method._isolatedBases == {}


def test_opt_in_transient_release_clears_raw_bases_and_preserves_anchors():
    method = KVCommTransformer(releaseCompletedRequestTransient=True)
    anchor = object()
    method._anchors["pool"] = [anchor]
    method._isolatedBases[(1,)] = object()
    method._isolatedBaseOrder = [(1,)]
    method.Reset()
    assert method._isolatedBases == {} and method._isolatedBaseOrder == []
    assert method._anchors["pool"] == [anchor]


def test_native_generation_retains_exact_output_and_reuses_without_output_prefill(monkeypatch):
    _, sampler = runtime()
    method = KVCommTransformer()
    method._gen = sampler
    samples = iter([7, 2, 8, 1])
    sampler._SampleNextToken = lambda *args: next(samples)
    result = method.Run([[3, 4, 5]], retainOutput=[True], maxNewTokens=5)[0]
    assert result.output_ids == [7, 2, 8] and result.output == "7 8"
    assert result.performance["num_output_tokens"] == 4
    segment = method._states[0]["segments"][0]
    assert segment["base_ids"] == (7, 2, 8)
    assert segment["base_cache"].get_seq_length() == 3
    key, value = _StackCache(segment["base_cache"])
    saved = key.clone(), value.clone()
    # Templates may prefill their own static markers; retained output itself
    # must remain the native cache generated above.
    monkeypatch.setattr(method, "_GetPrefixTemplate", lambda parts: None)
    monkeypatch.setattr(method, "_GetIsolatedBase", lambda ids: pytest.fail("output was re-prefilled"))
    sampler._SampleNextToken = lambda *args: 9
    method.Run([[10, 7, 2, 8, 11]], maxNewTokens=1)
    new_key, new_value = _StackCache(segment["base_cache"])
    torch.testing.assert_close(new_key, saved[0], rtol=0, atol=0)
    torch.testing.assert_close(new_value, saved[1], rtol=0, atol=0)


def test_anchor_correction_preserves_layer_zero_and_does_not_mutate_sources():
    method = KVCommTransformer()
    key = torch.ones(2, 1, 1, 2, 4)
    value = torch.full_like(key, 2)
    base = _NewCache(list(zip(key, value, strict=True)))
    context = {"ph_key_delta": torch.full_like(key, 3), "ph_value_delta": torch.full_like(value, 4)}
    anchor = {"ph_key_embedding": key, "ph_value_embedding": value,
              "contexts": {"run=0": context}}
    predicted, prefix = method._OffsetKvCachePair(base, None, [anchor], [0], consumerKey="run=0")
    corrected_key, corrected_value = _StackCache(predicted)
    assert prefix is None
    torch.testing.assert_close(corrected_key[0], key[0], rtol=0, atol=0)
    torch.testing.assert_close(corrected_value[0], value[0], rtol=0, atol=0)
    torch.testing.assert_close(corrected_key[1], key[1] + 3)
    torch.testing.assert_close(corrected_value[1], value[1] + 4)
    actual_key, actual_value = _StackCache(base)
    torch.testing.assert_close(actual_key, key, rtol=0, atol=0)
    torch.testing.assert_close(actual_value, value, rtol=0, atol=0)
