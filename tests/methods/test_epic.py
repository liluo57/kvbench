"""Tests for epic."""

import importlib.abc
import sys

import pytest
import torch

from helpers.backends.CacheBackend.Recompute import CacheHit, assemble_cache
from methods import EPIC
from tests.support.models import (
    assert_reordered_workflow_matches_full_recompute,
    bind,
    runtime,
)


def test_epic_retains_only_output_ids_with_local_positions_and_no_output_prefill(monkeypatch):
    backend, sampler = runtime()
    method = bind(EPIC(recomputeTokens=8), backend, sampler)
    method.Prepare([[[2, 3]]])
    monkeypatch.setattr(backend, "prefill_caches", lambda **kwargs: pytest.fail("output was re-prefilled"))
    samples = iter([14, 15])
    sampler._SampleNextToken = lambda *args: next(samples)
    prompt = [2, 3, 4, 5]
    result = method.Run([prompt], retainOutput=[True], maxNewTokens=2)[0]
    retained = method._states[0][-1]

    assert retained.ids == result.output_ids
    assert retained.ids
    assert retained.cache.seq_len == retained.cache.doc_len == len(retained.ids)
    assert retained.cache.layers[0].logical_positions.tolist() == [list(range(len(retained.ids)))]
    assert all(layer.key.shape[2] == len(result.output_ids) for layer in retained.cache.layers)

    # Matching the output IDs later must reuse the standalone fragment.
    later = [9, *retained.ids, 10]
    sampler._SampleNextToken = lambda *args: 16
    reused = method.Run([later], maxNewTokens=1)[0]
    assert reused.metadata["cached_tokens"] == len(retained.ids)


def test_epic_keeps_native_interior_special_ids_and_drops_terminal_eos():
    backend, sampler = runtime()
    samples = iter([2, 1])
    sampler._SampleNextToken = lambda *args: next(samples)
    method = bind(EPIC(), backend, sampler)
    result = method.Run([[7, 8, 9]], retainOutput=[True], maxNewTokens=4)[0]

    assert result.output_ids == [2]
    assert 2 in sampler.tokenizer.all_special_ids
    assert result.output == ""
    assert result.performance["num_output_tokens"] == 2
    assert method._states[0][-1].ids == [2]
    assert method._states[0][-1].cache.seq_len == 1


def test_sparse_repair_preserves_unselected_rows_and_full_cache_length():
    backend, sampler = runtime()
    method = bind(EPIC(recomputeTokens=1), backend, sampler)
    method.Prepare([[[2, 3, 4], [5, 6, 7]]])
    ids = torch.tensor([[2, 3, 4, 5, 6, 7, 8]])
    hits = [CacheHit(0, 3, method._states[0][0].cache), CacheHit(3, 6, method._states[0][1].cache)]
    assembled = assemble_cache(backend, ids, hits)
    original = assembled.clone()
    result = backend.recompute(input_ids=ids, cache=assembled, token_indices=[3, 6])
    assert result.cache.seq_len == len(ids[0])
    for before, after in zip(original.layers, result.cache.layers):
        torch.testing.assert_close(before.key[:, :, [0, 1, 2, 4, 5]], after.key[:, :, [0, 1, 2, 4, 5]], rtol=0, atol=0)
        torch.testing.assert_close(before.value[:, :, [0, 1, 2, 4, 5]], after.value[:, :, [0, 1, 2, 4, 5]], rtol=0, atol=0)
    assert method.Run([ids[0].tolist()], maxNewTokens=1)[0].metadata["selected_positions"] == [3, 6]


def test_terminal_eos_is_excluded_from_retained_ids_and_kv():
    backend, sampler = runtime()
    sampler._SampleNextToken = lambda *args: 1
    method = bind(EPIC(), backend, sampler)
    result = method.Run([[2, 3, 4]], retainOutput=[True], maxNewTokens=5)[0]
    assert result.output_ids == []
    assert result.performance["num_output_tokens"] == 1
    assert method._states == [[]]


@pytest.mark.parametrize("repair_tokens", [0, 100])
def test_reuse_ratio_counts_hits_independently_of_repair_and_terminal_replay(repair_tokens):
    backend, sampler = runtime()
    method = bind(EPIC(recomputeTokens=repair_tokens), backend, sampler)
    method.Prepare([[[2, 3, 4], [5, 6]]])
    for prompt, cached in [([2, 3, 4, 5, 6, 7], 5),
                           ([2, 3, 4], 3),
                           ([5, 6, 2, 3, 4, 5, 6, 7], 7),
                           ([9, 10], 0)]:
        result = method.Run([prompt], maxNewTokens=1)[0]
        assert result.metadata["cached_tokens"] == cached
        assert result.metadata["reuse_ratio"] == pytest.approx(cached / len(prompt))


def test_epic_can_prepare_and_run_with_kvpacket_imports_blocked(monkeypatch):
    class BlockPacket(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.startswith("kv_packet") or fullname.startswith("methods.KVPacket"):
                raise AssertionError(f"EPIC imported a KVPacket implementation: {fullname}")

    for name in list(sys.modules):
        if name.startswith("methods.KVPacket"):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(sys, "meta_path", [BlockPacket(), *sys.meta_path])
    backend, sampler = runtime()
    method = bind(EPIC(), backend, sampler)
    method.Prepare([[[2, 3, 4], [5, 6]]])
    result = method.Run([[5, 6, 2, 3, 4, 7]], maxNewTokens=2)[0]
    assert result.metadata["cached_tokens"] == 5


def test_reordering_repetition_and_fresh_spans_match_full_recompute():
    assert_reordered_workflow_matches_full_recompute(EPIC, {"recomputeTokens": 100})


@pytest.mark.parametrize("budget", [-1, True, 1.5])
def test_invalid_recompute_budget_is_rejected(budget):
    with pytest.raises(ValueError, match="recomputeTokens"):
        EPIC(recomputeTokens=budget)


@pytest.mark.parametrize("budget, expected", [(0, []), (2, [5, 6, 10]), (100, [5, 6, 7, 10])])
def test_selector_repairs_relocated_chunk_prefixes(budget, expected):
    hits = [CacheHit(0, 3, None), CacheHit(5, 8, None), CacheHit(10, 11, None)]
    assert EPIC(recomputeTokens=budget)._select(hits) == expected


def test_prepare_owns_ids_deduplicates_chunks_and_reset_discards_retained_output():
    backend, sampler = runtime()
    method = bind(EPIC(), backend, sampler)
    chunks = [[2, 3], [], [2, 3], [4, 5]]
    method.Prepare([chunks])
    chunks[0].append(6)
    assert [chunk.ids for chunk in method._states[0]] == [[2, 3], [4, 5]]
    sampler._SampleNextToken = lambda *args: 7
    method.Run([[2, 3, 8]], retainOutput=[True], maxNewTokens=1)
    assert len(method._states[0]) == 3
    method.Reset()
    assert method._states == []
    result = method.Run([[2, 3, 8]], maxNewTokens=1)[0]
    assert result.metadata["cached_tokens"] == 0


def test_batch_retention_is_case_local_and_generation_respects_remaining_window(context_limit):
    backend, sampler = runtime()
    sampler._SampleNextToken = lambda *args: 7
    method = bind(EPIC(), backend, sampler)
    results = method.Run([[2, 3, 4], [2, 3, 4, 5]], retainOutput=[True], maxNewTokens=8)
    assert [r.performance["num_output_tokens"] for r in results] == [2, 1]
    assert method._states[0][-1].ids == [7, 7]
    assert method._states[1] == []
    with pytest.raises(ValueError, match="MaxModelLen=5"):
        method.Run([[2] * 5])
