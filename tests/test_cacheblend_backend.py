from types import SimpleNamespace

import torch
import pytest

from methods.Cacheblend.backend.cacheblend_ops import (
    RetainedOutputCollector,
    capture_kv,
    indexed_causal_mask,
    register_retained_output,
)
from helpers.backends.Prompt import ComposeInterleavedReuse
from helpers.backends.Tokenization import OutputFragmentIds
from methods.Cacheblend.Worker import CacheBlendWorker


def test_native_capture_retains_decode_tokens_without_collecting_prefill():
    attention = SimpleNamespace(hack_kv=None)
    collector = RetainedOutputCollector([SimpleNamespace(self_attn=attention)])
    metadata = {"collect": False, "retain_collector": collector}
    key = torch.tensor([[1.0, 2.0]])
    value = torch.tensor([[3.0, 4.0]])

    capture_kv(attention, key, value, 0, metadata)
    assert attention.hack_kv is None
    assert collector.finish() is None

    capture_kv(attention, key, value, -1, metadata)
    key.add_(10)
    value.add_(10)
    capture_kv(attention, key, value, -1, metadata)

    retained = collector.finish()
    assert metadata["collect"] is False
    assert torch.equal(retained[0][0], torch.tensor([[1.0, 2.0], [11.0, 12.0]]))
    assert torch.equal(retained[0][1], torch.tensor([[3.0, 4.0], [13.0, 14.0]]))


def test_native_collect_owns_kv_and_does_not_capture_ordinary_decode():
    attention = SimpleNamespace(hack_kv=None)
    metadata = {"collect": True}
    key = torch.tensor([[1.0]])
    value = torch.tensor([[2.0]])
    capture_kv(attention, key, value, 0, metadata)
    key.zero_()
    value.zero_()
    assert attention.hack_kv[0].item() == 1
    assert attention.hack_kv[1].item() == 2

    metadata["collect"] = False
    capture_kv(attention, key, value, -1, metadata)
    assert attention.hack_kv[0].item() == 1


def test_retained_fragment_preserves_interior_specials_and_hits_exact_cache_key():
    output_ids = OutputFragmentIds([31, 99, 32, 2], terminalStopIds=[2])
    attention = SimpleNamespace(hack_kv=None)
    collector = RetainedOutputCollector([SimpleNamespace(self_attn=attention)])
    metadata = {"collect": False, "retain_collector": collector}
    for value in (1.0, 2.0, 3.0):
        capture_kv(
            attention,
            torch.tensor([[value]]),
            torch.tensor([[value + 10]]),
            -1,
            metadata,
        )
    retained = collector.finish()
    chunk_ids = {}
    chunk_kv = {}

    retained_tokens = register_retained_output(
        chunk_ids, chunk_kv, output_ids, retained
    )

    assert retained_tokens == len(output_ids) == 3
    assert chunk_ids[tuple(output_ids)] == [31, 99, 32]
    assert chunk_kv[tuple(output_ids)][0][0].flatten().tolist() == [1.0, 2.0, 3.0]
    parts = ComposeInterleavedReuse([output_ids], [7, *output_ids, 8])
    matched = [piece for index, piece in parts if index is not None]
    assert matched == [output_ids]
    assert tuple(matched[0]) in chunk_kv
    worker = object.__new__(CacheBlendWorker)
    worker._chunkIds = chunk_ids
    worker._chunkKv = chunk_kv
    segments, full_ids, sample_kv = worker._ResolveSegments(
        [[True, output_ids], [False, [77]]]
    )
    assert full_ids == [31, 99, 32, 77]
    assert [segment[0] for segment in segments] == [True, False]
    assert sample_kv is chunk_kv[tuple(output_ids)]


def test_retained_fragment_rejects_incomplete_kv_capture_without_truncating_ids():
    chunk_ids = {}
    chunk_kv = {}
    incomplete = [[torch.ones((2, 1)), torch.ones((2, 1))]]

    with pytest.raises(RuntimeError, match="fewer KV rows than native output IDs"):
        register_retained_output(chunk_ids, chunk_kv, [31, 32, 33], incomplete)

    assert chunk_ids == {}
    assert chunk_kv == {}


def test_sparse_mask_uses_absolute_positions_for_interleaved_repair():
    metadata = {
        "imp_indices": torch.tensor([1, 5, 12]),
        "org_seq_len": 13,
        "kv_cache_dtype": torch.float32,
        "_num_kv_heads": 2,
        "_num_queries_per_kv": 4,
    }
    mask = indexed_causal_mask(metadata, lambda: None)
    assert mask.shape == (1, 2, 4, 3, 13)
    assert mask.stride(-2) == 16
    for row, position in enumerate([1, 5, 12]):
        assert torch.all(mask[..., row, :position + 1] == 0)
        assert torch.all(mask[..., row, position + 1:] == torch.finfo(torch.float32).min)
