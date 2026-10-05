"""CacheClip retains generated primary KV rows as independently matchable chunks."""

from types import SimpleNamespace

import torch
from transformers.cache_utils import DynamicCache

from methods.CacheClip import CacheClip
from methods.CacheClip.Cache import _cache_layer_pairs, select_output_cache
from methods.CacheClip.Selector import AuxiliaryDocumentCache
from test_cacheclip import _FakeCache, _fake_method, _patch_fake_assembly
import importlib


module = importlib.import_module("methods.CacheClip.Method")


class _Rotary:
    def __call__(self, x, position_ids=None):
        # Identity RoPE keeps the cache easy to inspect while exercising the
        # same position-normalization path used for prepared chunks.
        shape = (x.shape[0], x.shape[-2], x.shape[-1])
        return x.new_ones(shape), x.new_zeros(shape)


def _model():
    return SimpleNamespace(
        config=SimpleNamespace(),
        model=SimpleNamespace(rotary_emb=_Rotary()),
        device=torch.device("cpu"),
    )


def _dynamic_cache(ids):
    cache = DynamicCache()
    key = torch.tensor(ids, dtype=torch.float32).view(1, 1, len(ids), 1).expand(1, 1, len(ids), 2).clone()
    cache.update(key, key + 100, 0)
    return cache


def test_output_cache_is_owned_and_normalized_to_prepared_source_positions():
    model = _model()
    original = _dynamic_cache([1, 2, 3, 4, 5])
    retained = select_output_cache(model, original, 3, 2, old_start=7)

    source_key = _cache_layer_pairs(original)[0][0]
    retained_key = _cache_layer_pairs(retained)[0][0]
    assert retained.get_seq_length() == 2
    assert retained_key.data_ptr() != source_key.data_ptr()
    assert retained_key.flatten().tolist() == source_key[:, :, 3:5].flatten().tolist()
    assert original.get_seq_length() == 5


def test_output_cache_relocates_native_rope_keys_to_shared_prefix_positions():
    class PositionRotary:
        def __call__(self, x, position_ids=None):
            angles = position_ids.to(x.dtype).unsqueeze(-1).expand(-1, -1, 2)
            return angles.cos(), angles.sin()

    model = _model()
    model.model.rotary_emb = PositionRotary()
    absolute_positions = torch.arange(5, dtype=torch.float32)
    # Native keys for raw [1, 0] at absolute prompt/decode positions.
    native_keys = torch.stack(
        (absolute_positions.cos(), absolute_positions.sin()), dim=-1
    ).view(1, 1, 5, 2)
    values = torch.arange(10, dtype=torch.float32).view(1, 1, 5, 2)
    cache = DynamicCache()
    cache.update(native_keys.clone(), values.clone(), 0)

    retained = select_output_cache(model, cache, 3, 2, old_start=7)
    key, value = _cache_layer_pairs(retained)[0]
    expected_positions = torch.tensor([7.0, 8.0])
    expected_keys = torch.stack(
        (expected_positions.cos(), expected_positions.sin()), dim=-1
    ).view(1, 1, 2, 2)
    torch.testing.assert_close(key, expected_keys)
    torch.testing.assert_close(value, values[:, :, 3:5], rtol=0, atol=0)
    torch.testing.assert_close(_cache_layer_pairs(cache)[0][0], native_keys, rtol=0, atol=0)


def test_fallback_output_is_retained_and_later_matched_without_primary_prefill(monkeypatch):
    method = _fake_method()
    method.sharedPrefixText = None
    method._gen.model = _model()
    method._gen.OutputTokenIds = lambda ids: list(ids[:-1]) if ids and ids[-1] == 0 else list(ids)
    method._auxiliary.prefill_documents = lambda prefix, docs: [
        AuxiliaryDocumentCache(_FakeCache([81]), [(0, 1)], 0)
    ]
    monkeypatch.setattr(module, "token_text_and_offsets", lambda tokenizer, ids: ("x", [(0, 1)] * len(ids)))

    def generate(ids, pastKeyValues=None, **kwargs):
        if kwargs.get("returnCache"):
            return "out", 0.1, 0.2, 3, _dynamic_cache([*ids, 81, 82, 0]), [81, 82, 0]
        return "out", 0.1, 0.2, 1, [90]

    method._gen.Generate = generate
    first = method.Run([[4, 5]], retainOutput=[True])[0]
    assert first.output_ids == [81, 82]
    assert method._states[0]["document_ids"] == [[81, 82]]
    assert method._states[0]["primary"][0].get_seq_length() == 2
    assert _cache_layer_pairs(method._states[0]["primary"][0])[0][0].flatten().tolist() == [81.0, 81.0, 82.0, 82.0]

    _patch_fake_assembly(monkeypatch)
    method._select = lambda *args: []
    later = method.Run([[99, 81, 82, 100]], retainOutput=[False])[0]
    assert later.output_ids == [90]
    assert all(ids not in ([81], [81, 82]) for ids, _past in method._gen.prefill_calls)
    assert method._states[0]["document_ids"] == [[81, 82]]


def test_empty_output_fragment_and_reset_do_not_leave_retained_chunks(monkeypatch):
    method = _fake_method()
    method.sharedPrefixText = None
    method._gen.model = _model()
    method._gen.OutputTokenIds = lambda ids: []
    method._gen.Generate = lambda ids, **kwargs: (
        "", 0.1, 0.2, 1, _dynamic_cache([*ids, 0]), [0]
    )
    monkeypatch.setattr(module, "token_text_and_offsets", lambda tokenizer, ids: ("", []))
    result = method.Run([[4]], retainOutput=[True])[0]
    assert result.output_ids == []
    assert method._states[0]["document_ids"] == []
    method.Reset()
    assert method._states == []


def test_retained_output_can_participate_in_an_interleaved_reuse(monkeypatch):
    method = _fake_method()
    method.sharedPrefixText = None
    method._gen.model = _model()
    method._gen.OutputTokenIds = lambda ids: list(ids[:-1]) if ids and ids[-1] == 0 else list(ids)
    method._auxiliary.prefill_documents = lambda prefix, docs: [
        AuxiliaryDocumentCache(_FakeCache([81, 82]), [(0, 1), (1, 2)], 0)
    ]
    monkeypatch.setattr(module, "token_text_and_offsets", lambda tokenizer, ids: ("xy", [(0, 1)] * len(ids)))
    method._gen.Generate = lambda ids, pastKeyValues=None, **kwargs: (
        ("out", 0.1, 0.2, 3, _dynamic_cache([*ids, 81, 82, 0]), [81, 82, 0])
        if kwargs.get("returnCache") else ("next", 0.1, 0.2, 1, [90])
    )

    method.Run([[4, 5]], retainOutput=[True])
    state = method._states[0]
    state["document_ids"].insert(0, [7, 8])
    state["primary_id_tensors"].insert(0, torch.tensor([[7, 8]]))
    state["primary"].insert(0, _dynamic_cache([7, 8]))
    state["primary_offsets"].insert(0, [(0, 1), (1, 2)])
    state["auxiliary"].insert(0, AuxiliaryDocumentCache(_FakeCache([7, 8]), [(0, 1), (1, 2)], 0))

    _patch_fake_assembly(monkeypatch)
    method._select_occurrences = lambda query, occurrences: []
    result = method.Run([[81, 82, 40, 7, 8, 41, 99]], retainOutput=[False])[0]

    assert result.output_ids == [90]
    assert result.metadata["n_cached_tokens"] == 4
    assert method._states[0]["document_ids"] == [[7, 8], [81, 82]]
