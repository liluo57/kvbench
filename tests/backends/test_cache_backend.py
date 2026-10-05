"""Shared cache geometry, RoPE alignment, and selective forward correctness."""

import pytest
import torch
import transformers

from helpers.backends.CacheBackend.Cache import KeyValue, KVCache, concat_caches
from helpers.backends.CacheBackend.Recompute import CacheHit, assemble_cache
from methods import EPIC, FullRecomputeTransformer, RandRecomputeTransformer
from tests.support.models import runtime


@pytest.mark.parametrize("family", ["llama", "mistral", "qwen2", "qwen3"])
def test_all_selected_rows_match_native_full_forward_and_every_cache_layer(family, monkeypatch):
    from helpers.backends.CacheBackend import Recompute

    monkeypatch.setattr(Recompute, "_QUERY_CHUNK_SIZE", 2)
    backend, _ = runtime(family)
    first, second = [2, 3, 4], [5, 6, 7, 8]
    cache = backend.prefill_caches(input_ids=torch.tensor([first]))[0]
    inputs = torch.tensor([[*second, 9, *first, 10]])
    assembled = assemble_cache(backend, inputs, [CacheHit(5, 8, cache)])
    source_keys = [layer.key.clone() for layer in cache.layers]
    repaired = backend.recompute(input_ids=inputs, cache=assembled,
                                 token_indices=list(range(inputs.shape[1])))
    with torch.no_grad():
        expected = backend.forward(input_ids=inputs)
    torch.testing.assert_close(repaired.logits, expected.logits[:, -1:])
    for actual, full in zip(repaired.cache.layers, expected.cache.layers):
        torch.testing.assert_close(actual.key, full.key)
        torch.testing.assert_close(actual.value, full.value)
    for layer, saved in zip(cache.layers, source_keys):
        torch.testing.assert_close(layer.key, saved, rtol=0, atol=0)


def test_qwen3_uses_checkpoint_head_dimension_instead_of_hidden_size_division():
    backend, _ = runtime(head_dim=12)
    ids = torch.tensor([[2, 3, 4, 5]])
    prepared = backend.prefill_caches(input_ids=ids)[0]
    result = backend.recompute(input_ids=ids, cache=prepared, token_indices=[0, 1, 2, 3])
    with torch.no_grad():
        expected = backend.forward(input_ids=ids)
    torch.testing.assert_close(result.logits, expected.logits[:, -1:])


@pytest.mark.parametrize("cls", [EPIC, RandRecomputeTransformer, FullRecomputeTransformer])
def test_initialize_prepare_run_and_close_reach_the_native_runtime(cls, monkeypatch):
    backend, sampler = runtime()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained",
                        lambda *args, **kwargs: backend.model)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        lambda *args, **kwargs: sampler.tokenizer)
    method = cls(dtype="float32")
    method.Initialize([0])
    method.Prepare([[[2, 3, 4], [5, 6]]])
    actual = method.Run([[5, 6, 2, 3, 4, 7]], maxNewTokens=2)[0]
    assert actual.metadata["n_input"] == 6
    assert actual.performance["num_output_tokens"] <= 2
    if cls is FullRecomputeTransformer:
        assert actual.metadata["full_prefill"]
        assert method._states == [[]]
    method.Close()
    assert method._backend is None and method._gen is None and method._states == []


class _Rotary:
    attention_scaling = 1.7

    def __call__(self, x, position_ids):
        angles = position_ids.to(x.dtype).unsqueeze(-1) * x.new_tensor([0.2, 0.35])
        angles = torch.cat([angles, angles], dim=-1)
        return angles.cos() * self.attention_scaling, angles.sin() * self.attention_scaling


def _rotated(raw, positions):
    # Complex multiplication is an independent reference for split-half RoPE.
    phase = positions.unsqueeze(1).unsqueeze(-1) * raw.new_tensor([0.2, 0.35])
    values = torch.complex(raw[..., :2], raw[..., 2:]) * torch.exp(1j * phase)
    return torch.cat([values.real, values.imag], dim=-1) * _Rotary.attention_scaling


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("delta", [-3, 0, 5])
def test_uniform_shift_preserves_physical_and_logical_spans_and_partial_rope(compressed, delta):
    logical = torch.tensor([[4, 5, 6, 7]])
    physical = logical[:, [0, 2]] if compressed else logical
    raw = torch.randn(1, 2, physical.shape[1], 4, dtype=torch.float64)
    unrotated = torch.randn(1, 2, physical.shape[1], 2, dtype=torch.float64)
    keys = torch.cat([unrotated, _rotated(raw, physical)], dim=-1)
    values = torch.randn_like(keys)
    source = KVCache([KeyValue(keys, values, logical)])
    shifted = source.shift(_Rotary(), delta, nope_dim=2)
    expected = torch.cat([unrotated, _rotated(raw, physical + delta)], dim=-1)
    torch.testing.assert_close(shifted[0].key, expected)
    torch.testing.assert_close(shifted[0].value, values, rtol=0, atol=0)
    assert shifted.seq_len == physical.shape[1] and shifted.doc_len == 4
    assert shifted[0].logical_positions.tolist() == (logical + delta).tolist()
    restored = shifted.shift(_Rotary(), -delta, nope_dim=2)
    torch.testing.assert_close(restored[0].key, keys)
    torch.testing.assert_close(source[0].key, keys, rtol=0, atol=0)
    if compressed:
        with pytest.raises(ValueError, match="compressed caches"):
            source.rerotate(_Rotary(), physical, physical + delta, nope_dim=2)


def test_clone_and_concatenation_own_tensors_and_keep_logical_positions():
    first = KVCache([KeyValue(torch.ones(1, 1, 2, 4), torch.ones(1, 1, 2, 4), torch.tensor([[0, 1, 2]]))])
    second = KVCache([KeyValue(torch.full((1, 1, 1, 4), 2.0), torch.ones(1, 1, 1, 4), torch.tensor([[7, 8]]))])
    combined = concat_caches([first, second])
    assert combined.seq_len == 3 and combined.doc_len == 5
    assert combined[0].logical_positions.tolist() == [[0, 1, 2, 7, 8]]
    cloned = combined.clone()
    cloned[0].key.zero_()
    cloned[0].value.zero_()
    cloned[0].logical_positions.zero_()
    assert combined[0].key.flatten().tolist() == [1.0] * 8 + [2.0] * 4
    assert combined[0].logical_positions.tolist() == [[0, 1, 2, 7, 8]]
    combined[0].key.zero_()
    assert torch.all(first[0].key == 1) and torch.all(second[0].key == 2)


@pytest.mark.parametrize("indices", [[], [0, 0, 2], [2, 1], [0, 1], [-1, 2], [0, 3]])
def test_sparse_recompute_rejects_invalid_positions(indices):
    backend, _ = runtime()
    inputs = torch.tensor([[2, 3, 4]])
    cache = backend.prefill_caches(input_ids=inputs)[0]
    with pytest.raises(ValueError, match="positions|final prompt token"):
        backend.recompute(input_ids=inputs, cache=cache, token_indices=indices)


@pytest.mark.parametrize("spans", [[], [(1, 3), (2, 4)], [(2, 4), (0, 2)], [(-1, 1)], [(3, 5)]])
def test_cache_assembly_rejects_empty_overlapping_unordered_or_out_of_bounds_hits(spans):
    backend, _ = runtime()
    cache = backend.prefill_caches(input_ids=torch.tensor([[2, 3]]))[0]
    hits = [CacheHit(start, end, cache) for start, end in spans]
    with pytest.raises(ValueError, match="at least one|ordered, disjoint"):
        assemble_cache(backend, torch.tensor([[2, 3, 4, 5]]), hits)
