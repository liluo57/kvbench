"""Numerical and workflow regressions for the unified native cache backend."""

import importlib.abc
import random
import sys

import pytest
import torch
import transformers

from helpers.backends.CacheBackend.HuggingFace import HFBackend
from helpers.backends.CacheBackend.Recompute import CacheHit, assemble_cache
from helpers.backends.TransformersHelper import TransformersGenerator
from methods import EPIC, FullRecomputeTransformer, RandRecomputeTransformer


class Tokenizer:
    eos_token_id = 1

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i) for i in ids if not skip_special_tokens or i != 1)

    def __call__(self, *args, **kwargs):
        raise AssertionError("native token IDs must not be retokenized")


def runtime(family="qwen3", *, head_dim=8):
    torch.manual_seed(7)
    prefix = {"llama": "Llama", "mistral": "Mistral", "qwen2": "Qwen2", "qwen3": "Qwen3"}[family]
    config = getattr(transformers, prefix + "Config")(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=head_dim, max_position_embeddings=256, sliding_window=None,
    )
    config._attn_implementation = "sdpa"
    model = getattr(transformers, prefix + "ForCausalLM")(config).eval()
    backend = HFBackend(model)
    sampler = TransformersGenerator.__new__(TransformersGenerator)
    sampler._torch = torch
    sampler.model = model
    sampler.tokenizer = Tokenizer()
    sampler.device = "cpu"
    sampler.samplingConfig = {"mode": "greedy"}
    sampler.eosIds = {1}
    return backend, sampler


def bind(method, backend, sampler):
    method._backend, method._gen = backend, sampler
    return method


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


@pytest.mark.parametrize("cls,kwargs", [(EPIC, {"recomputeTokens": 100}),
                                        (RandRecomputeTransformer, {"recomputeRatio": 1, "seed": 3})])
def test_native_token_workflow_reordering_repetition_and_interleaved_fresh_rows(cls, kwargs):
    backend, sampler = runtime()
    first, second = [2, 3, 4, 5], [6, 7, 8]
    method = bind(cls(**kwargs), backend, sampler)
    method.Prepare([[first, second]])
    saved = [[(layer.key.clone(), layer.value.clone()) for layer in item.cache.layers]
             for item in method._states[0]]
    baseline = bind(FullRecomputeTransformer(), backend, sampler)
    for prompt in [first + second + [9], [10] + second + [11] + first + second + [12],
                   second + first, [21, 22, 23]]:
        actual = method.Run([prompt], maxNewTokens=3)[0]
        expected = baseline.Run([prompt], maxNewTokens=3)[0]
        assert actual.output_ids == expected.output_ids
        assert actual.output == expected.output
        assert actual.metadata["n_input"] == len(prompt)
        assert actual.performance["num_output_tokens"] <= 3
        assert 0 < actual.performance["ttft"] <= actual.performance["total_time"]
    for chunk, layers in zip(method._states[0], saved):
        for layer, (key, value) in zip(chunk.cache.layers, layers):
            torch.testing.assert_close(layer.key, key, rtol=0, atol=0)
            torch.testing.assert_close(layer.value, value, rtol=0, atol=0)


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


def test_retained_generated_ids_and_cache_are_reusable_in_a_later_turn():
    backend, sampler = runtime()
    method = bind(EPIC(recomputeTokens=100), backend, sampler)
    prompt = [2, 3, 4, 5, 6]
    method.Prepare([[[2, 3], [4, 5]]])
    first = method.Run([prompt], retainOutput=[True], maxNewTokens=2)[0]
    retained = method._states[0][-1]
    assert retained.ids == prompt + first.output_ids
    assert retained.cache.seq_len == len(retained.ids)
    later = retained.ids + [9, 10]
    result = method.Run([later], maxNewTokens=2)[0]
    expected = bind(FullRecomputeTransformer(), backend, sampler).Run([later], maxNewTokens=2)[0]
    assert result.output_ids == expected.output_ids
    assert result.metadata["cached_tokens"] == len(retained.ids)


def test_terminal_eos_is_excluded_from_retained_ids_and_kv():
    backend, sampler = runtime()
    sampler._SampleNextToken = lambda *args: 1
    method = bind(EPIC(), backend, sampler)
    result = method.Run([[2, 3, 4]], retainOutput=[True], maxNewTokens=5)[0]
    assert result.output_ids == []
    assert result.performance["num_output_tokens"] == 1
    assert method._states[0][-1].cache.seq_len == 3


def test_random_policy_is_reproducible_without_changing_global_rng():
    state = random.getstate()
    method = RandRecomputeTransformer(recomputeRatio=0.5, seed=42)
    hits = [CacheHit(0, 4, None), CacheHit(7, 15, None), CacheHit(20, 24, None)]
    first = method._select(hits)
    assert len(first) == 6 and all(7 <= i < 15 or 20 <= i < 24 for i in first)
    assert first == method._select(hits)
    assert random.getstate() == state


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


def test_packet_api_and_methods_use_one_shared_backend_class():
    from methods.KVPacket.backend.kvpacket import HFBackend as PacketHFBackend

    assert PacketHFBackend is HFBackend


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


@pytest.mark.parametrize("cls,kwargs", [(EPIC, {"recomputeTokens": -1}),
                                        (EPIC, {"recomputeTokens": True}),
                                        (RandRecomputeTransformer, {"recomputeRatio": float("nan")}),
                                        (RandRecomputeTransformer, {"recomputeRatio": 1.1})])
def test_invalid_recompute_budgets_fail_before_model_loading(cls, kwargs):
    with pytest.raises(ValueError):
        cls(**kwargs)
