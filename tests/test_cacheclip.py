import importlib
import random
import threading
from types import SimpleNamespace

import pytest
import torch

from methods.CacheClip import CacheClip
from methods.CacheClipBackend import (
    AuxiliaryDocumentCache,
    AuxiliaryRuntime,
    _bucket_indices,
    _flashinfer_attention_forward,
    _flashinfer_single_attention_forward,
    _flashinfer_causal_bsr_metadata,
    _local_eager_attention_forward,
    _cache_layer_pairs,
    _new_dynamic_cache,
    _resolve_attention_backend,
    assemble_primary_cache,
    append_cached_primary_segment,
    PrimaryAssembly,
    recompute_selected_tokens,
    group_candidate_windows,
    map_auxiliary_to_primary_indices,
    select_cache_sequence,
    select_top_k_candidates,
    shared_prefix_token_length,
)

_cacheclip_module = importlib.import_module("methods.CacheClip")
_cacheclip_backend_module = importlib.import_module("methods.CacheClipBackend")


def _flashinfer_importable():
    try:
        from flashinfer.sparse import BlockSparseAttentionWrapper  # noqa: F401
    except Exception:
        return False
    return True


class _FakeGenerator:
    def __init__(self):
        self.model = SimpleNamespace(device=torch.device("cpu"))
        self.generate_calls = []

    @staticmethod
    def Encode(text, addSpecialTokens=True):
        return [1] if text else []

    def Generate(self, input_ids, pastKeyValues=None, **kwargs):
        self.generate_calls.append(kwargs)
        return "generated", 0.01, 0.02, 1


def _fake_method():
    method = object.__new__(CacheClip)
    method._gen = _FakeGenerator()
    method._auxiliary = object()
    method._states = []
    return method


def _prepared_state():
    return {
        "chunks": ["document"],
        "primary": [object()],
        "primary_ids": [torch.tensor([[7, 8]], dtype=torch.long)],
        "primary_offsets": [[]],
        "auxiliary": [object()],
        "prefix_length": 0,
    }


def _fake_assembly():
    return SimpleNamespace(
        cache=object(),
        context_ids=torch.tensor([[7, 8]], dtype=torch.long),
    )


def _patch_fake_repair(monkeypatch):
    monkeypatch.setattr(
        _cacheclip_module,
        "assemble_primary_cache",
        lambda *args: _fake_assembly(),
    )
    monkeypatch.setattr(
        _cacheclip_module,
        "recompute_selected_tokens",
        lambda *args: None,
    )


def test_cacheclip_prepare_uses_tokenizer_offset_helper_signature(monkeypatch):
    class FakeGenerator:
        tokenizer = object()
        model = SimpleNamespace(device=torch.device("cpu"))

        @staticmethod
        def Prefill(input_ids):
            return SimpleNamespace(past_key_values=object())

    class FakeAuxiliary:
        @staticmethod
        def prefill_documents(prefix, documents):
            return [object() for _ in documents]

    calls = []

    def tokenize_with_offsets(tokenizer, text):
        calls.append((tokenizer, text))
        return torch.tensor([[7, 8]], dtype=torch.long), [(0, 1), (1, 2)]

    method = object.__new__(CacheClip)
    method._gen = FakeGenerator()
    method._auxiliary = FakeAuxiliary()
    method._states = []
    monkeypatch.setattr(
        _cacheclip_module,
        "tokenize_with_offsets",
        tokenize_with_offsets,
    )

    method.Prepare([["document"]])

    assert calls == [(method._gen.tokenizer, "document")]
    assert method._states[0]["chunks"] == ["document"]


def test_cacheclip_buckets_legacy_tuple_auxiliary_caches():
    def tuple_cache(length):
        return tuple(
            (
                torch.zeros((1, 1, length, 2)),
                torch.zeros((1, 1, length, 2)),
            )
            for _ in range(2)
        )

    caches = [
        AuxiliaryDocumentCache(tuple_cache(3), [(0, 1)] * 3, 0),
        AuxiliaryDocumentCache(tuple_cache(3), [(0, 1)] * 3, 0),
    ]

    assert _bucket_indices(caches) == [[0, 1]]


def test_cacheclip_new_dynamic_cache_supports_transformers_446_constructor():
    cache = _new_dynamic_cache(SimpleNamespace())

    assert hasattr(cache, "get_seq_length")


def test_cacheclip_local_eager_attention_fallback_supports_gqa_and_mask():
    query = torch.zeros((1, 4, 1, 2))
    key = torch.zeros((1, 2, 3, 2))
    value = torch.ones((1, 2, 3, 2))
    mask = torch.zeros((1, 1, 1, 3))
    mask[:, :, :, 2] = torch.finfo(mask.dtype).min

    attended, weights = _local_eager_attention_forward(
        SimpleNamespace(training=False),
        query,
        key,
        value,
        mask,
        scaling=1.0,
        dropout=0.0,
    )

    assert attended.shape == query.shape
    assert weights.shape == (1, 4, 1, 3)
    assert torch.allclose(weights[..., 2], torch.zeros_like(weights[..., 2]))
    assert torch.allclose(weights.sum(dim=-1), torch.ones((1, 4, 1)))


def test_cacheclip_attention_backend_selection(monkeypatch):
    monkeypatch.delenv("CACHECLIP_ATTENTION_BACKEND", raising=False)
    assert _resolve_attention_backend() == "torch"

    monkeypatch.setenv("CACHECLIP_ATTENTION_BACKEND", "reference")
    assert _resolve_attention_backend() == "torch"
    monkeypatch.setenv("CACHECLIP_ATTENTION_BACKEND", "flashinfer")
    assert _resolve_attention_backend() == "flashinfer"
    monkeypatch.setenv("CACHECLIP_ATTENTION_BACKEND", "flashinfer_single")
    assert _resolve_attention_backend() == "flashinfer_single"

    monkeypatch.setenv("CACHECLIP_ATTENTION_BACKEND", "unknown")
    with pytest.raises(ValueError, match="CACHECLIP_ATTENTION_BACKEND"):
        _resolve_attention_backend()


def test_cacheclip_flashinfer_metadata_preserves_causal_prefixes():
    mask = torch.zeros((1, 1, 2, 5), dtype=torch.float32)
    mask[:, :, 0, 3:] = torch.finfo(mask.dtype).min
    mask[:, :, 1, 4:] = torch.finfo(mask.dtype).min

    indptr, indices, block_mask, padded_length = _flashinfer_causal_bsr_metadata(
        mask, block_size=2
    )

    assert indptr.tolist() == [0, 2, 4]
    assert indices.tolist() == [0, 1, 0, 1]
    assert block_mask.shape == (4, 1, 2)
    assert padded_length == 6
    assert block_mask[:, 0].tolist() == [[True, True], [True, False], [True, True], [True, True]]


def test_cacheclip_flashinfer_backend_fails_closed_without_flashinfer():
    if importlib.util.find_spec("flashinfer") is not None:
        pytest.skip("FlashInfer is installed; use the CUDA integration test")
    with pytest.raises(RuntimeError, match="BlockSparseAttentionWrapper is unavailable"):
        _flashinfer_attention_forward(
            SimpleNamespace(training=False),
            torch.zeros((1, 2, 1, 4)),
            torch.zeros((1, 1, 2, 4)),
            torch.zeros((1, 1, 2, 4)),
            torch.zeros((1, 1, 1, 2)),
            scaling=0.5,
        )


@pytest.mark.skipif(
    not _flashinfer_importable() or not torch.cuda.is_available(),
    reason="FlashInfer/CUDA is unavailable",
)
def test_cacheclip_flashinfer_matches_reference_small_random():
    device = torch.device("cuda")
    head_dim = 128
    torch.manual_seed(7)
    query = torch.randn((1, 8, 2, head_dim), device=device, dtype=torch.float16)
    key = torch.randn((1, 2, 16, head_dim), device=device, dtype=torch.float16)
    value = torch.randn((1, 2, 16, head_dim), device=device, dtype=torch.float16)
    mask = torch.zeros((1, 1, 2, 16), device=device, dtype=query.dtype)
    mask[:, :, 0, 4:] = torch.finfo(query.dtype).min
    mask[:, :, 1, 12:] = torch.finfo(query.dtype).min

    reference, _ = _local_eager_attention_forward(
        SimpleNamespace(training=False), query, key, value, mask, scaling=head_dim ** -0.5
    )
    actual, _ = _flashinfer_attention_forward(
        SimpleNamespace(training=False), query, key, value, mask, scaling=head_dim ** -0.5
    )
    # FlashInfer returns [query, heads, dim], while the local fallback used
    # here returns [batch, heads, query, dim].  The public wrapper normalizes
    # to the Transformers contract [batch, query, heads, dim].
    torch.testing.assert_close(
        actual, reference.transpose(1, 2).contiguous(), rtol=2e-2, atol=2e-2
    )


@pytest.mark.skipif(
    importlib.util.find_spec("flashinfer") is None or not torch.cuda.is_available(),
    reason="FlashInfer/CUDA is unavailable",
)
def test_cacheclip_single_flashinfer_matches_reference_small_random():
    device = torch.device("cuda")
    head_dim = 128
    torch.manual_seed(9)
    query = torch.randn((1, 8, 2, head_dim), device=device, dtype=torch.float16)
    key = torch.randn((1, 2, 16, head_dim), device=device, dtype=torch.float16)
    value = torch.randn((1, 2, 16, head_dim), device=device, dtype=torch.float16)
    mask = torch.zeros((1, 1, 2, 16), device=device, dtype=query.dtype)
    mask[:, :, 0, 4:] = torch.finfo(query.dtype).min
    mask[:, :, 1, 12:] = torch.finfo(query.dtype).min

    reference, _ = _local_eager_attention_forward(
        SimpleNamespace(training=False), query, key, value, mask, scaling=head_dim ** -0.5
    )
    actual, _ = _flashinfer_single_attention_forward(
        SimpleNamespace(training=False), query, key, value, mask, scaling=head_dim ** -0.5
    )
    torch.testing.assert_close(
        actual, reference.transpose(1, 2).contiguous(), rtol=2e-2, atol=2e-2
    )


def test_cacheclip_assembly_moves_context_ids_to_cache_device(monkeypatch):
    class FakeCache:
        def __init__(self):
            self.layers = []

        def update(self, key, value, layer_index):
            self.layers.append(SimpleNamespace(keys=key, values=value))

    monkeypatch.setattr(
        _cacheclip_backend_module,
        "_new_dynamic_cache",
        lambda config: FakeCache(),
    )
    monkeypatch.setattr(
        _cacheclip_backend_module,
        "_rotate_key",
        lambda key, *args: key,
    )
    key = torch.zeros((1, 1, 2, 2), device="meta")
    value = torch.zeros((1, 1, 2, 2), device="meta")
    model = SimpleNamespace(
        config=SimpleNamespace(),
        model=SimpleNamespace(rotary_emb=object()),
    )

    assembly = assemble_primary_cache(
        model,
        None,
        torch.empty((1, 0), dtype=torch.long),
        None,
        torch.empty((1, 0), dtype=torch.long),
        0,
        [((key, value),)],
        [torch.tensor([[7, 8]], dtype=torch.long)],
    )

    assert assembly.context_ids.device == key.device


def test_cacheclip_append_cached_primary_segment_realigns_and_appends(monkeypatch):
    class FakeCache:
        def __init__(self):
            self.layers = []

        def update(self, key, value, layer_index):
            self.layers.append(SimpleNamespace(keys=key, values=value))

    monkeypatch.setattr(
        _cacheclip_backend_module,
        "_new_dynamic_cache",
        lambda config: FakeCache(),
    )
    seen = []

    def rotate(key, rotary, old, new, nope_dim):
        seen.append((old.tolist(), new.tolist()))
        return key + 10

    monkeypatch.setattr(_cacheclip_backend_module, "_rotate_key", rotate)
    old_key = torch.zeros((1, 1, 2, 2))
    old_value = torch.ones((1, 1, 2, 2))
    new_key = torch.full((1, 1, 3, 2), 2.0)
    new_value = torch.full((1, 1, 3, 2), 3.0)
    current = FakeCache()
    current.update(old_key, old_value, 0)
    assembly = PrimaryAssembly(
        cache=current,
        context_ids=torch.tensor([[4, 5]], dtype=torch.long),
    )
    model = SimpleNamespace(
        config=SimpleNamespace(),
        model=SimpleNamespace(rotary_emb=object()),
    )

    document = FakeCache()
    document.update(new_key, new_value, 0)
    result = append_cached_primary_segment(
        model,
        assembly,
        document,
        torch.tensor([[6, 7, 8]], dtype=torch.long),
    )
    assert torch.equal(result.context_ids, torch.tensor([[4, 5, 6, 7, 8]]))
    assert torch.equal(result.cache.layers[0].keys, torch.cat([old_key, new_key + 10], dim=2))
    assert torch.equal(result.cache.layers[0].values, torch.cat([old_value, new_value], dim=2))
    assert seen[-1] == ([[0, 1, 2]], [[2, 3, 4]])


def test_cacheclip_dispatches_noncontiguous_matches_to_interleaved_runner(monkeypatch):
    method = _fake_method()
    method._run_interleaved = lambda state, prompt, parts, positions: "interleaved"
    method._run_contiguous = lambda state, prompt, parts, positions: "contiguous"
    monkeypatch.setattr(
        _cacheclip_module,
        "ComposeInterleavedReuse",
        lambda chunks, prompt: [(0, "A"), (None, "fresh"), (1, "C")],
    )

    assert method._run_one({"chunks": ["A", "C"]}, "A fresh C") == "interleaved"


def test_cacheclip_select_cache_sequence_supports_legacy_dynamic_cache(monkeypatch):
    class LegacyDynamicCache:
        def __init__(self, key, value):
            self.config = SimpleNamespace()
            self.key_cache = [key]
            self.value_cache = [value]

    class FakeDynamicCache:
        def __init__(self):
            self.key_cache = []
            self.value_cache = []

        def update(self, key, value, layer_index):
            self.key_cache.append(key)
            self.value_cache.append(value)

    key = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4)
    value = key + 100
    monkeypatch.setattr(
        _cacheclip_backend_module,
        "_new_dynamic_cache",
        lambda config: FakeDynamicCache(),
    )

    selected = select_cache_sequence(LegacyDynamicCache(key, value), 1, 3)

    assert torch.equal(selected.key_cache[0], key[:, :, 1:3, :])
    assert torch.equal(selected.value_cache[0], value[:, :, 1:3, :])


def test_cacheclip_recompute_passes_legacy_cache_tensors_to_attention(monkeypatch):
    class LlamaForCausalLM:
        pass

    class LegacyDynamicCache:
        def __init__(self, key, value):
            self.key_cache = [key]
            self.value_cache = [value]

    projection = torch.nn.Linear(4, 4, bias=False)
    projection.weight.data.copy_(torch.eye(4))
    attention = SimpleNamespace(
        head_dim=2,
        q_proj=projection,
        k_proj=projection,
        v_proj=projection,
        o_proj=projection,
        scaling=1.0,
        sliding_window=None,
    )
    decoder_layer = SimpleNamespace(
        input_layernorm=torch.nn.Identity(),
        self_attn=attention,
        post_attention_layernorm=torch.nn.Identity(),
        mlp=torch.nn.Identity(),
    )
    model = LlamaForCausalLM()
    model.config = SimpleNamespace()
    model.model = SimpleNamespace(
        embed_tokens=torch.nn.Embedding(10, 4),
        rotary_emb=lambda hidden, position_ids: (
            torch.ones((1, position_ids.size(1), 2)),
            torch.zeros((1, position_ids.size(1), 2)),
        ),
        layers=[decoder_layer],
    )
    key = torch.zeros((1, 2, 3, 2))
    value = torch.zeros((1, 2, 3, 2))
    assembly = PrimaryAssembly(
        cache=LegacyDynamicCache(key, value),
        context_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
    )
    llama_module = importlib.import_module("transformers.models.llama.modeling_llama")
    attention_calls = []

    monkeypatch.setattr(
        llama_module,
        "apply_rotary_pos_emb",
        lambda query, key, cos, sin: (query, key),
    )

    def fake_eager_attention(module, query, cache_key, cache_value, **kwargs):
        attention_calls.append((cache_key, cache_value))
        return torch.zeros_like(query), None

    monkeypatch.setattr(
        llama_module,
        "eager_attention_forward",
        fake_eager_attention,
        raising=False,
    )

    recompute_selected_tokens(model, assembly, [1])

    assert attention_calls[0][0] is key
    assert attention_calls[0][1] is value


@pytest.mark.parametrize("document_count", [1, 2])
def test_cacheclip_score_documents_normalizes_legacy_tuple_before_forward(document_count):
    class FakeTokenizer:
        @staticmethod
        def __call__(text, **kwargs):
            return {"input_ids": torch.tensor([[9]], dtype=torch.long)}

    class FakeModel:
        config = SimpleNamespace()

        def __init__(self):
            self.batch_sizes = []

        def __call__(self, input_ids, past_key_values, **kwargs):
            assert hasattr(past_key_values, "get_seq_length")
            self.batch_sizes.append(input_ids.size(0))
            cache_length = past_key_values.get_seq_length()
            assert kwargs["attention_mask"].shape == (
                input_ids.size(0), cache_length + input_ids.size(1)
            )
            assert kwargs["attention_mask"].all()
            assert kwargs["cache_position"].tolist() == list(
                range(cache_length, cache_length + input_ids.size(1))
            )
            assert kwargs["use_cache"] is False
            first_key, first_value = next(iter(_cacheclip_backend_module._cache_layer_pairs(past_key_values)))
            past_key_values.update(
                torch.zeros((input_ids.size(0), 1, input_ids.size(1), first_key.size(-1))),
                torch.zeros((input_ids.size(0), 1, input_ids.size(1), first_value.size(-1))),
                0,
            )
            attention = torch.ones(
                (
                    input_ids.size(0),
                    1,
                    input_ids.size(1),
                    cache_length + input_ids.size(1),
                )
            )
            return SimpleNamespace(attentions=[attention])

    def tuple_cache(length):
        return tuple(
            (
                torch.zeros((1, 1, length, 2)),
                torch.zeros((1, 1, length, 2)),
            )
            for _ in range(2)
        )

    model = FakeModel()
    runtime = object.__new__(AuxiliaryRuntime)
    runtime.model = model
    runtime.tokenizer = FakeTokenizer()
    caches = [
        AuxiliaryDocumentCache(tuple_cache(3), [(0, 1)] * 3, 0)
        for _ in range(document_count)
    ]

    scores = runtime.score_documents("query", caches)

    assert scores == [[1.0, 1.0, 1.0] for _ in range(document_count)]
    assert model.batch_sizes == [document_count]
    assert all(
        key.shape[-2] == 3
        for cache in caches
        for key, _ in _cacheclip_backend_module._cache_layer_pairs(cache.cache)
    )


def test_cacheclip_cached_attention_positions_start_after_past():
    from methods.CacheClipBackend import _cached_query_attention_kwargs

    input_ids = torch.tensor([[3, 4]], dtype=torch.long)

    kwargs = _cached_query_attention_kwargs(
        input_ids, SimpleNamespace(get_seq_length=lambda: 7)
    )

    assert kwargs["attention_mask"].shape == (1, 9)
    assert kwargs["cache_position"].tolist() == [7, 8]


def test_cacheclip_qwen2_rope_assembly_and_sparse_recompute(monkeypatch):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from helpers.backends.TransformersHelper import CacheLayerPairs

    monkeypatch.setenv("CACHECLIP_ATTENTION_BACKEND", "torch")
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    model = Qwen2ForCausalLM(config).eval()
    first_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    second_ids = torch.tensor([[5, 6, 7]], dtype=torch.long)
    with torch.no_grad():
        first_cache = model(input_ids=first_ids, use_cache=True).past_key_values
        second_cache = model(input_ids=second_ids, use_cache=True).past_key_values

    assembly = assemble_primary_cache(
        model,
        None,
        torch.empty((1, 0), dtype=torch.long),
        None,
        torch.empty((1, 0), dtype=torch.long),
        0,
        [list(CacheLayerPairs(first_cache)), list(CacheLayerPairs(second_cache))],
        [first_ids, second_ids],
    )

    assert assembly.context_ids.tolist() == [[1, 2, 3, 4, 5, 6, 7]]
    recompute_selected_tokens(model, assembly, [2, 5])
    assert all(torch.isfinite(key).all() for key, _ in CacheLayerPairs(assembly.cache))
    assert all(
        key.shape[-2] == 3
        for cache in caches
        for key, _ in _cacheclip_backend_module._cache_layer_pairs(cache.cache)
    )


def test_cacheclip_cached_attention_positions_start_after_past():
    from methods.CacheClipBackend import _cached_query_attention_kwargs

    input_ids = torch.tensor([[3, 4]], dtype=torch.long)

    kwargs = _cached_query_attention_kwargs(
        input_ids, SimpleNamespace(get_seq_length=lambda: 7)
    )

    assert kwargs["attention_mask"].shape == (1, 9)
    assert kwargs["cache_position"].tolist() == [7, 8]


def test_cacheclip_qwen2_rope_assembly_and_sparse_recompute(monkeypatch):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from helpers.backends.TransformersHelper import CacheLayerPairs

    monkeypatch.setenv("CACHECLIP_ATTENTION_BACKEND", "torch")
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    model = Qwen2ForCausalLM(config).eval()
    first_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    second_ids = torch.tensor([[5, 6, 7]], dtype=torch.long)
    with torch.no_grad():
        first_cache = model(input_ids=first_ids, use_cache=True).past_key_values
        second_cache = model(input_ids=second_ids, use_cache=True).past_key_values

    assembly = assemble_primary_cache(
        model,
        None,
        torch.empty((1, 0), dtype=torch.long),
        None,
        torch.empty((1, 0), dtype=torch.long),
        0,
        [list(CacheLayerPairs(first_cache)), list(CacheLayerPairs(second_cache))],
        [first_ids, second_ids],
    )

    assert assembly.context_ids.tolist() == [[1, 2, 3, 4, 5, 6, 7]]
    recompute_selected_tokens(model, assembly, [2, 5])
    assert all(torch.isfinite(key).all() for key, _ in _cache_layer_pairs(assembly.cache))


def test_cacheclip_selects_global_candidates_deterministically():
    scores = [[0.4, 0.9, 0.1], [0.9, 0.2]]
    assert select_top_k_candidates(scores, 0.4) == [[1], [0]]


def test_cacheclip_windows_never_cross_document_boundary():
    assert group_candidate_windows([[1, 2, 3, 9], [0, 1, 2, 3, 4]], [10, 5], 4, 3) == [list(range(1, 5)), list(range(0, 5))]


def test_cacheclip_window_grouping_matches_reference_for_random_candidates():
    rng = random.Random(1234)
    for _ in range(100):
        document_lengths = [rng.randrange(0, 64) for _ in range(rng.randrange(1, 6))]
        candidates_by_document = [
            [rng.randrange(length) for _ in range(rng.randrange(length + 1))]
            if length
            else []
            for length in document_lengths
        ]
        window_size = rng.randrange(1, 16)
        density_threshold = rng.randrange(1, 8)
        expected = []
        for candidates, document_length in zip(candidates_by_document, document_lengths):
            candidate_set = set(candidates)
            selected = set()
            for start in sorted(candidate_set):
                end = min(document_length, start + window_size)
                if sum(start <= candidate < end for candidate in candidate_set) >= density_threshold:
                    selected.update(range(start, end))
            expected.append(sorted(selected))
        assert group_candidate_windows(
            candidates_by_document,
            document_lengths,
            window_size,
            density_threshold,
        ) == expected


def test_cacheclip_maps_overlapping_tokenizer_spans():
    auxiliary = [(0, 2), (2, 5), (5, 8)]
    primary = [(0, 3), (3, 6), (6, 9)]
    assert map_auxiliary_to_primary_indices(auxiliary, primary, [1]) == [0, 1]


def test_cacheclip_shared_prefix_excludes_boundary_token():
    ids = [[10, 11, 20], [10, 11, 21]]
    offsets = [[(0, 1), (1, 2), (2, 4)], [(0, 1), (1, 2), (2, 4)]]
    assert shared_prefix_token_length(ids, offsets, 2) == 2


def test_cacheclip_run_hit_does_not_use_full_fallback(monkeypatch):
    method = _fake_method()
    method._select = lambda *args: []
    _patch_fake_repair(monkeypatch)
    fallback_calls = []
    monkeypatch.setattr(
        method,
        "_full_fallback",
        lambda prompt: fallback_calls.append(prompt),
    )

    result = method._run_one(_prepared_state(), "document question")

    assert result.output == "generated"
    assert fallback_calls == []


def test_cacheclip_run_miss_uses_explicit_full_fallback(monkeypatch):
    method = _fake_method()
    fallback_calls = []
    fallback_result = object()
    monkeypatch.setattr(
        method,
        "_full_fallback",
        lambda prompt: fallback_calls.append(prompt) or fallback_result,
    )

    result = method._run_one(_prepared_state(), "other question")

    assert result is fallback_result
    assert fallback_calls == ["other question"]


def test_cacheclip_selector_and_primary_assembly_futures_start_together(monkeypatch):
    method = _fake_method()
    started = []
    barrier = threading.Barrier(2)

    def selector(*args):
        started.append("cpu_selector")
        barrier.wait()
        return []

    def primary_assembly(*args):
        started.append("primary_assembly")
        barrier.wait()
        return _fake_assembly()

    method._select = selector
    monkeypatch.setattr(_cacheclip_module, "assemble_primary_cache", primary_assembly)
    monkeypatch.setattr(
        _cacheclip_module,
        "recompute_selected_tokens",
        lambda *args: None,
    )

    result = method._run_one(_prepared_state(), "document question")

    assert result.output == "generated"
    assert sorted(started) == ["cpu_selector", "primary_assembly"]


def test_cacheclip_constructor_has_no_method_metrics_and_unified_performance(monkeypatch):
    monkeypatch.setattr(_cacheclip_module, "DefaultModelPath", lambda: "/model")
    monkeypatch.setattr(_cacheclip_module, "ResolveSamplingConfig", lambda _: {})
    method = CacheClip(auxiliaryModelPath="/unused")
    result = method._result("answer", 0.1, 0.2, 3, 7)
    assert method.method_metrics == ("actual_recompute_ratio",)
    assert set(result.performance) == {"ttft", "num_output_tokens", "total_time"}
    assert result.performance["ttft"] <= result.performance["total_time"]


def test_cacheclip_retain_output_is_accepted_without_output_kv_registration(monkeypatch):
    method = _fake_method()
    method._states = [_prepared_state()]
    method._select = lambda *args: []
    _patch_fake_repair(monkeypatch)

    # KVCOMM can complete this action because retainOutput is a hint that a
    # Method may ignore, but generated output KV is not registered for reuse.
    results = method.Run(["document question"], retainOutput=[True])

    assert len(results) == 1
    assert set(results[0].performance) == {
        "ttft",
        "num_output_tokens",
        "total_time",
    }
    assert method._states[0]["chunks"] == ["document"]
    assert method._gen.generate_calls == [{}]


def test_cacheclip_rejects_unsupported_sparse_model_without_full_forward():
    class Unsupported:
        config = type("Config", (), {})()
        model = type("BaseModel", (), {"layers": []})()

    assembly = type("Assembly", (), {
        "context_ids": torch.zeros((1, 2), dtype=torch.long),
        "cache": type("Cache", (), {"layers": []})(),
    })()
    from methods.CacheClipBackend import recompute_selected_tokens
    with pytest.raises(RuntimeError, match="model-specific decoder-layer adapter"):
        recompute_selected_tokens(Unsupported(), assembly, [0])
