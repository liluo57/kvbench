import importlib
import random
import threading
from types import SimpleNamespace

import pytest
import torch

from methods.CacheClip import CacheClip
from methods.CacheClip.Cache import (
    PrimaryAssembly,
    _cache_layer_pairs,
    _new_dynamic_cache,
    append_cached_primary_segment,
    assemble_primary_cache,
    select_cache_sequence,
)
from methods.CacheClip.Recompute import (
    recompute_selected_tokens,
    _flashinfer_attention_forward,
    _flashinfer_single_attention_forward,
    _flashinfer_causal_bsr_metadata,
    _local_eager_attention_forward,
    _resolve_attention_backend,
)
from methods.CacheClip.Selector import (
    AuxiliaryDocumentCache,
    AuxiliaryRuntime,
    _bucket_indices,
    _cached_query_attention_kwargs,
    group_candidate_windows,
    map_auxiliary_to_primary_indices,
    select_top_k_candidates,
    token_text_and_offsets,
)

_cacheclip_module = importlib.import_module("methods.CacheClip.Method")
_cache_module = importlib.import_module("methods.CacheClip.Cache")
_selector_module = importlib.import_module("methods.CacheClip.Selector")
_recompute_module = importlib.import_module("methods.CacheClip.Recompute")


def _flashinfer_importable():
    try:
        from flashinfer.sparse import BlockSparseAttentionWrapper  # noqa: F401
    except Exception:
        return False
    return True


class _FakeCache:
    def __init__(self, ids=()):
        self.ids = list(ids)


class _FakeTokenizer:
    @staticmethod
    def decode(ids, **kwargs):
        assert isinstance(ids, list)
        assert all(isinstance(token, int) for token in ids)
        return "|".join(str(int(token)) for token in ids)


class _FakeGenerator:
    def __init__(self):
        self.model = SimpleNamespace(device=torch.device("cpu"))
        self.tokenizer = _FakeTokenizer()
        self.generate_calls = []
        self.prefill_calls = []
        self.encode_calls = []

    def Encode(self, text, addSpecialTokens=False):
        self.encode_calls.append(text)
        return [999] if text else []

    def Prefill(self, input_ids, pastKeyValues=None):
        ids = list(input_ids)
        self.prefill_calls.append((ids, pastKeyValues))
        merged = (pastKeyValues.ids if pastKeyValues is not None else []) + ids
        return SimpleNamespace(past_key_values=_FakeCache(merged))

    def Generate(self, input_ids, pastKeyValues=None, **kwargs):
        self.generate_calls.append((list(input_ids), pastKeyValues, kwargs))
        return "generated", 0.01, 0.02, 1, [901]


def _fake_method():
    method = object.__new__(CacheClip)
    method._gen = _FakeGenerator()
    method._auxiliary = SimpleNamespace()
    method._states = []
    method.requireReuse = False
    method.candidateRatio = 0.2
    method.windowSize = 8
    method.densityThreshold = 1
    return method


def _prepared_state():
    return {
        "document_ids": [[7, 8]],
        "primary_id_tensors": [torch.tensor([[7, 8]], dtype=torch.long)],
        "primary": [_FakeCache([7, 8])],
        "primary_offsets": [[(0, 1), (1, 2)]],
        "auxiliary": [AuxiliaryDocumentCache(_FakeCache([7, 8]), [(0, 1), (1, 2)], 0)],
        "old_start": 0,
    }


def _fake_assembly():
    return SimpleNamespace(
        cache=_FakeCache([7, 8]),
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


def test_cacheclip_prepare_prefills_authoritative_ids_with_optional_warmup_prefix(monkeypatch):
    class FakeGenerator:
        tokenizer = object()
        model = SimpleNamespace(device=torch.device("cpu"))
        encode_calls = []
        prefill_calls = []

        @classmethod
        def Encode(cls, text, addSpecialTokens=False):
            cls.encode_calls.append(text)
            return [999] if text else []

        @classmethod
        def Prefill(cls, input_ids):
            cls.prefill_calls.append(list(input_ids))
            return SimpleNamespace(past_key_values=object())

    class FakeAuxiliary:
        def __init__(self):
            self.calls = []

        def prefill_documents(self, prefix, documents):
            self.calls.append((prefix, list(documents)))
            return [object() for _ in documents]

    calls = []

    def token_text_and_offsets(tokenizer, ids):
        calls.append((tokenizer, list(ids)))
        return "document", [(0, 1), (1, 2)]

    method = object.__new__(CacheClip)
    method._gen = FakeGenerator()
    method._auxiliary = FakeAuxiliary()
    method._states = []
    method.sharedPrefixText = "warm "
    monkeypatch.setattr(
        _cacheclip_module,
        "token_text_and_offsets",
        token_text_and_offsets,
    )
    monkeypatch.setattr(
        _cacheclip_module,
        "select_cache_sequence",
        lambda cache, start, end: (cache, start, end),
    )

    method.Prepare([[[7, 8]]])

    assert calls == [(method._gen.tokenizer, [7, 8])]
    assert method._gen.encode_calls == ["warm "]
    assert method._gen.prefill_calls[0] == [999, 7, 8]
    assert method._states[0]["document_ids"] == [[7, 8]]
    assert method._states[0]["primary"][0][1:] == (1, 3)
    assert method._auxiliary.calls == [("warm ", ["document"])]


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
        _cache_module,
        "_new_dynamic_cache",
        lambda config: FakeCache(),
    )
    monkeypatch.setattr(
        _cache_module,
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
        _cache_module,
        "_new_dynamic_cache",
        lambda config: FakeCache(),
    )
    seen = []

    def rotate(key, rotary, old, new, nope_dim):
        seen.append((old.tolist(), new.tolist()))
        return key + 10

    monkeypatch.setattr(_cache_module, "_rotate_key", rotate)
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
    monkeypatch.setattr(method, "_run_interleaved", lambda *args: "interleaved")
    assert method._is_contiguous_match([(0, [1]), (None, [8]), (1, [2])], [0, 2]) is False
    assert method._is_contiguous_match([(None, [8]), (0, [1]), (1, [2])], [1, 2]) is True


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
        _cache_module,
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
            first_key, first_value = next(iter(_cache_layer_pairs(past_key_values)))
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
        for key, _ in _cache_layer_pairs(cache.cache)
    )


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
        [first_cache, second_cache],
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


def _state_for_documents(documents):
    return {
        "document_ids": [list(ids) for ids in documents],
        "primary_id_tensors": [torch.tensor([ids], dtype=torch.long) for ids in documents],
        "primary": [_FakeCache(ids) for ids in documents],
        "primary_offsets": [[(i, i + 1) for i in range(len(ids))] for ids in documents],
        "auxiliary": [
            AuxiliaryDocumentCache(_FakeCache(ids), [(i, i + 1) for i in range(len(ids))], 0)
            for ids in documents
        ],
        "old_start": 0,
    }


def _patch_fake_assembly(monkeypatch, *, context_override=None):
    def assemble(_model, _prefix_cache, prefix_ids, _document_caches, document_ids, *, old_start=0):
        ids = prefix_ids.tolist()[0] + [token for item in document_ids for token in item.tolist()[0]]
        if context_override is not None:
            ids = context_override(ids)
        return PrimaryAssembly(_FakeCache(ids), torch.tensor([ids], dtype=torch.long))

    def append(_model, assembly, _document_cache, document_ids, *, old_start=0):
        ids = assembly.context_ids.tolist()[0] + document_ids.tolist()[0]
        return PrimaryAssembly(_FakeCache(ids), torch.tensor([ids], dtype=torch.long))

    monkeypatch.setattr(_cacheclip_module, "assemble_primary_cache", assemble)
    monkeypatch.setattr(_cacheclip_module, "append_cached_primary_segment", append)
    monkeypatch.setattr(_cacheclip_module, "select_cache_sequence", lambda cache, start, end=None: (cache, start, end))
    monkeypatch.setattr(_cacheclip_module, "recompute_selected_tokens", lambda *args: None)


def test_cacheclip_run_hit_preserves_chunk_and_fresh_ids_and_output_ids(monkeypatch):
    method = _fake_method()
    method._states = [_state_for_documents([[7, 8]])]
    method._select = lambda *args: []
    _patch_fake_assembly(monkeypatch)

    result = method.Run([[7, 8, 42]], retainOutput=[False])[0]

    assert method._gen.encode_calls == []
    assert method._gen.generate_calls[0][0] == [42]
    assert method._gen.generate_calls[0][2]["returnTokenIds"] is True
    assert result.output_ids == [901]
    assert result.metadata["n_input"] == 3
    assert result.metadata["n_cached_tokens"] == 2
    assert result.metadata["reuse_ratio"] == pytest.approx(2 / 3)
    assert result.metadata["recompute_ratio"] == 0.0
    assert result.metadata["fallback_rate"] == 0.0


def test_cacheclip_composes_abc1abc2_without_reencoding(monkeypatch):
    method = _fake_method()
    method._states = [_state_for_documents([[10, 11], [12, 13]])]
    method._select = lambda *args: []
    _patch_fake_assembly(monkeypatch)

    result = method.Run([[10, 11, 12, 13, 99]])[0]

    assert method._gen.encode_calls == []
    assert method._gen.generate_calls[0][0] == [99]
    assert result.metadata["n_input"] == 5
    assert result.output_ids == [901]


def test_cacheclip_interleaved_reorders_and_repeats_prepared_chunks(monkeypatch):
    method = _fake_method()
    method._states = [_state_for_documents([[1, 2], [3, 4]])]
    selected_queries = []
    method._select_occurrences = lambda text, occurrences: selected_queries.append((text, len(occurrences))) or []
    _patch_fake_assembly(monkeypatch)

    result = method.Run([[3, 4, 9, 1, 2, 8, 3, 4, 7]])[0]

    assert selected_queries == [("9|8|7", 3)]
    assert method._gen.encode_calls == []
    assert method._gen.generate_calls[0][0] == [7]
    assert result.metadata["n_cached_tokens"] == 6
    assert result.metadata["n_input"] == 9
    assert result.metadata["reuse_ratio"] == pytest.approx(6 / 9)


def test_cacheclip_interleaved_tail_replay_uses_original_last_id(monkeypatch):
    method = _fake_method()
    method._states = [_state_for_documents([[7, 8], [3, 4]])]
    queries = []
    method._select_occurrences = lambda text, occurrences: queries.append(text) or []
    _patch_fake_assembly(monkeypatch)

    result = method.Run([[7, 8, 55, 3, 4]])[0]

    assert queries == ["55"]
    assert method._gen.generate_calls[0][0] == [4]
    assert method._gen.generate_calls[0][1][0].ids == [7, 8, 55, 3, 4]
    assert method._gen.generate_calls[0][1][1:] == (0, -1)
    assert result.metadata["n_cached_tokens"] == 4
    assert result.metadata["n_input"] == 5
    assert result.metadata["reuse_ratio"] == pytest.approx(4 / 5)
    assert result.metadata["recompute_ratio"] == pytest.approx(1 / 4)


def test_cacheclip_replay_at_contiguous_cached_tail_drops_one_cached_row(monkeypatch):
    method = _fake_method()
    method._states = [_state_for_documents([[7, 8]])]
    method._select = lambda *args: []
    _patch_fake_assembly(monkeypatch)

    result = method.Run([[7, 8]])[0]

    assert method._gen.generate_calls[0][0] == [8]
    assert result.metadata["n_cached_tokens"] == 2
    assert result.metadata["n_input"] == 2
    assert result.metadata["reuse_ratio"] == 1.0
    assert result.metadata["recompute_ratio"] == 0.5


def test_cacheclip_zero_match_is_the_only_full_prefill_and_is_strict_when_requested():
    method = _fake_method()
    method._states = [_prepared_state()]
    result = method.Run([[42, 43]])[0]
    assert method._gen.generate_calls[0][0] == [42, 43]
    assert method._gen.generate_calls[0][1] is None
    assert method._gen.encode_calls == []
    assert result.metadata["n_input"] == 2

    assert result.metadata["full_prefill"] is True
    assert result.metadata["fallback_rate"] == 1.0
    assert result.metadata["reuse_ratio"] == 0.0
    assert result.metadata["recompute_ratio"] == 1.0
    assert result.metadata["n_recomputed_tokens"] == 2

    method.requireReuse = True
    with pytest.raises(RuntimeError, match="no matching prepared token IDs"):
        method.Run([[42, 43]])


@pytest.mark.parametrize("selected", [[], [1], [1, 2]])
def test_cacheclip_dynamic_repair_ratio_excludes_fresh_prefix_and_does_not_reduce_hits(monkeypatch, selected):
    method = _fake_method()
    method._states = [_state_for_documents([[7, 8]])]
    method._select = lambda *args: selected
    _patch_fake_assembly(monkeypatch)

    result = method.Run([[42, 7, 8, 9]])[0]

    assert result.metadata["reuse_ratio"] == 0.5
    assert result.metadata["recompute_ratio"] == len(selected) / 2


def test_cacheclip_fallback_is_included_in_every_declared_metric(monkeypatch):
    from core.Worker import _RecordRunResults

    method = _fake_method()
    method._states = [_state_for_documents([[7, 8]])]
    method._select = lambda *args: []
    _patch_fake_assembly(monkeypatch)
    hit = method.Run([[7, 8, 9]])[0]
    miss = method.Run([[42, 43]])[0]
    scores = {name: [] for name in method.method_metrics}
    weights = {name: [] for name in method.method_metrics}
    _RecordRunResults([hit, miss], [], method, scores, weights)

    assert scores["reuse_ratio"] == [2 / 3, 0.0]
    assert weights["reuse_ratio"] == [3.0, 2.0]
    assert scores["recompute_ratio"] == [0.0, 1.0]
    assert scores["fallback_rate"] == [0.0, 1.0]


def test_cacheclip_run_budget_prefers_explicit_value_then_constructor_default():
    method = _fake_method()
    method._states = [_prepared_state()]
    method.maxNewTokens = 32

    method.Run([[42, 43]], maxNewTokens=None)
    assert method._gen.generate_calls[-1][2]["maxNewTokens"] == 32

    method.Run([[42, 43]], maxNewTokens=7)
    assert method._gen.generate_calls[-1][2]["maxNewTokens"] == 7


def test_cacheclip_run_without_constructor_budget_uses_common_default():
    method = _fake_method()
    method._states = [_prepared_state()]

    method.Run([[42, 43]], maxNewTokens=None)

    assert method._gen.generate_calls[-1][2]["maxNewTokens"] == 64


def test_cacheclip_raises_if_assembly_changes_ids_without_full_prefill(monkeypatch):
    method = _fake_method()
    method._states = [_state_for_documents([[7, 8]])]
    method._select = lambda *args: []
    _patch_fake_assembly(monkeypatch, context_override=lambda ids: ids + [123])

    with pytest.raises(RuntimeError, match="do not equal the input prompt"):
        method.Run([[7, 8, 42]])
    assert method._gen.generate_calls == []


def test_cacheclip_selector_and_primary_assembly_futures_start_together(monkeypatch):
    method = _fake_method()
    started = []
    barrier = threading.Barrier(2)

    def selector(*args):
        started.append("cpu_selector")
        barrier.wait(timeout=2)
        return []

    def primary_assembly(*args, **kwargs):
        started.append("primary_assembly")
        barrier.wait(timeout=2)
        return PrimaryAssembly(_FakeCache([7, 8]), torch.tensor([[7, 8]]))

    method._states = [_state_for_documents([[7, 8]])]
    method._select = selector
    monkeypatch.setattr(_cacheclip_module, "assemble_primary_cache", primary_assembly)
    monkeypatch.setattr(_cacheclip_module, "recompute_selected_tokens", lambda *args: None)

    result = method.Run([[7, 8, 9]])[0]

    assert result.output == "generated"
    assert sorted(started) == ["cpu_selector", "primary_assembly"]


def test_cacheclip_constructor_uses_common_performance_contract(monkeypatch):
    monkeypatch.setattr(_cacheclip_module, "DefaultModelPath", lambda: "/model")
    monkeypatch.setattr(_cacheclip_module, "ResolveSamplingConfig", lambda _: {})
    method = CacheClip(auxiliaryModelPath="/unused")
    assert method.method_metrics == ("reuse_ratio", "recompute_ratio", "fallback_rate")


def test_cacheclip_accepts_retain_output_and_returns_native_ids(monkeypatch):
    method = _fake_method()
    method._states = [_prepared_state()]
    method._select = lambda *args: []
    _patch_fake_assembly(monkeypatch)
    # This legacy fixture covers only the boolean hint API. Native cache
    # retention requires a real cache and is exercised in the dedicated test.
    method._gen.Generate = lambda *args, **kwargs: (
        "generated", 0.01, 0.02, 1, _FakeCache([7, 8, 901]), [901]
    )
    method._gen.OutputTokenIds = lambda ids: list(ids)
    monkeypatch.setattr(_cacheclip_module, "select_output_cache", lambda *args, **kwargs: _FakeCache([901]))
    monkeypatch.setattr(_cacheclip_module, "_cache_length", lambda cache: len(cache.ids))
    monkeypatch.setattr(_cacheclip_module, "token_text_and_offsets", lambda tokenizer, ids: ("output", [(0, 1)]))
    method._auxiliary.prefill_documents = lambda prefix, docs: [
        AuxiliaryDocumentCache(_FakeCache([901]), [(0, 1)], 0)
    ]
    method.sharedPrefixText = None

    results = method.Run([[7, 8, 9]], retainOutput=[True])

    assert results[0].output_ids == [901]
    assert method._states[0]["document_ids"][-1] == [901]


def test_cacheclip_rejects_unsupported_sparse_model_without_full_forward():
    class Unsupported:
        config = type("Config", (), {})()
        model = type("BaseModel", (), {"layers": []})()

    assembly = type("Assembly", (), {
        "context_ids": torch.zeros((1, 2), dtype=torch.long),
        "cache": type("Cache", (), {"layers": []})(),
    })()
    from methods.CacheClip.Recompute import recompute_selected_tokens
    with pytest.raises(RuntimeError, match="model-specific decoder-layer adapter"):
        recompute_selected_tokens(Unsupported(), assembly, [0])
