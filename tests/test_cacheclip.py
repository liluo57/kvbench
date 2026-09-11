import importlib
import threading
from types import SimpleNamespace

import pytest
import torch

from methods.CacheClip import CacheClip
from methods.CacheClipBackend import (
    group_candidate_windows,
    map_auxiliary_to_primary_indices,
    select_top_k_candidates,
    shared_prefix_token_length,
)

_cacheclip_module = importlib.import_module("methods.CacheClip")


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


def test_cacheclip_selects_global_candidates_deterministically():
    scores = [[0.4, 0.9, 0.1], [0.9, 0.2]]
    assert select_top_k_candidates(scores, 0.4) == [[1], [0]]


def test_cacheclip_windows_never_cross_document_boundary():
    assert group_candidate_windows([[1, 2, 3, 9], [0, 1, 2, 3, 4]], [10, 5], 4, 3) == [list(range(1, 5)), list(range(0, 5))]


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
    monkeypatch.setattr("methods.CacheClip.DefaultModelPath", lambda: "/model")
    monkeypatch.setattr("methods.CacheClip.ResolveSamplingConfig", lambda _: {})
    method = CacheClip(auxiliaryModelPath="/unused")
    result = method._result("answer", 0.1, 0.2, 3, 7)
    assert method.method_metrics == ()
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
