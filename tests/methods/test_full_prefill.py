"""Tests for full prefill."""

import sys
import types
from types import SimpleNamespace

import pytest

from methods.FullPrefill import FullPrefillTransformer, FullPrefillVllm
from tests.support.vllm import _Engine, _SamplingParams, _Tokenizer


def test_full_prefill_vllm_reads_max_num_seqs_from_config(monkeypatch):
    monkeypatch.setattr(
        "methods.FullPrefill.Get",
        lambda key, default=None: (
            {"MaxNumSeqs": 32}
            if key == "FullPrefillVllm"
            else default
        ),
    )

    method = FullPrefillVllm()

    assert method.maxNumSeqs == 32


def test_full_prefill_vllm_reports_native_prefix_hits(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(SamplingParams=_SamplingParams))
    method = object.__new__(FullPrefillVllm)
    method.llm = types.SimpleNamespace(llm_engine=_Engine(), get_tokenizer=lambda: _Tokenizer())
    method.samplingConfig = {"mode": "greedy"}

    result = method.Run([[7, 8, 9]], maxNewTokens=2)[0]

    assert result.metadata["num_cached_tokens"] == 3
    assert result.metadata["reuse_ratio"] == 1.0
    assert method.method_metrics == ("reuse_ratio",)


def test_transformer_batch_keeps_native_ids_order_budget_and_elapsed_time(clock):
    calls = []

    def generate(ids, **kwargs):
        calls.append((ids, kwargs))
        clock.advance(4)
        return str(ids), 2.0, 4.0, 1, [7]

    method = FullPrefillTransformer()
    method._gen = SimpleNamespace(Generate=generate)
    method.Prepare([[[1, 2]], [[3, 4]]])
    results = method.Run([[41, 42], [51]], retainOutput=[True, True], maxNewTokens=3)
    assert calls == [([41, 42], {"maxNewTokens": 3, "returnTokenIds": True}), ([51], {"maxNewTokens": 3, "returnTokenIds": True})]
    assert [r.output_ids for r in results] == [[7], [7]]
    assert [r.performance["ttft"] for r in results] == [2.0, 6.0]
    assert [r.performance["total_time"] for r in results] == [4.0, 8.0]
    assert [r.metadata["n_input"] for r in results] == [2, 1]
    assert all(r.metadata["reuse_ratio"] == 0 for r in results)


@pytest.mark.parametrize("value,error", [(True, TypeError), (1.5, TypeError), (0, ValueError)])
def test_vllm_rejects_invalid_configured_concurrency(monkeypatch, value, error):
    monkeypatch.setattr("methods.FullPrefill.Get", lambda *args: {"MaxNumSeqs": value})
    with pytest.raises(error, match="MaxNumSeqs"):
        FullPrefillVllm()
