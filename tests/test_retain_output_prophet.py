"""ProphetKV retains native output K/V as a standalone reusable chunk."""

import torch
import pytest

from methods.ProphetKV import ProphetKV, _CaseState, _DenseProphetRuntime
from test_prophet_runtime_contract import timing_clock


def test_prophet_retains_output_cache_without_prefilling_it(monkeypatch):
    method = ProphetKV(maxNewTokens=3)
    key = torch.arange(8, dtype=torch.float32).reshape(1, 1, 2, 4)
    value = key + 100
    runtime = type("Runtime", (), {})()
    runtime.max_new_tokens = 3
    runtime.full_generate = lambda ids, **kwargs: (
        "answer", 0.2, 1.0, 3, [21, 22], [(key, value)]
    )
    method._runtime = runtime
    method._states = [_CaseState([], [])]

    result = method.Run([[1, 2]], retainOutput=[True])[0]

    assert result.output_ids == [21, 22]
    assert method._states[0].chunks == [[21, 22]]
    cache = method._states[0].caches[0]
    assert cache.ids == result.output_ids
    assert cache.layers[0][0] is key
    assert cache.layers[0][1] is value

    # The newly registered output is directly discoverable by the normal
    # segment matcher, with no output-prefill step.
    from helpers.backends.Prompt import ComposeInterleavedReuse
    parts = ComposeInterleavedReuse(method._states[0].chunks, [9, 21, 22, 10])
    assert any(chunk_index == 0 and span == [21, 22] for chunk_index, span in parts)


def test_prophet_empty_output_fragment_does_not_register_a_chunk():
    method = ProphetKV()
    method._runtime = type("Runtime", (), {
        "max_new_tokens": 1,
        "full_generate": lambda self, ids, **kwargs: (
            "", 0.1, 0.2, 1, [], []
        ),
    })()
    method._states = [_CaseState([], [])]

    result = method.Run([[3]], retainOutput=[True])[0]

    assert result.output_ids == []
    assert method._states[0].chunks == []
    assert method._states[0].caches == []


def test_prophet_native_decode_rows_are_owned_and_reused_without_output_prefill(monkeypatch):
    from test_recompute_backend import runtime

    backend, sampler = runtime("llama")
    native = object.__new__(_DenseProphetRuntime)
    native.torch, native.device, native.model = torch, "cpu", backend.model
    base = backend.model.model
    native.layers, native.embed_tokens = base.layers, base.embed_tokens
    native.final_norm, native.lm_head = base.norm, backend.model.lm_head
    native.tokenizer, native.eos_ids = sampler.tokenizer, {1}
    native.sampling_config = {"mode": "greedy"}
    native.max_model_len, native.max_new_tokens = 256, 3
    native.flashinfer = None
    samples = iter([14, 15, 1])
    native._sample = lambda *args: next(samples)
    projected = []
    project = native._project

    def record_projection(*args):
        query, key, value = project(*args)
        projected.append((key.detach().clone(), value.detach().clone()))
        return query, key, value

    monkeypatch.setattr(native, "_project", record_projection)
    monkeypatch.setattr(native, "prefill_isolated", lambda *args: pytest.fail("output was re-prefilled"))
    method = ProphetKV(maxNewTokens=3, recomputeRatio=0)
    method._runtime = native
    first = method.Run([[2, 3, 4]], retainOutput=[True])[0]
    assert first.output_ids == [14, 15]
    assert first.performance["num_output_tokens"] == 3
    retained = method._states[0].caches[-1]
    n_layers = len(native.layers)
    output_projections = projected[-2 * n_layers:]
    saved = []
    for index, (key, value) in enumerate(retained.layers):
        expected_key = torch.cat([output_projections[index][0], output_projections[n_layers + index][0]], dim=2)
        expected_value = torch.cat([output_projections[index][1], output_projections[n_layers + index][1]], dim=2)
        torch.testing.assert_close(key, expected_key, rtol=0, atol=0)
        torch.testing.assert_close(value, expected_value, rtol=0, atol=0)
        assert key.untyped_storage().nbytes() == key.numel() * key.element_size()
        assert value.untyped_storage().nbytes() == value.numel() * value.element_size()
        saved.append((key.clone(), value.clone()))
    native._sample = lambda *args: 16
    later = method.Run([[20, 14, 15, 21]], maxNewTokens=1)[0]
    assert later.metadata["reuse_ratio"] == 0.5
    for (key, value), (old_key, old_value) in zip(retained.layers, saved):
        torch.testing.assert_close(key, old_key, rtol=0, atol=0)
        torch.testing.assert_close(value, old_value, rtol=0, atol=0)
    method.Reset()
    assert method._states == []


def test_prophet_rejects_native_cache_rows_that_disagree_with_output_ids():
    method = ProphetKV()
    key = torch.zeros(1, 1, 3, 4)
    state = _CaseState([], [])
    with pytest.raises(AssertionError, match="native output KV length"):
        method._retain_output(state, [21, 22], [(key, key)])
    assert state == _CaseState([], [])


def test_prophet_total_counts_output_retention_once(timing_clock, monkeypatch):
    from types import SimpleNamespace

    _, clock = timing_clock
    method = ProphetKV()
    key = torch.zeros(1, 1, 1, 4)

    def generate(ids, **kwargs):
        clock.now += 7
        return "answer", 2.0, 7.0, 1, [7], [(key, key)]

    method._runtime = SimpleNamespace(full_generate=generate)
    register = method._retain_output

    def delayed_register(*args):
        clock.now += 3
        return register(*args)

    monkeypatch.setattr(method, "_retain_output", delayed_register)
    result = method.Run([[2, 3]], retainOutput=[True])[0]
    assert result.performance["total_time"] == 10
    assert result.performance["ttft"] == 2
