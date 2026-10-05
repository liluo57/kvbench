"""Tests for prophet."""

from types import SimpleNamespace

import pytest
import torch

from core.Worker import _RunMethod
from helpers.backends.TransformersHelper import TransformersGenerator
from methods.ProphetKV import ProphetKV, _CaseState, _DenseProphetRuntime
from tests.support.models import runtime


def test_prophet_fixed_recompute_target_is_not_aggregated_as_a_method_metric():
    method = ProphetKV(recomputeRatio=0.2)
    assert "recompute_ratio" not in method.method_metrics
    assert method.recomputeRatio == pytest.approx(0.2)


@pytest.fixture
def timing_clock(monkeypatch):
    from importlib import import_module
    module = import_module("methods.ProphetKV")
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(module.time, "perf_counter", lambda: clock.now)
    return module, clock


def test_prophet_full_generate_total_includes_prefill(timing_clock):
    _, clock = timing_clock
    runtime = object.__new__(_DenseProphetRuntime)
    runtime.torch = torch
    runtime.max_model_len = 32
    runtime.device = "cpu"
    runtime.layers = [SimpleNamespace(self_attn=SimpleNamespace(
        num_key_value_heads=1, head_dim=2))]
    runtime.embed_tokens = SimpleNamespace(weight=torch.zeros(1))

    def recompute(*args):
        clock.now += 5
        return [], torch.zeros(2)

    def decode(*args):
        clock.now += 2
        return "answer", 0.75, 2.0, 1, [7]

    runtime.recompute = recompute
    runtime.decode_from_cache = decode
    _, ttft, total, _, _ = runtime.full_generate([1, 2])
    assert ttft == pytest.approx(5.75)
    assert total == pytest.approx(7.0)


@pytest.mark.parametrize("path", ["no_match", "fallback"])
def test_prophet_full_paths_total_includes_online_preparation(timing_clock, monkeypatch, path):
    module, clock = timing_clock
    method = ProphetKV(maxNewTokens=1)

    def generate(ids):
        clock.now += 7
        return "answer", 2.0, 7.0, 1, [7]

    method._runtime = SimpleNamespace(full_generate=generate)
    if path == "no_match":
        original = module.ComposeInterleavedReuse

        def match(*args):
            clock.now += 5
            return original(*args)

        monkeypatch.setattr(module, "ComposeInterleavedReuse", match)
        result = method.Run([[1, 2]])[0]
    else:
        clock.now = 5
        result = method._fallback([1, 2], "empty_query_span", request_start=0.0)

    assert result.performance["ttft"] == pytest.approx(7.0)
    assert result.performance["total_time"] == pytest.approx(12.0)
    assert result.output_ids == [7]


def test_prophet_reuse_total_counts_each_online_stage_once(timing_clock, monkeypatch):
    module, clock = timing_clock
    method = ProphetKV(maxNewTokens=1)
    original_match = module.ComposeInterleavedReuse
    original_cat = torch.cat
    original_topk = module._stable_topk

    def match(*args):
        clock.now += 5
        return original_match(*args)

    def cat(*args, **kwargs):
        clock.now += 2
        return original_cat(*args, **kwargs)

    def score(*args, **kwargs):
        clock.now += 3
        return [torch.tensor([0.2, 0.7])]

    def select(*args):
        clock.now += 1
        return original_topk(*args)

    def recompute(*args):
        clock.now += 4
        return [], torch.zeros(2)

    def decode(*args):
        clock.now += 2
        return "answer", 0.5, 2.0, 1, [7]

    monkeypatch.setattr(module, "ComposeInterleavedReuse", match)
    monkeypatch.setattr(torch, "cat", cat)
    monkeypatch.setattr(module, "_stable_topk", select)
    method._runtime = SimpleNamespace(
        torch=torch, layers=[object()], score_all_layers=score,
        recompute_segments=recompute, decode_from_cache=decode)
    kv = torch.ones(1, 1, 2, 1)
    method._states = [module._CaseState([[1, 2]], [module._ChunkCache([1, 2], [(kv, kv)])])]

    result = method.Run([[1, 2, 3]])[0]

    # Match 5 + K/V stitching 4 + score 3 + select 1 + recompute 4 + decode 2.
    assert result.performance["ttft"] == pytest.approx(17.5)
    assert result.performance["total_time"] == pytest.approx(19.0)
    assert result.output_ids == [7]


def test_prophet_uses_each_task_budget_and_keeps_direct_call_default():
    method = ProphetKV(maxNewTokens=8)
    observed = []
    runtime = SimpleNamespace(max_new_tokens=8)

    def generate(ids):
        count = runtime.max_new_tokens
        observed.append(count)
        return "answer", 0.01, 0.02, count, [7] * count

    runtime.full_generate = generate
    method._runtime = runtime
    for budget in [2, 5]:
        result = _RunMethod(method, [[1]], [False], budget)[0]
        assert result.performance["num_output_tokens"] == budget
        assert len(result.output_ids) == budget
    method.Run([[1]])
    assert observed == [2, 5, 8]


@pytest.mark.parametrize("extra", [
    {"repetition_penalty": 10.0}, {"presence_penalty": 3.0},
    {"frequency_penalty": 2.0}, {"min_p": 1.0}, {"typical_p": 0.2},
])
def test_prophet_sampling_matches_transformers_policy(extra, monkeypatch):
    policy = {"mode": "override", "do_sample": True, "temperature": 1.0,
              "top_k": 0, "top_p": 1.0, **extra}
    runtime = object.__new__(_DenseProphetRuntime)
    runtime.torch, runtime.sampling_config = torch, policy
    reference = object.__new__(TransformersGenerator)
    reference._torch, reference.samplingConfig = torch, policy
    probabilities = []

    def select(probs, *args, **kwargs):
        probabilities.append(probs.clone())
        return probs.argmax().reshape(1)

    monkeypatch.setattr(torch, "multinomial", select)
    logits = torch.tensor([2.0, 1.5, 0.0])
    runtime._sample(logits.clone(), [0, 0])
    reference._SampleNextToken(logits.clone(), [0, 0], None)
    torch.testing.assert_close(probabilities[0], probabilities[1])


@pytest.mark.parametrize("penalty", [
    {"repetition_penalty": 2.0},
    {"presence_penalty": 3.0},
    {"frequency_penalty": 3.0},
])
def test_prophet_deterministic_override_applies_history_penalties(penalty):
    runtime = object.__new__(_DenseProphetRuntime)
    runtime.torch = torch
    runtime.sampling_config = {
        "mode": "override", "do_sample": False, "temperature": 0.0, **penalty,
    }
    logits = torch.tensor([10.0, 8.0])
    original = logits.clone()
    assert runtime._sample(logits, [0]) == 1
    torch.testing.assert_close(logits, original)


def test_prophet_canonical_greedy_ignores_raw_penalties():
    runtime = object.__new__(_DenseProphetRuntime)
    runtime.torch = torch
    runtime.sampling_config = {
        "mode": "greedy", "do_sample": False, "temperature": 0.0,
        "repetition_penalty": 2.0, "presence_penalty": 100.0,
    }
    assert runtime._sample(torch.tensor([10.0, 8.0]), [0]) == 0


def test_prophet_seed_is_reset_per_request_without_changing_global_rng():
    runtime = object.__new__(_DenseProphetRuntime)
    runtime.torch = torch
    runtime.sampling_config = {"mode": "override", "do_sample": True, "seed": 23, "top_k": 0}
    runtime.max_new_tokens = 4
    runtime.max_model_len = 32
    runtime.eos_ids = {0, 1}
    runtime.decode = lambda ids: str(ids)
    torch.manual_seed(1)
    before = torch.get_rng_state().clone()
    first = runtime.decode_from_cache([], torch.tensor([0.0, 0.0]), [0])[0]
    assert torch.equal(before, torch.get_rng_state())
    torch.manual_seed(42)
    second = runtime.decode_from_cache([], torch.tensor([0.0, 0.0]), [0])[0]
    assert first == second


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


def test_prophet_bounds_each_request_budget(context_limit):
    method = ProphetKV(maxNewTokens=8)
    runtime = SimpleNamespace()
    observed = []

    def generate(ids):
        observed.append(runtime.max_new_tokens)
        return "answer", 0.01, 0.02, runtime.max_new_tokens, [7] * runtime.max_new_tokens

    runtime.full_generate = generate
    method._runtime = runtime
    results = method.Run([[2, 3, 4], [2, 3, 4, 5]], maxNewTokens=8)
    assert observed == [2, 1]
    assert [r.performance["num_output_tokens"] for r in results] == [2, 1]


def test_context_limit_rejects_prompts_and_chunks_before_runtime(context_limit):
    method = ProphetKV(maxModelLen=99)
    assert method.maxModelLen == 5
    method._runtime = SimpleNamespace(
        full_generate=lambda *args: pytest.fail("overlong prompt reached runtime"),
        prefill_isolated=lambda *args: pytest.fail("overlong chunk reached runtime"),
    )
    with pytest.raises(ValueError, match="MaxModelLen=5"):
        method.Prepare([[[2] * 6]])
    with pytest.raises(ValueError, match="MaxModelLen=5"):
        method.Run([[2] * 5], maxNewTokens=8)
