from types import SimpleNamespace
import pytest
import torch

from core.Worker import _RunMethod
from methods.ProphetKV import ProphetKV, _DenseProphetRuntime
from helpers.backends.TransformersHelper import TransformersGenerator


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
