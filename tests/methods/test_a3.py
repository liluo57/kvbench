"""Tests for a3."""

import hashlib
import importlib
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from methods import A3Repo
from methods.A3Repo.backend.utils import recover_pre_rope_keys
from methods.A3Repo.Worker import A3Worker

ROOT = Path(__file__).resolve().parents[2]


def test_constructor_uses_project_python_without_loading_backend():
    subprocess.run([sys.executable, "-B", "-c", '''
import sys
from methods import A3Repo
method = A3Repo()
assert str(method.workerPython) == sys.executable
assert method.repoRoot.is_relative_to(__import__('pathlib').Path.cwd())
assert not any(name.startswith('methods.A3Repo.backend.') for name in sys.modules)
import torch
assert not torch.cuda.is_initialized()
'''], cwd=ROOT, check=True, capture_output=True, text=True)


@pytest.mark.parametrize("family,head_dim", [
    ("llama", 8), ("mistral", 8), ("mistral", None), ("mistral", 16),
    ("qwen2", 8), ("qwen3", 8),
])
@pytest.mark.native
def test_native_models_preserve_hf_classes_and_selective_recompute(monkeypatch, family, head_dim):
    pytest.importorskip("flashinfer")
    import transformers
    from transformers.cache_utils import DynamicCache

    from methods.A3Repo.backend.models.loader import model_classes
    from methods.A3Repo.Worker import A3Worker

    prefixes = {"llama": "Llama", "mistral": "Mistral", "qwen2": "Qwen2", "qwen3": "Qwen3"}
    prefix = prefixes[family]
    hf_cls = getattr(transformers, prefix + "ForCausalLM")
    original_forward = hf_cls.forward
    original_cache_init = DynamicCache.__init__
    config = getattr(transformers, prefix + "Config")(
        vocab_size=32, hidden_size=32, intermediate_size=64,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, sliding_window=None, head_dim=head_dim,
        tie_word_embeddings=True,
    )
    config._attn_implementation = "sdpa"
    runtime_cls, precompute_cls = model_classes(config)
    precompute = precompute_cls(config).eval()
    with torch.device("meta"):
        runtime = runtime_cls(config)
    A3Worker._share_model_tensors(precompute, runtime)
    runtime.eval()
    assert getattr(transformers, prefix + "ForCausalLM") is hf_cls
    assert hf_cls.forward is original_forward
    assert DynamicCache.__init__ is original_cache_init
    source = dict(precompute.named_parameters(remove_duplicate=False))
    assert all(source[name] is tensor for name, tensor in runtime.named_parameters(remove_duplicate=False))
    assert runtime.lm_head.weight is runtime.model.embed_tokens.weight
    assert not any(tensor.is_meta for tensor in runtime.parameters())

    # Exercise the upstream model state machine on CPU, substituting only
    # the CUDA attention/mask kernels with mathematically equivalent SDPA.
    def prefill(q, k, v, causal, packed_custom_mask):
        q, k, v = [x.transpose(0, 1).unsqueeze(0) for x in (q, k, v)]
        out = F.scaled_dot_product_attention(q, k, v,
            attn_mask=packed_custom_mask, is_causal=causal)
        return out.squeeze(0).transpose(0, 1).contiguous()

    def mask(q, k, indices, mode):
        assert mode is True
        allowed = torch.arange(k.shape[2]).unsqueeze(0) <= indices.unsqueeze(1)
        return allowed, allowed

    module = importlib.import_module(runtime.model.layers[0].self_attn.forward.__module__)
    monkeypatch.setattr(module.flashinfer, "single_prefill_with_kv_cache", prefill)
    monkeypatch.setattr(module, "create_flashinfer_mask", mask)
    ids = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]])
    full_config = {"reuse_config": None, "drop_config": None, "other_config": {"decode": False}}

    with torch.no_grad():
        expected = precompute(input_ids=ids, use_cache=True, return_dict=True)
        kv = torch.stack([torch.stack([
            layer.self_attn.hack_kv[k][0] for layer in precompute.model.layers
        ]) for k in (0, 1)])
        full = runtime(input_ids=ids, use_cache=True, return_dict=True, extra_config=full_config)
        torch.testing.assert_close(full.logits, expected.logits)

        def reuse_config(ratio):
            return {"reuse_config": {"cat_kv": kv.clone(), "check": None,
                "fake_q": None, "mask": None, "recomp_ratio": ratio,
                "reuse": "debug", "causal": True}, "drop_config": None,
                "other_config": {"decode": False,
                    "data_params": {"prefix_len": 0, "last_len": 2, "doc_start_len": 0}}}

        all_rows = runtime(input_ids=ids, use_cache=True, return_dict=True,
                           extra_config=reuse_config(1.0))
        torch.testing.assert_close(all_rows.logits[:, -1], expected.logits[:, -1])
        selective_config = reuse_config(0.5)
        selective = runtime(input_ids=ids, use_cache=True, return_dict=True,
                            extra_config=selective_config)
        indices = selective_config["reuse_config"]["imp_indices"]
        assert indices.numel() == 5  # Half of six cached rows, plus two fresh rows.
        assert indices[-2:].tolist() == [6, 7]
        assert selective.logits.shape == (1, 5, 32)
        assert all(key.shape[-2] == 8 for key, _ in selective.past_key_values)
        selective_config["reuse_config"].update(check=None, cat_kv=None)
        selective_config["other_config"]["decode"] = True
        decoded = runtime(input_ids=torch.tensor([[10]]), position_ids=torch.tensor([[8]]),
                          past_key_values=selective.past_key_values, use_cache=True,
                          return_dict=True, extra_config=selective_config)
        assert decoded.logits.shape == (1, 1, 32)
        assert all(key.shape[-2] == 9 for key, _ in decoded.past_key_values)


def test_run_routes_native_ids_and_task_generation_budget(monkeypatch):
    method = A3Repo(maxNewTokens=8)
    method._chunks = [[[1, 2], [3, 4]]]
    method._wire_chunks = method._chunks
    requests = []

    def request(payload):
        requests.append(payload)
        return {"ok": True, "text": "answer", "n_input": 6, "num_tokens": 1}

    monkeypatch.setattr(method, "_request", request)
    method.Run([[1, 2, 3, 4, 5, 6]], maxNewTokens=3)
    assert requests[-1]["op"] == "reuse"
    assert requests[-1]["suffix_ids"] == [5, 6]
    assert requests[-1]["max_new_tokens"] == 3
    method.Run([[7, 1, 2, 3, 4, 5]])
    assert requests[-1]["op"] == "full"
    assert requests[-1]["input_ids"] == [7, 1, 2, 3, 4, 5]
    assert requests[-1]["max_new_tokens"] == 8


def test_fully_cached_prompt_keeps_a_fresh_query(monkeypatch):
    method = A3Repo()
    requests = []
    monkeypatch.setattr(method, "_request", lambda payload: requests.append(payload) or {
        "ok": True, "text": "answer", "num_tokens": 1,
    })
    # A shuffle task may prepare every chunk, including the question. Keep
    # the prompt intact while removing that final chunk from cache reuse.
    method._chunks = [[[1, 2], [3, 4], [5, 6]]]
    method._wire_chunks = method._chunks
    method.Run([[1, 2, 5, 6, 3, 4]])
    assert requests[-1]["op"] == "reuse"
    assert requests[-1]["suffix_ids"] == [3, 4]
    assert len(requests[-1]["chunks"]) == 2
    # One fully cached chunk has no reusable prefix left after reserving the
    # query, so it uses the existing full-prefill path.
    method._chunks = [[[1, 2]]]
    method._wire_chunks = method._chunks
    method.Run([[1, 2]])
    assert requests[-1]["op"] == "full"
    assert requests[-1]["input_ids"] == [1, 2]


def test_collect_finishes_gpu_work_before_prepare_returns():
    from types import SimpleNamespace

    from methods.A3Repo.Worker import A3Worker
    worker = A3Worker.__new__(A3Worker)
    events = []
    worker.chunk_kv = {}
    worker.torch = SimpleNamespace(cuda=SimpleNamespace(
        synchronize=lambda: events.append("synchronize")))

    def collect(text, ids=None):
        events.append((text, ids))
        worker.chunk_kv[text] = []

    worker._collect_one = collect
    assert worker.collect(["a", "b"], [[1], [2]]) == {"ok": True, "n_chunks": 2}
    assert events == [("a", [1]), ("b", [2]), "synchronize"]


def test_first_token_time_waits_for_the_token(monkeypatch):
    from types import SimpleNamespace

    import torch

    from methods.A3Repo.backend import utils
    events = []
    original_item = torch.Tensor.item

    def item(tensor):
        value = original_item(tensor)
        events.append(("token_ready", value))
        return value

    monkeypatch.setattr(torch.Tensor, "item", item)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor: tensor)
    monkeypatch.setattr(utils.time, "time", lambda: events.append("timestamp") or 100.0)
    model = lambda **kwargs: SimpleNamespace(
        logits=torch.tensor([[[0., 1., 2., 3.]]]), past_key_values=None)
    state = {"input_ids": torch.tensor([[1]]), "position_ids": torch.tensor([[0]]),
             "past_key_values": None}
    _, token, stop, first_time = utils.decode_step(None, model, state, {}, [3], 0)
    assert events[:2] == [("token_ready", 3), "timestamp"]
    assert stop and first_time == 100.0


def test_decode_returns_native_ids_including_terminal_stop(monkeypatch):
    from methods.A3Repo.backend import utils
    tokens = iter([7, 99, 8, 2])

    def step(**kwargs):
        token = next(tokens)
        return kwargs["input"], torch.tensor(token), token == 2, utils.time.time()

    monkeypatch.setattr(utils, "decode_step", step)
    text, _, _, ids = utils.decode(
        None, None, SimpleNamespace(decode=lambda ids, **kw: "display"), {},
        [2], 8, {"reuse_config": None, "other_config": {}},
        return_token_ids=True,
    )
    assert text == "display"
    assert ids == [7, 99, 8, 2]


def test_vanilla_can_report_generated_count_including_terminal_stop(monkeypatch):
    from contextlib import nullcontext

    from methods.A3Repo.backend import utils

    tokens = iter([7, 2])
    decoded = []

    class Token:
        def __init__(self, value):
            self.value = value

        def item(self):
            return self.value

    def step(**kwargs):
        token = next(tokens)
        return kwargs["input"], Token(token), token == 2, 1.0

    monkeypatch.setattr(utils.torch, "no_grad", nullcontext)
    monkeypatch.setattr(utils, "decode_step", step)

    def decode(ids, **_kwargs):
        decoded.extend(ids)
        return "visible"

    text, _, _, generated = utils.vanilla(
        None, None, SimpleNamespace(decode=decode), {}, [2], 4, {},
        return_token_ids=True,
    )
    assert text == "visible"
    assert decoded == [7]
    assert generated == [7, 2]


def test_a3_native_output_survives_next_agent_prompt(monkeypatch):
    from core.Result import Result
    from core.Workflow import ActionResult
    from helpers.backends import ModelAdapter
    from workflow import (
        AgentSpec,
        MultiAgentFullConnectionInput,
        MultiAgentFullConnectionWorkflow,
    )
    monkeypatch.setattr(ModelAdapter, "_tokenizer", lambda path: SimpleNamespace(
        encode=lambda text, **kwargs: [ord(c) for c in text]))
    workflow = MultiAgentFullConnectionWorkflow(1, MultiAgentFullConnectionInput(
        "TASK", [AgentSpec("A", "A {task}"), AgentSpec("B", "B {task}")],
        chatTemplate=False, modelPath="synthetic"))
    prepare = workflow.next()[0]
    workflow.observe([ActionResult(1, Result(), prepare.tag)])
    action = workflow.next()[0]
    method = A3Repo()
    monkeypatch.setattr(method, "_request", lambda payload: {
        "text": "unrelated display", "output_ids": [900, 999], "num_tokens": 2})
    result = method.Run([action.data])[0]
    workflow.observe([ActionResult(1, result, action.tag)])
    next_prompt = workflow.next()[0].data
    assert any(next_prompt[i:i + 2] == [900, 999] for i in range(len(next_prompt)))


def test_a3_total_time_includes_adapter_ipc_and_worker_preparation(monkeypatch):
    module = importlib.import_module("methods.A3Repo.Method")
    method = A3Repo()
    monkeypatch.setattr(module.time, "perf_counter", lambda: 12.0)
    result = method._result({"text": "answer", "output_ids": [7], "ttft": 0.1,
                             "generation_start": 11.0, "total_time": 0.2},
                            full=False, onlineStart=10.0)
    assert result.performance["ttft"] == pytest.approx(1.1)
    assert result.performance["total_time"] == pytest.approx(2.0)


def test_a3_resolved_sampling_runs_through_decode_with_history_seed_and_eos(monkeypatch):
    from methods.A3Repo.backend import utils
    from methods.A3Repo.Worker import A3Worker

    worker = A3Worker.__new__(A3Worker)
    worker.tokenizer = SimpleNamespace(eos_token_id=8, bos_token_id=9)
    worker.sampling_config = {
        "mode": "override", "do_sample": False, "temperature": 0.0,
        "top_p": 1.0, "top_k": 0, "min_p": 0.0, "typical_p": 1.0,
        "repetition_penalty": 2.0, "presence_penalty": 0.0,
        "frequency_penalty": 0.0, "seed": 17, "eos_token_id": 3,
    }
    policy = worker._sampling_policy()
    assert policy.do_sample is False and policy.repetition_penalty == 2.0

    seen = []
    seen_positions = []
    class Model:
        def __call__(self, **kwargs):
            seen.append(kwargs["input_ids"].tolist())
            seen_positions.append(kwargs["position_ids"].tolist())
            # Prompt includes token 1, so repetition penalty changes the
            # greedy choice from token 1 to token 2. Token 3 terminates next.
            logits = torch.tensor([[[0., 10., 8., 0.]]]) if len(seen) == 1 else torch.tensor([[[0., 0., 10., 8.]]])
            return SimpleNamespace(logits=logits, past_key_values=None)

    state = {"input_ids": torch.tensor([[1]]), "position_ids": torch.tensor([[0]]),
             "past_key_values": None}
    monkeypatch.setattr(utils.time, "time", lambda: 5.0)
    text, _, _, raw_ids = utils.decode(
        None, Model(), SimpleNamespace(decode=lambda ids, **kw: str(ids)), state,
        worker._stop_ids(), 5,
        {"reuse_config": None, "other_config": {"decode": False}},
        return_token_ids=True, sampling=policy,
    )
    assert raw_ids == [2, 3]
    assert text == "[2]"
    assert seen_positions == [[[0]], [[1]]]
    assert worker._stop_ids() == [3, 9]
    worker.sampling_config["eos_token_id"] = []
    assert worker._stop_ids() == [9]
    assert utils.time.time() == 5.0


def test_a3_rejects_unresolved_sampling_fields():
    from methods.A3Repo.Worker import A3Worker
    worker = A3Worker.__new__(A3Worker)
    worker.sampling_config = {"mode": "override", "beam_width": 4}
    with pytest.raises(ValueError, match="beam_width"):
        worker._sampling_policy()


def test_a3_seeded_sampling_generator_is_request_local_and_advances_across_steps(monkeypatch):
    from helpers.backends.CacheBackend.Sampling import SamplingConfig
    from methods.A3Repo.backend import utils

    original = utils.select_next_token
    generators = []
    def record(logits, sampling, history=(), generator=None):
        generators.append(generator)
        return original(logits, sampling, history, generator)
    monkeypatch.setattr(utils, "select_next_token", record)
    class Model:
        def __call__(self, **kwargs):
            return SimpleNamespace(logits=torch.zeros((1, 1, 16)), past_key_values=None)
    def run():
        state = {"input_ids": torch.tensor([[1, 2]]), "position_ids": torch.tensor([[0, 1]]),
                 "past_key_values": None}
        return utils.decode(None, Model(), SimpleNamespace(decode=lambda ids, **kw: ""), state,
            [], 4, {"reuse_config": None, "other_config": {}}, return_token_ids=True,
            sampling=SamplingConfig(do_sample=True, seed=4321))[3]
    first = run()
    first_generators = generators[:]
    generators.clear()
    second = run()
    assert first == second
    expected_generator = torch.Generator().manual_seed(4321)
    expected = [int(torch.multinomial(torch.full((16,), 1 / 16), 1,
                                      generator=expected_generator)) for _ in range(4)]
    assert first == expected
    assert len(first_generators) == 4 and len(set(map(id, first_generators))) == 1
    assert first_generators[0] is not generators[0]


class _StaticRotary:
    rope_type = "default"
    attention_scaling = 1.7

    def __call__(self, x, position_ids):
        phase = position_ids.to(dtype=torch.float32).unsqueeze(-1) * torch.tensor(
            [0.2, 0.35], device=x.device
        )
        phase = torch.cat((phase, phase), dim=-1)
        return phase.cos().to(x.dtype) * self.attention_scaling, phase.sin().to(x.dtype) * self.attention_scaling


def test_recover_pre_rope_keys_inverts_position_rotation_and_amplitude():
    raw = torch.randn(1, 2, 3, 4)
    positions = torch.tensor([[5, 6, 7]])
    rotary = _StaticRotary()
    cos, sin = rotary(raw, positions)
    from helpers.backends.CacheBackend.RoPE import apply_rotary

    rotated = apply_rotary(raw, cos, sin)
    recovered = recover_pre_rope_keys(rotated, rotary, positions)
    torch.testing.assert_close(recovered, raw)


def test_retain_output_registers_only_generated_native_kv_rows():
    rotary = _StaticRotary()
    attention = SimpleNamespace(rotary_emb=rotary)
    worker = object.__new__(A3Worker)
    worker.torch = torch
    worker.model = SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(self_attn=attention)]))
    raw = torch.randn(1, 2, 6, 4)
    positions = torch.arange(6).unsqueeze(0)
    cos, sin = rotary(raw, positions)
    from helpers.backends.CacheBackend.RoPE import apply_rotary

    key = apply_rotary(raw, cos, sin)
    value = torch.randn_like(key)
    worker.chunk_ids = {}
    worker.chunk_kv = {}

    worker._retain_output([12, 13], ((key, value),), prompt_len=4)

    name = "ids:" + hashlib.sha256(
        b"".join(token.to_bytes(4, "little", signed=True) for token in (12, 13))
    ).hexdigest()
    assert worker.chunk_ids[name] == [12, 13]
    torch.testing.assert_close(worker.chunk_kv[name][0][0], raw[0, :, 4:6])
    torch.testing.assert_close(worker.chunk_kv[name][0][1], value[0, :, 4:6])


def test_retain_output_rejects_dynamic_rope():
    rotary = _StaticRotary()
    rotary.rope_type = "dynamic_ntk"
    with pytest.raises(RuntimeError, match="dynamic RoPE"):
        recover_pre_rope_keys(torch.ones(1, 1, 1, 2), rotary, torch.zeros(1, 1, dtype=torch.long))


@pytest.mark.native
def test_tiny_qwen_runtime_retains_native_output_kv_and_reuses_it(monkeypatch, tmp_path):
    monkeypatch.setenv("FLASHINFER_WORKSPACE_BASE", str(tmp_path))
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("flashinfer")
    from helpers.backends.TransformersHelper import CacheLayerPairs
    from methods.A3Repo.backend.models.loader import model_classes
    from methods.A3Repo.Worker import A3Worker

    config = transformers.Qwen2Config(
        vocab_size=32, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, tie_word_embeddings=True,
    )
    config._attn_implementation = "sdpa"
    runtime_cls, precompute_cls = model_classes(config)
    precompute = precompute_cls(config).eval()
    runtime = runtime_cls(config).eval()
    A3Worker._share_model_tensors(precompute, runtime)

    def prefill(q, k, v, causal, packed_custom_mask):
        q, k, v = [x.transpose(0, 1).unsqueeze(0) for x in (q, k, v)]
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=packed_custom_mask, is_causal=causal
        )
        return out.squeeze(0).transpose(0, 1).contiguous()

    def mask(q, k, indices, mode):
        assert mode is True
        allowed = torch.arange(k.shape[2]).unsqueeze(0) <= indices.unsqueeze(1)
        return allowed, allowed

    module = importlib.import_module(runtime.model.layers[0].self_attn.forward.__module__)
    monkeypatch.setattr(module.flashinfer, "single_prefill_with_kv_cache", prefill)
    monkeypatch.setattr(module, "create_flashinfer_mask", mask)

    class Tokenizer:
        eos_token_id = 31
        bos_token_id = 1

        @staticmethod
        def decode(ids, skip_special_tokens=True):
            return ",".join(map(str, ids))

    worker = object.__new__(A3Worker)
    worker.torch = torch
    worker.model = runtime
    worker.precompute_model = precompute
    worker.tokenizer = Tokenizer()
    worker.args = SimpleNamespace(
        model="tiny", reuse_method="debug", recomp_ratio=0.15,
        max_model_len=64, max_new_tokens=2,
    )
    worker.sampling_config = {}
    worker.chunk_ids = {}
    worker.chunk_kv = {}
    utils = importlib.import_module("methods.A3Repo.backend.utils")
    monkeypatch.setattr(utils, "select_next_token", lambda *a, **k: 7)

    prompt = [2, 3, 4]
    with torch.no_grad():
        decoded = worker._decode(prompt, worker._full_config(), max_new_tokens=2, retain_output=True)
        output_ids, cache = decoded[5], decoded[6]
        assert output_ids == [7, 7]
        cache_pairs = list(CacheLayerPairs(cache))
        assert all(key.shape[-2] == len(prompt) + len(output_ids) for key, _ in cache_pairs)
        worker._retain_output(output_ids, cache, len(prompt))

        joined = torch.tensor([prompt + output_ids])
        precompute(input_ids=joined, use_cache=True, return_dict=True, past_key_values=None)
        for index, layer in enumerate(precompute.model.layers):
            expected_key = layer.self_attn.hack_kv[0][0, :, -len(output_ids):, :]
            torch.testing.assert_close(worker.chunk_kv[
                "ids:" + hashlib.sha256(b"".join(t.to_bytes(4, "little", signed=True) for t in output_ids)).hexdigest()
            ][index][0], expected_key, atol=3e-3, rtol=3e-3)

        retained_name = "ids:" + hashlib.sha256(
            b"".join(t.to_bytes(4, "little", signed=True) for t in output_ids)
        ).hexdigest()
        reused = worker.reuse([retained_name], "", suffix_ids=[8, 9], max_new_tokens=1)
        assert reused["a3_debug"]["doc_tokens"] == len(output_ids)
        assert reused["algorithm"] == "a3_reuse"


def test_method_registers_retained_output_ids_for_later_composition(monkeypatch):
    from methods.A3Repo.Method import A3Repo

    method = object.__new__(A3Repo)
    method.maxNewTokens = 1
    method.maxModelLen = 32
    method.recompRatio = 0.15
    method.repoRoot = Path("/tmp/a3")
    method._chunks = [[]]
    method._wire_chunks = [[]]
    requests = []

    def request(payload):
        requests.append(payload)
        return {"text": "answer", "output_ids": [17, 18], "ttft": 0.01,
                "total_time": 0.02, "num_tokens": 2, "n_input": 2}

    monkeypatch.setattr(method, "_request", request)
    result = method.Run([[4, 5]], retainOutput=[True])[0]
    assert requests[0]["retain_output"] is True
    assert result.output_ids == [17, 18]
    assert method._chunks[0][-1] == [17, 18]


def test_a3_worker_bounds_decode_and_rejects_overlong_collection(context_limit, monkeypatch):
    from methods.A3Repo.Worker import A3Worker

    utils = importlib.import_module("methods.A3Repo.backend.utils")
    observed = []

    def decode(*args, **kwargs):
        budget = args[5]
        observed.append(budget)
        return "answer", 0.01, 0.02, [7] * budget

    monkeypatch.setattr(utils, "decode", decode)
    worker = object.__new__(A3Worker)
    worker.sampling_config = {"mode": "greedy", "do_sample": False, "temperature": 0.0}
    worker.args = SimpleNamespace(max_model_len=5, max_new_tokens=8)
    worker._sampling_args = lambda: SimpleNamespace()
    worker.torch = SimpleNamespace(
        tensor=lambda ids, **kwargs: torch.tensor(ids),
        arange=lambda length, **kwargs: torch.arange(length), long=torch.long,
    )
    worker.model = None
    worker.tokenizer = SimpleNamespace(eos_token_id=1, bos_token_id=None)
    worker.chunk_kv = {}
    assert worker._decode([2, 3, 4], {}, 8)[3] == 2
    assert observed == [2]
    with pytest.raises(ValueError, match="max_model_len=5"):
        worker._decode([2] * 5, {})
    with pytest.raises(ValueError, match="max_model_len=5"):
        worker._collect_one("too long", [2] * 6)


def test_context_limit_rejects_prompt_before_worker_ipc(context_limit, monkeypatch):
    method = A3Repo(maxModelLen=99)
    assert method.maxModelLen == 5
    monkeypatch.setattr(method, "_request", lambda *args: pytest.fail("overlong prompt reached worker"))
    with pytest.raises(ValueError, match="MaxModelLen=5"):
        method.Run([[2] * 5], maxNewTokens=8)
