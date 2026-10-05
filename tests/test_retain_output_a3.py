import hashlib
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from methods.A3Repo.backend.utils import recover_pre_rope_keys
from methods.A3Repo.Worker import A3Worker


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


def test_tiny_qwen_runtime_retains_native_output_kv_and_reuses_it(monkeypatch, tmp_path):
    monkeypatch.setenv("FLASHINFER_WORKSPACE_BASE", str(tmp_path))
    transformers = pytest.importorskip("transformers")
    from methods.A3Repo.backend.models.loader import model_classes
    from methods.A3Repo.Worker import A3Worker
    from helpers.backends.TransformersHelper import CacheLayerPairs

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
