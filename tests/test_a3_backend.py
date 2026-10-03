import importlib
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]


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


def test_setup_uses_selected_project_python_and_no_checkout(monkeypatch):
    spec = importlib.util.spec_from_file_location("kvbench_setup_a3", ROOT / "setup.py")
    setup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup)
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(setup.subprocess, "run", run)
    monkeypatch.setattr(setup.shutil, "which", lambda name: name)
    setup.setup_a3(SimpleNamespace(python="/project/bin/python", rebuild=True,
        offline=True, no_index=True, find_links=["/wheels"]))
    install = commands[0]
    assert install[:3] == ["uv", "pip", "install"]
    assert install[install.index("--python") + 1] == "/project/bin/python"
    assert str(ROOT / "requirement.txt") in install
    assert str(ROOT / "methods/A3Repo/requirements.txt") in install
    assert "--offline" in install and "--no-index" in install
    assert not any("venv" in command or "git" in command for command in commands)


@pytest.mark.parametrize("family", ["llama", "mistral", "qwen2", "qwen3"])
def test_native_models_preserve_hf_classes_and_selective_recompute(monkeypatch, family):
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
        max_position_embeddings=128, sliding_window=None, head_dim=8,
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
    from methods import A3Repo
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
    from methods import A3Repo
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
