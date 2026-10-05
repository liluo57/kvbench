"""Tests for cacheblend repo."""

import importlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig

from helpers.backends.Prompt import ComposeInterleavedReuse
from helpers.backends.Tokenization import OutputFragmentIds
from methods import CacheblendRepo, NaiveCacheblendRepo
from methods.Cacheblend.backend.cacheblend_ops import (
    RetainedOutputCollector,
    capture_kv,
    indexed_causal_mask,
    register_retained_output,
)
from methods.Cacheblend.Worker import CacheBlendWorker, _cache_hit_token_count


def test_native_capture_retains_decode_tokens_without_collecting_prefill():
    attention = SimpleNamespace(hack_kv=None)
    collector = RetainedOutputCollector([SimpleNamespace(self_attn=attention)])
    metadata = {"collect": False, "retain_collector": collector}
    key = torch.tensor([[1.0, 2.0]])
    value = torch.tensor([[3.0, 4.0]])

    capture_kv(attention, key, value, 0, metadata)
    assert attention.hack_kv is None
    assert collector.finish() is None

    capture_kv(attention, key, value, -1, metadata)
    key.add_(10)
    value.add_(10)
    capture_kv(attention, key, value, -1, metadata)

    retained = collector.finish()
    assert metadata["collect"] is False
    assert torch.equal(retained[0][0], torch.tensor([[1.0, 2.0], [11.0, 12.0]]))
    assert torch.equal(retained[0][1], torch.tensor([[3.0, 4.0], [13.0, 14.0]]))


def test_native_collect_owns_kv_and_does_not_capture_ordinary_decode():
    attention = SimpleNamespace(hack_kv=None)
    metadata = {"collect": True}
    key = torch.tensor([[1.0]])
    value = torch.tensor([[2.0]])
    capture_kv(attention, key, value, 0, metadata)
    key.zero_()
    value.zero_()
    assert attention.hack_kv[0].item() == 1
    assert attention.hack_kv[1].item() == 2

    metadata["collect"] = False
    capture_kv(attention, key, value, -1, metadata)
    assert attention.hack_kv[0].item() == 1


def test_retained_fragment_preserves_interior_specials_and_hits_exact_cache_key():
    output_ids = OutputFragmentIds([31, 99, 32, 2], terminalStopIds=[2])
    attention = SimpleNamespace(hack_kv=None)
    collector = RetainedOutputCollector([SimpleNamespace(self_attn=attention)])
    metadata = {"collect": False, "retain_collector": collector}
    for value in (1.0, 2.0, 3.0):
        capture_kv(
            attention,
            torch.tensor([[value]]),
            torch.tensor([[value + 10]]),
            -1,
            metadata,
        )
    retained = collector.finish()
    chunk_ids = {}
    chunk_kv = {}

    retained_tokens = register_retained_output(
        chunk_ids, chunk_kv, output_ids, retained
    )

    assert retained_tokens == len(output_ids) == 3
    assert chunk_ids[tuple(output_ids)] == [31, 99, 32]
    assert chunk_kv[tuple(output_ids)][0][0].flatten().tolist() == [1.0, 2.0, 3.0]
    parts = ComposeInterleavedReuse([output_ids], [7, *output_ids, 8])
    matched = [piece for index, piece in parts if index is not None]
    assert matched == [output_ids]
    assert tuple(matched[0]) in chunk_kv
    worker = object.__new__(CacheBlendWorker)
    worker._chunkIds = chunk_ids
    worker._chunkKv = chunk_kv
    segments, full_ids, sample_kv = worker._ResolveSegments(
        [[True, output_ids], [False, [77]]]
    )
    assert full_ids == [31, 99, 32, 77]
    assert [segment[0] for segment in segments] == [True, False]
    assert sample_kv is chunk_kv[tuple(output_ids)]


def test_retained_fragment_rejects_incomplete_kv_capture_without_truncating_ids():
    chunk_ids = {}
    chunk_kv = {}
    incomplete = [[torch.ones((2, 1)), torch.ones((2, 1))]]

    with pytest.raises(RuntimeError, match="fewer KV rows than native output IDs"):
        register_retained_output(chunk_ids, chunk_kv, [31, 32, 33], incomplete)

    assert chunk_ids == {}
    assert chunk_kv == {}


def test_sparse_mask_uses_absolute_positions_for_interleaved_repair():
    metadata = {
        "imp_indices": torch.tensor([1, 5, 12]),
        "org_seq_len": 13,
        "kv_cache_dtype": torch.float32,
        "_num_kv_heads": 2,
        "_num_queries_per_kv": 4,
    }
    mask = indexed_causal_mask(metadata, lambda: None)
    assert mask.shape == (1, 2, 4, 3, 13)
    assert mask.stride(-2) == 16
    for row, position in enumerate([1, 5, 12]):
        assert torch.all(mask[..., row, :position + 1] == 0)
        assert torch.all(mask[..., row, position + 1:] == torch.finfo(torch.float32).min)


def test_cacheblend_reuse_counts_cached_terminal_suffix_and_repaired_rows():
    # The final cached segment may be replayed as CacheBlend's suffix, and any
    # cached segment may be selected by its repair top-k. Both remain hits.
    segments = [
        (True, [1, 2], None, 0, 2),
        (False, [3], None, 2, 3),
        (True, [4, 5, 6], None, 3, 6),
    ]
    assert _cache_hit_token_count(segments) == 5


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return list(range(len(text)))


def _worker():
    worker = CacheBlendWorker.__new__(CacheBlendWorker)
    worker.args = SimpleNamespace(max_model_len=16)
    worker._torch = torch
    worker.layers = [SimpleNamespace(), SimpleNamespace()]
    worker.engine = SimpleNamespace(
        model=SimpleNamespace(old_kvs=[[None, None], [None, None]])
    )
    worker.tokenizer = _Tokenizer()
    worker._assembledKv = []
    worker._assembledCapacity = 0
    kv = [
        [torch.ones(2, 3), torch.ones(2, 3)],
        [torch.ones(2, 3), torch.ones(2, 3)],
    ]
    worker._chunkIds = {(1, 2): [1, 2]}
    worker._chunkKv = {(1, 2): kv}
    return worker


def test_reserve_allocates_and_reuses_device_local_assembly_buffers():
    worker = _worker()

    report = worker.Reserve([[[True, [1, 2]], [False, [3, 4, 5]]]])

    assert report == {"ok": True, "capacity": 5, "requested": 5}
    assert len(worker._assembledKv) == 2
    assert worker._assembledKv[0][0].shape == (5, 3)
    assert worker._assembledKv[0][0].device == worker._chunkKv[(1, 2)][0][0].device
    first_buffer = worker._assembledKv[0][0]

    worker.Reserve([[[True, [1, 2]]]])
    assert worker._assembledKv[0][0] is first_buffer


def test_reserve_rejects_prompt_beyond_model_limit():
    worker = _worker()
    worker.args.max_model_len = 4

    try:
        worker.Reserve([[[True, [1, 2]], [False, [3, 4, 5]]]])
    except ValueError as exc:
        assert "exceeds max_model_len" in str(exc)
    else:
        raise AssertionError("oversized reserve unexpectedly succeeded")


def test_indexed_causal_mask_keeps_aligned_stride_for_single_selected_token():
    keyLen = 13
    cfm = {
        "imp_indices": torch.tensor([5]),
        "org_seq_len": keyLen,
        "kv_cache_dtype": torch.float32,
        "_num_kv_heads": 8,
        "_num_queries_per_kv": 4,
    }

    mask = indexed_causal_mask(cfm, lambda: None)

    assert mask.shape == (1, 8, 4, 1, keyLen)
    assert mask.stride(-2) == 16
    assert torch.all(mask[..., 0, :6] == 0)
    assert torch.all(mask[..., 0, 6:] == torch.finfo(torch.float32).min)


_helper = importlib.import_module("methods.Cacheblend.Worker")


_method = importlib.import_module("methods.Cacheblend.Method")


def test_worker_defaults_to_main_project_interpreter(monkeypatch):
    monkeypatch.setattr(_method, "Get", lambda *args: {})
    assert CacheblendRepo().workerPython == Path(sys.executable)


def test_worker_explicit_interpreter_override_resolves_from_project(monkeypatch):
    monkeypatch.setattr(_method, "Get", lambda *args: {"Python": "custom/bin/python"})
    root = Path(_method.__file__).resolve().parents[2]
    assert CacheblendRepo().workerPython == root / "custom/bin/python"


class _BrokenStream:
    def close(self):
        raise BrokenPipeError("closed peer")


class _ClosedProcess:
    def __init__(self):
        self.stdin = _BrokenStream()
        self.stdout = _BrokenStream()
        self.stderr = _BrokenStream()
        self.killed = False

    def poll(self):
        return 1

    def wait(self, timeout=None):
        return 1

    def kill(self):
        self.killed = True


def test_modelscope_consolidated_checkpoint_is_excluded_from_cacheblend_view(
    tmp_path, monkeypatch
):
    source = tmp_path / "model"
    source.mkdir()
    for name in (
        "config.json",
        "model.safetensors.index.json",
        "model-00001-of-00003.safetensors",
        "tokenizer.json",
    ):
        (source / name).write_text("placeholder")
    (source / "consolidated.safetensors").write_text("incompatible")
    staging_root = tmp_path / "staging"
    monkeypatch.setattr(_helper.tempfile, "gettempdir", lambda: str(staging_root))

    staged = Path(_helper._prepare_model_for_cacheblend(str(source)))

    assert staged != source
    assert not (staged / "consolidated.safetensors").exists()
    assert (staged / "model-00001-of-00003.safetensors").is_symlink()
    assert (staged / "config.json").resolve() == source / "config.json"


def test_close_suppresses_broken_pipe_from_dead_helper():
    method = CacheblendRepo()
    proc = _ClosedProcess()
    method._proc = proc
    method.Close()
    assert method._proc is None
    assert proc.stdin is None
    assert proc.stdout is None
    assert proc.stderr is None


def test_stderr_tail_uses_already_drained_lines():
    method = CacheblendRepo()
    method._stderrTail.extend(["first\n", "CUDA out of memory\n"])
    assert method._StderrTail().endswith("CUDA out of memory\n")


def test_run_reserves_gpu_assembly_buffer_before_fusing_batch():
    method = CacheblendRepo()
    method._chunks = [[[1]], [[2]]]
    requests = []

    def request(payload):
        requests.append(payload)
        if payload["op"] == "reserve":
            return {"ok": True, "capacity": 2}
        return {
            "ok": True,
            "text": "answer",
            "ttft": 0.1,
            "num_tokens": 1,
            "total_time": 0.2,
            "n_input": 2,
            "reuse_ratio": 0.5,
        }

    method._Request = request
    results = method.Run([[1, 3], [2, 4]])

    assert [item["op"] for item in requests] == ["reserve", "fuse", "fuse"]
    assert requests[0]["parts_batch"] == [
        [[True, [1]], [False, [3]]],
        [[True, [2]], [False, [4]]],
    ]
    assert [result.output for result in results] == ["answer", "answer"]


@pytest.fixture
def llama_backend(monkeypatch):
    backend = Path(__file__).resolve().parents[2] / "methods/Cacheblend/backend"
    monkeypatch.syspath_prepend(str(backend))
    pytest.importorskip("kvbench_cacheblend._C")
    # Import the loader first to resolve its Llama/Llava registry dependency.
    importlib.import_module("kvbench_cacheblend.model_executor.model_loader")
    llama = importlib.import_module("kvbench_cacheblend.model_executor.models.llama")
    vocab = importlib.import_module("kvbench_cacheblend.model_executor.layers.vocab_parallel_embedding")
    return llama, vocab


def make_model(llama_backend, monkeypatch, tp_size, rank, with_lora, tied, family="llama"):
    llama, vocab = llama_backend
    monkeypatch.setattr(vocab, "get_tensor_model_parallel_world_size", lambda: tp_size)
    monkeypatch.setattr(vocab, "get_tensor_model_parallel_rank", lambda: rank)
    config_class = LlamaConfig
    model_class = llama.LlamaForCausalLM
    if family == "qwen3":
        from transformers import Qwen3Config
        config_class = Qwen3Config
        model_class = importlib.import_module(
            "kvbench_cacheblend.model_executor.models.qwen3").Qwen3ForCausalLM
    config = config_class(vocab_size=515, hidden_size=8, intermediate_size=16,
                          num_hidden_layers=0, num_attention_heads=2,
                          num_key_value_heads=2, tie_word_embeddings=tied)
    lora = SimpleNamespace(lora_extra_vocab_size=256, max_loras=2,
                           lora_vocab_padding_size=256) if with_lora else None
    model = model_class(config, lora_config=lora)
    with torch.no_grad():
        model.model.embed_tokens.weight.fill_(-99)
        if hasattr(model, "lm_head"):
            model.lm_head.weight.fill_(-99)
    return model


def check_shard(module, source):
    expected = source[module.vocab_start_index:module.vocab_end_index]
    torch.testing.assert_close(module.weight[:len(expected)], expected)
    assert torch.all(module.weight[len(expected):] == -99)


@pytest.mark.parametrize("tp_size,rank", [(1, 0), (2, 0), (2, 1)])
@pytest.mark.parametrize("with_lora", [False, True])
@pytest.mark.parametrize("checkpoint", ["embedding_only", "head_first", "head_last"])
@pytest.mark.native
def test_tied_checkpoint_loads_each_vocab_shard(llama_backend, monkeypatch,
                                               tp_size, rank, with_lora, checkpoint):
    model = make_model(llama_backend, monkeypatch, tp_size, rank, with_lora, True)
    source = torch.arange(515 * 8, dtype=torch.float32).reshape(515, 8)
    weights = [("model.embed_tokens.weight", source)]
    redundant_head = ("lm_head.weight", torch.full_like(source, -7))
    if checkpoint == "head_first":
        weights.insert(0, redundant_head)
    elif checkpoint == "head_last":
        weights.append(redundant_head)

    model.load_weights(iter(weights))

    check_shard(model.model.embed_tokens, source)
    check_shard(model.lm_head, source)
    if with_lora and tp_size == 2:
        assert model.model.embed_tokens.weight.shape != model.lm_head.weight.shape


@pytest.mark.parametrize("tp_size,rank", [(1, 0), (2, 0), (2, 1)])
@pytest.mark.parametrize("with_lora", [False, True])
@pytest.mark.native
def test_untied_checkpoint_keeps_independent_head(llama_backend, monkeypatch,
                                                tp_size, rank, with_lora):
    model = make_model(llama_backend, monkeypatch, tp_size, rank, with_lora, False)
    embedding = torch.arange(515 * 8, dtype=torch.float32).reshape(515, 8)
    head = embedding + 7

    model.load_weights(iter([("lm_head.weight", head),
                            ("model.embed_tokens.weight", embedding)]))

    check_shard(model.model.embed_tokens, embedding)
    check_shard(model.lm_head, head)


@pytest.mark.native
def test_tied_head_logits_use_loaded_embedding(llama_backend, monkeypatch):
    model = make_model(llama_backend, monkeypatch, 1, 0, False, True)
    source = torch.arange(515 * 8, dtype=torch.float32).reshape(515, 8)
    model.load_weights(iter([("model.embed_tokens.weight", source)]))
    logits_module = importlib.import_module(
        "kvbench_cacheblend.model_executor.layers.logits_processor")
    monkeypatch.setattr(logits_module, "tensor_model_parallel_gather", lambda x: x)
    hidden = torch.ones(1, 8)
    metadata = SimpleNamespace(selected_token_indices=torch.tensor([0]), seq_groups=[])

    torch.testing.assert_close(model.compute_logits(hidden, metadata), hidden @ source.T)


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.native
def test_qwen3_inherited_loader_preserves_head_policy(llama_backend, monkeypatch, tied):
    model = make_model(llama_backend, monkeypatch, 1, 0, False, tied, family="qwen3")
    source = torch.arange(515 * 8, dtype=torch.float32).reshape(515, 8)
    head = source + 7
    model.load_weights(iter([("lm_head.weight", head),
                            ("model.embed_tokens.weight", source)]))
    check_shard(model.model.embed_tokens, source)
    if tied:
        assert not hasattr(model, "lm_head")
    else:
        check_shard(model.lm_head, head)
    logits_module = importlib.import_module(
        "kvbench_cacheblend.model_executor.layers.logits_processor")
    monkeypatch.setattr(logits_module, "tensor_model_parallel_gather", lambda x: x)
    hidden = torch.ones(1, 8)
    metadata = SimpleNamespace(selected_token_indices=torch.tensor([0]), seq_groups=[])
    expected = source if tied else head
    torch.testing.assert_close(model.compute_logits(hidden, metadata), hidden @ expected.T)


@pytest.mark.native
def test_main_transformers_optional_head_dim_falls_back_to_hidden_size():
    config_module = pytest.importorskip("kvbench_cacheblend.config")
    config = object.__new__(config_module.ModelConfig)
    config.hf_text_config = SimpleNamespace(
        head_dim=None, hidden_size=4096, num_attention_heads=32,
    )
    assert config.get_head_size() == 128
    config.hf_text_config.head_dim = 160
    assert config.get_head_size() == 160


def test_adapter_adds_parent_preparation_interval_once():
    method = CacheblendRepo()
    response = {"text": "answer", "ttft": 0.03, "generation_start": 1.02,
                "num_tokens": 1, "total_time": 0.04, "n_input": 1}
    result = method._Result(response, full=False, onlineStart=1.0)
    assert result.performance["ttft"] == pytest.approx(0.05)
    assert result.performance["total_time"] == pytest.approx(0.06)


def test_worker_ttft_starts_at_submission_before_scheduling(clock):
    worker = object.__new__(CacheBlendWorker)
    worker.args = SimpleNamespace(max_new_tokens=1)
    worker.terminalStopIds = []
    worker._SamplingParams = lambda count: object()
    worker._endRetainOutput = lambda: None

    def generate(**kwargs):
        clock.advance(0.04)
        return [SimpleNamespace(metrics=SimpleNamespace(arrival_time=0.0,
            first_scheduled_time=0.01, first_token_time=0.03),
            outputs=[SimpleNamespace(token_ids=[7], text="answer")])]

    worker.llm = SimpleNamespace(generate=generate)
    response = worker._Generate([1])
    assert response["ttft"] == pytest.approx(0.03)
    assert response["total_time"] == pytest.approx(0.04)


def test_naive_repo_disables_repair_and_rejects_nonzero_ratio():
    assert NaiveCacheblendRepo().recompRatio == 0
    with pytest.raises(ValueError, match="recompRatio=0"):
        NaiveCacheblendRepo(recompRatio=0.1)
