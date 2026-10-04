"""Load actual bundled Llama vocabulary weights without a CUDA context."""
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig


@pytest.fixture
def llama_backend(monkeypatch):
    backend = Path(__file__).resolve().parents[1] / "methods/Cacheblend/backend"
    monkeypatch.syspath_prepend(str(backend))
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
def test_untied_checkpoint_keeps_independent_head(llama_backend, monkeypatch,
                                                tp_size, rank, with_lora):
    model = make_model(llama_backend, monkeypatch, tp_size, rank, with_lora, False)
    embedding = torch.arange(515 * 8, dtype=torch.float32).reshape(515, 8)
    head = embedding + 7

    model.load_weights(iter([("lm_head.weight", head),
                            ("model.embed_tokens.weight", embedding)]))

    check_shard(model.model.embed_tokens, embedding)
    check_shard(model.lm_head, head)


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
