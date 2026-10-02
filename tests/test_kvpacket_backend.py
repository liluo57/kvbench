from __future__ import annotations

import importlib
import time
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from methods.KVPacket.Method import KVPacket
from methods.KVPacket.backend.kvpacket import (
    HFBackend,
    Mode,
    PacketPreprocessor,
    PacketSession,
    PacketStore,
    PacketWrapper,
    RawText,
    TextSourceEncoder,
    WrapperEntry,
    WrapperRegistry,
)
from methods.KVPacket.backend.kvpacket.backend.base import RerotationError


class _TinyTokenizer:
    """Deterministic character tokenizer for offline packet semantics tests."""

    eos_token_id = 1

    def __init__(self, vocab_size: int = 64):
        self.vocab_size = vocab_size
        self.delay_phrase: str | None = None
        self.delay_seconds = 0.0

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_tensors: str | None = None,
        return_offsets_mapping: bool = False,
    ):
        del add_special_tokens
        if self.delay_phrase and self.delay_phrase in text:
            time.sleep(self.delay_seconds)
        ids = [ord(char) % (self.vocab_size - 2) + 2 for char in text]
        if return_tensors == "pt":
            input_ids: object = torch.tensor([ids], dtype=torch.long)
        else:
            input_ids = ids
        encoded = {"input_ids": input_ids}
        if return_offsets_mapping:
            encoded["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return encoded

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        tokens = [token for token in token_ids if not (skip_special_tokens and token == self.eos_token_id)]
        return "".join(chr((token - 2) % (self.vocab_size - 2) + 32) for token in tokens)


@pytest.fixture

def tiny_runtime():
    torch.manual_seed(7)
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=256,
        use_cache=True,
    )
    config._name_or_path = "tiny-qwen3"
    model = Qwen3ForCausalLM(config).eval()
    backend = HFBackend(model)
    tokenizer = _TinyTokenizer()
    wrapper = PacketWrapper(
        1,
        1,
        backend.hidden_size,
        dtype=torch.float32,
        device="cpu",
        requires_grad=False,
    ).eval()
    store = PacketStore(backend=backend)
    registry = WrapperRegistry(
        compatibility=backend.compatibility,
        wrappers={"document": WrapperEntry(wrapper)},
    )
    preprocessor = PacketPreprocessor(
        backend=backend,
        wrapper_registry=registry,
        store=store,
        source_encoder=TextSourceEncoder(tokenizer),
    )
    assignments = {
        RawText(text, source_id=f"chunk_{text}"): "document"
        for text in ("alpha", "beta")
    }
    preprocessor.preprocess(assignments)
    return backend, tokenizer, store


def _session(backend, tokenizer, store):
    session = PacketSession(
        backend=backend,
        mode=Mode.SERVE,
        match="token",
        store=store,
        tokenizer=tokenizer,
    )
    if store.ids():
        session.register_store_chunks()
    return session


class _ForwardOnlyBackend:
    """Expose compute-backend methods while selecting PacketSession's decode loop."""

    def __init__(self, backend):
        self._backend = backend

    @property
    def compatibility(self):
        return self._backend.compatibility

    @property
    def device(self):
        return self._backend.device

    def assert_rerotatable(self):
        return self._backend.assert_rerotatable()

    def shift_cache(self, cache, offset):
        return self._backend.shift_cache(cache, offset)

    def shift_cache_flops(self, cache):
        return self._backend.shift_cache_flops(cache)

    def forward(self, **kwargs):
        return self._backend.forward(**kwargs)


def test_tiny_qwen3_packet_matching_reordering_repetition_and_generation(tiny_runtime):
    backend, tokenizer, store = tiny_runtime
    session = _session(backend, tokenizer, store)
    session.push("question beta then alpha and beta again")
    result = session.generate(max_new_tokens=2)

    assert len(result.token_ids) == 2
    assert result.cached_tokens == 4 + 5 + 4
    assert result.matched_chunk_ids == ["chunk_beta", "chunk_alpha", "chunk_beta"]
    assert result.flops > 0


def test_terminal_packet_match_is_forwarded_as_fresh_generation_prompt(tiny_runtime):
    backend, tokenizer, store = tiny_runtime

    exact_terminal = _session(backend, tokenizer, store)
    exact_terminal.push("beta")
    exact_result = exact_terminal.generate(max_new_tokens=2)
    assert len(exact_result.token_ids) == 2
    assert exact_result.prompt_len == len(tokenizer("beta")["input_ids"])
    assert exact_result.cached_tokens == 0
    assert exact_result.matched_chunk_ids == []

    trailing_after_reuse = _session(backend, tokenizer, store)
    trailing_after_reuse.push("alpha beta")
    mixed_result = trailing_after_reuse.generate(max_new_tokens=2)
    assert len(mixed_result.token_ids) == 2
    assert mixed_result.cached_tokens == len(tokenizer("alpha")["input_ids"])
    assert mixed_result.matched_chunk_ids == ["chunk_alpha"]


def test_kvbench_prepare_stores_authoritative_token_chunks(tiny_runtime):
    backend, tokenizer, _store = tiny_runtime
    method = KVPacket(allowUntrainedWrapper=True, headerLen=1, trailerLen=1)
    method._kv = importlib.import_module("methods.KVPacket.backend.kvpacket")
    method._backend = backend
    method._tokenizer = tokenizer
    method._wrapper = PacketWrapper(
        1,
        1,
        backend.hidden_size,
        dtype=torch.float32,
        device="cpu",
        requires_grad=False,
    ).eval()
    alpha_ids = tokenizer("alpha ")["input_ids"]
    spaced_ids = tokenizer(" alpha ")["input_ids"]
    method.Prepare([[[ ], alpha_ids, spaced_ids]])

    assert len(method._states) == 1
    assert method._states[0].chunks == [alpha_ids, spaced_ids]
    assert len(method._states[0].store.ids()) == 2

    prompt_ids = tokenizer("alpha beta")["input_ids"]
    result = method.Run([prompt_ids], maxNewTokens=2)[0]
    assert result.metadata["n_input"] == len(prompt_ids)
    assert result.metadata["cached_tokens"] == len(alpha_ids)


def test_kvbench_rejects_string_matching_without_task_text_and_offsets():
    with pytest.raises(ValueError, match="supports Match='token'"):
        KVPacket(match="string", allowUntrainedWrapper=True)


def test_packet_store_is_immutable_and_rope_shift_changes_only_returned_cache(tiny_runtime):
    backend, _, store = tiny_runtime
    original = store.get("chunk_alpha").packet
    source_key = original.layers[0].key.clone()
    source_positions = original.layers[0].logical_positions.clone()

    shifted = backend.shift_cache(original, 11)

    assert torch.equal(original.layers[0].key, source_key)
    assert torch.equal(original.layers[0].logical_positions, source_positions)
    assert torch.equal(
        shifted.layers[0].logical_positions,
        source_positions + 11,
    )
    assert not torch.equal(shifted.layers[0].key, source_key)


def test_sequence_length_dependent_rope_scaling_is_rejected(tiny_runtime):
    backend, _, _ = tiny_runtime
    backend.config.rope_scaling = {"rope_type": "longrope"}

    with pytest.raises(RerotationError, match="longrope"):
        backend.assert_rerotatable()


def test_unmatched_packet_session_matches_full_prefill_fallback(tiny_runtime):
    backend, tokenizer, populated_store = tiny_runtime
    empty_store = PacketStore(backend=backend)
    prompt = "fresh query with no cached document"

    full_result = _session(backend, tokenizer, empty_store)
    full_result.push(prompt)
    expected = full_result.generate(max_new_tokens=3)

    fallback = _session(backend, tokenizer, populated_store)
    fallback.push(prompt)
    actual = fallback.generate(max_new_tokens=3)

    assert actual.matched_chunk_ids == []
    assert actual.cached_tokens == 0
    assert actual.token_ids == expected.token_ids
    assert actual.text == expected.text


def test_backend_continuation_matches_framework_decode_at_each_logical_position(tiny_runtime):
    backend, tokenizer, _ = tiny_runtime
    empty_store = PacketStore(backend=backend)
    prompt = "nonzero logical start"
    positions: list[list[int]] = []

    def capture_positions(_module, _args, kwargs):
        positions.append(kwargs["position_ids"].detach().cpu().reshape(-1).tolist())

    handle = backend.model.register_forward_pre_hook(capture_positions, with_kwargs=True)
    try:
        native_session = _session(backend, tokenizer, empty_store)
        native_session.push(prompt)
        backend_result = native_session.generate(max_new_tokens=5, eos_token_id=1000)
        backend_positions = positions.copy()

        positions.clear()
        framework_session = PacketSession(
            backend=_ForwardOnlyBackend(backend),
            mode=Mode.SERVE,
            match="token",
            store=empty_store,
            tokenizer=tokenizer,
        )
        framework_session.push(prompt)
        framework_result = framework_session.generate(max_new_tokens=5, eos_token_id=1000)
        framework_positions = positions.copy()
    finally:
        handle.remove()

    assert backend_result.token_ids == framework_result.token_ids
    logical_start = len(tokenizer(prompt)["input_ids"])
    assert backend_positions[-5:] == [[logical_start + offset] for offset in range(5)]
    assert framework_positions[-5:] == [[logical_start + offset] for offset in range(5)]


def test_backend_continuation_keeps_nonzero_logical_offset_separate(tiny_runtime):
    backend, _, _ = tiny_runtime
    positions: list[list[int]] = []

    def capture_positions(_module, _args, kwargs):
        positions.append(kwargs["position_ids"].detach().cpu().reshape(-1).tolist())

    prefix = backend.forward(
        input_ids=torch.tensor([[2, 3]], dtype=torch.long),
        position_ids=torch.tensor([[0, 1]], dtype=torch.long),
    )
    handle = backend.model.register_forward_pre_hook(capture_positions, with_kwargs=True)
    try:
        result = backend.generate(
            input_ids=torch.tensor([[4]], dtype=torch.long),
            position_ids=torch.tensor([[9]], dtype=torch.long),
            cache=prefix.cache,
            max_new_tokens=3,
            eos_token_id=1000,
        )
    finally:
        handle.remove()

    assert positions == [[9], [10], [11], [12]]
    assert result.final_offset == 13
    assert result.cache.layers[0].logical_positions[0].tolist() == [0, 1, 9, 10, 11, 12]


def test_ttft_includes_request_matching_time(tiny_runtime):
    backend, tokenizer, store = tiny_runtime
    tokenizer.delay_phrase = "slow-query"
    tokenizer.delay_seconds = 0.05
    session = _session(backend, tokenizer, store)
    session.push("alpha slow-query")
    started_at = time.perf_counter()

    result = session.generate(max_new_tokens=1, started_at=started_at)

    assert result.matched_chunk_ids == ["chunk_alpha"]
    assert result.ttft >= 0.04


def test_kvpacket_fails_before_model_load_when_no_wrapper_is_configured(monkeypatch):
    method_module = importlib.import_module("methods.KVPacket.Method")
    monkeypatch.setattr(method_module, "Get", lambda *_args, **_kwargs: {})
    method = KVPacket(wrapperCheckpointPath="", allowUntrainedWrapper=False)

    with pytest.raises(RuntimeError, match="requires a model-compatible trained wrapper"):
        method.Initialize([0])


def test_legacy_wrapper_requires_matching_model_id_and_valid_geometry(monkeypatch):
    method_module = importlib.import_module("methods.KVPacket.Method")
    monkeypatch.setattr(
        method_module,
        "Get",
        lambda *_args, **_kwargs: {"WrapperModelId": None},
    )
    configured = KVPacket(allowUntrainedWrapper=True)
    assert configured.wrapperModelId is None

    method = KVPacket.__new__(KVPacket)
    method.wrapperModelId = "Qwen/Qwen3-4B-Instruct-2507"
    method._backend = SimpleNamespace(
        hidden_size=32,
        compatibility=SimpleNamespace(model_id="Qwen/Qwen3-4B"),
    )
    method._kv = SimpleNamespace(PacketWrapper=PacketWrapper)
    method._wrapper_format = "untrained"
    checkpoint = {
        "header": torch.zeros((1, 1, 32)),
        "trailer": torch.zeros((1, 1, 32)),
        "header_len": 1,
        "trailer_len": 1,
        "dim": 32,
    }

    with pytest.raises(ValueError, match="legacy KVPacket wrapper model mismatch"):
        method._load_legacy_wrapper(checkpoint, device="cpu", dtype=torch.float32)

    method._backend.compatibility.model_id = method.wrapperModelId
    checkpoint["header"] = torch.zeros((1, 2, 32))
    with pytest.raises(ValueError, match="header shape is invalid"):
        method._load_legacy_wrapper(checkpoint, device="cpu", dtype=torch.float32)
