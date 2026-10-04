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
from helpers.backends.CacheBackend.Sampling import SamplingConfig, select_next_token


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


def test_kvpacket_context_limit_counts_wrappers_and_keeps_terminal_match_fresh(tiny_runtime, monkeypatch):
    from core import Config
    original_get = Config.Get
    limit = 10
    monkeypatch.setattr(Config, "Get", lambda key, *args, **kwargs:
        limit if key == "MaxModelLen" else original_get(key, *args, **kwargs))
    backend, tokenizer, store = tiny_runtime
    with torch.no_grad():
        backend.model.lm_head.weight.zero_()  # greedy token 0 is not EOS
    method = KVPacket(allowUntrainedWrapper=True)
    method._backend, method._tokenizer = backend, tokenizer
    method._model, method._torch = backend.model, torch
    method._wrapper_format = "untrained"
    alpha_ids = tokenizer("alpha")["input_ids"]
    method._states = [SimpleNamespace(chunks=[alpha_ids], store=store)]

    result = method.Run([alpha_ids + [31]], maxNewTokens=8)[0]
    # Six original IDs plus the two wrapper positions leave two output slots.
    assert result.performance["num_output_tokens"] == 2
    assert result.metadata["cached_tokens"] == 5
    limit = 8
    with pytest.raises(ValueError, match="Wrapped KVPacket prompt"):
        method.Run([alpha_ids + [31]], maxNewTokens=8)
    limit = 7
    # A terminal match is fresh, so it consumes no wrapper positions.
    result = method.Run([alpha_ids], maxNewTokens=8)[0]
    assert result.performance["num_output_tokens"] == 2
    assert result.metadata["cached_tokens"] == len(alpha_ids)
    assert result.metadata["substituted_tokens"] == 0
    assert result.metadata["reuse_ratio"] == 1.0
    method._states = [SimpleNamespace(chunks=[], store=PacketStore(backend=backend))]
    baseline = method.Run([alpha_ids], maxNewTokens=8)[0]
    assert result.output_ids == baseline.output_ids
    assert result.performance["num_output_tokens"] == baseline.performance["num_output_tokens"]


def test_greedy_packet_selection_applies_history_penalties():
    logits = torch.tensor([[5.0, 4.0, 3.0]])
    assert select_next_token(logits, SamplingConfig(), history=[0]) == 0
    assert select_next_token(
        logits,
        SamplingConfig(do_sample=False, presence_penalty=2.0),
        history=[0],
    ) == 1
    assert select_next_token(
        logits,
        SamplingConfig(do_sample=False, frequency_penalty=1.1),
        history=[0, 0],
    ) == 1
    assert select_next_token(
        logits,
        SamplingConfig(do_sample=False, repetition_penalty=2.0),
        history=[0],
    ) == 1


def test_packet_typical_p_filter_keeps_the_most_typical_token():
    logits = torch.tensor([[2.0, 1.0, 0.0]])
    sampling = SamplingConfig(
        do_sample=True,
        top_k=0,
        typical_p=0.01,
    )
    generator = torch.Generator(device="cpu").manual_seed(9)
    assert select_next_token(logits, sampling, generator=generator) == 0


def test_packet_sampler_matches_transformers_helper_with_combined_filters():
    from helpers.backends.TransformersHelper import TransformersGenerator

    policy = {
        "mode": "override",
        "do_sample": True,
        "temperature": 0.8401051856315369,
        "top_k": 0,
        "typical_p": 0.6716978524525754,
        "top_p": 0.40545003212609204,
        "min_p": 0.0,
        "repetition_penalty": 1.25,
        "presence_penalty": 0.15,
        "frequency_penalty": 0.08,
    }
    logits = torch.tensor([[
        -0.070533216, 0.084036246, 0.699430108, -0.130221173,
        -0.365434378, -1.466285944, -0.267276198, 0.197312221,
        -0.401024878, -0.426097512, -0.291413903, 0.425464988,
    ]])
    history = [1, 1, 4, 7]
    helper = object.__new__(TransformersGenerator)
    helper._torch = torch
    helper.samplingConfig = policy
    seed = 71

    helper_generator = torch.Generator(device="cpu").manual_seed(seed)
    packet_generator = torch.Generator(device="cpu").manual_seed(seed)
    helper_token = helper._SampleNextToken(logits.clone()[0], history, helper_generator)
    packet_token = select_next_token(
        logits,
        SamplingConfig(
            do_sample=policy["do_sample"],
            temperature=policy["temperature"],
            top_k=policy["top_k"],
            typical_p=policy["typical_p"],
            top_p=policy["top_p"],
            min_p=policy["min_p"],
            repetition_penalty=policy["repetition_penalty"],
            presence_penalty=policy["presence_penalty"],
            frequency_penalty=policy["frequency_penalty"],
        ),
        history=history,
        generator=packet_generator,
    )
    assert helper_token == 0
    assert packet_token == helper_token


def test_native_backend_requires_full_history_for_cached_penalty_generation(tiny_runtime):
    backend, _, _ = tiny_runtime
    prefix = backend.forward(
        input_ids=torch.tensor([[2]], dtype=torch.long),
        position_ids=torch.tensor([[0]], dtype=torch.long),
    )
    with pytest.raises(ValueError, match="history_ids are required"):
        backend.generate(
            input_ids=torch.tensor([[3]], dtype=torch.long),
            position_ids=torch.tensor([[1]], dtype=torch.long),
            cache=prefix.cache,
            max_new_tokens=1,
            repetition_penalty=1.2,
        )


def test_kvpacket_rejects_unimplemented_resolved_sampling_field(monkeypatch):
    method_module = importlib.import_module("methods.KVPacket.Method")
    monkeypatch.setattr(
        method_module,
        "ResolveSamplingConfig",
        lambda _model_path: {
            "mode": "override",
            "do_sample": True,
            "future_warper": 0.5,
        },
    )
    with pytest.raises(ValueError, match="does not support resolved sampling.*future_warper"):
        KVPacket(allowUntrainedWrapper=True)


def test_kvpacket_passes_full_sampling_policy_and_seed_through_native_decode(
    tiny_runtime, monkeypatch
):
    backend, tokenizer, _ = tiny_runtime
    method = KVPacket(allowUntrainedWrapper=True)
    method._backend, method._tokenizer, method._model = backend, tokenizer, backend.model
    method._torch = torch
    method.samplingConfig = {
        "mode": "override",
        "do_sample": True,
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 0,
        "typical_p": 0.9,
        "repetition_penalty": 1.2,
        "presence_penalty": 0.2,
        "frequency_penalty": 0.1,
        "seed": 2718,
        "eos_token_id": 1,
    }
    original_generate = backend.generate
    observed = []

    def capture_generate(**kwargs):
        observed.append(kwargs)
        return original_generate(**kwargs)

    monkeypatch.setattr(backend, "generate", capture_generate)
    prompt = [7, 8, 9]
    first = method.Run([prompt], maxNewTokens=4)[0]
    method.Reset()
    second = method.Run([prompt], maxNewTokens=4)[0]

    assert first.output_ids == second.output_ids
    assert first.performance["num_output_tokens"] == second.performance["num_output_tokens"]
    assert observed
    native = observed[0]
    assert native["typical_p"] == 0.9
    assert native["repetition_penalty"] == 1.2
    assert native["presence_penalty"] == 0.2
    assert native["frequency_penalty"] == 0.1
    assert native["history_ids"][: len(prompt)] == prompt
    assert native["generator"] is not None


def test_kvpacket_greedy_run_uses_presence_penalty(
    tiny_runtime, monkeypatch
):
    backend, tokenizer, _ = tiny_runtime
    forward = backend.forward

    def forced_logits(**kwargs):
        output = forward(**kwargs)
        output.logits.fill_(-100.0)
        output.logits[..., 7] = 5.0
        output.logits[..., 8] = 4.0
        return output

    monkeypatch.setattr(backend, "forward", forced_logits)
    method = KVPacket(allowUntrainedWrapper=True)
    method._backend, method._tokenizer, method._model = backend, tokenizer, backend.model
    method._torch = torch
    method.samplingConfig = {
        "mode": "override",
        "do_sample": False,
        "temperature": 1.0,
        "presence_penalty": 2.0,
        "eos_token_id": 1,
    }

    result = method.Run([[7]], maxNewTokens=1)[0]
    assert result.output_ids == [8]
    assert result.performance["num_output_tokens"] == 1


def test_kvpacket_coverage_counts_repeated_matched_occurrences(tiny_runtime):
    backend, tokenizer, store = tiny_runtime
    method = KVPacket(allowUntrainedWrapper=True)
    method._backend, method._tokenizer, method._model = backend, tokenizer, backend.model
    method._torch = torch
    method._states = [SimpleNamespace(chunks=[], store=store)]

    alpha = tokenizer("alpha")["input_ids"]
    beta = tokenizer("beta")["input_ids"]
    prompt = beta + alpha + beta + [30]
    result = method.Run([prompt], maxNewTokens=2)[0]

    assert result.metadata["cached_tokens"] == len(beta) * 2 + len(alpha)
    assert result.metadata["reuse_ratio"] == pytest.approx(
        (len(beta) * 2 + len(alpha)) / len(prompt)
    )
    assert result.metadata["substituted_tokens"] <= result.metadata["cached_tokens"]


def test_kvbench_rejects_string_matching_without_task_text_and_offsets():
    with pytest.raises(ValueError, match="supports Match='token'"):
        KVPacket(match="string", allowUntrainedWrapper=True)


@pytest.mark.parametrize("override, expected_eos", [(None, 2), (3, 3)])
def test_kvbench_stops_on_first_eos_without_session_tokenizer(tiny_runtime, monkeypatch, override, expected_eos):
    backend, tokenizer, _ = tiny_runtime
    backend.model.generation_config.eos_token_id = 2
    forward = backend.forward

    def force_eos(**kwargs):
        output = forward(**kwargs)
        output.logits.fill_(-1000)
        output.logits[..., expected_eos] = 1000
        return output

    monkeypatch.setattr(backend, "forward", force_eos)
    monkeypatch.setattr(backend, "generate", lambda **kwargs: pytest.fail("continued after first EOS"))
    method = KVPacket(allowUntrainedWrapper=True)
    method._backend, method._tokenizer, method._model = backend, tokenizer, backend.model
    method._torch = torch
    if override is not None:
        method.samplingConfig = {"mode": "override", "do_sample": False, "eos_token_id": override}
    result = method.Run([[7]], maxNewTokens=3)[0]
    assert result.performance["num_output_tokens"] == 1
    assert result.output_ids == []


def test_kvbench_override_keeps_old_eos_at_generation_budget(tiny_runtime, monkeypatch):
    backend, tokenizer, _ = tiny_runtime
    tokenizer.eos_token_id = 2
    forward = backend.forward

    def force_old_eos(**kwargs):
        output = forward(**kwargs)
        output.logits.fill_(-1000)
        output.logits[..., 2] = 1000
        return output

    monkeypatch.setattr(backend, "forward", force_old_eos)
    method = KVPacket(allowUntrainedWrapper=True)
    method._backend, method._tokenizer, method._model = backend, tokenizer, backend.model
    method._torch = torch
    method.samplingConfig = {"mode": "override", "do_sample": False, "eos_token_id": 9}
    result = method.Run([[7]], maxNewTokens=1)[0]
    assert result.output_ids == [2]


def test_unversioned_checkpoint_requires_identical_actual_weights(tiny_runtime, tmp_path):
    from methods.KVPacket.backend.kvpacket import PacketTrainer
    from methods.KVPacket.backend.kvpacket.errors import BackendCompatibilityError
    backend, _, _ = tiny_runtime
    wrapper = PacketWrapper(1, 1, backend.hidden_size, dtype=torch.float32, requires_grad=True)
    registry = WrapperRegistry(compatibility=backend.compatibility, wrappers={"document": WrapperEntry(wrapper)})
    path = tmp_path / "trainer.pt"
    PacketTrainer(backend, registry, default_wrapper_id="document").save(path)

    same_model = Qwen3ForCausalLM(backend.model.config).eval()
    same_model.load_state_dict(backend.model.state_dict())
    same_backend = HFBackend(same_model)
    assert same_backend.compatibility == backend.compatibility
    PacketTrainer.load(path, backend=same_backend, device="cpu")
    with torch.no_grad():
        next(same_model.parameters()).add_(1)
    different_backend = HFBackend(same_model)
    assert different_backend.compatibility != backend.compatibility
    with pytest.raises(BackendCompatibilityError):
        PacketTrainer.load(path, backend=different_backend, device="cpu")


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
