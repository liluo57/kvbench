"""KVPacket can keep completed native output KV without wrapping or re-prefilling."""

import importlib
from types import SimpleNamespace

import torch

from methods.KVPacket.Method import KVPacket
from methods.KVPacket.backend.kvpacket import Mode, PacketSession, PacketStore, PacketWrapper
from methods.KVPacket.backend.kvpacket.kvcache.cache import KeyValue, KVCache, concat_caches
from test_kvpacket_backend import tiny_runtime


def _make_method(backend, tokenizer, chunks=()):
    method = KVPacket(allowUntrainedWrapper=True, headerLen=1, trailerLen=1)
    method._kv = importlib.import_module("methods.KVPacket.backend.kvpacket")
    method._backend = backend
    method._tokenizer = tokenizer
    method._model = backend.model
    method._torch = torch
    method._wrapper = PacketWrapper(
        1,
        1,
        backend.hidden_size,
        dtype=torch.float32,
        device="cpu",
        requires_grad=False,
    ).eval()
    method.samplingConfig = {"mode": "greedy"}
    if chunks:
        method.Prepare([list(chunks)])
    else:
        method._states = [
            SimpleNamespace(chunks=[], store=method._kv.PacketStore(backend=backend))
        ]
    return method


def _snapshot(packet):
    return [
        (layer.key.clone(), layer.value.clone(), layer.logical_positions.clone())
        for layer in packet.packet.layers
    ]


def _assert_snapshot(packet, saved):
    for layer, (key, value, positions) in zip(packet.packet.layers, saved, strict=True):
        torch.testing.assert_close(layer.key, key, rtol=0, atol=0)
        torch.testing.assert_close(layer.value, value, rtol=0, atol=0)
        torch.testing.assert_close(layer.logical_positions, positions, rtol=0, atol=0)


def test_kvpacket_retains_raw_output_cache_and_reuses_moved_repeated_ids(monkeypatch):
    backend, tokenizer, _ = tiny_runtime.__wrapped__()
    with torch.no_grad():
        backend.model.lm_head.weight.zero_()
    prepared_ids = tokenizer("alpha")["input_ids"]
    method = _make_method(backend, tokenizer, [prepared_ids])
    state = method._states[0]
    prepared = state.store.get(state.store.ids()[0])
    prepared_snapshot = _snapshot(prepared)

    # A generated fragment must come from the live decode cache, without a second
    # standalone prefill to manufacture a packet.
    monkeypatch.setattr(
        backend,
        "prefill_caches",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("output was re-prefilled")),
    )
    first = method.Run([[20, 21, 22]], retainOutput=[True], maxNewTokens=2)[0]
    assert first.output_ids == [0, 0]
    native_id = next(
        chunk_id
        for chunk_id in state.store.ids()
        if state.store.get(chunk_id).identity.wrapper_id == "native_output"
    )
    native = state.store.get(native_id)
    assert native.token_ids == first.output_ids
    assert native.identity.source_kind == "native_token_ids"
    assert native.body_len == native.wrapped_len == len(first.output_ids)
    assert native.packet.seq_len == native.packet.doc_len == len(first.output_ids)
    assert native.packet.layers[0].logical_positions.tolist() == [[0, 1]]
    native_snapshot = _snapshot(native)

    later = [30, *first.output_ids, 31, *first.output_ids, 32]
    reused = method.Run([later], maxNewTokens=1)[0]
    assert reused.metadata["cached_tokens"] == 2 * len(first.output_ids)
    assert reused.metadata["matched_chunk_ids"].count(native_id) == 2
    _assert_snapshot(native, native_snapshot)
    _assert_snapshot(prepared, prepared_snapshot)


def test_kvpacket_terminal_eos_only_does_not_register_native_packet():
    backend, tokenizer, _ = tiny_runtime.__wrapped__()
    with torch.no_grad():
        backend.model.lm_head.weight.zero_()
    method = _make_method(backend, tokenizer)
    method.samplingConfig = {"mode": "greedy", "eos_token_id": 0}

    result = method.Run([[20, 21]], retainOutput=[True], maxNewTokens=2)[0]

    assert result.output_ids == []
    assert method._states[0].store.ids() == []


def test_session_extracts_output_suffix_after_compressed_logical_prefix():
    backend, tokenizer, store = tiny_runtime.__wrapped__()
    source = store.get(store.ids()[0])
    prefix = KVCache([
        KeyValue(
            layer.key[:, :, :1].clone(),
            layer.value[:, :, :1].clone(),
            layer.logical_positions.clone(),
        )
        for layer in source.packet.layers
    ])
    output_ids = [14, 15]
    native = backend.prefill_caches(
        input_ids=torch.tensor([output_ids], dtype=torch.long)
    )[0]
    absolute_output = backend.shift_cache(native, source.wrapped_len)
    session = PacketSession(
        backend=backend,
        mode=Mode.SERVE,
        match="token",
        store=PacketStore(backend=backend),
        tokenizer=None,
    )
    session._cache = concat_caches([prefix, absolute_output])
    session._offset = source.wrapped_len + len(output_ids)

    retained = session.extract_output_cache(output_ids)

    assert retained is not None
    assert retained.seq_len == retained.doc_len == len(output_ids)
    assert retained.layers[0].logical_positions.tolist() == [[0, 1]]
    for actual, expected in zip(retained.layers, native.layers, strict=True):
        torch.testing.assert_close(actual.key, expected.key)
        torch.testing.assert_close(actual.value, expected.value, rtol=0, atol=0)


def test_kvpacket_retained_native_ids_win_wrapper_alias_and_retain_false_skips():
    backend, tokenizer, _ = tiny_runtime.__wrapped__()
    with torch.no_grad():
        backend.model.lm_head.weight.zero_()
    method = _make_method(backend, tokenizer, [[0, 0]])
    state = method._states[0]
    wrapped_id = state.store.ids()[0]

    first = method.Run([[20, 21]], retainOutput=[True], maxNewTokens=2)[0]
    native_id = next(
        chunk_id
        for chunk_id in state.store.ids()
        if state.store.get(chunk_id).identity.wrapper_id == "native_output"
    )
    assert first.output_ids == [0, 0]

    later = method.Run([[30, 0, 0, 31]], maxNewTokens=1)[0]
    assert later.metadata["matched_chunk_ids"] == [native_id]
    assert wrapped_id not in later.metadata["matched_chunk_ids"]

    count_before = len(state.store.ids())
    method.Run([[32, 33]], retainOutput=[False], maxNewTokens=2)
    assert len(state.store.ids()) == count_before


def test_kvpacket_replaces_same_native_ids_from_new_prompt_without_touching_prepared():
    backend, tokenizer, _ = tiny_runtime.__wrapped__()
    with torch.no_grad():
        backend.model.lm_head.weight.zero_()
    method = _make_method(backend, tokenizer, [[2, 3]])
    state = method._states[0]
    prepared_id = state.store.ids()[0]
    prepared = state.store.get(prepared_id)
    prepared_snapshot = _snapshot(prepared)

    first = method.Run([[20, 21]], retainOutput=[True], maxNewTokens=2)[0]
    native_id = next(
        chunk_id
        for chunk_id in state.store.ids()
        if state.store.get(chunk_id).identity.wrapper_id == "native_output"
    )
    assert first.output_ids == [0, 0]
    native_before = state.store.get(native_id)
    before_snapshot = _snapshot(native_before)

    second = method.Run([[30, 31]], retainOutput=[True], maxNewTokens=2)[0]
    native_after = state.store.get(native_id)
    assert second.output_ids == first.output_ids
    assert native_after is not native_before
    assert any(
        not torch.equal(old_key, layer.key)
        or not torch.equal(old_value, layer.value)
        for layer, (old_key, old_value, _positions) in zip(
            native_after.packet.layers, before_snapshot, strict=True
        )
    )
    _assert_snapshot(prepared, prepared_snapshot)


def test_kvpacket_mixed_output_and_terminal_eos_retains_matching_native_kv(monkeypatch):
    from helpers.backends.CacheBackend.Types import BackendGenerationOutput
    from helpers.backends.Tokenization import OutputFragmentIds
    from methods.KVPacket.backend.kvpacket.session import session as session_module

    backend, tokenizer, _ = tiny_runtime.__wrapped__()
    method = _make_method(backend, tokenizer)
    real_forward = backend.forward
    committed_cache = []

    def select_first(*args, **kwargs):
        return 14

    def generate_then_eos(*, input_ids, position_ids, cache, **kwargs):
        first = real_forward(
            input_ids=input_ids,
            position_ids=position_ids,
            cache=cache,
        )
        eos_position = position_ids + 1
        eos = real_forward(
            input_ids=torch.tensor([[1]], dtype=torch.long, device=input_ids.device),
            position_ids=eos_position,
            cache=first.cache,
        )
        committed_cache.append(eos.cache.clone())
        return BackendGenerationOutput(
            token_ids=[1], logits=[], cache=eos.cache, final_offset=int(eos_position.item()) + 1
        )

    monkeypatch.setattr(session_module, "select_next_token", select_first)
    monkeypatch.setattr(backend, "generate", generate_then_eos)
    prompt = [20, 21]
    result = method.Run([prompt], retainOutput=[True], maxNewTokens=3)[0]

    assert OutputFragmentIds([14, 1], method._terminalStopIds) == [14]
    assert result.output_ids == [14]
    state = method._states[0]
    native = next(
        state.store.get(chunk_id)
        for chunk_id in state.store.ids()
        if state.store.get(chunk_id).identity.wrapper_id == "native_output"
    )
    assert native.token_ids == [14]
    source = committed_cache[0]
    absolute_start = len(prompt)
    expected = backend.shift_cache(
        type(source)([
            type(layer)(
                layer.key[:, :, -2:-1].clone(),
                layer.value[:, :, -2:-1].clone(),
                torch.tensor([[absolute_start]], dtype=torch.long),
            )
            for layer in source.layers
        ]),
        -absolute_start,
    )
    for actual, want in zip(native.packet.layers, expected.layers, strict=True):
        torch.testing.assert_close(actual.key, want.key)
        torch.testing.assert_close(actual.value, want.value, rtol=0, atol=0)
