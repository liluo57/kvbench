import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from core.Sampling import (
    ResolveSamplingConfig,
    SglangSamplingParams,
    TransformersGenerationKwargs,
    VllmSamplingParams,
)
from methods.Naive import NaiveTransformer
from methods.KVComm import KVCommTransformer
from helpers.backends.TransformersHelper import TransformersGenerator


def _model_dir(tmp_path, *, config=None, generation=None):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(config or {}), encoding="utf-8"
    )
    if generation is not None:
        (model / "generation_config.json").write_text(
            json.dumps(generation), encoding="utf-8"
        )
    return model


def test_sampling_modes_resolve_model_and_yaml_values(tmp_path):
    model = _model_dir(
        tmp_path,
        config={"temperature": 0.4, "top_k": 7},
        generation={
            "do_sample": True,
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": 20,
        },
    )

    assert ResolveSamplingConfig(model, config={"ModelConfig": {"mode": "greedy"}}) == {
        "mode": "greedy",
        "do_sample": False,
        "temperature": 0.0,
    }
    assert ResolveSamplingConfig(model, config={"ModelConfig": {"mode": "default"}}) == {
        "mode": "default",
        "do_sample": True,
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 20,
    }
    assert ResolveSamplingConfig(
        model,
        config={
            "ModelConfig": {
                "mode": "override",
                "temp": 0.2,
                "topp": 0.7,
                "topk": 5,
            }
        },
    ) == {
        "mode": "override",
        "do_sample": True,
        "temperature": 0.2,
        "top_p": 0.7,
        "top_k": 5,
    }


def test_default_falls_back_to_config_json(tmp_path):
    model = _model_dir(
        tmp_path,
        config={"do_sample": True, "temperature": 0.6, "top_k": 11},
    )

    resolved = ResolveSamplingConfig(
        model, config={"ModelConfig": {"mode": "default"}}
    )

    assert resolved["temperature"] == 0.6
    assert resolved["top_k"] == 11
    assert resolved["top_p"] == 1.0


def test_backend_translations_keep_greedy_and_override_consistent():
    greedy = {"mode": "greedy", "do_sample": False, "temperature": 0.0}
    assert VllmSamplingParams(greedy, 12) == {
        "temperature": 0,
        "max_tokens": 12,
    }
    assert SglangSamplingParams(greedy, 12) == {
        "temperature": 0,
        "max_new_tokens": 12,
    }
    assert TransformersGenerationKwargs(greedy, 12) == {
        "max_new_tokens": 12,
        "do_sample": False,
    }

    # Canonical greedy is authoritative even if stale/raw penalty fields are
    # present; deterministic default/override policies retain useful penalties.
    canonical_with_raw = {**greedy, "repetition_penalty": 2.0, "top_p": 0.2}
    assert VllmSamplingParams(canonical_with_raw, 12) == {
        "temperature": 0, "max_tokens": 12,
    }
    assert TransformersGenerationKwargs(canonical_with_raw, 12) == {
        "max_new_tokens": 12, "do_sample": False,
    }
    for penalty in (
        {"repetition_penalty": 2.0},
        {"presence_penalty": 0.5},
        {"frequency_penalty": 0.5},
    ):
        deterministic = {"mode": "override", "do_sample": False, "temperature": 0.0,
                         "top_p": 0.2, **penalty}
        expected = {"temperature": 0, "max_tokens": 12, **penalty}
        assert VllmSamplingParams(deterministic, 12) == expected
        expected_sglang = {"temperature": 0, "max_new_tokens": 12, **penalty}
        assert SglangSamplingParams(deterministic, 12) == expected_sglang
        expected_hf = {"max_new_tokens": 12, "do_sample": False}
        if "repetition_penalty" in penalty:
            expected_hf.update(penalty)
        assert TransformersGenerationKwargs(deterministic, 12) == expected_hf
        assert "presence_penalty" not in expected_hf
        assert "frequency_penalty" not in expected_hf

    override = {
        "mode": "override",
        "do_sample": True,
        "temperature": 0.2,
        "top_p": 0.7,
        "top_k": 5,
    }
    assert VllmSamplingParams(override, 12) == {
        "max_tokens": 12,
        "temperature": 0.2,
        "top_p": 0.7,
        "top_k": 5,
    }


def test_sampling_mode_is_validated():
    with pytest.raises(ValueError, match="ModelConfig.mode"):
        ResolveSamplingConfig(config={"ModelConfig": {"mode": "random"}})


def test_default_sampling_reads_hub_metadata_and_uses_generation_precedence(tmp_path, monkeypatch):
    import transformers.utils.hub
    directory = _model_dir(tmp_path, config={"temperature": 0.4}, generation={
        "do_sample": True, "temperature": 0.7, "top_p": 0.9, "top_k": 20})
    requested = []

    def cached_file(repo, filename, **kwargs):
        assert repo == "org/model"
        requested.append(filename)
        return str(directory / filename)

    monkeypatch.setattr(transformers.utils.hub, "cached_file", cached_file)
    policy = ResolveSamplingConfig("org/model", config={"ModelConfig": {"mode": "default"}})
    assert policy == {"mode": "default", "do_sample": True, "temperature": 0.7,
                      "top_p": 0.9, "top_k": 20}
    assert set(requested) == {"config.json", "generation_config.json"}


def test_default_hub_sampling_does_not_hide_resolution_failure(monkeypatch):
    import transformers.utils.hub

    def unavailable(*args, **kwargs):
        raise OSError("offline metadata unavailable")

    monkeypatch.setattr(transformers.utils.hub, "cached_file", unavailable)
    with pytest.raises(OSError, match="offline metadata"):
        ResolveSamplingConfig("org/model", config={"ModelConfig": {"mode": "default"}})


@pytest.mark.parametrize("translate", [VllmSamplingParams, SglangSamplingParams])
@pytest.mark.parametrize("do_sample", [True, False])
def test_engine_backends_reject_eos_replacement_instead_of_ignoring_it(translate, do_sample):
    with pytest.raises(ValueError, match="eos_token_id replacement"):
        translate({"mode": "override", "do_sample": do_sample, "eos_token_id": 9}, 5)
    # Model defaults remain engine-owned; this is not an EOS replacement.
    assert translate({"mode": "default", "do_sample": False, "eos_token_id": 2}, 5)["temperature"] == 0


class _OutputCache:
    def __init__(self, length):
        self.layers = [(
            torch.zeros((1, 1, length, 2)),
            torch.zeros((1, 1, length, 2)),
        )]

    def get_seq_length(self):
        return self.layers[0][0].shape[-2]

    def update(self, key, value, _layer_index):
        old_key, old_value = self.layers[0]
        self.layers[0] = (
            torch.cat([old_key, key], dim=-2),
            torch.cat([old_value, value], dim=-2),
        )

    def crop(self, length):
        key, value = self.layers[0]
        self.layers[0] = (key[..., :length, :], value[..., :length, :])


def _naive_cache_registrar():
    method = object.__new__(NaiveTransformer)

    class Generator:
        tokenizer = SimpleNamespace(all_special_ids=[99])
        _torch = SimpleNamespace(no_grad=nullcontext)

        def __init__(self):
            self.prefill_calls = []
            self.generate_calls = []

        @staticmethod
        def OutputTokenIds(token_ids):
            values = list(token_ids)
            return values[:-1] if values and values[-1] == 99 else values

        @staticmethod
        def Forward(token_ids, cache):
            assert token_ids == [8]
            key, value = cache.layers[0]
            cache.layers[0] = (
                torch.cat([key, torch.ones((1, 1, 1, 2))], dim=-2),
                torch.cat([value, torch.ones((1, 1, 1, 2))], dim=-2),
            )
            return SimpleNamespace(past_key_values=cache)

        def Prefill(self, token_ids, pastKeyValues=None):
            self.prefill_calls.append(list(token_ids))
            cache = _OutputCache(0) if pastKeyValues is None else pastKeyValues
            token_count = len(token_ids)
            key = torch.zeros((1, 1, token_count, 2))
            value = torch.zeros((1, 1, token_count, 2))
            cache.update(key, value, 0)
            return SimpleNamespace(past_key_values=cache)

        def Generate(self, token_ids, *, pastKeyValues=None, returnTokenIds=False, **_kwargs):
            self.generate_calls.append((list(token_ids), pastKeyValues))
            return ("answer", 0.01, 0.02, 1, [10]) if returnTokenIds else ("answer", 0.01, 0.02, 1)

    method._gen = Generator()
    return method


def test_naive_retains_internal_special_with_exact_native_cache_rows():
    method = _naive_cache_registrar()
    state = {"segments": [], "ids": [], "caches": []}

    method._registerOutput(
        state,
        [7, 99, 8],
        _OutputCache(6),
        promptLength=3,
    )

    assert state["ids"] == [[7, 99, 8]]
    assert state["caches"][0][0][0].shape[-2] == 3
    assert method._gen.prefill_calls == []


def test_naive_requires_completed_native_cache_and_excludes_terminal_eos():
    method = _naive_cache_registrar()
    state = {"segments": [], "ids": [], "caches": []}

    method._registerOutput(
        state,
        [7, 8],
        _OutputCache(5),
        promptLength=3,
    )

    assert state["ids"] == [[7, 8]]
    assert state["caches"][0][0][0].shape[-2] == 2

    eos_state = {"segments": [], "ids": [], "caches": []}
    method._registerOutput(
        eos_state,
        [7, 8],
        _OutputCache(5),
        promptLength=3,
    )
    assert eos_state["ids"] == [[7, 8]]
    assert eos_state["caches"][0][0][0].shape[-2] == 2


def test_naive_native_output_cache_hits_next_prompt_without_full_prefill():
    method = _naive_cache_registrar()
    state = {"segments": [], "ids": [], "caches": []}
    method._registerOutput(
        state,
        [7, 99, 8],
        _OutputCache(6),
        promptLength=3,
    )
    assert state["ids"] == [[7, 99, 8]]
    method._gen.prefill_calls.clear()
    method._states = [state]

    result = method.Run([[1, 7, 99, 8, 9]])[0]

    assert result.metadata["reuse_ratio"] == pytest.approx(0.6)
    assert method._gen.prefill_calls == [[1], [9]]
    assert method._gen.generate_calls[0][0] == [9]


def test_naive_counts_cached_final_prompt_token_even_though_it_is_replayed():
    method = _naive_cache_registrar()
    assert method.method_metrics == ("reuse_ratio",)
    state = {"segments": [], "ids": [], "caches": []}
    method._registerOutput(state, [7, 99, 8], _OutputCache(6), promptLength=3)
    method._states = [state]

    result = method.Run([[7, 99, 8]])[0]

    assert result.metadata["reuse_ratio"] == pytest.approx(1.0)
    assert method._gen.generate_calls[0][0] == [8]
    assert method._gen.generate_calls[0][1].get_seq_length() == 2


def test_naive_no_match_reports_zero_reuse_ratio():
    method = _naive_cache_registrar()
    method._states = [{"segments": [], "ids": [], "caches": []}]

    result = method.Run([[1, 2]])[0]

    assert result.metadata["reuse_ratio"] == 0.0


def test_kvcomm_retains_internal_special_and_reuses_native_cache(monkeypatch):
    method = object.__new__(KVCommTransformer)

    class Generator:
        tokenizer = SimpleNamespace(all_special_ids=[99])
        _torch = SimpleNamespace(no_grad=nullcontext)
        prefill_calls = []

        @staticmethod
        def OutputTokenIds(token_ids):
            values = list(token_ids)
            return values[:-1] if values and values[-1] == 99 else values

        @staticmethod
        def Forward(*_args):
            raise AssertionError("already aligned native cache needs no completion")

        @staticmethod
        def Prefill(token_ids, pastKeyValues=None):
            Generator.prefill_calls.append(list(token_ids))
            cache = _OutputCache(0) if pastKeyValues is None else pastKeyValues
            count = len(token_ids)
            cache.update(
                torch.zeros((1, 1, count, 2)),
                torch.zeros((1, 1, count, 2)),
                0,
            )
            return SimpleNamespace(past_key_values=cache)

    method._gen = Generator()
    method._runIndex = 2
    method._ShiftCache = lambda cache, _offset: cache
    state = {"segments": []}
    monkeypatch.setattr("methods.KVComm._SliceCache", lambda cache, _start, _end: cache)

    method._RegisterOutput(
        state,
        [7, 99, 8],
        fullCache=_OutputCache(6),
        promptLength=3,
    )

    assert state["segments"][0]["base_ids"] == (7, 99, 8)
    assert state["segments"][0]["base_cache"] is not None
    assert Generator.prefill_calls == []

    from helpers.backends.Prompt import ComposeInterleavedReuse

    segment = state["segments"][0]
    matches = ComposeInterleavedReuse([list(segment["base_ids"])], [1, 7, 99, 8, 9])
    assert matches == [(None, [1]), (0, [7, 99, 8]), (None, [9])]
    parts = method._MakeParts(state["segments"], matches)
    method._GetPrefixTemplate = lambda _parts: None
    method._GetIsolatedBase = lambda _ids: pytest.fail(
        "retained visible output should reuse its rebuilt KV"
    )

    method._BuildBase(parts, [[1], [7, 99, 8], [9]])

    assert parts[1]["raw_cache"] is segment["base_cache"]


def test_kvcomm_equal_answers_keep_their_contextual_cache_identity():
    from helpers.backends.Prompt import ComposeInterleavedReuse
    method = object.__new__(KVCommTransformer)
    segments = [
        {"base_ids": (1,), "base_cache": object(), "placeholder_id": "user_question"},
        {"base_ids": (2,), "base_cache": object(), "placeholder_id": "agent_0_current"},
        {"base_ids": (2,), "base_cache": object(), "placeholder_id": "agent_1_current"},
    ]
    matches = ComposeInterleavedReuse([list(s["base_ids"]) for s in segments], [1, 2, 2])
    parts = method._MakeParts(segments, matches)
    assert [part["segment"] for part in parts] == segments
    assert parts[1]["segment"]["base_cache"] is not parts[2]["segment"]["base_cache"]
    # A single immutable prepared chunk remains reusable more than once.
    repeated = method._MakeParts(segments[:1], [(0, [1]), (0, [1])])
    assert all(part["segment"] is segments[0] for part in repeated)


def test_terminal_stop_override_replaces_default_and_keeps_explicit_stops():
    from helpers.backends.Tokenization import TerminalStopIdsFromTokenizer, OutputFragmentIds
    tokenizer = SimpleNamespace(eos_token_id=2)
    stops = TerminalStopIdsFromTokenizer(tokenizer, {"eos_token_id": 9}, [8])
    assert stops == [8, 9]
    assert OutputFragmentIds([10, 2], stops) == [10, 2]
    assert OutputFragmentIds([2, 10, 9], stops) == [2, 10]
    assert TerminalStopIdsFromTokenizer(tokenizer, {"eos_token_id": []}) == []
    assert TerminalStopIdsFromTokenizer(tokenizer) == [2]


def test_transformers_returncache_keeps_raw_ids_and_token_return_keeps_internal_special():
    generator = object.__new__(TransformersGenerator)
    generator._torch = SimpleNamespace(no_grad=nullcontext)
    generator._Generator = lambda: object()
    forwarded = []
    generated = iter([7, 99, 8])
    generator._SampleNextToken = lambda *_args: next(generated)
    generator.Prefill = lambda _ids: SimpleNamespace(
        logits=torch.zeros((1, 1, 1)), past_key_values=object()
    )
    def forward(ids, *_args, **_kwargs):
        forwarded.append(list(ids))
        return SimpleNamespace(logits=torch.zeros((1, 1, 1)), past_key_values=object())

    generator.Forward = forward
    generator.tokenizer = SimpleNamespace(
        all_special_ids=[99],
        decode=lambda ids, skip_special_tokens: "visible",
    )
    generator.eosIds = {88}

    cached = generator.Generate(
        [1], maxNewTokens=3, returnCache=True, returnTokenIds=True
    )
    assert cached[-1] == [7, 99, 8]
    assert forwarded[-1] == [8]  # final budget token receives its cache row

    forwarded.clear()
    generated = iter([7, 99, 8])
    visible = generator.Generate([1], maxNewTokens=3, returnTokenIds=True)
    assert visible[-1] == [7, 99, 8]
    assert forwarded[-1] == [99]  # no generated KV is retained on this call

    generator.eosIds = {88}
    generated = iter([7, 99, 88])
    stopped = generator.Generate([1], maxNewTokens=3, returnCache=True)
    assert stopped[-1] == [7, 99, 88]
    assert forwarded[-1] == [99]  # terminal EOS remains outside the KV fragment
