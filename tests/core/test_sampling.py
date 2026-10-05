"""Tests for sampling."""

import json

import pytest

from core.Sampling import (
    ResolveSamplingConfig,
    SglangSamplingParams,
    TransformersGenerationKwargs,
    VllmSamplingParams,
)


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
