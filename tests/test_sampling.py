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
