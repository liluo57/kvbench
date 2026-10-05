"""Tests for config."""

import pytest

from core.Config import DatasetDir, Get, LoadConfig, ModelPath


def test_config_loading_is_cached_per_path(tmp_path):
    configPath = tmp_path / "config.yaml"
    configPath.write_text("ModelPath: first\n", encoding="utf-8")

    first = LoadConfig(configPath)
    configPath.write_text("ModelPath: second\n", encoding="utf-8")

    assert first == {"ModelPath": "first"}
    assert LoadConfig(configPath) is first


def test_dataset_and_model_paths_use_explicit_config(tmp_path):
    datasetRoot = tmp_path / "datasets"
    expected = datasetRoot / "ruler"
    expected.mkdir(parents=True)
    config = {"DatasetPath": str(datasetRoot), "ModelPath": "/models/default"}

    assert DatasetDir("ruler", config=config) == expected
    assert ModelPath(config=config) == "/models/default"

    with pytest.raises(FileNotFoundError, match="missing"):
        DatasetDir("missing", config=config)


def test_get_honors_an_explicit_empty_config():
    assert Get("ModelPath", "fallback", config={}) == "fallback"
