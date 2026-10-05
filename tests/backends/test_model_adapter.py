"""Tests for model adapter."""

import json

from helpers.backends import ModelAdapter


def test_model_adapter_arch_family_reads_config_json(tmp_path):
    """``arch_family`` inspects ``<modelPath>/config.json`` ``architectures``."""
    modelDir = tmp_path / "model"
    modelDir.mkdir()
    (modelDir / "config.json").write_text(
        json.dumps({"architectures": ["Qwen3ForCausalLM"]}),
        encoding="utf-8",
    )
    ModelAdapter.arch_family.cache_clear()
    try:
        assert ModelAdapter.arch_family(str(modelDir)) == "qwen3"
    finally:
        ModelAdapter.arch_family.cache_clear()

    (modelDir / "config.json").write_text(
        json.dumps({"architectures": ["MuseGlimmerForCausalLM"]}),
        encoding="utf-8",
    )
    ModelAdapter.arch_family.cache_clear()
    try:
        assert ModelAdapter.arch_family(str(modelDir)) == "muse_glimmer"
    finally:
        ModelAdapter.arch_family.cache_clear()


def test_model_adapter_thinking_kwargs_translate_per_arch():
    """The CoT toggle is translated to the kwarg name the model's jinja reads."""
    ModelAdapter.set_arch_for_testing("qwen3")
    assert ModelAdapter._thinking_kwargs(False, "") == {"enable_thinking": False}
    assert ModelAdapter._thinking_kwargs(True, "") == {"enable_thinking": True}
    assert ModelAdapter._thinking_kwargs(None, "") == {}

    ModelAdapter.set_arch_for_testing("muse_glimmer")
    assert ModelAdapter._thinking_kwargs(False, "") == {"reasoning_strength": "low"}
    assert ModelAdapter._thinking_kwargs(True, "") == {"reasoning_strength": "high"}
    assert ModelAdapter._thinking_kwargs(None, "") == {}

    # Unknown archs ignore the kwarg entirely.
    ModelAdapter.set_arch_for_testing("other")
    assert ModelAdapter._thinking_kwargs(False, "") == {}
    assert ModelAdapter._thinking_kwargs(True, "") == {}

    ModelAdapter.set_arch_for_testing(None)


def test_model_adapter_render_chat_includes_assistant_generation_prompt(modelPath):
    """render_chat appends the assistant generation prompt regardless of body."""
    ModelAdapter.set_arch_for_testing("qwen3")
    try:
        out = ModelAdapter.render_chat(
            [{"role": "user", "content": "hi"}], modelPath=modelPath
        )
        assert "hi" in out
        assert "<|im_start|>assistant\n" in out
    finally:
        ModelAdapter.set_arch_for_testing(None)


def test_model_adapter_boundaries_reassemble_the_rendered_user_prompt(modelPath):
    ModelAdapter.set_arch_for_testing("qwen3")
    try:
        body = "boundary marker content"
        composed = (
            ModelAdapter.user_turn_prefix(modelPath)
            + body
            + ModelAdapter.assistant_turn_suffix(modelPath)
        )
        direct = ModelAdapter.render_chat(
            [{"role": "user", "content": body}], modelPath=modelPath
        )
        assert composed == direct
        assert ModelAdapter.user_turn_prefix(modelPath)
        assert body not in ModelAdapter.user_turn_prefix(modelPath)
    finally:
        ModelAdapter.set_arch_for_testing(None)


def test_model_adapter_boundaries_preserve_thinking_toggle(modelPath):
    ModelAdapter.set_arch_for_testing("qwen3_5")
    try:
        thinking = ModelAdapter.assistant_turn_suffix(modelPath, thinking=True)
        no_thinking = ModelAdapter.assistant_turn_suffix(modelPath, thinking=False)
        assert thinking != no_thinking
        assert "<think>" in thinking
        assert "</think>" in no_thinking
    finally:
        ModelAdapter.set_arch_for_testing(None)
