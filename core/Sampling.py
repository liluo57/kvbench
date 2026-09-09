"""Shared model sampling configuration.

The benchmark has several generation backends, but sampling is a property of
the model run rather than of a particular cache implementation.  This module
resolves the ``ModelConfig`` section from ``config.yaml`` and translates the
result into the small set of arguments understood by each backend.

``default`` follows the model's ``generation_config.json``.  A few model
repositories put generation fields directly in ``config.json`` instead, so
that file is also inspected as a fallback.  ``override`` takes the values
under ``ModelConfig`` in the repository config.  ``greedy`` is deliberately
backend-independent and always means argmax decoding.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

from . import Config


_MODES = {"greedy", "default", "override"}

# Keep this list intentionally small and portable.  These fields are shared by
# Transformers, vLLM, and SGLang; backend-specific request fields such as
# max_tokens are supplied by the caller.
_ALIASES = {
    "do_sample": ("do_sample", "dosample"),
    "temperature": ("temperature", "temp"),
    "top_p": ("top_p", "topp"),
    "top_k": ("top_k", "topk"),
    "min_p": ("min_p", "minp"),
    "typical_p": ("typical_p", "typicalp"),
    "repetition_penalty": ("repetition_penalty", "repetitionpenalty"),
    "presence_penalty": ("presence_penalty", "presencepenalty"),
    "frequency_penalty": ("frequency_penalty", "frequencypenalty"),
    "seed": ("seed",),
    "eos_token_id": ("eos_token_id", "eos"),
}

_DEFAULTS = {
    # These are the standard Transformers GenerationConfig defaults.  Model
    # files normally provide the values explicitly; these make a missing or
    # partial generation config deterministic and consistent across backends.
    "do_sample": False,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 50,
}


def _CanonicalValues(source: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract supported sampling fields from a mapping, with aliases."""
    result: Dict[str, Any] = {}
    lowered = {str(key).lower(): value for key, value in source.items()}
    for canonical, aliases in _ALIASES.items():
        for alias in aliases:
            if alias in lowered:
                result[canonical] = lowered[alias]
                break
    return result


def _ReadJson(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _ModelGenerationValues(modelPath: Optional[Union[str, Path]]) -> Dict[str, Any]:
    """Read generation values from a local model directory.

    ``generation_config.json`` is the canonical HuggingFace location.  The
    fallback to ``config.json`` is needed for model exports that embed those
    fields there, and also matches the way several vLLM model configs expose
    them.
    """
    if not modelPath:
        return {}
    modelDir = Path(modelPath).expanduser()
    if not modelDir.is_dir():
        return {}

    modelConfig = _ReadJson(modelDir / "config.json")
    values: Dict[str, Any] = {}
    for container in (
        modelConfig,
        modelConfig.get("text_config"),
        modelConfig.get("generation_config"),
    ):
        if isinstance(container, Mapping):
            values.update(_CanonicalValues(container))

    # The dedicated generation config takes precedence over fields embedded in
    # the model config, just as AutoModel/GenerationConfig does.
    values.update(_CanonicalValues(_ReadJson(modelDir / "generation_config.json")))
    return values


def _Validate(values: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate and normalize user/model values before backend translation."""
    result = dict(values)

    if not isinstance(result.get("do_sample", False), bool):
        raise TypeError("ModelConfig.do_sample must be a bool")

    for key in ("temperature", "top_p", "min_p", "typical_p"):
        if key not in result or result[key] is None:
            continue
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"ModelConfig.{key} must be a number")
        if key == "temperature" and value < 0:
            raise ValueError("ModelConfig.temperature must be non-negative")
        if key != "temperature" and not 0 <= value <= 1:
            raise ValueError(f"ModelConfig.{key} must be between 0 and 1")
        result[key] = float(value)

    if "top_k" in result and result["top_k"] is not None:
        value = result["top_k"]
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("ModelConfig.top_k must be an integer")
        if value < -1:
            raise ValueError("ModelConfig.top_k must be -1 or non-negative")

    for key in ("repetition_penalty", "presence_penalty", "frequency_penalty"):
        if key not in result or result[key] is None:
            continue
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"ModelConfig.{key} must be a number")
        if key == "repetition_penalty" and value <= 0:
            raise ValueError("ModelConfig.repetition_penalty must be positive")
        result[key] = float(value)

    if "seed" in result and result["seed"] is not None:
        value = result["seed"]
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("ModelConfig.seed must be an integer")

    return {key: value for key, value in result.items() if value is not None}


def GreedySamplingConfig() -> Dict[str, Any]:
    """Return the canonical backend-neutral greedy configuration."""
    return {"mode": "greedy", "do_sample": False, "temperature": 0.0}


def ResolveSamplingConfig(
    modelPath: Optional[Union[str, Path]] = None,
    *,
    config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve ``ModelConfig`` for ``modelPath``.

    The returned dictionary is JSON-serializable because it is also passed to
    the out-of-process CacheBlend helper.
    """
    source = Config.LoadConfig() if config is None else config
    raw = source.get("ModelConfig", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise TypeError("ModelConfig must be a mapping")

    mode = str(raw.get("mode", "greedy")).strip().lower()
    if mode not in _MODES:
        raise ValueError(
            f"ModelConfig.mode must be one of {sorted(_MODES)} (got {mode!r})"
        )
    if mode == "greedy":
        return GreedySamplingConfig()

    if mode == "default":
        values = dict(_DEFAULTS)
        values.update(_ModelGenerationValues(modelPath))
    else:
        # An override is a complete sampling policy.  Unspecified fields use
        # the portable defaults, not whichever backend happens to be running.
        values = dict(_DEFAULTS)
        values["do_sample"] = True
        values.update(_CanonicalValues(raw))

    values = _Validate(values)
    return {"mode": mode, **values}


def IsGreedy(samplingConfig: Optional[Mapping[str, Any]]) -> bool:
    """Whether a resolved or legacy sampling config requires argmax."""
    if not samplingConfig:
        return True
    return (
        samplingConfig.get("mode") == "greedy"
        or not bool(samplingConfig.get("do_sample", False))
        or float(samplingConfig.get("temperature", 1.0)) == 0
    )


def VllmSamplingParams(
    samplingConfig: Optional[Mapping[str, Any]], maxTokens: int
) -> Dict[str, Any]:
    """Translate the shared config to ``vllm.SamplingParams`` kwargs."""
    if IsGreedy(samplingConfig):
        return {"temperature": 0, "max_tokens": maxTokens}

    config = samplingConfig or {}
    result: Dict[str, Any] = {"max_tokens": maxTokens}
    for key in (
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "repetition_penalty",
        "presence_penalty",
        "frequency_penalty",
        "seed",
    ):
        if key in config:
            result[key] = config[key]
    return result


def SglangSamplingParams(
    samplingConfig: Optional[Mapping[str, Any]], maxNewTokens: int
) -> Dict[str, Any]:
    """Translate the shared config to SGLang ``sampling_params``."""
    if IsGreedy(samplingConfig):
        return {"temperature": 0, "max_new_tokens": maxNewTokens}

    config = samplingConfig or {}
    result: Dict[str, Any] = {"max_new_tokens": maxNewTokens}
    for key in (
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "repetition_penalty",
        "presence_penalty",
        "frequency_penalty",
        "seed",
    ):
        if key in config:
            result[key] = config[key]
    return result


def TransformersGenerationKwargs(
    samplingConfig: Optional[Mapping[str, Any]], maxNewTokens: int
) -> Dict[str, Any]:
    """Translate the shared config to ``transformers.generate`` kwargs."""
    config = samplingConfig or GreedySamplingConfig()
    result: Dict[str, Any] = {
        "max_new_tokens": maxNewTokens,
        "do_sample": not IsGreedy(config),
    }
    if not IsGreedy(config):
        for key in (
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "typical_p",
            "repetition_penalty",
        ):
            if key in config:
                result[key] = config[key]
    if "eos_token_id" in config:
        result["eos_token_id"] = config["eos_token_id"]
    return result


__all__ = [
    "GreedySamplingConfig",
    "IsGreedy",
    "ResolveSamplingConfig",
    "SglangSamplingParams",
    "TransformersGenerationKwargs",
    "VllmSamplingParams",
]
