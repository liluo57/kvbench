"""Token-native HYPIC and LinearKV over the bundled hybrid-attention runtime."""
from __future__ import annotations

import math
import os
import time
from typing import Any, Optional, Sequence

from core.Config import Get, MaxModelLen, ModelPath
from core.Method import Method, ResolveMaxNewTokens
from core.Result import NumOutputTokensKey, Result, TotalTimeKey, TtftKey
from core.Sampling import ResolveSamplingConfig, SglangSamplingParams
from helpers.backends.ModelAdapter import arch_family, _tokenizer
from helpers.backends.Prompt import ComposeInterleavedReuse
from helpers.backends.Tokenization import OutputFragmentIds, TerminalStopIdsFromTokenizer

_PIC_MODES = {"addition", "transition", "transition_rope", "transition_rope_recompute"}
_WARMUP_TAIL = "\n[KVBench HYPIC cache warmup]\n"
_QWEN35_SYSTEM = "You are a helpful assistant."


def _MaxMambaCacheSize() -> Optional[int]:
    cfg = Get("Hypic", {}) or {}
    if not isinstance(cfg, dict):
        raise TypeError("Hypic in config.yaml must be a mapping")
    value = cfg.get("MaxMambaCacheSize", 32)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("Hypic.MaxMambaCacheSize must be an integer or -1")
    if value == -1:
        return None
    if value < 1:
        raise ValueError("Hypic.MaxMambaCacheSize must be positive or -1")
    return value


def _BuildHypicEngineKwargs(
    modelPath: str, gpuIds: Sequence[int], *, dtype: str,
    maxModelLen: int, memFractionStatic: float, picMode: str,
    fullPrefill: bool, seamTokens: float = 8, recompRatio: Optional[float] = None,
    selector: str = "epic", selectorLayer: Optional[int] = None,
) -> dict[str, Any]:
    options = dict(
        model_path=modelPath, dtype=dtype, tp_size=len(gpuIds),
        context_length=maxModelLen, max_prefill_tokens=maxModelLen,
        max_running_requests=1, mem_fraction_static=memFractionStatic,
        trust_remote_code=True, enable_multimodal=False,
        page_size=1, chunked_prefill_size=-1,
        cuda_graph_backend_prefill="disabled", log_level="error",
        attention_backend="flashinfer",
        pic_enable=not fullPrefill,
    )
    maxMamba = _MaxMambaCacheSize()
    if maxMamba is not None:
        options["max_mamba_cache_size"] = maxMamba
    if fullPrefill:
        options.update(disable_radix_cache=False,
                       mamba_radix_cache_strategy="no_buffer",
                       disable_overlap_schedule=True)
    else:
        options.update(pic_mode=picMode, pic_seam_sink=seamTokens)
        if picMode == "linearkv":
            options["pic_epic_ratio"] = recompRatio
            options["pic_linearkv_selector"] = selector
            options["pic_linearkv_selector_layer"] = selectorLayer
            # Selector probes and replay share request-private slots. Keep
            # their execution on the same stream without scheduler overlap.
            if selector != "epic":
                options["disable_overlap_schedule"] = True
    return options


def _CreateHypicEngine(modelPath, gpuIds, **kwargs):
    # CUDA and scheduler processes must be initialized only in the Method worker.
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in gpuIds)
    from .backend.kvbench_hypic.srt.entrypoints.engine import Engine
    return Engine(**_BuildHypicEngineKwargs(modelPath, gpuIds, **kwargs))


def _TokenIds(value: Sequence[int]) -> list[int]:
    if not isinstance(value, (list, tuple)) or any(
        isinstance(token, bool) or not isinstance(token, int) or token < 0
        for token in value
    ):
        raise TypeError("HYPIC inputs must be sequences of non-negative integer token IDs")
    return list(value)


def _PackSegments(parts: Sequence[Sequence[int]]) -> tuple[list[int], list[tuple[int, int]]]:
    ids: list[int] = []
    boundaries: list[tuple[int, int]] = []
    for part in parts:
        if part:
            start = len(ids)
            ids.extend(part)
            boundaries.append((start, len(ids)))
    return ids, boundaries


class _PicMethod(Method):
    backend = "hypic"
    method_metrics = ("reuse_ratio",)
    maxCaseBatchSize = 1

    def __init__(self, gpuNums=1, perfWeight=1.0, *, memFractionStatic=0.80,
                 dtype="bfloat16", picMode="transition_rope_recompute",
                 seamTokens=8, recompRatio=None, fullPrefill=False, tag=None,
                 selector="epic", selectorLayer=None,
                 **legacy):
        super().__init__(gpuNums=gpuNums, perfWeight=perfWeight, tag=tag)
        legacy.pop("maxNewTokens", None)
        legacy.pop("separator", None)
        if legacy:
            raise TypeError(f"unexpected keyword argument {next(iter(legacy))!r}")
        if isinstance(memFractionStatic, bool) or not 0 < float(memFractionStatic) < 1:
            raise ValueError("memFractionStatic must be between 0 and 1")
        if not isinstance(fullPrefill, bool):
            raise TypeError("fullPrefill must be a bool")
        if isinstance(seamTokens, bool) or not math.isfinite(float(seamTokens)) or seamTokens < 0:
            raise ValueError("seamTokens must be a finite non-negative number")
        _MaxMambaCacheSize()
        self.modelPath = ModelPath()
        self.samplingConfig = ResolveSamplingConfig(self.modelPath)
        self.maxModelLen = MaxModelLen()
        self.memFractionStatic = float(memFractionStatic)
        self.dtype, self.picMode = dtype, picMode
        self.seamTokens, self.recompRatio = float(seamTokens), recompRatio
        self.fullPrefill = fullPrefill
        self.selector, self.selectorLayer = selector, selectorLayer
        self.engine = None
        self._states: list[list[list[int]]] = []
        self._systemIds: list[int] = []
        self._tailIds: list[int] = []
        self._terminalStopIds: list[int] = []
        self._tokenizer = None

    def Initialize(self, gpuIds: Sequence[int]) -> None:
        super().Initialize(gpuIds)
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in self.gpuIds)
        if not self.modelPath:
            raise ValueError("ModelPath is empty; set it in config.yaml")
        from .backend.kvbench_hypic.srt.utils.hf_transformers_utils import get_config
        config = get_config(self.modelPath, trust_remote_code=True)
        textConfig = getattr(config, "text_config", config)
        if getattr(config, "quantization_config", None) or getattr(textConfig, "quantization_config", None):
            raise ValueError("The bundled HYPIC backend requires an unquantized FP16/BF16 checkpoint")
        if self.picMode == "linearkv":
            architectures = getattr(config, "architectures", []) or []
            if not any(arch.startswith("Qwen3_5") for arch in architectures):
                raise ValueError(
                    "LinearKV requires a Qwen3_5 architecture (including compatible "
                    "Qwen3.6 checkpoints); OlmoHybridForCausalLM is not implemented"
                )
        self._tokenizer = _tokenizer(self.modelPath)
        self._terminalStopIds = TerminalStopIdsFromTokenizer(self._tokenizer, self.samplingConfig)
        self._systemIds = self._tokenizer.encode(_QWEN35_SYSTEM, add_special_tokens=False)
        self._tailIds = self._tokenizer.encode(_WARMUP_TAIL, add_special_tokens=False)
        self.engine = _CreateHypicEngine(
            self.modelPath, self.gpuIds, dtype=self.dtype,
            maxModelLen=self.maxModelLen, memFractionStatic=self.memFractionStatic,
            picMode=self.picMode, fullPrefill=self.fullPrefill,
            seamTokens=self.seamTokens, recompRatio=self.recompRatio,
            selector=self.selector, selectorLayer=self.selectorLayer,
        )

    def _Warmup(self, chunk: list[int]) -> None:
        if self.picMode == "linearkv":
            parts = [chunk]
        else:
            # HYPIC's system/tail priming is offline only. Run does not rewrite
            # the Task's rendered prompt or re-tokenize its prepared fragments.
            parts = ([self._systemIds] if arch_family(self.modelPath) == "qwen3_5" else [])
            parts = [*parts, chunk, self._tailIds]
        ids, segments = _PackSegments(parts)
        self._Generate(ids, segments, maxNewTokens=1)

    def Prepare(self, data: list[list[list[int]]]) -> None:
        if len(data) > 1:
            raise ValueError("HYPIC and LinearKV accept one case per cache lifecycle")
        self._states = [[_TokenIds(chunk) for chunk in chunks if chunk] for chunks in data]
        if not self.fullPrefill:
            for chunks in self._states:
                for chunk in chunks:
                    self._Warmup(chunk)

    def Run(self, data: list[list[int]], retainOutput=None, maxNewTokens=None) -> list[Result]:
        if len(data) > 1:
            raise ValueError("HYPIC and LinearKV accept one case per cache lifecycle")
        if retainOutput is not None and any(
            bool(retainOutput[index])
            for index in range(min(len(data), len(retainOutput)))
        ):
            raise NotImplementedError(
                "HYPIC and LinearKV do not support native output KV retention; "
                "output tokens cannot be retained by warmup or re-prefill"
            )
        maxNewTokens = ResolveMaxNewTokens(maxNewTokens)
        if len(self._states) != len(data):
            self._states = [[] for _ in data]
        results = []
        for index, prompt in enumerate(data):
            started = time.perf_counter()
            ids = _TokenIds(prompt)
            if not ids:
                raise ValueError("HYPIC prompt token IDs cannot be empty")
            prepared = self._states[index]
            parts = [(None, ids)] if self.fullPrefill else ComposeInterleavedReuse(prepared, ids)
            _, segments = _PackSegments([part for _, part in parts])
            output, outputIds, ttft, total, meta = self._Generate(
                ids, None if self.fullPrefill else segments,
                maxNewTokens=maxNewTokens, onlineStart=started,
            )
            cached = min(len(ids), max(0, int(meta.get("cached_tokens", 0) or 0)))
            metadata = dict(backend=self.backend, n_input=len(ids),
                            num_cached_tokens=cached, reuse_ratio=cached / len(ids))
            if self.fullPrefill:
                metadata["full_prefill"] = True
            else:
                metadata.update(pic_mode=self.picMode, n_pic_segments=len(segments),
                                matched_prepared_segments=sum(i is not None for i, _ in parts))
                if self.picMode == "linearkv":
                    metadata.update(selector=self.selector, target_recompute_ratio=self.recompRatio)
                    selections = meta.get("linearkv_selection") or []
                    if selections:
                        metadata.update(selections[-1])
            result = Result(output=output, output_ids=outputIds, metadata=metadata,
                            performance={TtftKey: ttft, TotalTimeKey: total,
                                         NumOutputTokensKey: int(meta["completion_tokens"])})
            results.append(result)
        return results

    def _Generate(self, ids, segments, *, maxNewTokens, onlineStart=None):
        if self.engine is None:
            raise RuntimeError(f"{type(self).__name__} is not initialized")
        started = time.perf_counter() if onlineStart is None else onlineStart
        ttft, final, cached = None, None, 0
        for item in self.engine.generate(
            input_ids=ids, pic_segments=segments,
            sampling_params=SglangSamplingParams(self.samplingConfig, maxNewTokens), stream=True,
        ):
            final = item
            if item.get("output_ids") and ttft is None:
                ttft = time.perf_counter() - started
            cached = max(cached, int((item.get("meta_info") or {}).get("cached_tokens", 0) or 0))
        total = time.perf_counter() - started
        if final is None:
            raise RuntimeError("HYPIC returned no generation output")
        meta = dict(final.get("meta_info") or {})
        rawOutputIds = list(final.get("output_ids") or [])
        completionTokens = meta.get("completion_tokens")
        if (
            isinstance(completionTokens, bool)
            or not isinstance(completionTokens, int)
            or completionTokens < 0
        ):
            # Compatibility with lightweight/fake engines that omit SGLang's
            # cumulative completion count. Count before terminal stop stripping.
            completionTokens = len(rawOutputIds)
        meta["completion_tokens"] = completionTokens
        stopIds = list(self._terminalStopIds)
        reason = meta.get("finish_reason") or {}
        if isinstance(reason, dict) and reason.get("type") == "stop":
            matched = reason.get("matched")
            if isinstance(matched, int) and not isinstance(matched, bool):
                stopIds.append(matched)
        outputIds = OutputFragmentIds(rawOutputIds, stopIds)
        meta["cached_tokens"] = cached
        output = (self._tokenizer.decode(outputIds, skip_special_tokens=True)
                  if self._tokenizer is not None else final.get("text") or "")
        return output, outputIds, total if ttft is None else ttft, total, meta

    def Reset(self) -> None:
        self._states = []
        if self.engine is not None:
            response = self.engine.flush_cache()
            if hasattr(response, "success") and not response.success:
                raise RuntimeError(f"HYPIC cache flush failed: {response.message}")

    def Close(self) -> None:
        self._states = []
        engine, self.engine = self.engine, None
        if engine is not None:
            engine.shutdown()


class HypicMethod(_PicMethod):
    name = "hypic"

    def __init__(self, gpuNums=1, perfWeight=1.0, *, picMode="transition_rope_recompute", **kwargs):
        if picMode not in _PIC_MODES:
            raise ValueError(f"unknown picMode={picMode!r}; expected one of {sorted(_PIC_MODES)}")
        super().__init__(gpuNums, perfWeight, picMode=picMode, **kwargs)


class LinearKVMethod(_PicMethod):
    name = "linearkv"
    method_metrics = ("reuse_ratio", "recompute_ratio", "num_recomputed_tokens")

    def __init__(self, gpuNums=1, perfWeight=1.0, *, recompRatio=0.2,
                 selector="epic", selectorLayer=None, tag=None, **kwargs):
        if isinstance(recompRatio, bool) or not math.isfinite(float(recompRatio)) or not 0 <= recompRatio <= 1:
            raise ValueError("recompRatio must be a finite number in [0, 1]")
        if not isinstance(selector, str) or selector.lower() not in {"epic", "cacheblend", "prophetkv"}:
            raise ValueError("selector must be epic, cacheblend, or prophetkv")
        selector = selector.lower()
        if selectorLayer is not None and (
            isinstance(selectorLayer, bool) or not isinstance(selectorLayer, int) or selectorLayer < 0
        ):
            raise ValueError("selectorLayer must be a non-negative layer index or None")
        if selectorLayer is not None and selector != "cacheblend":
            raise ValueError("selectorLayer is only supported by the CacheBlend selector")
        if tag is None:
            tag = "full_prefill" if kwargs.get("fullPrefill") else f"{selector},r={float(recompRatio):g}"
            if selectorLayer is not None and not kwargs.get("fullPrefill"):
                tag += f",layer={selectorLayer}"
        super().__init__(gpuNums, perfWeight, picMode="linearkv", recompRatio=float(recompRatio),
                         selector=selector, selectorLayer=selectorLayer, tag=tag, **kwargs)
