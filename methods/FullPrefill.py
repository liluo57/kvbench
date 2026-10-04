"""FullPrefill — the baseline that recomputes the whole prompt every query.

``Prepare`` stores no reusable document state; ``Run(data)`` submits the complete
prompt. Native backend prefix-cache hits, when enabled, are reported separately.
Two backends:

- :class:`FullPrefillTransformer` — plain HuggingFace ``transformers``.
- :class:`FullPrefillVllm` — the system vLLM.

This is the correctness / TTFT baseline that cacheblend's ``blend_niah.py``
compares its fused runs against. The constructor declares a strict GPU count;
the Engine assigns concrete ids when the worker initializes.

The vLLM backend submits the batch as concurrent requests. The Transformers
backend processes prompts sequentially through the manual token-by-token
``TransformersGenerator.Generate`` loop to measure the first-token boundary.

``Prepare`` is a no-op — FullPrefill recomputes everything, so warm-up text is
irrelevant.
"""

import time
from typing import List, Mapping, Optional, Sequence

from core.Config import Get, MaxModelLen, ModelPath as DefaultModelPath
from core.Method import Method, ResolveMaxNewTokens
from core.Result import NumOutputTokensKey, Result, TotalTimeKey, TtftKey
from core.Sampling import ResolveSamplingConfig

from helpers.backends.TransformersHelper import TransformersGenerator
from helpers.backends.VllmHelper import CreateLlm, GenerateBatch


def _ConfiguredMaxNumSeqs() -> Optional[int]:
    """Read the optional FullPrefillVllm max sequence count from config."""
    config = Get("FullPrefillVllm", {}) or {}
    if not isinstance(config, Mapping):
        raise TypeError("FullPrefillVllm config must be a mapping")

    value = config.get("MaxNumSeqs")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            "FullPrefillVllm.MaxNumSeqs must be a positive integer "
            f"(got {type(value).__name__}: {value!r})"
        )
    if value < 1:
        raise ValueError(
            "FullPrefillVllm.MaxNumSeqs must be a positive integer "
            f"(got {value})"
        )
    return value


class _FullPrefillBase(Method):
    """Shared FullPrefill logic. ``Prepare`` is a no-op (the whole prompt is
    recomputed); each backend's ``Run`` returns one result per prompt."""

    #: backend identifier for ``Result.metadata``.
    backend = "transformers"
    method_metrics = ("reuse_ratio",)

    def __init__(
        self,
        *,
        gpuNums: int,
        perfWeight: float,
        maxGpuNums: int | None,
        dtype: str,
        tag: Optional[str] = None,
    ):
        super().__init__(
            gpuNums=gpuNums,
            perfWeight=perfWeight,
            maxGpuNums=maxGpuNums,
            tag=tag,
        )
        # Model path is config-only — switch models via config.yaml.
        self.modelPath = DefaultModelPath()
        self.samplingConfig = ResolveSamplingConfig(self.modelPath)
        self.dtype = dtype

    # ---------------------------------------------------------------- Method
    def Prepare(self, data: List[List[List[int]]]) -> None:
        pass  # FullPrefill recomputes everything; warm-up is irrelevant

    def Run(
        self,
        data: List[List[int]],
        retainOutput: Optional[List[bool]] = None,
        maxNewTokens: Optional[int] = None,
    ) -> List[Result]:
        raise NotImplementedError

    def _Result(self, text, ttft, total, nTokens, *, metadata, outputIds=None) -> Result:
        return Result(
            output=text,
            performance={
                TtftKey: ttft,
                NumOutputTokensKey: nTokens,
                TotalTimeKey: total,
            },
            metadata={"backend": self.backend, "reuse_ratio": 0.0, **metadata},
            output_ids=outputIds,
        )

    def Close(self) -> None:
        if hasattr(self, "_gen"):
            self._gen = None
        if hasattr(self, "llm"):
            self.llm = None
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


class FullPrefillTransformer(_FullPrefillBase):
    """Full recomputation of the whole prompt over plain ``transformers``."""

    name = "full_prefill"
    backend = "transformers"

    def __init__(
        self,
        gpuNums: int = 1,
        perfWeight: float = 1.0,
        *,
        dtype: str = "bfloat16",
        tag: Optional[str] = None,
    ):
        super().__init__(
            gpuNums=gpuNums,
            perfWeight=perfWeight,
            maxGpuNums=1,
            dtype=dtype,
            tag=tag,
        )
        self._gen = None

    def Initialize(self, gpuIds: Sequence[int]) -> None:
        super().Initialize(gpuIds)
        self._gen = TransformersGenerator(
            self.modelPath,
            self.gpuIds,
            dtype=self.dtype,
            samplingConfig=self.samplingConfig,
        )

    def Run(
        self,
        data: List[List[int]],
        retainOutput: Optional[List[bool]] = None,
        maxNewTokens: Optional[int] = None,
    ) -> List[Result]:
        runStart = time.perf_counter()
        maxNewTokens = ResolveMaxNewTokens(maxNewTokens)
        # Task supplies the canonical segmented-token sequence.  Keep all
        # prompt tokenization outside the online Method boundary.
        idsList = [list(ids) for ids in data]
        # Process prompts sequentially and include time spent serving earlier
        # prompts in the same Run call. Generate() measures the first-token
        # boundary; GenerateBatch() also uses this sequential generation loop.
        batchOut = []
        for index, ids in enumerate(idsList):
            generationStart = time.perf_counter()
            generated = self._gen.Generate(
                ids, maxNewTokens=maxNewTokens, returnTokenIds=True
            )
            text, backendTtft, total, nTokens, outputIds = generated
            batchOut.append(
                (
                    text,
                    generationStart - runStart + backendTtft,
                    generationStart - runStart + total,
                    nTokens,
                    outputIds,
                )
            )
        return [
            self._Result(
                text, ttft, total, nTokens, metadata={"n_input": len(ids)},
                outputIds=outputIds,
            )
            for (text, ttft, total, nTokens, outputIds), ids in zip(batchOut, idsList)
        ]


class FullPrefillVllm(_FullPrefillBase):
    """Full recomputation of the whole prompt on the system vLLM."""

    name = "full_prefill_vllm"
    backend = "vllm"

    def __init__(
        self,
        gpuNums: int = 1,
        perfWeight: float = 1.0,
        *,
        gpuMemoryUtilization: float = 0.7,
        enforceEager: bool = False,
        chatTemplate: Optional[str] = None,
        # Pass ``True`` for multimodal checkpoints that ship a vision tower
        # (Qwen3.5 / Qwen3.8 etc., arch = ``*ForConditionalGeneration``) when
        # only the language model will be exercised. Saves the vision encoder
        # weights + mm_profiler scratch on each rank.
        languageModelOnly: bool = False,
        maxNumSeqs: Optional[int] = None,
        tag: Optional[str] = None,
    ):
        super().__init__(
            gpuNums=gpuNums,
            perfWeight=perfWeight,
            maxGpuNums=None,
            dtype="bfloat16",
            tag=tag,
        )
        self.gpuMemoryUtilization = gpuMemoryUtilization
        self.maxModelLen = MaxModelLen()
        self.enforceEager = enforceEager
        # None -> CreateLlm auto-detects ``<model>/chat_template.jinja``.
        self.chatTemplate = chatTemplate
        self.languageModelOnly = languageModelOnly
        # An explicit constructor value is useful for programmatic callers;
        # normal runs read the backend limit from config.yaml.
        self.maxNumSeqs = (
            _ConfiguredMaxNumSeqs() if maxNumSeqs is None else maxNumSeqs
        )
        self.llm = None

    def Initialize(self, gpuIds: Sequence[int]) -> None:
        super().Initialize(gpuIds)
        self.llm = CreateLlm(
            self.modelPath,
            self.gpuIds,
            gpuMemoryUtilization=self.gpuMemoryUtilization,
            maxModelLen=self.maxModelLen,
            tensorParallelSize=self.gpuNums,
            enforceEager=self.enforceEager,
            chatTemplate=self.chatTemplate,
            languageModelOnly=self.languageModelOnly,
            maxNumSeqs=self.maxNumSeqs,
        )

    def Run(
        self,
        data: List[List[int]],
        retainOutput: Optional[List[bool]] = None,
        maxNewTokens: Optional[int] = None,
    ) -> List[Result]:
        runStart = time.perf_counter()
        maxNewTokens = ResolveMaxNewTokens(maxNewTokens)
        batchOut = GenerateBatch(
            self.llm,
            [list(ids) for ids in data],
            maxNewTokens,
            self.samplingConfig,
            onlineStart=runStart,
        )
        results = []
        for generation, ids in zip(batchOut, data):
            nInput = len(ids)
            results.append(
                self._Result(
                    generation.text,
                    generation.ttft,
                    generation.totalTime,
                    generation.numTokens,
                    metadata={
                        "n_input": nInput,
                        "num_cached_tokens": generation.numCached,
                        "reuse_ratio": (
                            generation.numCached / nInput if nInput else 0.0
                        ),
                        "finish_reason": generation.finishReason,
                        "stop_reason": generation.stopReason,
                    },
                    outputIds=generation.outputIds,
                )
            )
        return results
