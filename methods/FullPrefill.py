"""FullPrefill — the baseline that recomputes the whole prompt every query.

Nothing is cached or reused: ``Run(data)`` answers the complete prompt ``data``
from scratch. Two backends:

- :class:`FullPrefillTransformer` — plain HuggingFace ``transformers``.
- :class:`FullPrefillVllm` — the system vLLM.

This is the correctness / TTFT baseline that cacheblend's ``blend_niah.py``
compares its fused runs against. The constructor declares a strict GPU count;
the Engine assigns concrete ids when the worker initializes.

Both ``Run`` methods process their whole batch in one call:

- the vLLM backend submits every prompt as a concurrent request and steps them
  together (:func:`helpers.VllmHelper.GenerateBatch`);
- the transformers backend pads the prompts into one batched ``model.generate``
  (:meth:`helpers.TransformersHelper.TransformersGenerator.GenerateBatch`).

``Prepare`` is a no-op — FullPrefill recomputes everything, so warm-up text is
irrelevant.
"""

from typing import List, Optional, Sequence

from core.Config import ModelPath as DefaultModelPath
from core.Method import Method
from core.Result import NumOutputTokensKey, Result, TotalTimeKey, TtftKey

from helpers.TransformersHelper import TransformersGenerator
from helpers.VllmHelper import CreateLlm, EncodeIds, GenerateBatch


class _FullPrefillBase(Method):
    """Shared FullPrefill logic. ``Prepare`` is a no-op (the whole prompt is
    recomputed); ``Run`` is implemented per backend and processes the whole
    batch at once."""

    #: backend identifier for ``Result.metadata``.
    backend = "transformers"

    def __init__(
        self,
        *,
        gpuNums: int,
        perfWeight: float,
        maxGpuNums: int | None,
        maxNewTokens: int,
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
        self.maxNewTokens = maxNewTokens
        self.dtype = dtype

    # ---------------------------------------------------------------- Method
    def Prepare(self, data: List[List[str]]) -> None:
        pass  # FullPrefill recomputes everything; warm-up is irrelevant

    def Run(self, data: List[str], retainOutput: Optional[List[bool]] = None) -> List[Result]:
        raise NotImplementedError

    def _Result(self, text, ttft, total, nTokens, *, metadata) -> Result:
        return Result(
            output=text,
            performance={
                TtftKey: ttft,
                NumOutputTokensKey: nTokens,
                TotalTimeKey: total,
            },
            metadata={"backend": self.backend, **metadata},
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
        maxNewTokens: int = 64,
        dtype: str = "bfloat16",
        tag: Optional[str] = None,
    ):
        super().__init__(
            gpuNums=gpuNums,
            perfWeight=perfWeight,
            maxGpuNums=1,
            maxNewTokens=maxNewTokens,
            dtype=dtype,
            tag=tag,
        )
        self._gen = None

    def Initialize(self, gpuIds: Sequence[int]) -> None:
        super().Initialize(gpuIds)
        self._gen = TransformersGenerator(
            self.modelPath,
            self.gpuIds,
            maxNewTokens=self.maxNewTokens,
            dtype=self.dtype,
        )

    def Run(self, data: List[str], retainOutput: Optional[List[bool]] = None) -> List[Result]:
        idsList = [self._gen.Encode(d) for d in data]
        # The KVCOMM protocol is request-sequential (batch size 1). Generate()
        # exposes the real first-token boundary; GenerateBatch() can only
        # approximate TTFT from total generation time and must not be used for
        # latency comparisons.
        batchOut = [self._gen.Generate(ids) for ids in idsList]
        return [
            self._Result(
                text, ttft, total, nTokens, metadata={"n_input": len(ids)}
            )
            for (text, ttft, total, nTokens), ids in zip(batchOut, idsList)
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
        maxNewTokens: int = 64,
        gpuMemoryUtilization: float = 0.7,
        maxModelLen: int = 40960,
        enforceEager: bool = False,
        chatTemplate: Optional[str] = None,
        tag: Optional[str] = None,
    ):
        super().__init__(
            gpuNums=gpuNums,
            perfWeight=perfWeight,
            maxGpuNums=None,
            maxNewTokens=maxNewTokens,
            dtype="bfloat16",
            tag=tag,
        )
        self.gpuMemoryUtilization = gpuMemoryUtilization
        self.maxModelLen = maxModelLen
        self.enforceEager = enforceEager
        # None -> CreateLlm auto-detects ``<model>/chat_template.jinja``.
        self.chatTemplate = chatTemplate
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
        )

    def Run(self, data: List[str], retainOutput: Optional[List[bool]] = None) -> List[Result]:
        batchOut = GenerateBatch(self.llm, data, self.maxNewTokens)
        results = []
        for (text, ttft, nTokens, total, numCached), prompt in zip(batchOut, data):
            nInput = len(EncodeIds(self.llm, prompt))
            results.append(
                self._Result(
                    text,
                    ttft,
                    total,
                    nTokens,
                    metadata={
                        "n_input": nInput,
                        "num_cached_tokens": numCached,
                        "reuse_ratio": (numCached / nInput) if nInput else 0.0,
                    },
                )
            )
        return results
