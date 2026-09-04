"""HYPIC-backed position-independent cache method.

This adapter drives the in-process ``sglang.Engine`` from the HYPIC checkout.
``Prepare`` submits the original-order reusable segments once.  ``Run`` finds
those segments in the complete prompt (including reordered/interleaved uses),
inserts HYPIC's out-of-band separator between spans, and generates from the
resulting position-independent cache composition.  HYPIC removes the separator
before tokenization, so it is never visible to the model.

HYPIC v1 deliberately never caches the last segment of a request.  Prepare
therefore appends a small throwaway tail so every user-provided segment is
eligible for caching.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from core.Config import Get, ModelPath as DefaultModelPath
from core.Method import Method
from core.Result import NumOutputTokensKey, Result, TotalTimeKey, TtftKey
from helpers.backends.Prompt import ComposeInterleavedReuse


_PIC_MODES = {
    "addition",
    "transition",
    "transition_rope",
    "transition_rope_recompute",
}
_DEFAULT_MAX_MAMBA_CACHE_SIZE = 128
_WARMUP_TAIL = "\n[KVBench HYPIC cache warmup]\n"


def _HypicRepoPath() -> Path:
    config = Get("Hypic", {}) or {}
    return Path(config.get("RepoPath") or "/root/hypic").expanduser().resolve()


def _CreateHypicEngine(
    modelPath: str,
    gpuIds: Sequence[int],
    *,
    dtype: str,
    maxModelLen: int,
    memFractionStatic: float,
    picMode: str,
    separator: str,
    maxMambaCacheSize: int,
    fullPrefill: bool,
):
    """Import HYPIC lazily and create its SGLang engine on ``gpuIds``.

    Keeping the import here is important: Method objects are constructed in
    KVBench's coordinator, while CUDA and HYPIC's child processes must only be
    initialized in the assigned method worker.
    """
    repo = _HypicRepoPath()
    pythonDir = repo / "python"
    if not (pythonDir / "sglang").is_dir():
        raise FileNotFoundError(
            f"HYPIC Python package not found under {pythonDir}; "
            "set Hypic.RepoPath in config.yaml"
        )

    # HYPIC's scheduler children inherit both this import path and the physical
    # GPU visibility mask.  Within the child, tp ranks address logical 0..N-1.
    sys.path.insert(0, str(pythonDir))
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in gpuIds)

    import sglang as sgl

    loaded = Path(sgl.__file__).resolve()
    if pythonDir not in loaded.parents:
        raise RuntimeError(
            f"imported sglang from {loaded}, expected the HYPIC checkout at {pythonDir}"
        )

    engineKwargs = dict(
        model_path=modelPath,
        dtype=dtype,
        tp_size=len(gpuIds),
        context_length=maxModelLen,
        max_prefill_tokens=maxModelLen,
        max_running_requests=1,
        mem_fraction_static=memFractionStatic,
        trust_remote_code=True,
        enable_multimodal=False,
        page_size=1,
        chunked_prefill_size=-1,
        cuda_graph_backend_prefill="disabled",
        log_level="error",
    )
    if fullPrefill:
        # Match HYPIC's own full-recompute control: neither PIC nor the normal
        # radix prefix cache may satisfy any part of the measured request.
        engineKwargs.update(
            pic_enable=False,
            disable_radix_cache=True,
            mamba_radix_cache_strategy="no_buffer",
        )
    else:
        engineKwargs.update(
            pic_enable=True,
            pic_mode=picMode,
            pic_separator_str=separator,
            max_mamba_cache_size=maxMambaCacheSize,
        )
    return sgl.Engine(**engineKwargs)


def _SegmentedPrompt(parts: Iterable[str], separator: str) -> str:
    """Join non-empty spans with HYPIC's tokenizer-level separator."""
    spans = [part for part in parts if part]
    if any(separator in part for part in spans):
        raise ValueError(
            f"prompt text contains the configured HYPIC separator {separator!r}"
        )
    return separator.join(spans)


class HypicMethod(Method):
    """Position-independent reuse through HYPIC's patched SGLang runtime."""

    name = "hypic"
    backend = "hypic-sglang"
    method_metrics = ("reuse_ratio",)

    # HYPIC v1 accepts one string per request (its tokenizer explicitly rejects
    # batched text), and each benchmark case owns a cache warmup/reset cycle.
    maxCaseBatchSize = 1

    def __init__(
        self,
        gpuNums: int = 1,
        perfWeight: float = 1.0,
        *,
        maxNewTokens: int = 64,
        maxModelLen: int = 25600,
        memFractionStatic: float = 0.80,
        dtype: str = "bfloat16",
        picMode: str = "addition",
        separator: str = "<<PIC_SEP>>",
        maxMambaCacheSize: int = _DEFAULT_MAX_MAMBA_CACHE_SIZE,
        fullPrefill: bool = False,
        tag: Optional[str] = None,
    ):
        super().__init__(
            gpuNums=gpuNums,
            perfWeight=perfWeight,
            maxGpuNums=None,
            tag=tag,
        )
        if isinstance(maxNewTokens, bool) or not isinstance(maxNewTokens, int):
            raise TypeError("maxNewTokens must be an integer")
        if maxNewTokens < 1:
            raise ValueError("maxNewTokens must be at least 1")
        if isinstance(maxModelLen, bool) or not isinstance(maxModelLen, int):
            raise TypeError("maxModelLen must be an integer")
        if maxModelLen < 1:
            raise ValueError("maxModelLen must be at least 1")
        if not 0.0 < float(memFractionStatic) < 1.0:
            raise ValueError("memFractionStatic must be between 0 and 1")
        if picMode not in _PIC_MODES:
            raise ValueError(
                f"unknown picMode={picMode!r}; expected one of {sorted(_PIC_MODES)}"
            )
        if not separator:
            raise ValueError("separator must not be empty")
        if isinstance(maxMambaCacheSize, bool) or not isinstance(
            maxMambaCacheSize, int
        ):
            raise TypeError("maxMambaCacheSize must be an integer")
        if maxMambaCacheSize < 1:
            raise ValueError("maxMambaCacheSize must be at least 1")
        if not isinstance(fullPrefill, bool):
            raise TypeError("fullPrefill must be a bool")

        self.modelPath = DefaultModelPath()
        self.maxNewTokens = maxNewTokens
        self.maxModelLen = maxModelLen
        self.memFractionStatic = float(memFractionStatic)
        self.dtype = dtype
        self.picMode = picMode
        self.separator = separator
        self.maxMambaCacheSize = maxMambaCacheSize
        self.fullPrefill = fullPrefill
        self.engine = None
        self._states: List[Dict[str, Any]] = []

    def Initialize(self, gpuIds: Sequence[int]) -> None:
        super().Initialize(gpuIds)
        if not self.modelPath:
            raise ValueError("ModelPath is empty; set it in config.yaml")
        self.engine = _CreateHypicEngine(
            self.modelPath,
            self.gpuIds,
            dtype=self.dtype,
            maxModelLen=self.maxModelLen,
            memFractionStatic=self.memFractionStatic,
            picMode=self.picMode,
            separator=self.separator,
            maxMambaCacheSize=self.maxMambaCacheSize,
            fullPrefill=self.fullPrefill,
        )

    def Prepare(self, data: List[List[str]]) -> None:
        """Prefill every case's original-order segments into PICache."""
        self._states = []
        if self.fullPrefill:
            return
        for chunks in data:
            prepare = [chunk for chunk in (chunks or []) if chunk]
            self._states.append({"prepare": prepare})
            if not prepare:
                continue

            # PICache excludes the final segment by design.  The disposable
            # tail makes the final prepared chunk cacheable as well.
            warmup = _SegmentedPrompt([*prepare, _WARMUP_TAIL], self.separator)
            self._Generate(warmup, maxNewTokens=1)

    def Run(
        self,
        data: List[str],
        retainOutput: Optional[List[bool]] = None,
    ) -> List[Result]:
        # HYPIC owns cache lifetime; generated-output retention is not exposed
        # by PICache yet.
        _ = retainOutput
        results: List[Result] = []
        for index, runInput in enumerate(data):
            if self.fullPrefill:
                output, ttft, total, nTokens, meta = self._Generate(
                    runInput, maxNewTokens=self.maxNewTokens
                )
                results.append(
                    self._Result(
                        output,
                        ttft,
                        total,
                        nTokens,
                        meta,
                        fullPrefill=True,
                    )
                )
                continue

            prepare = (
                self._states[index]["prepare"]
                if index < len(self._states)
                else []
            )
            parts = ComposeInterleavedReuse(prepare, runInput)
            matched = [
                prepareIndex
                for prepareIndex, _ in parts
                if prepareIndex is not None
            ]

            # With no prepared match, leave the prompt byte-for-byte unchanged.
            # It becomes one PIC miss segment, equivalent to full prefill.
            prompt = (
                _SegmentedPrompt((text for _, text in parts), self.separator)
                if matched
                else runInput
            )
            output, ttft, total, nTokens, meta = self._Generate(
                prompt, maxNewTokens=self.maxNewTokens
            )

            results.append(
                self._Result(
                    output,
                    ttft,
                    total,
                    nTokens,
                    meta,
                    nPicSegments=len(parts) if matched else 1,
                    matchedPreparedSegments=len(matched),
                )
            )
        return results

    def _Result(
        self,
        output: str,
        ttft: float,
        total: float,
        nTokens: int,
        meta: Dict[str, Any],
        *,
        fullPrefill: bool = False,
        nPicSegments: int = 1,
        matchedPreparedSegments: int = 0,
    ) -> Result:
        nInput = int(meta.get("prompt_tokens", 0) or 0)
        numCached = (
            0
            if fullPrefill
            else int(meta.get("cached_tokens", 0) or 0)
        )
        metadata: Dict[str, Any] = {
            "backend": self.backend,
            "n_input": nInput,
            "num_cached_tokens": numCached,
            "reuse_ratio": (numCached / nInput) if nInput else 0.0,
        }
        if fullPrefill:
            metadata["full_prefill"] = True
        else:
            metadata.update(
                {
                    "pic_mode": self.picMode,
                    "n_pic_segments": nPicSegments,
                    "matched_prepared_segments": matchedPreparedSegments,
                }
            )
        return Result(
            output=output,
            performance={
                TtftKey: ttft,
                NumOutputTokensKey: nTokens,
                TotalTimeKey: total,
            },
            metadata=metadata,
        )

    def _Generate(
        self, prompt: str, *, maxNewTokens: int
    ) -> Tuple[str, float, float, int, Dict[str, Any]]:
        """Generate one request and measure TTFT at the first streamed token."""
        if self.engine is None:
            raise RuntimeError("HypicMethod is not initialized")

        started = time.perf_counter()
        ttft: Optional[float] = None
        final: Optional[Dict[str, Any]] = None
        maxCached = 0

        stream = self.engine.generate(
            prompt,
            sampling_params={"temperature": 0, "max_new_tokens": maxNewTokens},
            stream=True,
        )
        for chunk in stream:
            final = chunk
            outputIds = chunk.get("output_ids") or []
            if outputIds and ttft is None:
                ttft = time.perf_counter() - started
            chunkMeta = chunk.get("meta_info") or {}
            maxCached = max(maxCached, int(chunkMeta.get("cached_tokens", 0) or 0))

        total = time.perf_counter() - started
        if final is None:
            raise RuntimeError("HYPIC returned no generation output")

        meta = dict(final.get("meta_info") or {})
        meta["cached_tokens"] = max(
            maxCached, int(meta.get("cached_tokens", 0) or 0)
        )
        outputIds = list(final.get("output_ids") or [])
        nTokens = int(
            meta.get("completion_tokens", len(outputIds)) or len(outputIds)
        )
        return (
            final.get("text") or "",
            float(ttft if ttft is not None else total),
            total,
            nTokens,
            meta,
        )

    def Reset(self) -> None:
        self._states = []
        if self.engine is None:
            return
        response = self.engine.flush_cache()
        if hasattr(response, "success") and not response.success:
            raise RuntimeError(
                f"HYPIC cache flush failed: {getattr(response, 'message', response)}"
            )

    def Close(self) -> None:
        self._states = []
        engine, self.engine = self.engine, None
        if engine is not None:
            engine.shutdown()
