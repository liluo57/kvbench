"""Naive — cheap KV reuse, no repair, over plain ``transformers``.

The counterpart to CacheBlend *before* its check phase: prepared chunks are
prefilled once in ``Prepare`` and reused wherever they reappear in ``Run``.
Fresh spans between reused chunks are prefetched against the stitched prefix;
cached chunks themselves are never recomputed.

A single prepared segment is prefilled in one pass. Multiple prepared chunks
are each prefilled in isolation, so cross-chunk attention is missing — the gap
that CacheBlend's ``recomp_ratio > 0`` later repairs.

Naive is the *control group* for CacheBlend's repair: it does *nothing* to fix
the recombination, so its accuracy is exactly what KV stitching breaks.

For a RUN whose ``retainOutput`` hint is true, Naive also keeps the generated
tokens and their KV as another reusable segment. Later RUN steps of the same
case can therefore reuse earlier agent output as well as prepared input.
"""

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.Config import ModelPath as DefaultModelPath
from core.Method import Method, ResolveMaxNewTokens
from core.Result import NumOutputTokensKey, Result, TotalTimeKey, TtftKey
from core.Sampling import ResolveSamplingConfig

from helpers.backends.Prompt import ComposeInterleavedReuse
from helpers.backends.TransformersHelper import CacheLayerPairs, TransformersGenerator


class NaiveTransformer(Method):
    """Cheap KV reuse over plain ``transformers`` (the naive control group).

    Prepared chunks are cached independently in ``Prepare``. ``Run`` stitches
    those KVs in prompt order and prefills only the fresh spans between them.
    No cached token is repaired or recomputed. When ``retainOutput`` is true,
    the generated output and its KV are registered for later RUN steps.
    """

    name = "naive"
    method_metrics = ("reuse_ratio",)
    backend = "transformers"
    # Each Case owns differently-shaped DynamicCache tensors and Run processes
    # them sequentially, so a larger Case batch only extends their lifetime.
    maxCaseBatchSize = 1

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
            tag=tag,
        )
        # Model path is config-only — switch models via config.yaml.
        self.modelPath = DefaultModelPath()
        self.samplingConfig = ResolveSamplingConfig(self.modelPath)
        self.dtype = dtype
        self._gen = None

        #: Per-case state accumulated by :meth:`Prepare` and :meth:`Run`.
        #: ``segments``, ``caches`` and ``ids`` stay aligned and contain both
        #: prepared chunks and outputs retained for possible future reuse.
        self._states: List[Dict[str, Any]] = []

    def Initialize(self, gpuIds: Sequence[int]) -> None:
        super().Initialize(gpuIds)
        self._gen = TransformersGenerator(
            self.modelPath,
            self.gpuIds,
            dtype=self.dtype,
            samplingConfig=self.samplingConfig,
        )

    def Close(self) -> None:
        self._states = []
        self._gen = None
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    # ---------------------------------------------------------------- Method
    def Prepare(self, data: List[List[List[int]]]) -> None:
        self._states = []

        for chunks in data:
            prepare = [list(chunk) for chunk in (chunks or [])]
            caches, ids = self._prefillChunks(prepare)

            self._states.append(
                {
                    "segments": prepare,
                    "caches": caches,
                    "ids": ids,
                }
            )

    def Run(
        self,
        data: List[List[int]],
        retainOutput: Optional[List[bool]] = None,
        maxNewTokens: Optional[int] = None,
    ) -> List[Result]:
        runStart = time.perf_counter()
        maxNewTokens = ResolveMaxNewTokens(maxNewTokens)
        # Naive processes the batch sequentially: every case builds a
        # differently-sized DynamicCache, so the past tensors cannot be packed
        # into one batched ``transformers`` call.
        if len(self._states) != len(data):
            self._states = [{"segments": [], "caches": [], "ids": []} for _ in data]
        results = []

        for i, runTokens in enumerate(data):
            runInput = list(runTokens)
            state = self._states[i]
            outputIds = None
            outputCacheTime = 0.0

            retain = bool(retainOutput[i]) if retainOutput is not None and i < len(retainOutput) else False
            prepare: List[List[int]] = state["segments"]
            caches: List[Any] = state["caches"]
            preparedIds: List[List[int]] = state["ids"]

            parts = ComposeInterleavedReuse(prepare, runInput)

            if not any(
                prepareIndex is not None
                for prepareIndex, _ in parts
            ):
                # Nothing reusable appears in this prompt.
                ids = runInput
                generationStart = time.perf_counter()
                generated = self._gen.Generate(
                    ids,
                    maxNewTokens=maxNewTokens,
                    returnCache=retain,
                    returnTokenIds=True,
                )
                if retain:
                    text, backendTtft, total, nTokens, fullCache, rawOutputIds = generated
                    outputIds = self._gen.OutputTokenIds(rawOutputIds)
                    outputCacheTime = self._registerOutput(
                        state, outputIds, fullCache, len(ids)
                    )
                else:
                    text, backendTtft, total, nTokens, outputIds = generated

                ttft = generationStart - runStart + backendTtft
                total = generationStart - runStart + total + outputCacheTime

                nInput = len(ids)
                reuseRatio = 0.0

            else:
                from transformers.cache_utils import DynamicCache

                past = DynamicCache()
                lastId: Optional[int] = None
                reusedTokens = 0

                for prepareIndex, span in parts:
                    if prepareIndex is None:
                        # Fresh span: compute it against everything assembled
                        # before it.
                        ids = list(span)

                        if not ids:
                            continue

                        out = self._gen.Prefill(
                            ids,
                            pastKeyValues=(
                                past
                                if past.get_seq_length() > 0
                                else None
                            ),
                        )

                        past = out.past_key_values

                    else:
                        # Reusable span: append its isolated KV directly.
                        ids = preparedIds[prepareIndex]
                        cache = caches[prepareIndex]

                        if not ids or cache is None:
                            continue

                        for layerIndex, (k, v) in enumerate(
                            CacheLayerPairs(cache)
                        ):
                            past.update(k, v, layerIndex)

                        reusedTokens += len(ids)

                    lastId = ids[-1]

                nInput = past.get_seq_length()
                if nInput == 0 or lastId is None:
                    ids = runInput
                    generationStart = time.perf_counter()
                    generated = self._gen.Generate(
                        ids,
                        maxNewTokens=maxNewTokens,
                        returnCache=retain,
                        returnTokenIds=True,
                    )
                    if retain:
                        text, backendTtft, total, nTokens, fullCache, rawOutputIds = generated
                        outputIds = self._gen.OutputTokenIds(rawOutputIds)
                        outputCacheTime = self._registerOutput(
                            state, outputIds, fullCache, len(ids)
                        )
                    else:
                        text, backendTtft, total, nTokens, outputIds = generated

                    ttft = generationStart - runStart + backendTtft
                    total = generationStart - runStart + total + outputCacheTime

                    nInput = len(ids)
                    reuseRatio = 0.0

                else:
                    generationStart = time.perf_counter()
                    decoded = self._decodeFromCache(
                        past,
                        lastId,
                        runInput,
                        retainOutput=retain,
                        maxNewTokens=maxNewTokens,
                    )
                    if retain:
                        text, decodeTtft, decodeTotal, nTokens, fullCache, rawOutputIds = decoded
                        outputIds = self._gen.OutputTokenIds(rawOutputIds)
                        outputCacheTime = self._registerOutput(
                            state, outputIds, fullCache, nInput
                        )
                    else:
                        text, decodeTtft, decodeTotal, nTokens, outputIds = decoded

                    # Fresh intermediate prefill happens before Generate(), so
                    # include it in both TTFT and total time.
                    ttft = generationStart - runStart + decodeTtft
                    total = generationStart - runStart + decodeTotal + outputCacheTime

                    reuseRatio = (
                        reusedTokens / nInput
                        if nInput
                        else 0.0
                    )

            results.append(
                self._Result(
                    text,
                    ttft,
                    total,
                    nTokens,
                    metadata={
                        "reuse_ratio": reuseRatio,
                        "n_input": nInput,
                    },
                    output_ids=outputIds,
                )
            )

        return results

    def _Result(self, text, ttft, total, nTokens, *, metadata, output_ids=None) -> Result:
        return Result(
            output=text,
            performance={
                TtftKey: ttft,
                NumOutputTokensKey: nTokens,
                TotalTimeKey: total,
            },
            metadata={
                "backend": self.backend,
                **metadata,
            },
            output_ids=output_ids,
        )

    def Reset(self) -> None:
        self._states = []

    # ------------------------------------------------------------ transformers
    def _decodeFromCache(
        self,
        past: Any,
        lastId: Optional[int],
        fallbackPrompt: List[int],
        retainOutput: bool = False,
        maxNewTokens: Optional[int] = None,
    ):
        """Decode from a fully assembled prompt cache.

        The final prompt token is removed from the cache and fed back as the
        first decode step, so generation sees exactly one uncached token.
        """
        n = past.get_seq_length()

        if n > 1 and lastId is not None:
            past.crop(n - 1)
            return self._gen.Generate(
                [lastId],
                pastKeyValues=past,
                maxNewTokens=ResolveMaxNewTokens(maxNewTokens),
                returnCache=retainOutput,
                returnTokenIds=True,
            )

        return self._gen.Generate(
            fallbackPrompt,
            maxNewTokens=ResolveMaxNewTokens(maxNewTokens),
            returnCache=retainOutput,
            returnTokenIds=True,
        )

    def _registerOutput(
        self,
        state: Dict[str, Any],
        outputIds: List[int],
        fullCache: Any,
        promptLength: int,
    ) -> float:
        """Register the native output fragment with its exact generated KV.

        The helper completes the final sampled non-stop token before return.
        A terminal stop EOS is excluded from both the retained IDs and cache.
        Every other token, including internal special IDs, remains verbatim.
        """
        started = time.perf_counter()
        if not outputIds:
            return time.perf_counter() - started
        tokenIds = list(outputIds)
        cache = fullCache
        cacheStart = int(promptLength)
        expectedLength = cacheStart + len(tokenIds)
        if cache is None or int(cache.get_seq_length()) != expectedLength:
            raise AssertionError("native output KV length differs from retained token IDs")
        pairs = []
        for k, v in CacheLayerPairs(cache):
            if k.shape[-2] != expectedLength or v.shape[-2] != expectedLength:
                raise AssertionError("native output KV layer length differs from retained token IDs")
            pairs.append((k[:, :, cacheStart:expectedLength, :].detach(), v[:, :, cacheStart:expectedLength, :].detach()))
        state["segments"].append(tokenIds)
        state["ids"].append(tokenIds)
        state["caches"].append(pairs)
        return time.perf_counter() - started

    def _prefillChunks(
        self,
        chunks: List[List[int]],
    ) -> Tuple[List[Any], List[List[int]]]:
        """Prefill each prepared chunk independently.

        Returns two lists aligned with ``chunks``:
        ``caches[i]`` is the isolated KV for ``chunks[i]`` (or ``None`` for an
        empty/tokenless chunk), and ``ids[i]`` is that chunk's token ids.
        """
        caches: List[Any] = []
        idsList: List[List[int]] = []

        for chunk in chunks:
            ids = list(chunk)

            idsList.append(ids)

            if not ids:
                caches.append(None)
                continue

            out = self._gen.Prefill(ids)
            caches.append(out.past_key_values)

        return caches, idsList
