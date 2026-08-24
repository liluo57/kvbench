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
from core.Method import Method
from core.Result import NumOutputTokensKey, Result, TotalTimeKey, TtftKey

from helpers.Prompt import ComposeInterleavedReuse
from helpers.TransformersHelper import CacheLayerPairs, TransformersGenerator


class NaiveTransformer(Method):
    """Cheap KV reuse over plain ``transformers`` (the naive control group).

    Prepared chunks are cached independently in ``Prepare``. ``Run`` stitches
    those KVs in prompt order and prefills only the fresh spans between them.
    No cached token is repaired or recomputed. When ``retainOutput`` is true,
    the generated output and its KV are registered for later RUN steps.
    """

    name = "naive"
    backend = "transformers"
    # Each Case owns differently-shaped DynamicCache tensors and Run processes
    # them sequentially, so a larger Case batch only extends their lifetime.
    maxCaseBatchSize = 1

    def __init__(
        self,
        gpuNums: int = 1,
        perfWeight: float = 1.0,
        modelPath: str | None = None,
        *,
        maxNewTokens: int = 64,
        dtype: str = "bfloat16",
        tag: Optional[str] = None,
    ):
        super().__init__(
            gpuNums=gpuNums,
            perfWeight=perfWeight,
            maxGpuNums=1,
            tag=tag,
        )
        self.modelPath = modelPath or DefaultModelPath()
        self.maxNewTokens = maxNewTokens
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
            maxNewTokens=self.maxNewTokens,
            dtype=self.dtype,
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
    def Prepare(self, data: List[List[str]]) -> None:
        self._states = []

        for chunks in data:
            prepare = list(chunks or [])
            caches, ids = self._prefillChunks(prepare)

            self._states.append(
                {
                    "segments": prepare,
                    "caches": caches,
                    "ids": ids,
                }
            )

    def Run(self, data: List[str], retainOutput: Optional[List[bool]] = None) -> List[Result]:
        # Naive processes the batch sequentially: every case builds a
        # differently-sized DynamicCache, so the past tensors cannot be packed
        # into one batched ``transformers`` call.
        if len(self._states) != len(data):
            self._states = [{"segments": [], "caches": [], "ids": []} for _ in data]
        results = []

        for i, runInput in enumerate(data):
            state = self._states[i]

            retain = bool(retainOutput[i]) if retainOutput is not None and i < len(retainOutput) else False
            prepare: List[str] = state["segments"]
            caches: List[Any] = state["caches"]
            preparedIds: List[List[int]] = state["ids"]

            parts = ComposeInterleavedReuse(prepare, runInput)

            if not any(
                prepareIndex is not None
                for prepareIndex, _ in parts
            ):
                # Nothing reusable appears in this prompt.
                ids = self._gen.Encode(runInput)
                generated = self._gen.Generate(ids, returnCache=retain)
                if retain:
                    text, ttft, total, nTokens, fullCache, outputIds = generated
                    self._registerOutput(state, text, outputIds, fullCache)
                else:
                    text, ttft, total, nTokens = generated

                nInput = len(ids)
                reuseRatio = 0.0

            else:
                from transformers.cache_utils import DynamicCache

                t0 = time.perf_counter()

                past = DynamicCache()
                lastId: Optional[int] = None
                reusedTokens = 0

                for prepareIndex, span in parts:
                    if prepareIndex is None:
                        # Fresh span: compute it against everything assembled
                        # before it.
                        ids = self._gen.Encode(
                            span,
                            addSpecialTokens=False,
                        )

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
                prefillTime = time.perf_counter() - t0

                if nInput == 0 or lastId is None:
                    ids = self._gen.Encode(runInput)
                    generated = self._gen.Generate(ids, returnCache=retain)
                    if retain:
                        text, ttft, total, nTokens, fullCache, outputIds = generated
                        self._registerOutput(state, text, outputIds, fullCache)
                    else:
                        text, ttft, total, nTokens = generated

                    nInput = len(ids)
                    reuseRatio = 0.0

                else:
                    decoded = self._decodeFromCache(
                        past, lastId, runInput, retainOutput=retain
                    )
                    if retain:
                        text, decodeTtft, decodeTotal, nTokens, fullCache, outputIds = decoded
                        self._registerOutput(state, text, outputIds, fullCache)
                    else:
                        text, decodeTtft, decodeTotal, nTokens = decoded

                    # Fresh intermediate prefill happens before Generate(), so
                    # include it in both TTFT and total time.
                    ttft = prefillTime + decodeTtft
                    total = prefillTime + decodeTotal

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
                )
            )

        return results

    def _Result(self, text, ttft, total, nTokens, *, metadata) -> Result:
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
        )

    def Reset(self) -> None:
        self._states = []

    # ------------------------------------------------------------ transformers
    def _decodeFromCache(
        self,
        past: Any,
        lastId: Optional[int],
        fallbackPrompt: str,
        retainOutput: bool = False,
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
                returnCache=retainOutput,
            )

        ids = self._gen.Encode(fallbackPrompt)
        return self._gen.Generate(ids, returnCache=retainOutput)

    def _registerOutput(self, state: Dict[str, Any], text: str, outputIds: List[int], fullCache: Any) -> None:
        """Register generated tokens as a reusable segment when cache is available."""
        if not outputIds or fullCache is None:
            return
        pairs = []
        for k, v in CacheLayerPairs(fullCache):
            pairs.append((k[:, :, -len(outputIds):, :].detach(), v[:, :, -len(outputIds):, :].detach()))
        state["segments"].append(text)
        state["ids"].append(outputIds)
        state["caches"].append(pairs)

    def _prefillChunks(
        self,
        chunks: List[str],
    ) -> Tuple[List[Any], List[List[int]]]:
        """Prefill each prepared chunk independently.

        Returns two lists aligned with ``chunks``:
        ``caches[i]`` is the isolated KV for ``chunks[i]`` (or ``None`` for an
        empty/tokenless chunk), and ``ids[i]`` is that chunk's token ids.
        """
        caches: List[Any] = []
        idsList: List[List[int]] = []

        for chunk in chunks:
            ids = self._gen.Encode(
                chunk,
                addSpecialTokens=False,
            )

            idsList.append(ids)

            if not ids:
                caches.append(None)
                continue

            out = self._gen.Prefill(ids)
            caches.append(out.past_key_values)

        return caches, idsList
