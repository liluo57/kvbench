"""Naive — cheap KV reuse, no repair, over plain ``transformers``.

The counterpart to CacheBlend *before* its check phase: the shared context's KV
is prefilled once in ``Prepare`` and every query in ``Run`` is answered against
that cache. No tokens are ever recomputed. The shared context is prefilled the
way the task splits it: a single segment in one pass (positions are correct, so
reuse is lossless); a chunked context (e.g. NIAHShuffleTask's A/B/C) is
prefilled per chunk in isolation and concatenated, so cross-chunk attention is
missing — the gap that CacheBlend's ``recomp_ratio > 0`` later repairs.

Naive is the *control group* for CacheBlend's repair: it does *nothing* to fix
the recombination — so its accuracy is exactly what the KV stitching breaks,
which is what CacheBlend's check phase is meant to recover. When the prompt no
longer starts with the cached context (NIAHShuffleTask's shuffled order) it does
*not* detect the change: it blindly serves the stale stitched KV and decodes the
answer out of it, leaving the recombination un-repaired. It recomputes from
scratch only when nothing was warmed up (NIAHTask / VTTask / CWETask pass no
shared context).
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from core.Config import ModelPath as DefaultModelPath
from core.Method import Method
from core.Result import NumOutputTokensKey, Result, TotalTimeKey, TtftKey

from helpers.Prompt import ComposeReuse
from helpers.TransformersHelper import CacheLayerPairs, TransformersGenerator


class NaiveTransformer(Method):
    """Cheap KV reuse over plain ``transformers`` (the naive control group).

    The shared context's KV is cached once in ``Prepare``; ``Run`` strips it off
    the complete prompt and decodes only the fresh suffix against the cache.
    When the prompt no longer starts with the cache (niah_shuffle's shuffled
    order) it blindly decodes the answer from the stitched cache instead of
    recomputing; it only recomputes when nothing was warmed up.
    """

    name = "naive"
    backend = "transformers"

    def __init__(
        self,
        gpuIds: Union[str, list[int]] = "0",
        modelPath: str | None = None,
        *,
        maxNewTokens: int = 64,
        dtype: str = "bfloat16",
        tag: Optional[str] = None,
    ):
        super().__init__(tag=tag)
        self.gpuIds = gpuIds
        self.modelPath = modelPath or DefaultModelPath()
        self.maxNewTokens = maxNewTokens
        self._gen = TransformersGenerator(
            self.modelPath, gpuIds, maxNewTokens=maxNewTokens, dtype=dtype
        )
        #: Per-case state accumulated by :meth:`Prepare` and consumed by
        #: :meth:`Run`: each entry is ``{"prepare", "past", "last_id"}`` (the
        #: concatenated context KV — DynamicCache or None — and its last token).
        self._states: List[Dict[str, Any]] = []

    # ---------------------------------------------------------------- Method
    def Prepare(self, data: List[List[str]]) -> None:
        self._states = []
        for chunks in data:
            prepare = list(chunks or [])
            past, lastId = self._prefillChunks(prepare)
            self._states.append(
                {"prepare": prepare, "past": past, "last_id": lastId}
            )

    def Run(self, data: List[str]) -> List[Result]:
        # Naive processes the batch sequentially: every case's cached-path
        # decode uses its own DynamicCache, whose sequence length differs per
        # case, so the past tensors cannot be packed into one batched
        # ``transformers`` call. (The vLLM methods — CacheBlend, FullPrefillVllm
        # — do batch natively via concurrent engine requests.)
        results = []
        for i, run_input in enumerate(data):
            state = self._states[i]
            prepare: List[str] = state["prepare"]
            past: Any = state["past"]
            lastId: Optional[int] = state["last_id"]
            order, suffix = ComposeReuse(prepare, run_input)
            if past is None:
                # Nothing was warmed up (niah / vt / cwe): recompute the whole
                # prompt from scratch — the recompute baseline.
                ids = self._gen.Encode(run_input)
                text, ttft, total, nTokens = self._gen.Generate(ids)
                nInput = len(ids)
                reuseRatio = 0.0
            elif order is not None:
                # Found reusable chunks forming a prefix of run_input
                if suffix:
                    # The prompt starts with cached context and has fresh tokens
                    # after it: decode only the suffix against the stitched KV.
                    ids = self._gen.Encode(suffix)
                    text, ttft, total, nTokens = self._gen.Generate(
                        ids, pastKeyValues=past
                    )
                    nInput = len(ids) + past.get_seq_length()
                    reuseRatio = 1.0
                else:
                    # Prompt is exactly the cached context (nothing fresh to fuse).
                    # Naive serves the stitched KV, decoding the answer from cache.
                    nInput = past.get_seq_length()
                    text, ttft, total, nTokens = self._decodeFromCache(
                        past, lastId, prepare
                    )
                    reuseRatio = 1.0
            else:
                # No chunks match the start of run_input (e.g., shuffled order).
                # Naive does not detect the change and blindly serves the
                # stitched KV, decoding the answer from the cache — exactly what
                # the recombination breaks.
                nInput = past.get_seq_length()  # before _decodeFromCache crops
                text, ttft, total, nTokens = self._decodeFromCache(
                    past, lastId, prepare
                )
                reuseRatio = 1.0
            results.append(
                self._Result(
                    text,
                    ttft,
                    total,
                    nTokens,
                    metadata={"reuse_ratio": reuseRatio, "n_input": nInput},
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
            metadata={"backend": self.backend, **metadata},
        )

    def Reset(self) -> None:
        self._states = []

    # ------------------------------------------------------------ transformers
    def _decodeFromCache(
        self, past: Any, lastId: Optional[int], prepare: List[str]
    ):
        """Blindly decode the answer straight out of the stitched cache.

        The cache ends at the last context token (the answer prefix for
        niah_shuffle); naive drops that token from the cache, feeds it back as
        the first decode step, and keeps decoding — the shuffled ``run`` prompt
        is ignored entirely. ``DynamicCache.crop`` trims in place (returns
        ``None``) and is only reached once per case (each case owns its own
        cache), so it is safe here.
        """
        n = past.get_seq_length()
        if n > 1 and lastId is not None:
            past.crop(n - 1)
            return self._gen.Generate([lastId], pastKeyValues=past)
        # Degenerate: nothing meaningful cached — recompute the whole prompt.
        ids = self._gen.Encode("".join(prepare))
        return self._gen.Generate(ids)

    def _prefillChunks(self, chunks: List[str]) -> Tuple[Any, Optional[int]]:
        """Prefill the shared context once, concatenating per-chunk KVs.

        A single segment is prefilled in one pass (positions are correct, so
        reuse is lossless). Multiple chunks are each prefilled in isolation —
        chunk i does not attend to chunks 0..i-1 — and their KVs are simply
        concatenated, the naive knowledge-base setup CacheBlend repairs.
        Returns ``(None, None)`` when there is nothing to prefill, else
        ``(acc, lastId)`` — the concatenated DynamicCache and its last token.
        """
        if not chunks:
            return None, None
        from transformers.cache_utils import DynamicCache

        acc = DynamicCache()
        lastId: Optional[int] = None
        for chunk in chunks:
            ids = self._gen.Encode(chunk, addSpecialTokens=False)
            if not ids:
                continue
            # Chunked prefill keeps peak memory bounded for long contexts
            # (the eager attention matrix would otherwise blow up).
            out = self._gen.Prefill(ids)
            cache = out.past_key_values  # DynamicCache (or tuple of (K, V))
            for li, (k, v) in enumerate(CacheLayerPairs(cache)):
                acc.update(k, v, li)
            lastId = ids[-1]
        return acc, lastId
