"""KVCOMM anchor-based KV reuse over the KVBench transformers backend.

This module deliberately contains only the algorithmic part of KVCOMM.  The
model, tokenizer, generation loop, GPU placement, and sampling policy remain
owned by :class:`TransformersGenerator`.

The official implementation represents a prompt as a sequence of static
prefix segments and placeholders.  KVBench already exposes the same shape
through ``Prepare`` plus ``ComposeInterleavedReuse``: prepared text (including
retained agent outputs) is a placeholder and the text between placeholders is
the prefix.  We adapt the official anchor operations to those pieces here.

The first implementation supports ordinary dense-attention models whose
Transformers cache has one ``(K, V)`` tensor pair per layer and whose model
exposes a standard ``rotary_emb``.  Hybrid/state-space caches are rejected at
the point where KV reuse is requested instead of being silently stitched.
"""

from __future__ import annotations

import hashlib
import math
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.Config import ModelPath as DefaultModelPath
from core.Method import Method, ResolveMaxNewTokens
from core.Result import NumOutputTokensKey, Result, TotalTimeKey, TtftKey
from core.Sampling import ResolveSamplingConfig

from helpers.backends.Prompt import ComposeInterleavedReuse
from helpers.backends.TransformersHelper import CacheLayerPairs, TransformersGenerator


def _CacheLength(cache: Any) -> int:
    """Get a cache length across the DynamicCache API generations."""
    if cache is None:
        return 0
    getter = getattr(cache, "get_seq_length", None)
    if callable(getter):
        try:
            value = getter()
        except TypeError:
            value = getter(0)
        if value is not None:
            return int(value)
    for key, _ in CacheLayerPairs(cache):
        if key is not None:
            return int(key.shape[-2])
    return 0


def _CachePairs(cache: Any) -> List[Tuple[Any, Any]]:
    """Return dense layer pairs, rejecting unsupported cache layouts."""
    if cache is None:
        return []
    pairs = list(CacheLayerPairs(cache))
    if not pairs:
        return []
    for layer, (key, value) in enumerate(pairs):
        if key is None or value is None or not hasattr(key, "shape") or not hasattr(value, "shape"):
            raise RuntimeError(
                "KVCommTransformer requires a dense attention DynamicCache; "
                f"layer {layer} has no ordinary key/value tensors"
            )
        if key.ndim != 4 or value.ndim != 4:
            raise RuntimeError(
                "KVCommTransformer requires 4-D per-layer K/V tensors "
                "([batch, heads, sequence, head_dim])"
            )
    return pairs


def _NewCache(pairs: Sequence[Tuple[Any, Any]]) -> Any:
    """Build a current Transformers DynamicCache from layer tensors."""
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache()
    for layer, (key, value) in enumerate(pairs):
        cache.update(key, value, layer)
    return cache


def _SliceCache(cache: Any, start: int, end: int) -> Any:
    """Functional sequence slice, preserving the DynamicCache contract."""
    pairs = _CachePairs(cache)
    return _NewCache(
        [
            (key[..., start:end, :].clone(), value[..., start:end, :].clone())
            for key, value in pairs
        ]
    )


def _ConcatCaches(caches: Sequence[Any]) -> Any:
    """Concatenate dense caches along their sequence dimension."""
    usable = [cache for cache in caches if cache is not None and _CacheLength(cache) > 0]
    if not usable:
        return None
    import torch

    pair_lists = [_CachePairs(cache) for cache in usable]
    layer_count = len(pair_lists[0])
    if any(len(pairs) != layer_count for pairs in pair_lists[1:]):
        raise RuntimeError("KVCommTransformer saw inconsistent layer counts in DynamicCache")
    result = []
    for layer in range(layer_count):
        keys = [pairs[layer][0] for pairs in pair_lists]
        values = [pairs[layer][1] for pairs in pair_lists]
        result.append((torch.cat(keys, dim=-2), torch.cat(values, dim=-2)))
    return _NewCache(result)


def _StackCache(cache: Any) -> Tuple[Any, Any]:
    """Stack ``(K, V)`` as ``[layers, batch, heads, sequence, dim]``."""
    import torch

    pairs = _CachePairs(cache)
    if not pairs:
        raise RuntimeError("cannot stack an empty DynamicCache")
    lengths = {int(key.shape[-2]) for key, _ in pairs}
    if len(lengths) != 1:
        raise RuntimeError(
            "KVCommTransformer requires equal sequence lengths in all attention layers"
        )
    return torch.stack([key for key, _ in pairs]), torch.stack(
        [value for _, value in pairs]
    )


def _RotateHalf(tensor: Any) -> Any:
    half = tensor.shape[-1] // 2
    return __import__("torch").cat((-tensor[..., half:], tensor[..., :half]), dim=-1)


class KVCommTransformer(Method):
    """Online KVCOMM reuse for one dense transformers model on one GPU.

    ``Prepare`` records text only.  The request-dependent work needed to make
    a base cache is intentionally performed in ``Run`` so it belongs to the
    KVBench TTFT interval.  Anchors are the only long-lived tensors: each
    stores base placeholder embeddings and placeholder/prefix KV deltas, not a
    full prompt cache.
    """

    name = "kvcomm"
    backend = "transformers"
    method_metrics = ("reuse_ratio",)
    maxCaseBatchSize = 1

    def __init__(
        self,
        gpuNums: int = 1,
        perfWeight: float = 1.0,
        *,
        dtype: str = "bfloat16",
        threshold: float = 0.3,
        maxAnchorNum: int = 20,
        windowSize: int = 5,
        releaseCompletedRequestTransient: bool = False,
        tag: Optional[str] = None,
    ):
        super().__init__(
            gpuNums=gpuNums,
            perfWeight=perfWeight,
            maxGpuNums=1,
            tag=tag,
        )
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise TypeError("threshold must be a number")
        if not 0 <= float(threshold) <= 1:
            raise ValueError("threshold must be between 0 and 1")
        if isinstance(maxAnchorNum, bool) or not isinstance(maxAnchorNum, int) or maxAnchorNum < 1:
            raise ValueError("maxAnchorNum must be a positive integer")
        if isinstance(windowSize, bool) or not isinstance(windowSize, int) or windowSize < 1:
            raise ValueError("windowSize must be a positive integer")
        if not isinstance(releaseCompletedRequestTransient, bool):
            raise TypeError("releaseCompletedRequestTransient must be a bool")

        # ModelPath and sampling resolution are configuration-only.  Model and
        # CUDA initialization remains exclusively in Initialize().
        self.modelPath = DefaultModelPath()
        self.samplingConfig = ResolveSamplingConfig(self.modelPath)
        self.dtype = dtype
        self.threshold = float(threshold)
        self.maxAnchorNum = maxAnchorNum
        self.windowSize = windowSize
        # The official runtime keeps exact-message raw KV available for a
        # later identical message.  Keep that general behavior by default.
        # HumanEval has a verified unique-message workload, so its runner may
        # explicitly enable request-final transient release without changing
        # persistent KVCOMM anchors or any prediction/eviction semantics.
        self.releaseCompletedRequestTransient = releaseCompletedRequestTransient
        self._gen: Optional[TransformersGenerator] = None

        # Persistent online state.  Each key identifies one semantic
        # placeholder (``user_question`` / ``agent_i_current``) inside one
        # workload.  This mirrors official ``anchors[ph_id][message]``: a
        # placeholder has one V-bounded pool, while consumer-specific deltas
        # live inside each anchor instead of duplicating the whole pool for
        # every prompt in which that placeholder appears.
        self._anchors: Dict[str, List[Dict[str, Any]]] = {}
        # Official ``anchor_dict[ph_id][message]`` records the prediction even
        # when the message is not materialized as an anchor.  It must survive
        # Reset alongside anchors so exact-message handling remains identical.
        self._anchorFlags: Dict[str, Dict[str, bool]] = {}
        self._anchorSerial = 0

        # Official KVCOMM keeps the prompt's static text segments in the
        # per-agent shared KV store.  Keep a bounded structural equivalent
        # here.  These entries contain only template/fresh-prefix KV; actual
        # request text is still computed in Run and remains part of TTFT.
        self._prefixTemplates: Dict[str, Dict[str, Any]] = {}
        self._prefixTemplateOrder: List[str] = []
        self._maxPrefixTemplates = 32

        # Isolated placeholder KV is deterministic for a token sequence.  By
        # default this mirrors official input/response shared memory exactly,
        # including future exact-message hits.  Workloads whose messages are
        # proven unique may opt into request-final release via the constructor.
        self._isolatedBases: Dict[Tuple[int, ...], Any] = {}
        self._isolatedBaseOrder: List[Tuple[int, ...]] = []

        # Request/case-local state.  Reset clears all of this but deliberately
        # leaves _anchors intact: KVCOMM learns online across Cases.
        self._states: List[Dict[str, Any]] = []
        self._runIndex = 0
        self._caseSerial = 0
        # Diagnostic-only instrumentation.  It is intentionally opt-in so
        # ordinary benchmark runs have exactly the same request path and
        # timing semantics.
        self._memoryDebug = os.environ.get("KVCOMM_MEMORY_DEBUG") == "1"

    # ---------------------------------------------------------------- Method
    def Initialize(self, gpuIds: Sequence[int]) -> None:
        super().Initialize(gpuIds)
        self._gen = TransformersGenerator(
            self.modelPath,
            self.gpuIds,
            dtype=self.dtype,
            samplingConfig=self.samplingConfig,
        )

    def Prepare(self, data: List[List[str]]) -> None:
        # PREPARE is only a case-local registration point.  In particular, do
        # not tokenize/prefill here: those operations depend on the current
        # RUN prompt and must be visible in its TTFT.
        self._states = []
        for chunks in data:
            prepared = [str(text) for text in list(chunks or []) if str(text)]
            segments = []
            for index, text in enumerate(prepared):
                segments.append(
                    {
                        "text": text,
                        "base_ids": None,
                        "base_cache": None,
                        # The KVCOMM workloads prepare the shared user request
                        # first. Keep a deterministic fallback for any future
                        # caller that prepares more than one segment.
                        "placeholder_id": (
                            "user_question" if index == 0 else f"prepared_{index}"
                        ),
                    }
                )
            # Official anchors are keyed by the request message. This key is
            # used only to update the same anchor as it flows through later
            # agents; it is deliberately not part of the workload namespace,
            # so distinct dataset samples still share one online pool.
            messageText = prepared[0] if prepared else f"case:{self._caseSerial}"
            self._states.append(
                {
                    "segments": segments,
                    "message_key": hashlib.sha256(
                        messageText.encode("utf-8")
                    ).hexdigest(),
                    "namespace": None,
                }
            )
        self._runIndex = 0

    def Run(
        self,
        data: List[str],
        retainOutput: Optional[List[bool]] = None,
        maxNewTokens: Optional[int] = None,
    ) -> List[Result]:
        # This is intentionally the first request-timing operation.  Every
        # operation below, including matching and fallback prefill, is TTFT
        # relevant when it happens before the first generated token.
        results = []
        runStart = time.perf_counter()
        maxNewTokens = ResolveMaxNewTokens(maxNewTokens)

        if len(self._states) != len(data):
            self._states = [
                {
                    "segments": [],
                    "message_key": f"case:{self._caseSerial}",
                    "namespace": None,
                }
                for _ in data
            ]

        import torch

        with torch.inference_mode():
            for index, runInput in enumerate(data):
                requestStart = runStart if len(data) == 1 else time.perf_counter()
                state = self._states[index]
                retain = bool(
                    retainOutput is not None
                    and index < len(retainOutput)
                    and retainOutput[index]
                )
                result = self._RunOne(
                    state,
                    str(runInput),
                    retain,
                    maxNewTokens,
                    requestStart,
                )
                results.append(result)
                self._runIndex += 1
        return results

    def Reset(self) -> None:
        # Anchors intentionally survive this hook.  Worker calls Reset after
        # every Case, but KVCOMM's online anchor pool must serve later samples.
        # Only current-case segments, base caches, and run-local counters die.
        self._DebugMemory("before_reset")
        self._states = []
        if self.releaseCompletedRequestTransient:
            self.ReleaseCompletedRequestTransient()
        self._runIndex = 0
        self._caseSerial += 1
        self._DebugMemory("after_reset")

    def Close(self) -> None:
        self._states = []
        self._anchors.clear()
        self._anchorFlags.clear()
        self._prefixTemplates.clear()
        self._prefixTemplateOrder.clear()
        self._isolatedBases.clear()
        self._isolatedBaseOrder.clear()
        self._gen = None
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def ReleaseCompletedRequestTransient(self) -> None:
        """Release raw exact-message KV after a completed request.

        ``_isolatedBases`` is the KVBench adapter's equivalent of the
        official input/response raw-cache dictionaries.  It is not an anchor
        store: it contains position-zero KV used to build the current request
        base representation.  This method deliberately does not touch
        ``_anchors``, ``_prefixTemplates``, or any anchor metadata.

        The operation is opt-in because deleting this cache changes the
        optimization for a future *identical* message.  KVBench enables it
        only for HumanEval, whose sampled prompts are unique; therefore no
        future request can legally hit one of these raw entries.
        """
        self._isolatedBases.clear()
        self._isolatedBaseOrder.clear()

    # ------------------------------------------------------------ request path
    def _RunOne(
        self,
        state: Dict[str, Any],
        runInput: str,
        retain: bool,
        maxNewTokens: int,
        requestStart: float,
    ) -> Result:
        if self._gen is None:
            raise RuntimeError("KVCommTransformer.Initialize() must be called before Run()")

        segments = state["segments"]
        matches = ComposeInterleavedReuse(
            [segment["text"] for segment in segments], runInput
        )
        reusable = any(segmentIndex is not None for segmentIndex, _ in matches)
        if not reusable:
            return self._DenseResult(
                runInput,
                retain,
                maxNewTokens,
                requestStart,
                anchorFallbacks=0,
            )

        parts = self._MakeParts(segments, matches)
        fullIds = self._gen.Encode(runInput)
        pieceIds = self._TokenizeParts(runInput, fullIds, parts)

        # Boundary-sensitive BPE tokenization means separately encoding each
        # text piece is not equivalent to encoding the complete prompt.  Use
        # the full prompt's offset mapping to partition its actual token IDs;
        # if this tokenizer cannot provide offsets, retain the safe dense path.
        if pieceIds is None:
            return self._DenseResult(
                runInput,
                retain,
                maxNewTokens,
                requestStart,
                anchorFallbacks=len([p for p in parts if p["kind"] == "reuse"]),
            )

        # Retained agent responses already carry the isolated, position-zero
        # KV returned by their generation.  The official runtime stores that
        # response cache in shared memory and fetches it later; keep its token
        # IDs as the reusable segment's canonical IDs instead of re-encoding
        # the response at the surrounding prompt boundary.
        for index, part in enumerate(parts):
            segment = part.get("segment")
            cachedIds = segment.get("base_ids") if segment is not None else None
            if cachedIds is not None:
                pieceIds[index] = list(cachedIds)

        def denseFallback(materializeParts: Optional[set[int]] = None) -> Result:
            # Generate the current response from fullIds exactly, so fallback
            # is equivalent to FullPrefill for this request.  Materialize the
            # next anchor from the canonical segmented layout: this keeps the
            # official placeholder/prefix shapes stable without changing the
            # current response's dense semantics.
            return self._DenseWithBase(
                runInput,
                fullIds,
                parts,
                retain,
                maxNewTokens,
                requestStart,
                anchorFallbacks=len(materializeParts or ()),
                anchorIds=assembledIds,
                anchorPromptLength=len(assembledIds),
                anchorParts=parts,
                materializeParts=materializeParts,
            )

        # The official implementation tokenizes each text segment around a
        # placeholder independently and then concatenates those segment IDs.
        # Reusing the raw rendered-prompt IDs here makes the token count of a
        # suffix depend on the generated placeholder text's BPE boundary;
        # that in turn makes otherwise-compatible prefix deltas look like
        # different shapes and forces a whole-request fallback.  Keep the
        # actual placeholder IDs, but use the persistent canonical IDs for
        # fresh text segments, matching ``locate_placeholder`` plus
        # ``prepare_prefix_kv_segments`` in the official runtime.
        template = self._GetPrefixTemplate(parts)
        if template is not None:
            pieceIds = [
                list(template["piece_ids"][index])
                if part["kind"] == "fresh"
                else ids
                for index, (part, ids) in enumerate(zip(parts, pieceIds))
            ]
        assembledIds = [token for ids in pieceIds for token in ids]
        if not assembledIds:
            return self._DenseResult(
                runInput,
                retain,
                maxNewTokens,
                requestStart,
                anchorFallbacks=len([p for p in parts if p["kind"] == "reuse"]),
            )

        self._BuildBase(parts, pieceIds)
        namespaceParts = state.get("namespace")
        if namespaceParts is None:
            # The first RUN skeleton is stable across Cases of one Task and
            # excludes the concrete question. Keeping it on the case state
            # lets all later agent/decision RUNs share the same semantic
            # placeholder pools while naturally isolating MMLU/GSM8K/
            # HumanEval without a new core BeginTask hook.
            namespaceParts = self._NamespaceParts(parts)
            state["namespace"] = namespaceParts
        consumerKey = self._ConsumerKey()
        messageKey = str(state.get("message_key", f"case:{self._caseSerial}"))
        for part in parts:
            if part["kind"] != "reuse":
                continue
            placeholderId = str(part["segment"]["placeholder_id"])
            part["pool_key"] = self._PoolKey(namespaceParts, placeholderId)
            part["consumer_key"] = consumerKey
            part["message_key"] = messageKey

        # ``predict_as_anchor`` uses its top-p subset only to update anchor
        # activation counters.  Official ``offset_kv_cache_pair`` separately
        # blends *all* length-compatible anchors.  Keep both lists explicit so
        # those two semantics cannot be accidentally conflated.
        choices: Dict[int, Tuple[List[int], List[int]]] = {}
        materializeParts: set[int] = set()
        for partIndex, part in enumerate(parts):
            if part["kind"] != "reuse":
                continue
            pool = self._anchors.get(part["pool_key"], [])
            selected = self._SelectAnchors(
                part["raw_cache"],
                part.get("prefix_cache"),
                pool,
                poolKey=part["pool_key"],
                consumerKey=consumerKey,
                messageKey=messageKey,
            )
            if selected is None:
                materializeParts.add(partIndex)
            else:
                choices[partIndex] = selected

        # Prediction happens per placeholder before the official runtime asks
        # ``has_active_anchor`` whether the whole request must be dense.  Thus
        # a shareable placeholder's top-p anchors gain activation counts even
        # when a different placeholder forces whole-request fallback.  These
        # counters affect only the later LFU eviction policy.
        for partIndex, (_, activated) in choices.items():
            pool = self._anchors[parts[partIndex]["pool_key"]]
            for anchorIndex in activated:
                pool[anchorIndex]["hits"] += 1

        if materializeParts:
            # Official KVCOMM changes the whole request to dense prefill when
            # any placeholder needs a new anchor, but set_anchor materializes
            # only those placeholders whose prediction flag is true.
            return denseFallback(materializeParts)

        predictedParts = list(parts)
        anchorHits = 0
        for partIndex, (compatible, activated) in choices.items():
            part = parts[partIndex]
            pool = self._anchors[part["pool_key"]]
            predictedPlaceholder, predictedPrefix = self._OffsetKvCachePair(
                part["base_cache"],
                part.get("prefix_cache"),
                pool,
                compatible,
                consumerKey=consumerKey,
            )
            partCopy = dict(part)
            partCopy["cache"] = predictedPlaceholder
            predictedParts[partIndex] = partCopy
            anchorHits += len(activated)

            prefixIndex = part.get("prefix_index")
            if prefixIndex is not None and predictedPrefix is not None:
                prefixCopy = dict(predictedParts[prefixIndex])
                prefixCopy["cache"] = predictedPrefix
                predictedParts[prefixIndex] = prefixCopy

        predictedCache = _ConcatCaches(
            [part["cache"] for part in predictedParts if part.get("cache") is not None]
        )
        nInput = len(assembledIds)
        if predictedCache is None or _CacheLength(predictedCache) != nInput or nInput <= 1:
            # This is a defensive correctness boundary for unusual cache
            # implementations; it never reports a partial reuse ratio.
            return denseFallback()

        lastId = assembledIds[-1]
        predictedCache.crop(nInput - 1)
        preprocessTime = time.perf_counter() - requestStart
        generated = self._gen.Generate(
            [lastId],
            pastKeyValues=predictedCache,
            maxNewTokens=maxNewTokens,
            returnCache=retain,
        )
        if retain:
            text, generationTtft, generationTotal, nTokens, fullCache, outputIds = generated
            self._RegisterOutput(
                state,
                text,
                outputIds,
                fullCache=fullCache,
                promptLength=nInput,
            )
        else:
            text, generationTtft, generationTotal, nTokens = generated

        reusedTokens = sum(
            len(part["ids"])
            for part in parts
            if part["kind"] == "reuse"
        )
        total = time.perf_counter() - requestStart
        ttft = preprocessTime + generationTtft
        return self._Result(
            text,
            ttft,
            max(total, preprocessTime + generationTotal),
            nTokens,
            {
                "reuse_ratio": reusedTokens / nInput if nInput else 0.0,
                "n_input": nInput,
                "reused_tokens": reusedTokens,
                "anchor_hits": anchorHits,
                "anchor_fallbacks": 0,
                "anchor_count": self._AnchorCount(),
            },
        )

    def _DenseResult(
        self,
        prompt: str,
        retain: bool,
        maxNewTokens: int,
        requestStart: float,
        *,
        anchorFallbacks: int,
    ) -> Result:
        ids = self._gen.Encode(prompt)
        preprocessTime = time.perf_counter() - requestStart
        generated = self._gen.Generate(
            ids,
            maxNewTokens=maxNewTokens,
            returnCache=retain,
        )
        if retain:
            text, generationTtft, generationTotal, nTokens, fullCache, outputIds = generated
            # The text registration is useful even when this request had no
            # reusable segment; its output can become one in the next RUN.
            state = self._states[0] if len(self._states) == 1 else None
            if state is not None:
                self._RegisterOutput(
                    state,
                    text,
                    outputIds,
                    fullCache=fullCache,
                    promptLength=len(ids),
                )
        else:
            text, generationTtft, generationTotal, nTokens = generated
        return self._Result(
            text,
            preprocessTime + generationTtft,
            max(time.perf_counter() - requestStart, preprocessTime + generationTotal),
            nTokens,
            {
                "reuse_ratio": 0.0,
                "n_input": len(ids),
                "reused_tokens": 0,
                "anchor_hits": 0,
                "anchor_fallbacks": anchorFallbacks,
                "anchor_count": self._AnchorCount(),
            },
        )

    def _DenseWithBase(
        self,
        prompt: str,
        fullIds: List[int],
        parts: List[Dict[str, Any]],
        retain: bool,
        maxNewTokens: int,
        requestStart: float,
        *,
        anchorFallbacks: int,
        anchorIds: Optional[List[int]] = None,
        anchorPromptLength: Optional[int] = None,
        anchorParts: Optional[List[Dict[str, Any]]] = None,
        materializeParts: Optional[set[int]] = None,
    ) -> Result:
        preprocessTime = time.perf_counter() - requestStart
        # Always request the cache on this path: it is needed to materialize
        # anchors.  The generated output cache is discarded when retain=False.
        generated = self._gen.Generate(
            fullIds,
            maxNewTokens=maxNewTokens,
            returnCache=True,
        )
        text, generationTtft, generationTotal, nTokens, fullCache, outputIds = generated
        if retain:
            state = self._states[0] if len(self._states) == 1 else None
            if state is not None:
                self._RegisterOutput(
                    state,
                    text,
                    outputIds,
                    fullCache=fullCache,
                    promptLength=len(fullIds),
                )
        if materializeParts:
            # KVBench generates from the exact full-prompt tokenization but
            # anchors use the official independently-tokenized placeholder
            # layout. Do that adapter-only canonical prefill after generation,
            # matching official set_anchor's post-first-token lifecycle. Drop
            # the generation full cache first so dense fallback never holds
            # two contextual full-prompt caches at once.
            del fullCache
            canonicalIds = fullIds if anchorIds is None else anchorIds
            canonicalOutput = self._gen.Prefill(canonicalIds)
            self._MaterializeAnchors(
                canonicalOutput.past_key_values,
                (
                    len(canonicalIds)
                    if anchorPromptLength is None
                    else anchorPromptLength
                ),
                parts if anchorParts is None else anchorParts,
                materializeParts=materializeParts,
            )
            del canonicalOutput
        total = time.perf_counter() - requestStart
        return self._Result(
            text,
            preprocessTime + generationTtft,
            max(total, preprocessTime + generationTotal),
            nTokens,
            {
                # The dense path computes every input token contextually; base
                # isolated work does not make these tokens ``reused``.
                "reuse_ratio": 0.0,
                "n_input": len(fullIds),
                "reused_tokens": 0,
                "anchor_hits": 0,
                "anchor_fallbacks": anchorFallbacks,
                "anchor_count": self._AnchorCount(),
            },
        )

    def _MakeParts(
        self,
        segments: List[Dict[str, Any]],
        matches: List[Tuple[Optional[int], str]],
    ) -> List[Dict[str, Any]]:
        parts: List[Dict[str, Any]] = []
        slot = 0
        # ComposeInterleavedReuse is intentionally content-addressed. When two
        # agents emit identical text, every occurrence can therefore resolve
        # to the first matching segment. Official KVCOMM retains the source
        # placeholder identity (agent_i_current), so recover that identity by
        # pairing equal-text occurrences with retained segments in workflow
        # order. MultiAgentFullConnection emits each prior output once and in
        # that same order.
        segmentIndicesByText: Dict[str, List[int]] = {}
        for index, segment in enumerate(segments):
            segmentIndicesByText.setdefault(str(segment["text"]), []).append(index)
        consumedByText: Dict[str, int] = {}
        for segmentIndex, text in matches:
            if not text:
                continue
            if segmentIndex is None:
                parts.append({"kind": "fresh", "text": text, "slot": None})
            else:
                candidates = segmentIndicesByText.get(text, [])
                occurrence = consumedByText.get(text, 0)
                if occurrence < len(candidates):
                    segmentIndex = candidates[occurrence]
                consumedByText[text] = occurrence + 1
                parts.append(
                    {
                        "kind": "reuse",
                        "text": text,
                        "segment": segments[segmentIndex],
                        "slot": slot,
                    }
                )
                slot += 1
        return parts

    def _TokenizeParts(
        self,
        prompt: str,
        fullIds: List[int],
        parts: List[Dict[str, Any]],
    ) -> Optional[List[List[int]]]:
        """Partition full-prompt token IDs by character-span ownership.

        KVBench passes complete rendered prompts to ``Run``.  Encoding each
        reusable segment independently changes BPE tokens at boundaries (a
        common example is ``" For"`` versus ``"For"``), which would make a
        stitched cache address a different token sequence.  Fast tokenizers
        expose offsets, so assign every complete-prompt token to the span where
        that token starts.  A token crossing a boundary is consequently kept
        whole in one piece and the concatenated IDs remain exactly equal to
        the real prompt IDs.
        """
        tokenizer = getattr(self._gen, "tokenizer", None)
        if tokenizer is None:
            return None
        char = 0
        for part in parts:
            part["char_start"] = char
            char += len(part["text"])
            part["char_end"] = char
        if char != len(prompt):
            return None
        try:
            encoded = tokenizer(
                prompt,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            offsets = encoded["offset_mapping"]
            encodedIds = encoded["input_ids"]
        except (AttributeError, KeyError, NotImplementedError, TypeError, ValueError):
            return None

        if offsets and isinstance(offsets[0], (list, tuple)) and offsets[0] and isinstance(offsets[0][0], (list, tuple)):
            offsets = offsets[0]
        if encodedIds and isinstance(encodedIds[0], list):
            encodedIds = encodedIds[0]
        if len(offsets) != len(fullIds) or list(encodedIds) != list(fullIds):
            return None

        owners: List[int] = []
        partIndex = 0
        for start, _ in offsets:
            start = int(start)
            while (
                partIndex + 1 < len(parts)
                and start >= parts[partIndex]["char_end"]
            ):
                partIndex += 1
            owners.append(partIndex)

        result: List[List[int]] = []
        for partIndex in range(len(parts)):
            ids = [
                token
                for token, owner in zip(fullIds, owners)
                if owner == partIndex
            ]
            result.append(ids)
        return result if [token for ids in result for token in ids] == list(fullIds) else None

    def _TemplateKey(self, parts: List[Dict[str, Any]]) -> str:
        """Return a structural prompt key with slot-specific placeholders."""
        pieces = []
        for part in parts:
            if part["kind"] == "reuse":
                pieces.append(self._TemplateMarker(int(part["slot"])))
            else:
                pieces.append(part["text"])
        return "".join(pieces)

    @staticmethod
    def _TemplateMarker(slot: int) -> str:
        # Keep this ordinary text rather than a model-specific special token.
        # It is used only to build the canonical prefix cache and never sent
        # to generation as user-visible prompt content.
        return f"[KVCOMM_PLACEHOLDER_{slot}]"

    def _TouchPrefixTemplate(self, key: str) -> None:
        if key in self._prefixTemplateOrder:
            self._prefixTemplateOrder.remove(key)
        self._prefixTemplateOrder.append(key)

    def _GetPrefixTemplate(self, parts: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Build/fetch the official-style static prefix KV for one topology.

        ``prepare_prefix_kv_segments`` in the official implementation runs a
        template containing placeholder markers once per agent and stores the
        resulting text-segment KV.  The actual request's placeholder cache is
        inserted later, with RoPE offset correction and anchor deltas.  This is
        the same separation, expressed with KVBench's interleaved parts.
        """
        if self._gen is None:
            return None
        key = self._TemplateKey(parts)
        cached = self._prefixTemplates.get(key)
        if cached is not None:
            self._TouchPrefixTemplate(key)
            return cached

        templateParts = []
        for part in parts:
            if part["kind"] == "reuse":
                text = self._TemplateMarker(int(part["slot"]))
            else:
                text = part["text"]
            templateParts.append(
                {
                    "kind": part["kind"],
                    "text": text,
                    "slot": part.get("slot"),
                }
            )

        templateText = "".join(part["text"] for part in templateParts)
        templateIds = self._gen.Encode(templateText)
        templatePieceIds = self._TokenizeParts(
            templateText, templateIds, templateParts
        )
        if templatePieceIds is None:
            return None

        templateCursor = 0
        for part, ids in zip(templateParts, templatePieceIds):
            part["start"] = templateCursor
            part["end"] = templateCursor + len(ids)
            templateCursor = part["end"]

        freshIndices = [
            index
            for index, part in enumerate(templateParts)
            if part["kind"] == "fresh" and templatePieceIds[index]
        ]
        templateCaches: Dict[int, Any] = {}
        if freshIndices:
            # This is topology-only work.  On the first topology occurrence it
            # happens inside Run and is therefore included in TTFT; later
            # requests fetch the detached cache from this bounded store.
            templateOutput = self._gen.Prefill(templateIds)
            fullTemplateCache = templateOutput.past_key_values
            for index in freshIndices:
                part = templateParts[index]
                templateCaches[index] = _SliceCache(
                    fullTemplateCache, part["start"], part["end"]
                )

        entry = {
            "parts": templateParts,
            "piece_ids": templatePieceIds,
            "caches": templateCaches,
        }
        self._prefixTemplates[key] = entry
        self._TouchPrefixTemplate(key)
        while len(self._prefixTemplateOrder) > self._maxPrefixTemplates:
            evicted = self._prefixTemplateOrder.pop(0)
            self._prefixTemplates.pop(evicted, None)
        return entry

    def _GetIsolatedBase(self, ids: List[int]) -> Any:
        """Fetch or compute one message/response's position-zero KV cache."""
        key = tuple(int(token) for token in ids)
        cached = self._isolatedBases.get(key)
        if cached is not None:
            if key in self._isolatedBaseOrder:
                self._isolatedBaseOrder.remove(key)
            self._isolatedBaseOrder.append(key)
            return cached

        rawCache = self._gen.Prefill(ids).past_key_values
        self._isolatedBases[key] = rawCache
        self._isolatedBaseOrder.append(key)
        return rawCache

    def _BuildBase(
        self,
        parts: List[Dict[str, Any]],
        pieceIds: List[List[int]],
    ) -> Any:
        """Build the official KVCOMM base cache for this request.

        Reusable pieces are prefetched in isolation, then their K is RoPE
        shifted to the current absolute position.  Static fresh pieces come
        from the persistent canonical template cache, just like official
        ``prepare_prefix_kv_segments``.  If boundary tokenization prevents
        that exact cache match, the safe contextual prefill path is retained.
        Any first-time template or dynamic prefill still happens in Run and is
        therefore part of TTFT.
        """
        template = self._GetPrefixTemplate(parts)
        # Static segments before the first incompatible boundary remain valid.
        # Once a fresh segment's BPE boundary differs from the canonical
        # template, later fresh text is context-dependent and must be
        # prefetched against the assembled actual prefix.
        dynamicFreshTail = template is None
        assembled: List[Any] = []
        cursor = 0
        for index, (part, ids) in enumerate(zip(parts, pieceIds)):
            part["ids"] = ids
            part["start"] = cursor
            part["end"] = cursor + len(ids)
            if not ids:
                part["cache"] = None
                part["base_cache"] = None
                part["raw_cache"] = None
                cursor = part["end"]
                continue

            if part["kind"] == "reuse":
                segment = part.get("segment")
                storedIds = segment.get("base_ids") if segment is not None else None
                storedCache = segment.get("base_cache") if segment is not None else None
                if (
                    storedCache is not None
                    and storedIds is not None
                    and list(storedIds) == list(ids)
                ):
                    # This is the official response-cache path.  It avoids a
                    # second prefill for a retained output whose KV was
                    # already produced by the preceding RUN.
                    rawCache = storedCache
                else:
                    rawCache = self._GetIsolatedBase(ids)
                    if segment is not None and storedIds is None:
                        # Official shared input memory records one tokenization
                        # and one isolated KV for ``message`` and every later
                        # destination agent consumes that same pair.  A raw
                        # KVBench prompt boundary can otherwise assign a
                        # boundary-crossing BPE token to opposite sides in two
                        # RUNs, giving one message inconsistent placeholder
                        # lengths (and invalid anchor deltas).  Canonicalize on
                        # the first online use; this work still occurs in Run
                        # and remains inside TTFT.
                        segment["base_ids"] = list(ids)
                        segment["base_cache"] = rawCache
                shiftedCache = self._ShiftCache(rawCache, cursor)
                part["raw_cache"] = rawCache
                part["base_cache"] = shiftedCache
                part["cache"] = shiftedCache
                assembled.append(shiftedCache)
            else:
                staticPart = False
                if not dynamicFreshTail:
                    templatePart = template["parts"][index]
                    templateCache = template["caches"].get(index)
                    staticPart = bool(
                        templateCache is not None
                        and list(ids) == list(template["piece_ids"][index])
                    )
                    if staticPart:
                        # Shift canonical template keys from marker positions
                        # to actual request positions. Values are contextualized
                        # by the canonical placeholder layout; anchor prefix
                        # deltas correct the content-dependent part.
                        part["cache"] = self._ShiftCache(
                            templateCache,
                            cursor - int(templatePart["start"]),
                        )
                        part["base_cache"] = part["cache"]
                        assembled.append(part["cache"])

                if not staticPart:
                    dynamicFreshTail = True
                    output = self._gen.Prefill(
                        ids,
                        pastKeyValues=(
                            _ConcatCaches(assembled)
                            if assembled
                            else None
                        ),
                    )
                    fullPieceCache = output.past_key_values
                    part["cache"] = _SliceCache(fullPieceCache, cursor, part["end"])
                    part["base_cache"] = part["cache"]
                    assembled.append(part["cache"])

            cursor = part["end"]

        for index, part in enumerate(parts):
            if part["kind"] != "reuse":
                continue
            prefixIndex = index + 1 if index + 1 < len(parts) and parts[index + 1]["kind"] == "fresh" else None
            part["prefix_index"] = prefixIndex
            part["prefix_cache"] = parts[prefixIndex]["base_cache"] if prefixIndex is not None else None
            if prefixIndex is not None:
                parts[prefixIndex]["prefix_owner"] = index
        # Every caller consumes the aligned per-part caches above. Building a
        # second concatenated full-prompt base here retained a large tensor
        # that was never read and inflated dense-fallback peak memory.

    # --------------------------------------------------------------- anchors
    def _NamespaceParts(self, parts: List[Dict[str, Any]]) -> str:
        # The first RUN's fresh spans contain stable workflow instructions
        # after the concrete task has been removed. This becomes the workload
        # namespace shared by every later RUN in the same Case; _PoolKey then
        # separates semantic placeholders within that workload.
        skeleton = []
        for part in parts:
            if part["kind"] == "reuse":
                skeleton.append("<reusable-placeholder>")
            else:
                skeleton.append(part["text"])
        # Do not include the request/run counter here.  The counter is only a
        # local execution detail; including it would create a fresh namespace
        # for every agent/case and defeat KVCOMM's online cross-case learning.
        # The first-RUN topology separates different KVCOMM Tasks.
        return "".join(skeleton)

    def _PoolKey(self, namespace: str, placeholderId: str) -> str:
        digest = hashlib.sha256(namespace.encode("utf-8")).hexdigest()[:24]
        return f"{digest}:placeholder={placeholderId}"

    def _ConsumerKey(self) -> str:
        """Stable destination-agent identity for the sequential workflow."""
        return f"run={self._runIndex}"

    def _SelectAnchors(
        self,
        rawCache: Any,
        prefixCache: Any,
        pool: List[Dict[str, Any]],
        *,
        poolKey: str,
        consumerKey: str,
        messageKey: str,
    ) -> Optional[Tuple[List[int], List[int]]]:
        """Port ``predict_as_anchor``: entropy decides dense vs reuse.

        Return ``(compatible, activated)``.  The first list is consumed by
        official ``offset_kv_cache_pair``; the top-p second list affects only
        anchor activation counters and therefore future eviction.
        """
        if rawCache is None:
            return None
        import torch

        _, candidateValue = _StackCache(rawCache)
        candidateLength = int(candidateValue.shape[-2])
        flags = self._anchorFlags.setdefault(poolKey, {})
        exactMessage = messageKey in flags
        sameMessage = next(
            (anchor for anchor in pool if anchor.get("message_key") == messageKey),
            None,
        )
        if (
            exactMessage
            and flags[messageKey]
            and (
                sameMessage is None
                or consumerKey not in sameMessage.get("contexts", {})
            )
        ):
            # Official has_active_anchor forces dense prefill until this exact
            # anchor has acquired the destination node's delta fields.
            return None

        eligible = []
        for index, anchor in enumerate(pool):
            context = anchor.get("contexts", {}).get(consumerKey)
            if context is None:
                continue
            if int(anchor["ph_value_embedding"].shape[-2]) < candidateLength:
                continue
            eligible.append(index)
        if prefixCache is not None:
            prefixShape = _StackCache(prefixCache)[0].shape
            eligible = [
                index
                for index in eligible
                if pool[index]["contexts"][consumerKey].get("pf_key_delta") is not None
                and tuple(
                    pool[index]["contexts"][consumerKey]["pf_key_delta"].shape[-2:]
                ) == tuple(prefixShape[-2:])
            ]
        if exactMessage:
            # The official exact-message fast path bypasses anchor prediction;
            # offset_kv_cache_pair then weights every length-compatible anchor.
            return eligible, []
        if len(pool) < 2 or len(eligible) < 2:
            flags[messageKey] = True
            return None

        candidate = candidateValue[..., :candidateLength, :]
        anchors = torch.stack(
            [pool[index]["ph_value_embedding"][..., :candidateLength, :] for index in eligible]
        )
        diff = (candidate.unsqueeze(0) - anchors).norm(
            2, dim=(1, 2, 3, 4, 5)
        )
        similarity = torch.softmax(-diff.float(), dim=0)
        entropy = -(similarity * (similarity + 1e-40).log2()).sum()
        maxEntropy = self.threshold * math.log2(len(eligible))
        if float(entropy) > maxEntropy:
            flags[messageKey] = True
            return None

        sortedSimilarity, sortedIndices = torch.sort(similarity, descending=True)
        cumulative = torch.cumsum(sortedSimilarity, dim=0)
        # Match the official cutoff exactly: select the shortest prefix whose
        # cumulative similarity reaches top_p=0.9.  In the unusual case where
        # the first anchor already exceeds top_p, the official implementation
        # keeps the full candidate list (rather than silently changing its
        # anchor-selection semantics here).
        cutoffCandidates = (cumulative < 0.9).nonzero(as_tuple=True)[0]
        cutoffIndex = (
            int(cutoffCandidates[-1].item())
            if cutoffCandidates.numel()
            else len(sortedSimilarity) - 1
        )
        activated = [
            eligible[int(index)]
            for index in sortedIndices[: cutoffIndex + 1]
        ]
        flags[messageKey] = False
        return eligible, activated

    def _OffsetKvCachePair(
        self,
        basePlaceholder: Any,
        basePrefix: Any,
        pool: List[Dict[str, Any]],
        selected: List[int],
        *,
        consumerKey: str,
    ) -> Tuple[Any, Optional[Any]]:
        """Port ``offset_kv_cache_pair`` using the current base position.

        The official implementation matches against the already RoPE-shifted
        base placeholder. Anchor embeddings are likewise the shifted base
        captured by ``set_anchor`` in the latest materializing consumer.
        """
        import torch

        if not selected:
            # Official returns untouched copies when no anchor covers the
            # exact-message placeholder length.
            return (
                _NewCache(
                    [(key.clone(), value.clone()) for key, value in _CachePairs(basePlaceholder)]
                ),
                (
                    _NewCache(
                        [(key.clone(), value.clone()) for key, value in _CachePairs(basePrefix)]
                    )
                    if basePrefix is not None
                    else None
                ),
            )

        baseKey, baseValue = _StackCache(basePlaceholder)
        placeholderLength = int(baseKey.shape[-2])
        used = [pool[index] for index in selected]
        contexts = [anchor["contexts"][consumerKey] for anchor in used]

        anchorKeyPrefix = torch.stack(
            [anchor["ph_key_embedding"][..., -placeholderLength:, :] for anchor in used]
        )
        anchorValuePrefix = torch.stack(
            [anchor["ph_value_embedding"][..., -placeholderLength:, :] for anchor in used]
        )
        currentKey = baseKey[..., -placeholderLength:, :]
        currentValue = baseValue[..., -placeholderLength:, :]

        # Official prefix weights are per head/dimension and compare the tail
        # of the placeholder representation.
        weightsKeyPrefix = torch.softmax(
            -(currentKey.unsqueeze(0) - anchorKeyPrefix).norm(2, dim=-2).float(), dim=0
        ).unsqueeze(-2)
        weightsValuePrefix = torch.softmax(
            -(currentValue.unsqueeze(0) - anchorValuePrefix).norm(2, dim=-2).float(), dim=0
        ).unsqueeze(-2)

        # Official placeholder weights are per token, derived from absolute
        # base/value proximity.  Official anchor deltas are sliced from the
        # beginning when a candidate is shorter than the stored anchor.
        anchorKeyPlaceholder = torch.stack(
            [anchor["ph_key_embedding"][..., :placeholderLength, :] for anchor in used]
        )
        anchorValuePlaceholder = torch.stack(
            [anchor["ph_value_embedding"][..., :placeholderLength, :] for anchor in used]
        )
        weightsKeyPlaceholder = torch.softmax(
            -(currentKey.unsqueeze(0) - anchorKeyPlaceholder).abs().mean(
                dim=(-5, -4, -3, -1), keepdim=True
            ).float(),
            dim=0,
        )
        weightsValuePlaceholder = torch.softmax(
            -(currentValue.unsqueeze(0) - anchorValuePlaceholder).abs().mean(
                dim=(-5, -4, -3, -1), keepdim=True
            ).float(),
            dim=0,
        )

        phKeyDelta = torch.stack(
            [context["ph_key_delta"][..., :placeholderLength, :] for context in contexts]
        )
        phValueDelta = torch.stack(
            [context["ph_value_delta"][..., :placeholderLength, :] for context in contexts]
        )
        predictedKey = baseKey + (
            weightsKeyPlaceholder * phKeyDelta
        ).sum(0).to(baseKey.dtype)
        predictedValue = baseValue + (
            weightsValuePlaceholder * phValueDelta
        ).sum(0).to(baseValue.dtype)
        # Preserve layer 0 exactly as in the official KVCOMM correction.
        predictedKey[0] = baseKey[0]
        predictedValue[0] = baseValue[0]
        predictedPlaceholder = _NewCache(
            list(zip(list(predictedKey), list(predictedValue)))
        )

        predictedPrefix = None
        if basePrefix is not None:
            prefixKey, prefixValue = _StackCache(basePrefix)
            prefixDeltasKey = torch.stack([context["pf_key_delta"] for context in contexts])
            prefixDeltasValue = torch.stack([context["pf_value_delta"] for context in contexts])
            predictedPrefixKey = prefixKey + (
                weightsKeyPrefix * prefixDeltasKey
            ).sum(0).to(prefixKey.dtype)
            predictedPrefixValue = prefixValue + (
                weightsValuePrefix * prefixDeltasValue
            ).sum(0).to(prefixValue.dtype)
            predictedPrefixKey[0] = prefixKey[0]
            predictedPrefixValue[0] = prefixValue[0]
            predictedPrefix = _NewCache(
                list(
                    zip(
                        list(predictedPrefixKey),
                        list(predictedPrefixValue),
                    )
                )
            )
        return predictedPlaceholder, predictedPrefix

    def _MaterializeAnchors(
        self,
        fullCache: Any,
        promptLength: int,
        parts: List[Dict[str, Any]],
        *,
        materializeParts: set[int],
    ) -> None:
        """Port ``set_anchor`` with official per-placeholder storage.

        One message anchor owns its base embedding once. Deltas for each
        destination RUN are updated in ``contexts``; this is the KVBench
        equivalent of the official ``{node_id}_ph_*`` / ``{node_id}_pf_*``
        fields on the same ``anchors[ph_id][message]`` entry.
        """
        for index, part in enumerate(parts):
            if (
                index not in materializeParts
                or part["kind"] != "reuse"
                or part.get("raw_cache") is None
            ):
                continue
            prefixIndex = part.get("prefix_index")
            realPlaceholder = _SliceCache(
                fullCache, part["start"], part["end"]
            )
            realPrefix = None
            if prefixIndex is not None and parts[prefixIndex].get("ids"):
                realPrefix = _SliceCache(
                    fullCache,
                    parts[prefixIndex]["start"],
                    parts[prefixIndex]["end"],
                )
            rawKey, _ = _StackCache(part["raw_cache"])
            baseKey, baseValue = _StackCache(part["base_cache"])
            realKey, realValue = _StackCache(realPlaceholder)
            if rawKey.shape[-2] != realKey.shape[-2]:
                continue

            prefixKeyDelta = prefixValueDelta = None
            if realPrefix is not None and part.get("prefix_cache") is not None:
                basePrefixKey, basePrefixValue = _StackCache(part["prefix_cache"])
                realPrefixKey, realPrefixValue = _StackCache(realPrefix)
                if basePrefixKey.shape != realPrefixKey.shape:
                    # Prefix deltas must be shape-compatible for weighted
                    # interpolation. A later request can still use the ph-only
                    # anchor when its prefix is empty, but never guess here.
                    continue
                prefixKeyDelta = (realPrefixKey - basePrefixKey).detach()
                prefixValueDelta = (realPrefixValue - basePrefixValue).detach()

            key = part["pool_key"]
            pool = self._anchors.setdefault(key, [])
            if len(pool) > self.maxAnchorNum:
                # Preserve the official strict ``> max_anchor_num`` check.
                # Consequently V=20 reaches a bounded steady state of 21
                # entries; changing this to >= would alter paper semantics.
                candidates = pool[: min(self.windowSize, len(pool))]
                victim = min(range(len(candidates)), key=lambda i: candidates[i]["hits"])
                del pool[victim]

            context = {
                "ph_key_delta": (realKey - baseKey).detach(),
                "ph_value_delta": (realValue - baseValue).detach(),
                "pf_key_delta": prefixKeyDelta,
                "pf_value_delta": prefixValueDelta,
            }
            messageKey = str(part["message_key"])
            existing = next(
                (anchor for anchor in pool if anchor.get("message_key") == messageKey),
                None,
            )
            if existing is None:
                pool.append(
                    {
                        # Official set_anchor stores the RoPE-shifted base and
                        # later destination agents update it in place.
                        "message_key": messageKey,
                        "ph_key_embedding": baseKey.detach(),
                        "ph_value_embedding": baseValue.detach(),
                        "contexts": {part["consumer_key"]: context},
                        "hits": 0,
                        "serial": self._anchorSerial,
                    }
                )
                self._anchorSerial += 1
            else:
                # Official ``anchor_store[ph_id][message].update(entry)`` adds
                # this destination node's deltas to the existing message
                # anchor, without consuming another slot from V.
                existing["ph_key_embedding"] = baseKey.detach()
                existing["ph_value_embedding"] = baseValue.detach()
                existing.setdefault("contexts", {})[part["consumer_key"]] = context

    def _ShiftCache(self, cache: Any, offset: int) -> Any:
        """Apply the official RoPE position correction to cache keys only."""
        import torch

        if offset == 0:
            return _NewCache([(key.clone(), value.clone()) for key, value in _CachePairs(cache)])
        model = self._gen.model
        modelCore = getattr(model, "model", None)
        rotary = getattr(modelCore, "rotary_emb", None)
        if rotary is None:
            raise RuntimeError(
                "KVCommTransformer requires a model.model.rotary_emb for "
                "position-correct KV reuse; this model architecture is unsupported"
            )
        pairs = _CachePairs(cache)
        reference = pairs[0][0]
        positionIds = torch.full(
            (reference.shape[0], reference.shape[-2]),
            int(offset),
            dtype=torch.long,
            device=reference.device,
        )
        try:
            cos, sin = rotary(reference, positionIds)
        except TypeError:
            cos, sin = rotary(x=reference, position_ids=positionIds)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        stackedKeys = torch.stack([key for key, _ in pairs])
        while cos.ndim < stackedKeys.ndim:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        rotatedKeys = stackedKeys * cos + _RotateHalf(stackedKeys) * sin
        return _NewCache(
            [
                (rotatedKeys[layer], value.clone())
                for layer, (_, value) in enumerate(pairs)
            ]
        )

    # --------------------------------------------------------------- utilities
    def _RegisterOutput(
        self,
        state: Dict[str, Any],
        text: str,
        outputIds: List[int],
        *,
        fullCache: Any = None,
        promptLength: Optional[int] = None,
    ) -> None:
        """Register retained generated text for later workflow RUNs.

        The official backend retains response KV for the following agent.  HF
        generation returns a cache that is one token short of the generated
        text (the last token has not yet been fed back through the model), so
        complete that one-token cache before storing it.  The cache is rotated
        back to position zero; ``_BuildBase`` applies the new absolute offset
        when the segment appears in the next prompt.  This is case-local state,
        not a persistent anchor, and is cleared by ``Reset``.
        """
        if not outputIds or not text:
            return
        baseCache = None
        if fullCache is not None and promptLength is not None:
            # ``text`` is what the workflow inserts into the next prompt;
            # decode(skip_special_tokens=True) can remove EOS/chat markers from
            # ``outputIds``.  Align the stored cache to those actual text ids,
            # otherwise the next request would necessarily take the dense path.
            textIds = self._gen.Encode(text)
            if outputIds[: len(textIds)] == textIds:
                expectedLength = int(promptLength) + len(textIds)
                cache = fullCache
                cacheLength = _CacheLength(cache)
                # Generate stops after producing the final token, so its
                # returned past cache normally contains the preceding output
                # tokens.  Feed the final text token once when needed.
                if cacheLength == expectedLength - 1:
                    cache = self._gen.Forward([textIds[-1]], cache).past_key_values
                    cacheLength = _CacheLength(cache)
                if cacheLength == expectedLength:
                    outputCache = _SliceCache(cache, int(promptLength), expectedLength)
                    baseCache = self._ShiftCache(outputCache, -int(promptLength))
        state["segments"].append(
            {
                "text": str(text),
                "base_ids": self._gen.Encode(text),
                "base_cache": baseCache,
                # RUN order is stable and maxCaseBatchSize=1, so this is the
                # same semantic identity as official agent_<node_id>_current.
                "placeholder_id": f"agent_{self._runIndex}_current",
            }
        )

    def _DebugMemory(self, phase: str) -> None:
        """Log tracked KV lifetimes without changing the execution path.

        The current KVBench adapter has no official ``message`` dictionaries;
        the closest raw-cache stores are ``_isolatedBases`` and the
        case-local retained-response caches in ``_states``.  This helper is
        called only from ``Reset`` (after a request has finished), never from
        the timed ``Run`` path, and never calls ``gc`` or ``empty_cache``.
        """
        if not self._memoryDebug or self._gen is None:
            return
        try:
            import torch

            def tensorBytes(value: Any, seen: Optional[set[int]] = None) -> int:
                if seen is None:
                    seen = set()
                if torch.is_tensor(value):
                    identity = id(value)
                    if identity in seen:
                        return 0
                    seen.add(identity)
                    return int(value.numel() * value.element_size())
                if isinstance(value, dict):
                    return sum(
                        tensorBytes(key, seen) + tensorBytes(item, seen)
                        for key, item in value.items()
                    )
                if isinstance(value, (list, tuple, set)):
                    return sum(tensorBytes(item, seen) for item in value)
                if hasattr(value, "layers"):
                    total = 0
                    for key, item in CacheLayerPairs(value):
                        total += tensorBytes(key, seen) + tensorBytes(item, seen)
                    return total
                return 0

            stateSegments = [
                segment
                for state in self._states
                for segment in state.get("segments", [])
            ]
            responseEntries = sum(
                1
                for segment in stateSegments
                if segment.get("base_cache") is not None
                and str(segment.get("placeholder_id", "")).startswith("agent_")
            )
            inputEntries = len(self._isolatedBases)
            prefixEntries = len(self._prefixTemplates)
            anchorPools = len(self._anchors)
            anchors = self._AnchorCount()
            anchorPoolMax = max(
                (len(pool) for pool in self._anchors.values()), default=0
            )
            anchorContexts = sum(
                len(anchor.get("contexts", {}))
                for pool in self._anchors.values()
                for anchor in pool
            )
            anchorFlags = sum(len(flags) for flags in self._anchorFlags.values())
            anchorBytes = tensorBytes(self._anchors)
            isolatedBytes = tensorBytes(self._isolatedBases)
            prefixBytes = tensorBytes(self._prefixTemplates)
            stateBytes = tensorBytes(self._states)
            trackedBytes = tensorBytes(
                [self._anchors, self._isolatedBases, self._prefixTemplates, self._states]
            )
            allocated = int(torch.cuda.memory_allocated())
            reserved = int(torch.cuda.memory_reserved())
            print(
                "[KVCOMM MEMORY] "
                f"sample={self._caseSerial} phase={phase} "
                f"allocated={allocated} reserved={reserved} "
                f"isolated_base_entries={inputEntries} "
                f"current_response_entries={responseEntries} "
                f"prefix_template_entries={prefixEntries} "
                f"anchor_pools={anchorPools} anchors={anchors} "
                f"anchor_pool_max={anchorPoolMax} "
                f"anchor_contexts={anchorContexts} "
                f"anchor_flags={anchorFlags} "
                "weight_entries=0 "
                f"anchor_bytes={anchorBytes} isolated_bytes={isolatedBytes} "
                f"prefix_bytes={prefixBytes} state_bytes={stateBytes} "
                f"tracked_bytes={trackedBytes}",
                flush=True,
            )
        except Exception as exc:  # diagnostics must never affect a benchmark
            print(f"[KVCOMM MEMORY] instrumentation_error={exc!r}", flush=True)

    def _AnchorCount(self) -> int:
        return sum(len(pool) for pool in self._anchors.values())

    def _Result(
        self,
        text: str,
        ttft: float,
        total: float,
        nTokens: int,
        metadata: Dict[str, Any],
    ) -> Result:
        return Result(
            output=text,
            performance={
                TtftKey: ttft,
                NumOutputTokensKey: nTokens,
                TotalTimeKey: total,
            },
            metadata={"backend": self.backend, **metadata},
        )
