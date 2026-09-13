"""Dependency analysis over the existing transformer FullPrefill path.

The method treats the strings received by ``Prepare`` as the reusable units.
``Prompt.ComposeInterleavedReuseSpans`` locates those exact units in the RUN
prompt; no document boundaries are guessed from prompt delimiters.

Generation is deliberately performed by the unmodified
``TransformersGenerator.Generate`` path.  Attention instrumentation runs in a
separate prefill pass and returns only selected query rows, so it cannot alter
the KV used for generation and does not materialize the full attention tensor.
Independent document prefills use the document's full-prompt token positions
for RoPE, even though their causal cache contains only that document.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from difflib import SequenceMatcher
import importlib
import os
import time
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from core.Result import Result
from helpers.backends.Prompt import ComposeInterleavedReuseSpans
from helpers.backends.TransformersHelper import CacheLayerPairs
from methods.FullPrefill import FullPrefillTransformer


_MetricNames = (
    "missing_attention",
    "value_weighted_dependency",
    "missing_attention_skip_k",
    "value_weighted_dependency_skip_k",
    "kv_deviation",
)


@dataclass(frozen=True)
class _DocumentOccurrence:
    prepareIndex: int
    text: str
    charStart: int
    charEnd: int
    tokenStart: int
    tokenEnd: int
    # (full-prompt token index, independent-chunk token index).  A tokenizer
    # may merge text across a character boundary, so these sequences are not
    # always identical at the first/last token of a chunk.
    tokenPairs: Tuple[Tuple[int, int], ...]
    independentIds: Tuple[int, ...]


def _AlignTokenPairs(
    fullTokenIndices: Sequence[int],
    fullTokenIds: Sequence[int],
    independentIds: Sequence[int],
) -> List[Tuple[int, int]]:
    """Align equal token IDs while keeping boundary mismatches out.

    Isolated tokenization normally differs from full-prompt tokenization only
    at the two chunk edges.  Strip equal edges first so long documents do not
    needlessly pass their entire token sequence through ``SequenceMatcher``.
    The matcher remains as a fallback for a tokenizer that changes more than
    one edge token.
    """
    prefix = 0
    while (
        prefix < len(fullTokenIds)
        and prefix < len(independentIds)
        and fullTokenIds[prefix] == independentIds[prefix]
    ):
        prefix += 1

    fullEnd = len(fullTokenIds)
    independentEnd = len(independentIds)
    while (
        fullEnd > prefix
        and independentEnd > prefix
        and fullTokenIds[fullEnd - 1] == independentIds[independentEnd - 1]
    ):
        fullEnd -= 1
        independentEnd -= 1

    pairs = [
        (fullTokenIndices[index], index)
        for index in range(prefix)
    ]
    middleFull = fullTokenIds[prefix:fullEnd]
    middleIndependent = independentIds[prefix:independentEnd]
    for fullOffset, independentOffset, size in SequenceMatcher(
        a=middleFull,
        b=middleIndependent,
        autojunk=False,
    ).get_matching_blocks():
        pairs.extend(
            (
                fullTokenIndices[prefix + fullOffset + delta],
                prefix + independentOffset + delta,
            )
            for delta in range(size)
        )

    suffixLength = len(fullTokenIds) - fullEnd
    pairs.extend(
        (
            fullTokenIndices[fullEnd + delta],
            independentEnd + delta,
        )
        for delta in range(suffixLength)
    )
    return pairs


class _DependencyStats:
    """Streaming layer/head/token aggregation for one sample."""

    def __init__(self, tokenInfo: Mapping[int, Tuple[int, int]], skipK: int):
        # tokenInfo[full_token_index] = (document_start, document_local_index)
        self.tokenInfo = dict(tokenInfo)
        self.skipK = skipK
        self.queryRows: List[int] = []
        self.queryPositions: List[int] = []
        self.docStarts: List[int] = []
        self.skipRows: List[bool] = []

        self.missingAttentionSum = 0.0
        self.valueDependencySum = 0.0
        self.missingAttentionSkipSum = 0.0
        self.valueDependencySkipSum = 0.0
        self.headCount = 0
        self.skipHeadCount = 0

    def SetQueryRows(self, queryStart: int, queryLength: int) -> None:
        self.queryRows = []
        self.queryPositions = []
        self.docStarts = []
        self.skipRows = []

        for position in sorted(self.tokenInfo):
            if queryStart <= position < queryStart + queryLength:
                docStart, localIndex = self.tokenInfo[position]
                self.queryRows.append(position - queryStart)
                self.queryPositions.append(position)
                self.docStarts.append(docStart)
                self.skipRows.append(localIndex >= self.skipK)

    def Record(
        self,
        torch,
        qwenModule,
        module,
        query,
        key,
        value,
        scaling: float,
    ) -> None:
        if not self.queryRows:
            return

        # Qwen3 uses GQA.  Attention probabilities have num_attention_heads,
        # while the cache stores num_key_value_heads; repeat V to the same head
        # space used by alpha[l,h,t,s].
        keyFull = qwenModule.repeat_kv(key, module.num_key_value_groups)
        valueFull = qwenModule.repeat_kv(value, module.num_key_value_groups)
        valueNorm = valueFull.float().norm(dim=-1).unsqueeze(2)
        keyFull = keyFull.float()
        queryFull = query.float()
        keyLength = keyFull.shape[-2]
        keyPositions = torch.arange(
            keyLength, device=query.device, dtype=torch.long
        )

        # Limit the temporary attention tensor to a small number of selected
        # query rows.  This is the important memory distinction from
        # output_attentions=True, which retains every query/key pair.
        rowBatchSize = 64
        for offset in range(0, len(self.queryRows), rowBatchSize):
            rows = self.queryRows[offset: offset + rowBatchSize]
            positions = self.queryPositions[offset: offset + rowBatchSize]
            docStarts = self.docStarts[offset: offset + rowBatchSize]
            skipRows = self.skipRows[offset: offset + rowBatchSize]

            rowTensor = torch.tensor(rows, device=query.device, dtype=torch.long)
            queryRows = queryFull.index_select(2, rowTensor)
            logits = torch.matmul(
                queryRows,
                keyFull.transpose(-1, -2),
            ) * float(scaling)

            queryPositions = torch.tensor(
                positions, device=query.device, dtype=torch.long
            )
            causal = keyPositions.unsqueeze(0) <= queryPositions.unsqueeze(1)
            logits = logits.masked_fill(
                ~causal.unsqueeze(0).unsqueeze(0),
                float("-inf"),
            )
            alpha = torch.softmax(logits, dim=-1, dtype=torch.float32)

            # Each independent cache contains only its own prepared chunk.
            # Consequently all full-prompt keys before the occurrence are
            # missing.  Tokens inside the occurrence are independently visible.
            missing = keyPositions.unsqueeze(0) < torch.tensor(
                docStarts, device=query.device, dtype=torch.long
            ).unsqueeze(1)
            missing = missing & causal
            missing = missing.unsqueeze(0).unsqueeze(0)

            missingMass = (alpha * missing).sum(dim=-1)
            valueNumerator = (
                alpha * valueNorm * missing
            ).sum(dim=-1)
            valueDenominator = (alpha * valueNorm).sum(dim=-1)
            valueDependency = torch.where(
                valueDenominator > 1e-12,
                valueNumerator / valueDenominator,
                torch.zeros_like(valueNumerator),
            ).clamp_(0.0, 1.0)

            heads = alpha.shape[1]
            self.missingAttentionSum += float(missingMass.sum().item())
            self.valueDependencySum += float(valueDependency.sum().item())
            self.headCount += len(rows) * heads

            skipMask = torch.tensor(
                skipRows, device=query.device, dtype=torch.bool
            )
            if bool(skipMask.any()):
                self.missingAttentionSkipSum += float(
                    missingMass[:, :, skipMask].sum().item()
                )
                self.valueDependencySkipSum += float(
                    valueDependency[:, :, skipMask].sum().item()
                )
                self.skipHeadCount += int(skipMask.sum().item()) * heads

    def Values(self) -> Dict[str, Optional[float]]:
        if self.headCount == 0:
            return {
                "missing_attention": None,
                "value_weighted_dependency": None,
                "missing_attention_skip_k": None,
                "value_weighted_dependency_skip_k": None,
            }

        return {
            "missing_attention": self.missingAttentionSum / self.headCount,
            "value_weighted_dependency": self.valueDependencySum / self.headCount,
            "missing_attention_skip_k": (
                self.missingAttentionSkipSum / self.skipHeadCount
                if self.skipHeadCount
                else None
            ),
            "value_weighted_dependency_skip_k": (
                self.valueDependencySkipSum / self.skipHeadCount
                if self.skipHeadCount
                else None
            ),
        }


@contextmanager
def _CaptureQwenAttention(model, stats: _DependencyStats) -> Iterator[None]:
    """Install a temporary Qwen3 attention interface for selected rows."""
    qwenModule = importlib.import_module(
        "transformers.models.qwen3.modeling_qwen3"
    )
    attentionFunctions = getattr(qwenModule, "ALL_ATTENTION_FUNCTIONS", None)
    eager = getattr(qwenModule, "eager_attention_forward", None)
    if attentionFunctions is None or eager is None:
        raise RuntimeError(
            "DependencyAnalysisMethod requires the Qwen3 attention interface"
        )

    baseModel = getattr(model, "model", model)
    config = baseModel.config
    originalImplementation = getattr(config, "_attn_implementation", None)
    originalAttention = attentionFunctions.get_interface(
        originalImplementation,
        eager,
    )
    captureKey = f"kvbench_dependency_{id(stats)}"

    def captureAttention(
        module,
        query,
        key,
        value,
        attentionMask,
        scaling,
        dropout=0.0,
        **kwargs,
    ):
        capture = kwargs.pop("dependency_capture", None)
        output = originalAttention(
            module,
            query,
            key,
            value,
            attentionMask,
            scaling=scaling,
            dropout=dropout,
            **kwargs,
        )
        if capture is not None:
            capture.Record(
                __import__("torch"),
                qwenModule,
                module,
                query,
                key,
                value,
                scaling,
            )
        return output

    previous = attentionFunctions._global_mapping.get(captureKey)
    attentionFunctions.register(captureKey, captureAttention)
    config._attn_implementation = captureKey
    try:
        yield
    finally:
        config._attn_implementation = originalImplementation
        if previous is None:
            attentionFunctions._global_mapping.pop(captureKey, None)
        else:
            attentionFunctions._global_mapping[captureKey] = previous


class DependencyAnalysisMethod(FullPrefillTransformer):
    """FullPrefill generation plus RAG document dependency measurements."""

    name = "dependency_analysis"
    method_metrics = _MetricNames
    maxCaseBatchSize = 1

    def __init__(
        self,
        gpuNums: int = 1,
        perfWeight: float = 1.0,
        *,
        maxNewTokens: int = 64,
        dtype: str = "bfloat16",
        skipK: int = 16,
        tag: Optional[str] = None,
    ):
        super().__init__(
            gpuNums=gpuNums,
            perfWeight=perfWeight,
            maxNewTokens=maxNewTokens,
            dtype=dtype,
            tag=tag,
        )
        if isinstance(skipK, bool) or not isinstance(skipK, int):
            raise TypeError("skipK must be an integer")
        if skipK < 0:
            raise ValueError("skipK must not be negative")
        self.skipK = skipK
        self._chunks: List[List[str]] = []

    def Prepare(self, data: List[List[str]]) -> None:
        self._chunks = [list(chunks or []) for chunks in data]

    def Run(
        self,
        data: List[str],
        retainOutput: Optional[List[bool]] = None,
    ) -> List[Result]:
        if len(self._chunks) != len(data):
            self._chunks = [[] for _ in data]

        results: List[Result] = []
        for index, prompt in enumerate(data):
            ids = self._gen.Encode(prompt)
            chunks = self._chunks[index]
            start = time.perf_counter()

            # Do not keep the returned tuple alive: it also owns a reference
            # to the full input KV cache.  The cache is needed for
            # ``kv_deviation`` but must not overlap with the second, full
            # prefill used for attention dependency measurements.
            text, ttft, _, nTokens, fullCache, _ = self._gen.Generate(
                ids,
                returnCache=True,
            )
            # Pass ownership through a mutable holder so _Analyze can clear
            # the only caller-side reference before the attention prefill.
            fullCacheHolder = [fullCache]
            del fullCache

            metadata = self._Analyze(
                prompt,
                ids,
                chunks,
                fullCacheHolder,
            )
            total = time.perf_counter() - start
            metadata.update({
                "n_input": len(ids),
            })

            results.append(
                self._Result(
                    text,
                    ttft,
                    total,
                    nTokens,
                    metadata=metadata,
                )
            )

        return results

    def Reset(self) -> None:
        self._chunks = []

    def _Analyze(
        self,
        prompt: str,
        ids: List[int],
        chunks: List[str],
        fullCacheHolder: List[Any],
    ) -> Dict[str, Any]:
        occurrences, tokenInfo = self._Occurrences(prompt, ids, chunks)
        diagnostics = {
            "num_reusable_documents": len(occurrences),
            "num_reusable_tokens": sum(
                len(occurrence.tokenPairs)
                for occurrence in occurrences
            ),
            "num_valid_tokens_after_skip_k": sum(
                sum(
                    1
                    for _, independentIndex in occurrence.tokenPairs
                    if independentIndex >= self.skipK
                )
                for occurrence in occurrences
            ),
        }

        if not occurrences:
            return {
                **diagnostics,
                "missing_attention": None,
                "value_weighted_dependency": None,
                "missing_attention_skip_k": None,
                "value_weighted_dependency_skip_k": None,
                "kv_deviation": None,
            }

        # Generation returns a full-prompt KV cache.  Compute the cache metric
        # first, then release that cache before running another long prefill;
        # retaining both caches is enough to OOM on the longest GovReport
        # sample even though each operation fits by itself.
        kvDeviation = self._KvDeviation(
            occurrences,
            fullCacheHolder[0],
        )
        fullCacheHolder[0] = None

        stats = _DependencyStats(tokenInfo, self.skipK)
        self._CaptureFullAttention(ids, tokenInfo, stats)
        values = stats.Values()
        values["kv_deviation"] = kvDeviation
        return {
            **diagnostics,
            **values,
        }

    def _Occurrences(
        self,
        prompt: str,
        ids: List[int],
        chunks: List[str],
    ) -> Tuple[List[_DocumentOccurrence], Dict[int, Tuple[int, int]]]:
        if not chunks:
            return [], {}

        tokenizer = self._gen.tokenizer
        try:
            encoded = tokenizer(
                prompt,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
        except (TypeError, NotImplementedError) as exc:
            raise RuntimeError(
                "DependencyAnalysisMethod requires a fast tokenizer with "
                "offset mappings"
            ) from exc

        tokenIds = encoded["input_ids"]
        offsets = encoded.get("offset_mapping")
        if tokenIds and isinstance(tokenIds[0], list):
            tokenIds = tokenIds[0]
            if offsets is not None:
                offsets = offsets[0]
        if list(tokenIds) != list(ids):
            raise RuntimeError(
                "tokenizer IDs differ between Encode() and offset mapping"
            )
        if offsets is None:
            raise RuntimeError("tokenizer did not return offset mappings")

        occurrences: List[_DocumentOccurrence] = []
        tokenInfo: Dict[int, Tuple[int, int]] = {}
        usedTokens = set()

        for prepareIndex, text, charStart, charEnd in ComposeInterleavedReuseSpans(
            chunks,
            prompt,
        ):
            if prepareIndex is None or not text:
                continue

            overlapping = []
            for tokenIndex, offset in enumerate(offsets):
                if offset is None or len(offset) != 2:
                    continue
                tokenStart, tokenEnd = offset
                if tokenEnd > charStart and tokenStart < charEnd:
                    overlapping.append((tokenIndex, tokenStart, tokenEnd))

            if not overlapping:
                continue

            tokenStart = overlapping[0][0]
            tokenEnd = overlapping[-1][0] + 1
            if [index for index, _, _ in overlapping] != list(
                range(tokenStart, tokenEnd)
            ):
                raise RuntimeError("reusable chunk token span is not contiguous")
            independentIds = list(
                self._gen.Encode(text, addSpecialTokens=False)
            )

            # The full prompt and the isolated chunk have different tokenizer
            # boundaries.  BPEs can therefore produce e.g. ``'.One'`` in the
            # former and ``'.'``, ``'One'`` in the latter.  Align by exact
            # token ID instead of comparing rows by offset; unmatched edge
            # tokens are intentionally excluded from both measurements.
            fullTokenIndices = [index for index, _, _ in overlapping]
            fullTokenIds = [ids[index] for index in fullTokenIndices]
            tokenPairs = _AlignTokenPairs(
                fullTokenIndices,
                fullTokenIds,
                independentIds,
            )

            # A crossing token can occur in the offset span of both adjacent
            # reusable chunks.  Attribute an exact match only once in the
            # full prompt, avoiding duplicate query rows and KV samples.
            tokenPairs = [
                (fullIndex, independentIndex)
                for fullIndex, independentIndex in tokenPairs
                if fullIndex not in usedTokens
            ]
            usedTokens.update(fullIndex for fullIndex, _ in tokenPairs)

            occurrence = _DocumentOccurrence(
                prepareIndex=prepareIndex,
                text=text,
                charStart=charStart,
                charEnd=charEnd,
                tokenStart=tokenStart,
                tokenEnd=tokenEnd,
                tokenPairs=tuple(tokenPairs),
                independentIds=tuple(independentIds),
            )
            occurrences.append(occurrence)
            if tokenPairs:
                # localIndex is the position in the isolated cache.  This is
                # also the correct coordinate for skipK after edge alignment.
                docStart = tokenPairs[0][0]
                for fullIndex, independentIndex in tokenPairs:
                    tokenInfo[fullIndex] = (docStart, independentIndex)

        return occurrences, tokenInfo

    def _CaptureFullAttention(
        self,
        ids: List[int],
        tokenInfo: Mapping[int, Tuple[int, int]],
        stats: _DependencyStats,
    ) -> None:
        torch = self._gen._torch
        model = self._gen.model
        baseModel = getattr(model, "model", model)
        # SDPA may fall back to a materialized query x key attention tensor.
        # Keep this analysis-only prefill smaller than normal generation so a
        # long prompt cannot turn that temporary tensor into a multi-GB peak.
        configuredChunk = int(
            os.environ.get("KVBENCH_DEPENDENCY_PREFILL_CHUNK", "1024")
        )
        if configuredChunk < 1:
            raise ValueError(
                "KVBENCH_DEPENDENCY_PREFILL_CHUNK must be at least 1"
            )
        chunkSize = min(max(1, int(self._gen.prefillChunk)), configuredChunk)
        past = None

        with torch.no_grad(), _CaptureQwenAttention(model, stats):
            for start in range(0, len(ids), chunkSize):
                end = min(start + chunkSize, len(ids))
                stats.SetQueryRows(start, end - start)
                inputTensor = torch.tensor(
                    [ids[start:end]],
                    device=self._gen.device,
                    dtype=torch.long,
                )
                positionIds = torch.arange(
                    start,
                    end,
                    device=self._gen.device,
                    dtype=torch.long,
                ).unsqueeze(0)
                output = baseModel(
                    input_ids=inputTensor,
                    position_ids=positionIds,
                    past_key_values=past,
                    use_cache=True,
                    dependency_capture=stats,
                )
                past = output.past_key_values

        del past

    def _IndependentCache(
        self,
        ids: List[int],
        positionStart: int,
    ):
        torch = self._gen._torch
        model = self._gen.model
        baseModel = getattr(model, "model", model)
        chunkSize = max(1, int(self._gen.prefillChunk))
        past = None

        with torch.no_grad():
            for start in range(0, len(ids), chunkSize):
                end = min(start + chunkSize, len(ids))
                inputTensor = torch.tensor(
                    [ids[start:end]],
                    device=self._gen.device,
                    dtype=torch.long,
                )
                positionIds = torch.arange(
                    positionStart + start,
                    positionStart + end,
                    device=self._gen.device,
                    dtype=torch.long,
                ).unsqueeze(0)
                output = baseModel(
                    input_ids=inputTensor,
                    position_ids=positionIds,
                    past_key_values=past,
                    use_cache=True,
                )
                past = output.past_key_values

        return past

    def _KvDeviation(
        self,
        occurrences: Sequence[_DocumentOccurrence],
        fullCache,
    ) -> Optional[float]:
        if fullCache is None:
            raise RuntimeError("FullPrefill did not return an input KV cache")

        torch = self._gen._torch
        functional = torch.nn.functional
        fullPairs = list(CacheLayerPairs(fullCache))
        total = 0.0
        count = 0

        for occurrence in occurrences:
            if not occurrence.tokenPairs:
                continue

            ids = list(occurrence.independentIds)
            independentCache = self._IndependentCache(
                ids,
                # If the first isolated token only matches a later full
                # token, shift its RoPE origin so every comparable token has
                # its full-prompt absolute position.
                occurrence.tokenPairs[0][0] - occurrence.tokenPairs[0][1],
            )
            independentPairs = list(CacheLayerPairs(independentCache))
            if len(independentPairs) != len(fullPairs):
                raise RuntimeError("full and independent cache layer counts differ")

            for (fullK, fullV), (independentK, independentV) in zip(
                fullPairs,
                independentPairs,
            ):
                fullIndices = torch.tensor(
                    [fullIndex for fullIndex, _ in occurrence.tokenPairs],
                    device=fullK.device,
                    dtype=torch.long,
                )
                independentIndices = torch.tensor(
                    [
                        independentIndex
                        for _, independentIndex in occurrence.tokenPairs
                    ],
                    device=independentK.device,
                    dtype=torch.long,
                )
                fullK = fullK.index_select(2, fullIndices)
                fullV = fullV.index_select(2, fullIndices)
                independentK = independentK.index_select(2, independentIndices)
                independentV = independentV.index_select(2, independentIndices)
                if fullK.shape != independentK.shape or fullV.shape != independentV.shape:
                    raise RuntimeError(
                        "full and independent KV token shapes differ"
                    )

                deltaK = (
                    1.0
                    - functional.cosine_similarity(
                        fullK.float(),
                        independentK.float(),
                        dim=-1,
                        eps=1e-8,
                    )
                ).clamp_min_(0.0)
                deltaV = (
                    1.0
                    - functional.cosine_similarity(
                        fullV.float(),
                        independentV.float(),
                        dim=-1,
                        eps=1e-8,
                    )
                ).clamp_min_(0.0)
                total += float(((deltaK + deltaV) * 0.5).sum().item())
                count += int(deltaK.numel())

            del independentCache

        return total / count if count else None


# Keep both names available without requiring a methods/__init__.py change.
DependencyAnalysisTransformer = DependencyAnalysisMethod

__all__ = [
    "DependencyAnalysisMethod",
    "DependencyAnalysisTransformer",
]
