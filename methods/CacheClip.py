"""CacheClip method with CPU selection and sparse primary-cache repair.

``Prepare`` is the offline document-KV build phase. It prepares independent
primary and auxiliary document caches without contributing to online timing.
``Run`` matches prepared documents, selects candidate windows with the CPU
auxiliary model, assembles the primary GPU cache, selectively recomputes the
selected tokens, and generates from the repaired cache. The CPU selection and
GPU primary-cache assembly overlap in the two futures submitted by ``Run``.

CacheClip uses shared-prefix token mapping and density-based window grouping
before sparse recomputation. Both contiguous and interleaved cached documents
are supported; full-prompt generation is used only when no reusable match can
be formed or the prompt has no valid generation boundary.

The method intentionally exposes only KVBench's unified ``Result.performance``
fields: ``ttft``, ``num_output_tokens``, and ``total_time``. No auxiliary
selector metrics are added. ``retainOutput`` remains part of the common Method
signature, but CacheClip does not currently retain generated output KV because
its prepared-state format has no safe output-segment registration path.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from time import perf_counter
from typing import Any, List, Optional, Sequence

import torch

from core.Config import ModelPath as DefaultModelPath
from core.Method import Method
from core.Result import NumOutputTokensKey, Result, TotalTimeKey, TtftKey
from core.Sampling import ResolveSamplingConfig
from helpers.backends.Prompt import ComposeInterleavedReuse
from helpers.backends.TransformersHelper import TransformersGenerator

from .CacheClipBackend import (
    AuxiliaryRuntime,
    PrimaryAssembly,
    append_cached_primary_segment,
    assemble_primary_cache,
    continuation_offsets_from_full_sequence,
    group_candidate_windows,
    map_auxiliary_to_primary_indices,
    recompute_selected_tokens,
    select_top_k_candidates,
    select_cache_sequence,
    shared_prefix_token_length,
    tokenize_with_offsets,
)


def recompute_diagnostics(selected_indices: Sequence[int], context_tokens: int) -> dict[str, Any]:
    """Return per-run selection counts without promoting them to metrics."""
    selected_count = len(set(int(index) for index in selected_indices))
    denominator = max(0, int(context_tokens))
    return {
        "n_recomputed_tokens": selected_count,
        "n_reusable_context_tokens": denominator,
        "actual_recompute_ratio": (
            selected_count / denominator if denominator else 0.0
        ),
    }


class CacheClip(Method):
    """CacheClip over Transformers with a CPU SmolLM2-135M selector.

    The primary model runs on the configured GPU and the auxiliary selector
    runs on CPU. Documents are cached independently during ``Prepare``; an
    online query reuses matching documents, maps auxiliary windows to primary
    token positions, repairs only selected primary tokens, and then generates.
    """

    name = "cacheclip"
    backend = "transformers"
    maxCaseBatchSize = 1
    # Aggregate the realized post-grouping recomputation ratio so experiment
    # runners can verify the configured target without exposing selector
    # timing internals as public KVBench metrics.
    method_metrics = ("actual_recompute_ratio",)

    def __init__(
        self,
        gpuNums: int = 1,
        perfWeight: float = 1.0,
        *,
        maxNewTokens: int = 64,
        dtype: str = "bfloat16",
        candidateRatio: float = 0.2,
        windowSize: int = 8,
        densityThreshold: int = 5,
        auxiliaryModelPath: Optional[str] = None,
        auxiliaryThreads: int = 12,
        requireReuse: bool = False,
        sharedPrefixText: Optional[str] = None,
        tag: Optional[str] = None,
    ):
        super().__init__(gpuNums=gpuNums, perfWeight=perfWeight, maxGpuNums=1, tag=tag)
        if not 0 <= float(candidateRatio) <= 1:
            raise ValueError("candidateRatio must be in [0, 1]")
        if windowSize <= 0 or densityThreshold <= 0:
            raise ValueError("windowSize and densityThreshold must be positive")
        self.modelPath = DefaultModelPath()
        self.samplingConfig = ResolveSamplingConfig(self.modelPath)
        self.maxNewTokens = maxNewTokens
        self.dtype = dtype
        self.candidateRatio = float(candidateRatio)
        self.windowSize = windowSize
        self.densityThreshold = densityThreshold
        self.auxiliaryModelPath = auxiliaryModelPath or os.environ.get(
            "KVBENCH_CACHECLIP_AUX_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct"
        )
        self.auxiliaryThreads = auxiliaryThreads
        # Experiment runners can fail closed instead of silently turning a
        # selective-reuse case into a full-prefill case.  The default keeps
        # existing callers' behavior unchanged.
        self.requireReuse = bool(requireReuse)
        self.sharedPrefixText = sharedPrefixText
        self._gen: Optional[TransformersGenerator] = None
        self._auxiliary: Optional[AuxiliaryRuntime] = None
        self._states: list[dict[str, Any]] = []

    def Initialize(self, gpuIds: Sequence[int]) -> None:
        """Load the primary GPU generator and CPU auxiliary selector."""
        super().Initialize(gpuIds)
        self._gen = TransformersGenerator(
            self.modelPath,
            self.gpuIds,
            maxNewTokens=self.maxNewTokens,
            dtype=self.dtype,
            samplingConfig=self.samplingConfig,
        )
        self._auxiliary = AuxiliaryRuntime.load(
            self.auxiliaryModelPath, self.auxiliaryThreads
        )

    def Prepare(self, data: List[List[str]]) -> None:
        """Build document KV caches offline without recording performance time.

        Each document receives an independent primary cache and an auxiliary
        cache. A shared character prefix is converted to a token-safe prefix
        length so later cache assembly never includes a partial boundary token.
        """
        if self._gen is None or self._auxiliary is None:
            raise RuntimeError("CacheClip must be initialized before Prepare")
        self._states = []
        for chunks in data:
            documents = list(chunks or [])
            if not documents:
                self._states.append({"chunks": [], "primary": [], "auxiliary": [], "prefix_length": 0})
                continue
            shared_prefix_text = getattr(self, "sharedPrefixText", None)
            if shared_prefix_text is not None:
                prefix = str(shared_prefix_text)
                if any(not chunk.startswith(prefix) for chunk in documents):
                    raise ValueError(
                        "configured CacheClip sharedPrefixText must prefix every document"
                    )
                if any(len(chunk) <= len(prefix) for chunk in documents):
                    raise ValueError(
                        "configured CacheClip sharedPrefixText must leave content in every document"
                    )
            else:
                prefix = self._common_prefix(documents)
            primary_caches = []
            primary_ids = []
            primary_offsets = []
            full_ids = []
            full_offsets = []
            for chunk in documents:
                ids, offsets = tokenize_with_offsets(self._gen.tokenizer, chunk)
                full_ids.append(ids[0].tolist())
                full_offsets.append(offsets)
                primary_ids.append(ids)
                primary_offsets.append(offsets)
                primary_caches.append(self._gen.Prefill(ids[0].tolist()).past_key_values)
            prefix_length = shared_prefix_token_length(
                full_ids, full_offsets, len(prefix)
            ) if prefix else 0
            continuation_primary_offsets = [
                continuation_offsets_from_full_sequence(offsets, prefix_length, len(prefix))
                for offsets in full_offsets
            ]
            auxiliary_caches = self._auxiliary.prefill_documents(
                prefix, [chunk[len(prefix):] for chunk in documents]
            )
            self._states.append({
                "chunks": documents,
                "primary": primary_caches,
                "primary_ids": primary_ids,
                "primary_offsets": continuation_primary_offsets,
                "auxiliary": auxiliary_caches,
                "prefix_length": prefix_length,
                "prefix": prefix,
            })

    def Run(self, data: List[str], retainOutput: Optional[List[bool]] = None) -> List[Result]:
        """Run online CacheClip and return only KVBench's unified metrics.

        The CPU auxiliary selector and GPU primary-cache assembly are submitted
        as separate futures and overlap. After both complete, selected tokens
        undergo sparse model-specific recomputation before generation. The
        common ``retainOutput`` hint is accepted for Method compatibility but
        is intentionally ignored: this implementation cannot safely register
        generated output KV as another prepared segment yet.
        """
        if self._gen is None or self._auxiliary is None:
            raise RuntimeError("CacheClip must be initialized before Run")
        _ = retainOutput
        if len(self._states) != len(data):
            self._states = [{"chunks": [], "primary": [], "auxiliary": [], "prefix_length": 0} for _ in data]
        results = []
        for index, prompt in enumerate(data):
            state = self._states[index]
            result = self._run_one(state, prompt)
            results.append(result)
        return results

    def _run_one(self, state: dict[str, Any], prompt: str) -> Result:
        chunks = state.get("chunks", [])
        parts = ComposeInterleavedReuse(chunks, prompt)
        matched_positions = [position for position, (chunk_index, _) in enumerate(parts) if chunk_index is not None]
        if not matched_positions:
            return self._full_fallback(prompt)
        if not self._is_contiguous_match(parts, matched_positions):
            return self._run_interleaved(state, prompt, parts, matched_positions)
        return self._run_contiguous(state, prompt, parts, matched_positions)

    def _run_contiguous(
        self,
        state: dict[str, Any],
        prompt: str,
        parts,
        matched_positions: list[int],
    ) -> Result:
        """Run the original fast path when all cached parts are adjacent."""
        first_match, last_match = matched_positions[0], matched_positions[-1]
        prefix_text = "".join(text for chunk_index, text in parts[:first_match] if chunk_index is None)
        query_text = "".join(text for chunk_index, text in parts[last_match + 1:] if chunk_index is None)
        if not query_text:
            return self._full_fallback(prompt)
        ordered_indices = [parts[position][0] for position in matched_positions]
        primary_indices = [state["primary"][chunk_index] for chunk_index in ordered_indices]
        primary_ids = [state["primary_ids"][chunk_index] for chunk_index in ordered_indices]
        auxiliary_caches = [state["auxiliary"][chunk_index] for chunk_index in ordered_indices]
        prefix_cache = None
        prefix_ids = torch.empty((1, 0), dtype=torch.long, device=self._gen.model.device)
        online_start = perf_counter()
        if prefix_text:
            prefix_ids = torch.tensor(
                [self._gen.Encode(prefix_text, addSpecialTokens=False)],
                dtype=torch.long,
                device=self._gen.model.device,
            )
            prefix_cache = self._gen.Prefill(prefix_ids[0].tolist()).past_key_values
        shared_prefix_length_value = state["prefix_length"]
        shared_prefix_cache = None
        shared_prefix_ids = torch.empty((1, 0), dtype=torch.long, device=self._gen.model.device)
        if shared_prefix_length_value:
            shared_prefix_cache = select_cache_sequence(
                primary_indices[0], 0, shared_prefix_length_value
            )
            # ``primary_ids`` are tokenization outputs kept on CPU during
            # Prepare; assembly concatenates them with any online prefix ids
            # on the model device.  Move the shared-prefix slice explicitly so
            # multi-document tasks (e.g. MuSiQue) do not mix CPU/CUDA tensors.
            shared_prefix_ids = primary_ids[0][:, :shared_prefix_length_value].to(
                self._gen.model.device
            )
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="cacheclip") as executor:
            selection_future = executor.submit(
                self._select,
                query_text,
                auxiliary_caches,
                state,
                ordered_indices,
                prefix_text,
            )
            primary_future = executor.submit(
                assemble_primary_cache,
                self._gen.model,
                prefix_cache,
                prefix_ids,
                shared_prefix_cache,
                shared_prefix_ids,
                shared_prefix_length_value,
                primary_indices,
                primary_ids,
            )
            selected_indices = selection_future.result()
            assembly = primary_future.result()
        recompute_selected_tokens(self._gen.model, assembly, selected_indices)
        query_ids = self._gen.Encode(query_text, addSpecialTokens=False)
        if not query_ids:
            return self._full_fallback(prompt)
        generation_start = perf_counter()
        text, generation_ttft, generation_total, token_count = self._gen.Generate(
            query_ids, pastKeyValues=assembly.cache
        )
        preparation_time = generation_start - online_start
        return self._result(
            text,
            preparation_time + generation_ttft,
            preparation_time + generation_total,
            token_count,
            len(assembly.context_ids) + len(query_ids),
            recompute_diagnostics(selected_indices, assembly.context_ids.size(1)),
        )

    def _run_interleaved(
        self,
        state: dict[str, Any],
        prompt: str,
        parts,
        matched_positions: list[int],
    ) -> Result:
        """Run a prompt containing cached documents separated by fresh spans.

        Fresh spans before or between cached chunks are prefetched online with
        the already assembled cache as ``past_key_values``.  Cached chunks are
        appended after RoPE re-alignment.  The final fresh suffix remains the
        generation query; when the prompt ends in a cached chunk, its last
        token is replayed from a cache truncated by one token so generation
        still has a correct next-token boundary.
        """
        last_match = matched_positions[-1]
        query_text = "".join(
            text for chunk_index, text in parts[last_match + 1:]
            if chunk_index is None
        )
        # If the prompt ends in a cached part, use that part as the selector
        # query.  It normally contains the question/assistant boundary (e.g.
        # VT/CWE shuffle tasks), while the last-token replay below provides the
        # generation boundary.
        selection_text = "".join(
            text for chunk_index, text in parts if chunk_index is None
        ) or parts[last_match][1]
        if not selection_text:
            selection_text = prompt
        device = self._gen.model.device
        assembly: PrimaryAssembly | None = None
        occurrences: list[dict[str, Any]] = []
        online_start = perf_counter()

        for part_index, (chunk_index, text) in enumerate(parts[: last_match + 1]):
            if chunk_index is None:
                ids = self._gen.Encode(text, addSpecialTokens=False)
                if not ids:
                    continue
                ids_tensor = torch.tensor([ids], dtype=torch.long, device=device)
                if assembly is None:
                    output = self._gen.Prefill(ids)
                    assembly = PrimaryAssembly(
                        cache=output.past_key_values,
                        context_ids=ids_tensor,
                    )
                else:
                    output = self._gen.Prefill(ids, pastKeyValues=assembly.cache)
                    assembly = PrimaryAssembly(
                        cache=output.past_key_values,
                        context_ids=torch.cat([assembly.context_ids, ids_tensor], dim=1),
                    )
                continue

            ids = state["primary_ids"][chunk_index].to(device)
            if ids.numel() == 0:
                continue
            cache = state["primary"][chunk_index]
            global_start = assembly.context_ids.size(1) if assembly is not None else 0
            if assembly is None:
                empty = torch.empty((1, 0), dtype=torch.long, device=device)
                assembly = assemble_primary_cache(
                    self._gen.model,
                    None,
                    empty,
                    None,
                    empty,
                    0,
                    [cache],
                    [ids],
                )
            else:
                assembly = append_cached_primary_segment(
                    self._gen.model,
                    assembly,
                    cache,
                    ids,
                )
            occurrences.append({
                "chunk_index": chunk_index,
                "global_start": global_start,
                "auxiliary": state["auxiliary"][chunk_index],
                "primary_offsets": state["primary_offsets"][chunk_index],
                "prefix_length": int(state.get("prefix_length", 0)),
            })

        if assembly is None or not occurrences:
            return self._full_fallback(prompt)

        selected_indices = self._select_occurrences(selection_text, occurrences)
        recompute_selected_tokens(self._gen.model, assembly, selected_indices)

        if query_text:
            query_ids = self._gen.Encode(query_text, addSpecialTokens=False)
            if not query_ids:
                return self._full_fallback(prompt)
            generation_cache = assembly.cache
            input_count = assembly.context_ids.size(1) + len(query_ids)
        else:
            # All prompt text is already represented by the assembled cache.
            # Replay the final token from a cache shortened by one position to
            # obtain the next-token logits without duplicating that token.
            if assembly.context_ids.size(1) == 0:
                return self._full_fallback(prompt)
            query_ids = [int(assembly.context_ids[0, -1].item())]
            generation_cache = select_cache_sequence(assembly.cache, 0, -1)
            input_count = assembly.context_ids.size(1)

        generation_start = perf_counter()
        text, generation_ttft, generation_total, token_count = self._gen.Generate(
            query_ids,
            pastKeyValues=generation_cache,
        )
        preparation_time = generation_start - online_start
        return self._result(
            text,
            preparation_time + generation_ttft,
            preparation_time + generation_total,
            token_count,
            input_count,
            recompute_diagnostics(selected_indices, assembly.context_ids.size(1)),
        )

    def _select(self, query: str, auxiliary_caches, state, ordered_indices, prefix_text: str = "") -> list[int]:
        """Select auxiliary windows and map them to primary cache positions."""
        scores = self._auxiliary.score_documents(query, auxiliary_caches)
        candidates = select_top_k_candidates(scores, self.candidateRatio)
        grouped = group_candidate_windows(
            candidates,
            [len(cache.continuation_offsets) for cache in auxiliary_caches],
            self.windowSize,
            self.densityThreshold,
        )
        selected = []
        cursor = 0
        prefix_length = state["prefix_length"]
        for local_index, (cache, windows, primary_offsets) in enumerate(
            zip(auxiliary_caches, grouped, [state["primary_offsets"][index] for index in ordered_indices])
        ):
            mapped = map_auxiliary_to_primary_indices(
                cache.continuation_offsets, primary_offsets, windows
            )
            selected.extend(prefix_length + cursor + item for item in mapped)
            cursor += len(primary_offsets)
        prefix_offset = len(self._gen.Encode(prefix_text, addSpecialTokens=False)) if prefix_text else 0
        return sorted(set(prefix_offset + item for item in selected))

    def _select_occurrences(
        self,
        query: str,
        occurrences: list[dict[str, Any]],
    ) -> list[int]:
        """Select and map tokens for cached occurrences in an interleaved run."""
        scores = self._auxiliary.score_documents(
            query,
            [item["auxiliary"] for item in occurrences],
        )
        candidates = select_top_k_candidates(scores, self.candidateRatio)
        grouped = group_candidate_windows(
            candidates,
            [len(item["auxiliary"].continuation_offsets) for item in occurrences],
            self.windowSize,
            self.densityThreshold,
        )
        # ``primary_offsets`` and auxiliary continuation offsets omit the
        # common prefix computed during Prepare, while the interleaved path
        # assembles each cached chunk in full.  Restore that local offset when
        # mapping selected continuation tokens to global positions.
        prefix_length = int(occurrences[0].get("prefix_length", 0))
        selected: list[int] = []
        for occurrence, windows in zip(occurrences, grouped):
            mapped = map_auxiliary_to_primary_indices(
                occurrence["auxiliary"].continuation_offsets,
                occurrence["primary_offsets"],
                windows,
            )
            selected.extend(
                occurrence["global_start"] + prefix_length + item
                for item in mapped
            )
        return sorted(set(selected))

    @staticmethod
    def _common_prefix(chunks: list[str]) -> str:
        if len(chunks) < 2:
            return ""
        prefix = chunks[0]
        for chunk in chunks[1:]:
            limit = min(len(prefix), len(chunk))
            end = 0
            while end < limit and prefix[end] == chunk[end]:
                end += 1
            prefix = prefix[:end]
        return "" if len(prefix) >= min(len(chunk) for chunk in chunks) else prefix

    @staticmethod
    def _is_contiguous_match(parts, matched_positions) -> bool:
        first, last = matched_positions[0], matched_positions[-1]
        return all(parts[position][0] is not None for position in range(first, last + 1))

    def _full_fallback(self, prompt: str) -> Result:
        """Generate the full prompt only when reuse cannot be formed."""
        if self.requireReuse:
            raise RuntimeError(
                "CacheClip strict reuse is enabled, but this prompt could not "
                "be served through cached selective recomputation"
            )
        ids = self._gen.Encode(prompt)
        text, ttft, total, token_count = self._gen.Generate(ids)
        return self._result(text, ttft, total, token_count, len(ids))

    @staticmethod
    def _result(
        text: str,
        ttft: float,
        total: float,
        token_count: int,
        input_count: int,
        diagnostics: Optional[dict[str, Any]] = None,
    ) -> Result:
        """Build a Result without adding CacheClip-specific metrics."""
        metadata = {"backend": "transformers", "n_input": input_count}
        if diagnostics:
            metadata.update(diagnostics)
        return Result(
            output=text,
            performance={
                TtftKey: float(ttft),
                NumOutputTokensKey: int(token_count),
                TotalTimeKey: float(total),
            },
            metadata=metadata,
        )

    def Reset(self) -> None:
        """Discard prepared document state while keeping loaded models."""
        self._states = []

    def Close(self) -> None:
        """Release prepared state and both model runtimes."""
        self._states = []
        self._auxiliary = None
        self._gen = None
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


__all__ = ["CacheClip"]
