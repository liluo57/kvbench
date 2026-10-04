"""CacheClip over authoritative primary-model token IDs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from time import perf_counter
from typing import Any, List, Optional, Sequence

import torch

from core.Config import ModelPath as DefaultModelPath
from core.Method import Method, ResolveMaxNewTokens
from core.Result import NumOutputTokensKey, Result, TotalTimeKey, TtftKey
from core.Sampling import ResolveSamplingConfig
from helpers.backends.Prompt import ComposeInterleavedReuse
from helpers.backends.TransformersHelper import TransformersGenerator

from .Cache import (
    PrimaryAssembly,
    append_cached_primary_segment,
    assemble_primary_cache,
    select_cache_sequence,
)
from .Recompute import recompute_selected_tokens
from .Selector import (
    AuxiliaryRuntime,
    group_candidate_windows,
    map_auxiliary_to_primary_indices,
    select_top_k_candidates,
    token_text_and_offsets,
)


def recompute_diagnostics(selected_indices: Sequence[int], cached_tokens: int) -> dict[str, Any]:
    selected_count = len(set(int(index) for index in selected_indices))
    denominator = max(0, int(cached_tokens))
    return {
        "n_recomputed_tokens": selected_count,
        "n_cached_tokens": denominator,
        "recompute_ratio": selected_count / denominator if denominator else 0.0,
    }


class CacheClip(Method):
    """CPU auxiliary selection with sparse repair of primary-model KV rows."""

    name = "cacheclip"
    backend = "transformers"
    maxCaseBatchSize = 1
    method_metrics = ("reuse_ratio", "recompute_ratio", "fallback_rate")

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
        self.requireReuse = bool(requireReuse)
        self.sharedPrefixText = sharedPrefixText
        self._gen: Optional[TransformersGenerator] = None
        self._auxiliary: Optional[AuxiliaryRuntime] = None
        self._states: list[dict[str, Any]] = []

    def Initialize(self, gpuIds: Sequence[int]) -> None:
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

    def Prepare(self, data: List[List[List[int]]]) -> None:
        """Cache each authoritative document ID sequence, excluding warmup P."""
        if self._gen is None or self._auxiliary is None:
            raise RuntimeError("CacheClip must be initialized before Prepare")

        prefix_text = self.sharedPrefixText or ""
        prefix_ids = self._gen.Encode(prefix_text, addSpecialTokens=False) if prefix_text else []
        old_start = len(prefix_ids)
        states = []
        for chunks in data:
            document_ids = [list(ids) for ids in chunks]
            primary_caches: list[Any] = []
            primary_id_tensors: list[torch.Tensor] = []
            primary_offsets: list[list[tuple[int, int]]] = []
            document_texts: list[str] = []
            nonempty_indices: list[int] = []
            for index, ids in enumerate(document_ids):
                text, offsets = token_text_and_offsets(self._gen.tokenizer, ids)
                document_texts.append(text)
                primary_offsets.append(offsets)
                primary_id_tensors.append(torch.tensor([ids], dtype=torch.long))
                if not ids:
                    primary_caches.append(None)
                    continue
                prefill_ids = prefix_ids + ids
                full_cache = self._gen.Prefill(prefill_ids).past_key_values
                if old_start:
                    full_cache = select_cache_sequence(
                        full_cache, old_start, old_start + len(ids)
                    )
                primary_caches.append(full_cache)
                nonempty_indices.append(index)

            auxiliary_caches: list[Any] = [None] * len(document_ids)
            if nonempty_indices:
                built = self._auxiliary.prefill_documents(
                    prefix_text, [document_texts[index] for index in nonempty_indices]
                )
                for index, cache in zip(nonempty_indices, built):
                    auxiliary_caches[index] = cache
            states.append({
                "document_ids": document_ids,
                "primary_id_tensors": primary_id_tensors,
                "primary": primary_caches,
                "primary_offsets": primary_offsets,
                "auxiliary": auxiliary_caches,
                "old_start": old_start,
            })
        self._states = states

    def Run(
        self,
        data: List[List[int]],
        retainOutput: Optional[List[bool]] = None,
        maxNewTokens: Optional[int] = None,
    ) -> List[Result]:
        """Run over token IDs without decoding/re-encoding the primary prompt."""
        run_start = perf_counter()
        if self._gen is None or self._auxiliary is None:
            raise RuntimeError("CacheClip must be initialized before Run")
        max_new_tokens = ResolveMaxNewTokens(
            maxNewTokens if maxNewTokens is not None else getattr(self, "maxNewTokens", None)
        )
        _ = retainOutput
        if not self._states:
            states = [{"document_ids": [], "primary_id_tensors": [], "primary": [], "primary_offsets": [],
                       "auxiliary": [], "old_start": 0} for _ in data]
        elif len(self._states) != len(data):
            raise RuntimeError(
                f"CacheClip state has {len(self._states)} case(s), received {len(data)} prompt(s)"
            )
        else:
            states = self._states
        return [
            self._run_one(state, list(prompt), run_start, max_new_tokens)
            for state, prompt in zip(states, data)
        ]

    def _run_one(
        self, state: dict[str, Any], prompt_ids: list[int], run_start: float,
        max_new_tokens: int,
    ) -> Result:
        parts = ComposeInterleavedReuse(state["document_ids"], prompt_ids)
        matches = [i for i, (chunk_index, _) in enumerate(parts) if chunk_index is not None]
        if not matches:
            if self.requireReuse:
                raise RuntimeError("CacheClip strict reuse found no matching prepared token IDs")
            return self._full_prefill(prompt_ids, run_start, max_new_tokens)
        if self._is_contiguous_match(parts, matches):
            return self._run_contiguous(state, prompt_ids, parts, matches, run_start, max_new_tokens)
        return self._run_interleaved(state, prompt_ids, parts, matches, run_start, max_new_tokens)

    @staticmethod
    def _is_contiguous_match(parts, matches) -> bool:
        first, last = matches[0], matches[-1]
        return all(parts[pos][0] is not None for pos in range(first, last + 1))

    @staticmethod
    def _join_ids(parts) -> list[int]:
        return [token for _, ids in parts for token in ids]

    def _run_contiguous(self, state, prompt_ids, parts, matches, run_start, max_new_tokens):
        first, last = matches[0], matches[-1]
        fresh_prefix = self._join_ids(parts[:first])
        query_ids = self._join_ids(parts[last + 1:])
        ordered_indices = [parts[pos][0] for pos in matches]
        document_caches = [state["primary"][index] for index in ordered_indices]
        document_ids = [state["document_ids"][index] for index in ordered_indices]
        document_tensors = [state["primary_id_tensors"][index] for index in ordered_indices]
        auxiliary_caches = [state["auxiliary"][index] for index in ordered_indices]
        prefix_tensor = torch.tensor([fresh_prefix], dtype=torch.long, device=self._gen.model.device)
        prefix_cache = self._gen.Prefill(fresh_prefix).past_key_values if fresh_prefix else None
        query_for_selection = query_ids or document_ids[-1]
        query_text = self._gen.tokenizer.decode(
            query_for_selection, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="cacheclip") as executor:
            selector_future = executor.submit(
                self._select, query_text, auxiliary_caches,
                [state["primary_offsets"][index] for index in ordered_indices],
                len(fresh_prefix),
            )
            assembly_future = executor.submit(
                assemble_primary_cache,
                self._gen.model,
                prefix_cache,
                prefix_tensor,
                document_caches,
                document_tensors,
                old_start=state["old_start"],
            )
            selected = selector_future.result()
            assembly = assembly_future.result()
        return self._generate_from_assembly(
            assembly, prompt_ids, query_ids, selected, run_start,
            max_new_tokens,
            sum(len(state["document_ids"][index]) for index in ordered_indices),
        )

    def _run_interleaved(self, state, prompt_ids, parts, matches, run_start, max_new_tokens):
        last_match = matches[-1]
        query_ids = self._join_ids(parts[last_match + 1:])
        fresh_ids = [ids for chunk_index, ids in parts if chunk_index is None]
        selector_ids = [token for ids in fresh_ids for token in ids]
        if not selector_ids:
            selector_ids = list(parts[last_match][1])
        query_text = self._gen.tokenizer.decode(
            selector_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

        device = self._gen.model.device
        assembly: Optional[PrimaryAssembly] = None
        occurrences: list[dict[str, Any]] = []
        cached_count = 0
        for chunk_index, ids in parts[:last_match + 1]:
            if chunk_index is None:
                if not ids:
                    continue
                ids_tensor = torch.tensor([ids], dtype=torch.long, device=device)
                if assembly is None:
                    cache = self._gen.Prefill(ids).past_key_values
                    assembly = PrimaryAssembly(cache=cache, context_ids=ids_tensor)
                else:
                    cache = self._gen.Prefill(ids, pastKeyValues=assembly.cache).past_key_values
                    assembly = PrimaryAssembly(
                        cache=cache, context_ids=torch.cat([assembly.context_ids, ids_tensor], dim=1)
                    )
                continue

            prepared_ids = state["document_ids"][chunk_index]
            if ids != prepared_ids:
                raise RuntimeError("CacheClip matcher returned IDs different from prepared document")
            if not ids:
                continue
            cache = state["primary"][chunk_index]
            global_start = 0 if assembly is None else assembly.context_ids.size(1)
            if assembly is None:
                empty_ids = torch.empty((1, 0), dtype=torch.long, device=device)
                assembly = assemble_primary_cache(
                    self._gen.model, None, empty_ids, [cache],
                    [state["primary_id_tensors"][chunk_index].to(device)],
                    old_start=state["old_start"],
                )
            else:
                assembly = append_cached_primary_segment(
                    self._gen.model, assembly, cache,
                    torch.tensor([ids], dtype=torch.long, device=device),
                    old_start=state["old_start"],
                )
            occurrences.append({
                "global_start": global_start,
                "auxiliary": state["auxiliary"][chunk_index],
                "primary_offsets": state["primary_offsets"][chunk_index],
                "primary_length": len(ids),
            })
            cached_count += len(ids)

        if assembly is None:
            raise RuntimeError("CacheClip matched documents but assembled no cache")
        selected = self._select_occurrences(query_text, occurrences)
        return self._generate_from_assembly(
            assembly, prompt_ids, query_ids, selected, run_start,
            max_new_tokens, max(0, cached_count),
        )

    def _select(self, query_text, auxiliary_caches, primary_offsets, prefix_count):
        scores = self._auxiliary.score_documents(query_text, auxiliary_caches)
        candidates = select_top_k_candidates(scores, self.candidateRatio)
        grouped = group_candidate_windows(
            candidates,
            [len(cache.continuation_offsets) for cache in auxiliary_caches],
            self.windowSize,
            self.densityThreshold,
        )
        selected = []
        cursor = 0
        for cache, windows, offsets in zip(auxiliary_caches, grouped, primary_offsets):
            mapped = map_auxiliary_to_primary_indices(
                cache.continuation_offsets, offsets, windows
            )
            selected.extend(prefix_count + cursor + index for index in mapped)
            cursor += len(offsets)
        return sorted(set(selected))

    def _select_occurrences(self, query_text, occurrences):
        scores = self._auxiliary.score_documents(
            query_text, [item["auxiliary"] for item in occurrences]
        )
        candidates = select_top_k_candidates(scores, self.candidateRatio)
        grouped = group_candidate_windows(
            candidates,
            [len(item["auxiliary"].continuation_offsets) for item in occurrences],
            self.windowSize,
            self.densityThreshold,
        )
        selected = []
        for occurrence, windows in zip(occurrences, grouped):
            mapped = map_auxiliary_to_primary_indices(
                occurrence["auxiliary"].continuation_offsets,
                occurrence["primary_offsets"], windows,
            )
            selected.extend(occurrence["global_start"] + index for index in mapped)
        return sorted(set(selected))

    def _generate_from_assembly(
        self, assembly, prompt_ids, query_ids, selected, run_start,
        max_new_tokens, cached_count,
    ):
        context_ids = self._ids(assembly.context_ids)
        if context_ids + query_ids != prompt_ids:
            raise RuntimeError("CacheClip cache plus fresh IDs do not equal the input prompt")
        recompute_selected_tokens(self._gen.model, assembly, selected)
        if query_ids:
            generation_cache = assembly.cache
            generation_ids = query_ids
            generation_count = len(context_ids) + len(query_ids)
        else:
            if not context_ids:
                raise RuntimeError("CacheClip cannot replay a terminal ID from an empty cache")
            generation_cache = select_cache_sequence(assembly.cache, 0, -1)
            generation_ids = [prompt_ids[-1]]
            generation_count = len(context_ids)
        generation_start = perf_counter()
        text, backend_ttft, total, token_count, output_ids = self._gen.Generate(
            generation_ids,
            pastKeyValues=generation_cache,
            maxNewTokens=max_new_tokens,
            returnTokenIds=True,
        )
        # A replayed last cached token is still a cache hit, and is counted
        # once among repaired rows even if the selector already chose it.
        repaired = set(selected)
        if not query_ids:
            repaired.add(len(context_ids) - 1)
        diagnostics = recompute_diagnostics(repaired, cached_count)
        diagnostics.update(
            reuse_ratio=cached_count / len(prompt_ids),
            full_prefill=False,
            fallback_rate=0.0,
        )
        online = generation_start - run_start
        return Result(
            output=text,
            output_ids=output_ids,
            performance={
                TtftKey: online + backend_ttft,
                NumOutputTokensKey: token_count,
                TotalTimeKey: online + total,
            },
            metadata={"backend": "transformers", "n_input": generation_count, **diagnostics},
        )

    def _full_prefill(self, prompt_ids: list[int], run_start: float, max_new_tokens: int) -> Result:
        if self.requireReuse:
            raise RuntimeError("CacheClip strict reuse found no matching prepared token IDs")
        generation_start = perf_counter()
        text, backend_ttft, total, token_count, output_ids = self._gen.Generate(
            prompt_ids, maxNewTokens=max_new_tokens, returnTokenIds=True
        )
        online = generation_start - run_start
        return Result(
            output=text,
            output_ids=output_ids,
            performance={
                TtftKey: online + backend_ttft,
                NumOutputTokensKey: token_count,
                TotalTimeKey: online + total,
            },
            metadata={
                "backend": "transformers", "n_input": len(prompt_ids),
                "n_cached_tokens": 0, "n_recomputed_tokens": len(prompt_ids),
                "reuse_ratio": 0.0, "recompute_ratio": 1.0,
                "full_prefill": True, "fallback_rate": 1.0,
                "fallback_reason": "no_matching_prepared_tokens",
            },
        )

    @staticmethod
    def _ids(input_ids: torch.Tensor) -> list[int]:
        return [int(token) for token in input_ids[0].tolist()]

    def Reset(self) -> None:
        self._states = []

    def Close(self) -> None:
        self._states = []
        self._auxiliary = None
        self._gen = None
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


__all__ = ["CacheClip", "recompute_diagnostics"]
