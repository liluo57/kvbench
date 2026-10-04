"""Shared token-ID lifecycle for selective KV-cache recomputation methods."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Optional

from core.Config import MaxModelLen, ModelPath
from core.Method import Method, ResolveMaxNewTokens
from core.Result import Result, TtftKey, TotalTimeKey, NumOutputTokensKey
from core.Sampling import ResolveSamplingConfig
from .Prompt import ComposeInterleavedReuse


@dataclass
class PreparedChunk:
    ids: list[int]
    cache: Any


class CacheRecomputeMethod(Method):
    """Own a model and prepared raw KV; subclasses only choose repair rows."""

    backend = "hf_cache"
    method_metrics = ("reuse_ratio",)
    maxCaseBatchSize = 1
    fullRecompute = False

    def __init__(self, gpuNums=1, perfWeight=1.0, *, maxNewTokens=64,
                 dtype="bfloat16", tag: Optional[str] = None):
        super().__init__(gpuNums=gpuNums, perfWeight=perfWeight, maxGpuNums=1, tag=tag)
        self.modelPath = ModelPath()
        self.samplingConfig = ResolveSamplingConfig(self.modelPath)
        self.maxNewTokens = ResolveMaxNewTokens(maxNewTokens)
        self.dtype = dtype
        self._gen = None
        self._backend = None
        self._states: list[list[PreparedChunk]] = []

    def Initialize(self, gpuIds):
        super().Initialize(gpuIds)
        import torch
        from .TransformersHelper import TransformersGenerator
        from .CacheBackend.HuggingFace import HFBackend

        self._gen = TransformersGenerator(
            self.modelPath, gpuIds=self.gpuIds if torch.cuda.is_available() else None,
            device="cpu", maxNewTokens=self.maxNewTokens, dtype=self.dtype,
            samplingConfig=self.samplingConfig,
        )
        self._backend = HFBackend(self._gen.model)

    def Prepare(self, data: list[list[list[int]]]):
        import torch

        if self._backend is None:
            raise RuntimeError("initialize the recomputation method before Prepare")
        self._states = []
        for chunks in data:
            state = []
            seen = set()
            for chunk in chunks:
                ids = list(chunk)
                if len(ids) > MaxModelLen():
                    raise ValueError(f"Prepared chunk exceeds MaxModelLen={MaxModelLen()}")
                if any(not isinstance(token, int) or isinstance(token, bool) for token in ids):
                    raise TypeError("Prepare requires native integer token IDs")
                if not ids or tuple(ids) in seen or self.fullRecompute:
                    continue
                seen.add(tuple(ids))
                cache = self._backend.prefill_caches(
                    input_ids=torch.tensor([ids], dtype=torch.long, device=self._backend.device),
                )[0]
                if cache.seq_len != len(ids) or cache.doc_len != len(ids):
                    raise ValueError("recomputation requires complete, uncompressed chunk caches")
                state.append(PreparedChunk(ids, cache))
            self._states.append(state)
        if self._backend.device.type == "cuda":
            torch.cuda.synchronize(self._backend.device)

    def _select(self, hits):
        raise NotImplementedError("a selective method must choose repair positions")

    def Run(self, data: list[list[int]], retainOutput=None, maxNewTokens=None):
        entered = time.perf_counter()
        import torch
        from .CacheBackend.Recompute import CacheHit, assemble_cache

        if self._backend is None or self._gen is None:
            raise RuntimeError("initialize the recomputation method before Run")
        budget = self.maxNewTokens if maxNewTokens is None else ResolveMaxNewTokens(maxNewTokens)
        if len(self._states) != len(data):
            self._states = [[] for _ in data]
        results = []
        with torch.inference_mode():
            for case_index, prompt in enumerate(data):
                started = entered if case_index == 0 else time.perf_counter()
                ids = list(prompt)
                if not ids:
                    raise ValueError("recomputation requires a non-empty prompt")
                if len(ids) >= MaxModelLen():
                    raise ValueError(f"Prompt must leave room for generation within MaxModelLen={MaxModelLen()}")
                case_budget = min(budget, MaxModelLen() - len(ids))
                if any(not isinstance(token, int) or isinstance(token, bool) for token in ids):
                    raise TypeError("Run requires native integer token IDs")
                inputs = torch.tensor([ids], dtype=torch.long, device=self._backend.device)
                hits, fresh = [], []
                offset = 0
                prepared = self._states[case_index]
                parts = ComposeInterleavedReuse([chunk.ids for chunk in prepared], ids)
                for chunk_index, span in parts:
                    end = offset + len(span)
                    if chunk_index is None or self.fullRecompute:
                        fresh.extend(range(offset, end))
                    else:
                        hits.append(CacheHit(offset, end, prepared[chunk_index].cache))
                    offset = end
                repaired = sorted(set(self._select(hits))) if hits else []
                selected = sorted(set(fresh + repaired + [len(ids) - 1]))
                if hits:
                    cache = assemble_cache(self._backend, inputs, hits)
                    out = self._backend.recompute(input_ids=inputs, cache=cache,
                                                  token_indices=selected)
                else:
                    out = self._backend.forward(input_ids=inputs)
                cache = out.cache
                logits = out.logits[0, -1]
                generated, history = [], list(ids)
                generator = self._gen._Generator()
                ttft = None
                for step in range(case_budget):
                    token = self._gen._SampleNextToken(logits, history, generator)
                    if ttft is None:
                        ttft = time.perf_counter() - started
                    generated.append(token)
                    history.append(token)
                    if token in self._gen.eosIds or step + 1 == case_budget:
                        break
                    out = self._backend.forward(
                        input_ids=inputs.new_tensor([[token]]), cache=cache,
                        position_ids=inputs.new_tensor([[len(ids) + step]]),
                    )
                    cache, logits = out.cache, out.logits[0, -1]
                output_ids = self._gen.OutputTokenIds(generated)
                if not self.fullRecompute and retainOutput and retainOutput[case_index]:
                    if generated and generated[-1] not in self._gen.eosIds:
                        cache = self._backend.forward(
                            input_ids=inputs.new_tensor([[generated[-1]]]), cache=cache,
                            position_ids=inputs.new_tensor([[len(ids) + len(output_ids) - 1]]),
                        ).cache
                    self._states[case_index].append(PreparedChunk(ids + output_ids, cache))
                cached_positions = {pos for hit in hits for pos in range(hit.start, hit.end)}
                # Reuse measures cache-hit coverage, independently of repair.
                reused = len(cached_positions)
                results.append(Result(
                    output=self._gen.tokenizer.decode(generated, skip_special_tokens=True),
                    output_ids=output_ids,
                    performance={TtftKey: float(ttft), NumOutputTokensKey: len(generated),
                                 TotalTimeKey: time.perf_counter() - started},
                    metadata={"backend": self.backend, "n_input": len(ids),
                              "cached_tokens": len(cached_positions), "reused_tokens": reused,
                              "reuse_ratio": reused / len(ids),
                              "recomputed_tokens": len(selected) if hits else len(ids),
                              "selected_positions": selected if hits else list(range(len(ids))),
                              "full_prefill": not bool(hits)},
                ))
        return results

    def Reset(self):
        self._states = []

    def Close(self):
        self.Reset()
        self._backend = None
        self._gen = None
        import torch

        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
