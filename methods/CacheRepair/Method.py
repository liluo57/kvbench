"""KVBench adapter for the vendored CacheRepair implementation.

The upstream repairer consumes independently prefetched chunks. This adapter
uses KVBench's own PREPARE/RUN token IDs and existing interleaved matcher,
repairs the assembled prefix, then prefills an uncached query suffix against
that repaired prefix.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, List, Optional, Sequence

from core.Config import Get, MaxModelLen, ModelPath as DefaultModelPath
from core.Method import Method, ResolveMaxNewTokens
from core.Result import NumOutputTokensKey, Result, TotalTimeKey, TtftKey
from core.Sampling import ResolveSamplingConfig
from helpers.backends.Prompt import ComposeInterleavedReuse
from helpers.backends.TransformersHelper import TransformersGenerator


class CacheRepairMethod(Method):
    """CacheRepair over the project Transformers backend.

    CacheRepair uses a target-specific learned repairer, so ``ModelPath`` must
    identify the matching Qwen2.5-3B-Instruct checkpoint. Repairer location
    and size come from the ``CacheRepair`` section in ``config.yaml``. Output-
    cache retention is unsupported in repair mode.
    """

    name = "cacherepair"
    backend = "transformers"
    maxCaseBatchSize = 1
    method_metrics = ("reuse_ratio", "repair_ms")

    def __init__(
        self,
        gpuNums: int = 1,
        perfWeight: float = 1.0,
        *,
        fullPrefill: bool = False,
        dtype: str = "bfloat16",
        tag: Optional[str] = None,
    ):
        super().__init__(
            gpuNums=gpuNums,
            perfWeight=perfWeight,
            maxGpuNums=1,
            tag=tag,
        )
        if not isinstance(fullPrefill, bool):
            raise TypeError("fullPrefill must be a bool")

        repairer = Get("CacheRepair", {}) or {}
        if not isinstance(repairer, Mapping):
            raise TypeError("CacheRepair in config.yaml must be a mapping")
        repairer_path = repairer.get("RepairerPath", "gwang3456/cacherepair")
        if not isinstance(repairer_path, str) or not repairer_path.strip():
            raise ValueError("CacheRepair.RepairerPath must be a non-empty string")
        size = str(repairer.get("RepairerSize", "M")).upper()
        if size not in {"S", "M", "L"}:
            raise ValueError("CacheRepair.RepairerSize must be 'S', 'M', or 'L'")

        self.modelPath = DefaultModelPath()
        self.samplingConfig = ResolveSamplingConfig(self.modelPath)
        self.repairerPath = repairer_path.strip()
        self.repairerSize = size
        self.fullPrefill = fullPrefill
        self.dtype = dtype

        self._gen: Optional[TransformersGenerator] = None
        self._torch = None
        self._compileStale = None
        self._materialize = None
        self._toHfCache = None
        self._repairer = None
        self._invFreq = None
        self._ropeScaling = 1.0
        self._device = None
        self._cacheDtype = None
        self._warmedBuckets: set[int] = set()
        self._states: List[dict[str, Any]] = []

    def Initialize(self, gpuIds: Sequence[int]) -> None:
        super().Initialize(gpuIds)
        self.Reset()
        self._warmedBuckets.clear()
        self._gen = TransformersGenerator(
            self.modelPath,
            self.gpuIds,
            dtype=self.dtype,
            samplingConfig=self.samplingConfig,
        )
        import torch

        self._torch = torch
        target = self._gen.model
        embedding = target.get_input_embeddings().weight
        self._device = embedding.device
        self._cacheDtype = embedding.dtype
        self._warm_target()
        if self.fullPrefill:
            return

        from .backend.cacherepair.cache import (
            compile_stale,
            materialize,
            to_hf_cache,
        )
        from .backend.cacherepair.model import load_repairer

        repairer_path = Path(self.repairerPath).expanduser()
        checkpoint_subfolder = (
            None
            if repairer_path.is_dir() and (repairer_path / "config.json").is_file()
            else f"Qwen2.5-3B-Instruct/{self.repairerSize}"
        )
        self._repairer = load_repairer(
            self.repairerPath,
            target,
            subfolder=checkpoint_subfolder,
        )
        self._repairer.to(dtype=self._cacheDtype).eval()
        self._repairer.use_serving_attention()

        rotary = target.model.rotary_emb
        self._invFreq = rotary.inv_freq.detach()
        self._ropeScaling = float(rotary.attention_scaling)
        self._compileStale = compile_stale
        self._materialize = materialize
        self._toHfCache = to_hf_cache
        self._warm_repair()

    def Prepare(self, data: List[List[List[int]]]) -> None:
        """Compile independent stale KV for the supplied PREPARE segments."""
        if self.fullPrefill:
            self._states = []
            return
        if self._gen is None or self._compileStale is None:
            raise RuntimeError("CacheRepair must be initialized before Prepare")

        self._states = []
        for chunks in data:
            segments = [list(chunk) for chunk in (chunks or [])]
            nonempty = [chunk for chunk in segments if chunk]
            offsets: List[Optional[int]] = []
            cursor = 0
            for chunk in segments:
                offsets.append(cursor if chunk else None)
                cursor += len(chunk)

            stale_cpu = None
            if nonempty:
                stale, tokens, chunk_ids = self._compileStale(
                    self._gen.model, nonempty
                )
                stale_cpu = (
                    stale[0]
                    .to(device="cpu", dtype=self._cacheDtype)
                    .contiguous()
                )
                if len(nonempty) > 1:
                    self._warm_repair_bucket(stale, tokens, chunk_ids)
                del stale

            self._states.append(
                {
                    "segments": segments,
                    "offsets": offsets,
                    "stale": stale_cpu,
                }
            )

    def Run(
        self,
        data: List[List[int]],
        retainOutput: Optional[List[bool]] = None,
        maxNewTokens: Optional[int] = None,
    ) -> List[Result]:
        if self._gen is None:
            raise RuntimeError("CacheRepair must be initialized before Run")

        retain = retainOutput or []
        if not self.fullPrefill and any(bool(value) for value in retain):
            raise ValueError(
                "CacheRepair does not support retainOutput; use it with a "
                "single-turn RAG workflow"
            )

        max_new_tokens = ResolveMaxNewTokens(maxNewTokens)
        run_start = time.perf_counter()
        if self.fullPrefill:
            return [
                self._full_prefill(list(prompt), run_start, max_new_tokens)
                for prompt in data
            ]
        if self._repairer is None:
            raise RuntimeError("CacheRepair repairer is not initialized")

        if not self._states:
            states = [self._empty_state() for _ in data]
        elif len(self._states) != len(data):
            raise RuntimeError(
                f"CacheRepair state has {len(self._states)} case(s), "
                f"received {len(data)} prompt(s)"
            )
        else:
            states = self._states

        return [
            self._run_one(state, list(prompt), run_start, max_new_tokens)
            for state, prompt in zip(states, data)
        ]

    def Reset(self) -> None:
        self._states = []

    def Close(self) -> None:
        self.Reset()
        self._warmedBuckets.clear()
        self._repairer = None
        self._gen = None
        self._invFreq = None
        try:
            if self._torch is not None and self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - shutdown cleanup is best effort
            pass

    @staticmethod
    def _empty_state() -> dict[str, Any]:
        return {"segments": [], "offsets": [], "stale": None}

    @staticmethod
    def _bucket(length: int) -> int:
        return max(128, 1 << (length - 1).bit_length())

    def _warm_target(self) -> None:
        """Warm prefill and first-token forward without consuming sampling RNG."""
        if MaxModelLen() < 2:
            return
        token = self._gen.Encode("Warmup")[0]
        with self._torch.no_grad():
            prefix = self._gen.Prefill([token] * min(128, MaxModelLen() - 1))
            output = self._gen.Forward([token], prefix.past_key_values)
            output.logits[0, -1].argmax().item()
        self._torch.cuda.synchronize(self._device)

    def _warm_repair(self) -> None:
        """Warm every supported repair bucket and its online cache consumer.

        Synthetic KV keeps warmup independent of Tasks and their data. Two
        chunks model a fresh chat header plus one prepared document, including
        the single-PREPARE-chunk case. Padding exercises the serving mask path.
        """
        torch = self._torch
        max_prefix = min(
            MaxModelLen() - 2,
            self._repairer.target_config.max_position_embeddings,
        )
        if max_prefix < 2:
            return
        token = self._gen.Encode("Warmup")[0]
        # Fresh spans use upstream compile_stale rather than the helper's
        # Prefill path. Warm its explicit position/mask and canonicalization
        # kernels, as well as the host-to-device cache transfer used by RUN.
        stale, tokens, chunk_ids = self._compileStale(
            self._gen.model, [[token] * min(24, max_prefix)]
        )
        host = stale[0].to(device="cpu", dtype=self._cacheDtype).contiguous()
        transferred = host.unsqueeze(0).to(self._device, non_blocking=True)
        torch.cuda.synchronize(self._device)
        del transferred, host, stale, tokens, chunk_ids

        bucket = 128
        while bucket <= self._bucket(max_prefix):
            valid = min(bucket - 1, max_prefix)
            tokens = torch.full(
                (1, valid), token, dtype=torch.long, device=self._device
            )
            chunk_ids = torch.ones_like(tokens)
            chunk_ids[:, : min(24, valid // 2)] = 0
            stale = torch.zeros(
                (1, valid, *self._repairer.sigma_stale.shape),
                dtype=self._cacheDtype,
                device=self._device,
            )
            self._warm_repair_bucket(stale, tokens, chunk_ids)
            with torch.no_grad():
                global_kv = self._materialize(
                    stale,
                    self._invFreq,
                    self._ropeScaling,
                    dtype=self._cacheDtype,
                )
                past = self._toHfCache(global_kv)
                query_length = min(7, MaxModelLen() - valid - 2)
                if query_length:
                    past = self._gen.Prefill(
                        [token] * query_length, pastKeyValues=past
                    ).past_key_values
                output = self._gen.Forward([token], past)
                output.logits[0, -1].argmax().item()
            torch.cuda.synchronize(self._device)
            del output, past, global_kv, stale, tokens, chunk_ids
            bucket *= 2

    def _warm_repair_bucket(self, stale, tokens, chunk_ids) -> None:
        """Compile a serving shape outside the timed RUN interval."""
        torch = self._torch
        bucket = self._bucket(tokens.shape[1])
        if bucket in self._warmedBuckets:
            return
        padded_chunk_ids = torch.nn.functional.pad(
            chunk_ids, (0, bucket - tokens.shape[1]), value=-1
        )
        block_mask = self._repairer.prepare_block_mask(
            padded_chunk_ids, valid_len=tokens.shape[1]
        )
        self._repairer.repair(
            stale.to(dtype=self._cacheDtype),
            tokens,
            chunk_ids,
            bucket=bucket,
            block_mask=block_mask,
        )
        torch.cuda.synchronize(self._device)
        self._warmedBuckets.add(bucket)

    def _run_one(
        self,
        state: dict[str, Any],
        prompt_ids: List[int],
        run_start: float,
        max_new_tokens: int,
    ) -> Result:
        if not prompt_ids:
            raise ValueError("CacheRepair requires a non-empty RUN prompt")

        parts = ComposeInterleavedReuse(state["segments"], prompt_ids)
        if [token for _, span in parts for token in span] != prompt_ids:
            raise RuntimeError("KVBench prompt matcher did not preserve RUN token IDs")
        if not any(chunk_index is not None for chunk_index, _ in parts):
            return self._full_prefill(prompt_ids, run_start, max_new_tokens)

        # A final unmatched span is the online query. Its tokens are prefetched
        # after repair; the last prompt token is fed to Generate for next-token
        # sampling. If the prompt ends on a cached segment, remove its last row
        # from the repaired prefix and feed that same final token online.
        if parts and parts[-1][0] is None:
            query_span = list(parts[-1][1])
            prefix_parts = list(parts[:-1])
            query_prefix = query_span[:-1]
            generation_token = query_span[-1]
        else:
            prefix_parts = list(parts)
            last_index, last_span = prefix_parts[-1]
            generation_token = last_span[-1]
            if len(last_span) == 1:
                prefix_parts.pop()
            else:
                prefix_parts[-1] = (last_index, list(last_span[:-1]))
            query_prefix = []

        torch = self._torch
        stale_parts = []
        token_ids: List[int] = []
        chunk_id_parts = []
        reused_tokens = 0
        for chunk_index, span in prefix_parts:
            span = list(span)
            if not span:
                continue

            if chunk_index is None:
                stale, _, _ = self._compileStale(self._gen.model, [span])
                chunk_stale = stale[0].unsqueeze(0).to(dtype=self._cacheDtype)
                del stale
            else:
                prepared = state["segments"][chunk_index]
                if prepared[: len(span)] != span:
                    raise RuntimeError(
                        "CacheRepair matcher returned IDs different from "
                        "the prepared segment"
                    )
                offset = state["offsets"][chunk_index]
                if offset is None or state["stale"] is None:
                    raise RuntimeError("CacheRepair matched a segment without cached KV")
                chunk_stale = (
                    state["stale"][offset : offset + len(span)]
                    .unsqueeze(0)
                    .to(
                        device=self._device,
                        dtype=self._cacheDtype,
                        non_blocking=True,
                    )
                )
                reused_tokens += len(span)

            chunk_number = len(stale_parts)
            stale_parts.append(chunk_stale)
            token_ids.extend(span)
            chunk_id_parts.append(
                torch.full(
                    (1, len(span)),
                    chunk_number,
                    dtype=torch.long,
                    device=self._device,
                )
            )

        if reused_tokens == 0 or not stale_parts:
            return self._full_prefill(prompt_ids, run_start, max_new_tokens)

        expected_prefix = prompt_ids[: len(prompt_ids) - len(query_prefix) - 1]
        if token_ids != expected_prefix:
            raise RuntimeError(
                "CacheRepair assembled prefix IDs do not match the RUN prompt"
            )

        tokens = torch.tensor([token_ids], dtype=torch.long, device=self._device)
        chunk_ids = torch.cat(chunk_id_parts, dim=1)
        stale = torch.cat(stale_parts, dim=1)
        if stale.shape[1] != tokens.shape[1]:
            raise RuntimeError("CacheRepair stale KV length does not match prefix tokens")

        repair_ms = 0.0
        fixed = stale
        if len(stale_parts) > 1:
            bucket = self._bucket(tokens.shape[1])
            padded_chunk_ids = torch.nn.functional.pad(
                chunk_ids, (0, bucket - tokens.shape[1]), value=-1
            )
            start = torch.cuda.Event(enable_timing=True)
            done = torch.cuda.Event(enable_timing=True)
            start.record()
            block_mask = self._repairer.prepare_block_mask(
                padded_chunk_ids, valid_len=tokens.shape[1]
            )
            fixed = self._repairer.repair(
                stale,
                tokens,
                chunk_ids,
                bucket=bucket,
                block_mask=block_mask,
            )
            done.record()

        global_kv = self._materialize(
            fixed,
            self._invFreq,
            self._ropeScaling,
            dtype=self._cacheDtype,
        )
        past = self._toHfCache(global_kv)

        # The query tail remains a normal causal prefill against repaired KV.
        if query_prefix:
            past = self._gen.Prefill(query_prefix, pastKeyValues=past).past_key_values

        if len(stale_parts) > 1:
            torch.cuda.synchronize(self._device)
            repair_ms = float(start.elapsed_time(done))

        generation_start = time.perf_counter()
        text, backend_ttft, backend_total, n_tokens, output_ids = self._gen.Generate(
            [generation_token],
            pastKeyValues=past,
            maxNewTokens=max_new_tokens,
            returnTokenIds=True,
        )
        online_before_generation = generation_start - run_start
        return Result(
            output=text,
            output_ids=output_ids,
            performance={
                TtftKey: online_before_generation + backend_ttft,
                NumOutputTokensKey: n_tokens,
                TotalTimeKey: online_before_generation + backend_total,
            },
            metadata={
                "reuse_ratio": reused_tokens / len(prompt_ids),
                "n_input": len(prompt_ids),
                "repair_ms": repair_ms,
                "repaired_tokens": len(token_ids) if len(stale_parts) > 1 else 0,
                "full_prefill": False,
            },
        )

    def _full_prefill(
        self,
        prompt_ids: List[int],
        run_start: float,
        max_new_tokens: int,
    ) -> Result:
        generation_start = time.perf_counter()
        text, backend_ttft, backend_total, n_tokens, output_ids = self._gen.Generate(
            prompt_ids,
            maxNewTokens=max_new_tokens,
            returnTokenIds=True,
        )
        online_before_generation = generation_start - run_start
        return Result(
            output=text,
            output_ids=output_ids,
            performance={
                TtftKey: online_before_generation + backend_ttft,
                NumOutputTokensKey: n_tokens,
                TotalTimeKey: online_before_generation + backend_total,
            },
            metadata={
                "reuse_ratio": 0.0,
                "n_input": len(prompt_ids),
                "repair_ms": 0.0,
                "repaired_tokens": 0,
                "full_prefill": True,
            },
        )
