"""CacheBlend method — vLLM 0.25 + LMCache in-process CacheBlend.

This is the in-process alternative to :class:`methods.CacheblendRepo`, which
continues to drive the original authors' patched vLLM in a helper subprocess.
This version drives vLLM 0.25.1's built-in LMCache connector
(``LMCacheConnectorV1``, ``kv_role="kv_both"``) with blending enabled, in the
same interpreter — no external repo, no host-level ``sitecustomize.py``.

The wiring lmcache 0.5.2 needs to blend (model registration) is applied from
this framework module via :mod:`helpers.VllmCacheblendPatches`; see its
docstring for the details.
Only active while this method is running (gated by the env var), so the rest of
the framework's vLLM path is untouched.

How the patches reach the spawned EngineCore (vLLM's engine runs in a child
process): vLLM spawns the EngineCore with ``multiprocessing``, which re-runs the
**main module's** top-level imports (``from methods import ...``) in the child.
The env var set here in ``Initialize`` is inherited by the child, so the
module-level guard below re-applies the patches there, before the child imports
the vLLM model runner — without any host-level file.

The engine calls, once per batch (``Prepare`` gets the batch's warm-up chunk
lists, ``Run`` the batch's prompts, ``Reset`` clears the whole batch):

    Prepare(chunks)  -> for each case, encode each chunk at the token level,
                        joined by the literal ``' # # '`` sep ids, and prefill
                        it once so lmcache stores per-chunk KV segments.
    Run(prompt)      -> per case: 1. ComposeReuse finds the longest prefix of
                            ``prompt`` explainable by prepared chunks (preferring
                            original order), returns (order, suffix). If found,
                            assemble reordered context tokens + fresh suffix
                            tokens and generate; lmcache reuses cached chunk KV
                            and blends the boundary.
                        2. no match -> generate the whole prompt (no reuse,
                            e.g. empty prepare or no chunks match the start).
    Reset()          -> drop the per-case token state (segment keys are
                        content-hashed, so stale cached KV is simply never hit).
    Close()          -> drop the LLM + free GPU cache.

Note on the shuffle tasks (NIAHShuffleTask etc.): their ``run_input`` is the
prepared chunks in a different order, so the old ``SplitReuseParts`` would find
no prefix. Now ``ComposeReuse`` handles this by searching for reordered chunks
after trying the original order first. LMCache keys segments by *content hash*
(position-independent), so the shuffled sep-joined stream still hits ~100% of
the stored chunk KV; the blender recomputes the important (reordered) tokens it
detects. Only a prompt that cannot be explained by the prepared chunks at all
falls back to a full prefill. The naive control (blindly serving the stale
un-shuffled KV, no re-detection) is what the shuffle tests fail.

A throwaway **warmup** request is issued during worker initialization: on this class of
hosts lmcache's *first* store in a fresh EngineCore writes corrupt KV
(nondeterministic — EOS or wrong content on later reuse); every later store is
clean. Warming up makes the real stores always clean.
"""

import hashlib
import os
import time
from typing import Dict, List, Optional, Sequence

from core.Config import ModelPath as DefaultModelPath
from core.Method import Method
from core.Result import NumOutputTokensKey, Result, TotalTimeKey, TtftKey

from helpers.backends.Prompt import ComposeReuse
from helpers.backends.VllmCacheblendPatches import (
    ApplyPatches,
    BuildContextTokens,
    CreateBlendLlm,
    EncodeText,
    SepTokens,
    SetBlendEnv,
    Warmup,
)

# The spawned EngineCore child re-runs this module's import (vLLM re-executes
# the main module's top-level `from methods import ...`), and the parent's
# ``LMCACHE_ENABLE_BLENDING`` is inherited, so the patches self-apply in the
# child before the vLLM model runner is imported there. In the coordinator the
# env is not set; the worker's ``Initialize`` applies the patches explicitly.
if os.environ.get("LMCACHE_ENABLE_BLENDING") or os.environ.get(
    "LMCACHE_EC_ENABLE_BLENDING"
):
    ApplyPatches()


class CacheblendLmcache(Method):
    name = "cacheblend_lmcache"

    #: Reuse rate is cacheblend's method metric: the share of the run stream the
    #: engine actually served from lmcache KV — ``num_cached_tokens`` reported by
    #: the scheduler for the request, over the input length (see :meth:`Run`).
    #: It is *not* the constructed context share, which stays ~1.0 even when the
    #: cache could not store anything (memory pressure) and zero tokens were hit.
    #: Naive may record ``reuse_ratio`` as diagnostic metadata but does not
    #: declare it; FullPrefill does neither. Their report entries therefore
    #: have no ``method_metrics`` section.
    method_metrics = ("reuse_ratio",)

    def __init__(
        self,
        gpuNums: int = 1,
        perfWeight: float = 1.0,
        *,
        maxNewTokens: int = 64,
        maxModelLen: int = 40960,
        gpuMemoryUtilization: float = 0.7,
        recompRatio: float = 0.15,
        maxLocalCpuSize: float = 5.0,
        dtype: str = "bfloat16",
        enforceEager: bool = True,
        # Kept for signature compatibility with the old worker subprocess
        # implementation; no longer used.
        numLayers: int = 28,
        repoRoot=None,
        workerPython=None,
        startTimeout: float = 1800.0,
        tag: Optional[str] = None,
    ):
        super().__init__(gpuNums=gpuNums, perfWeight=perfWeight, tag=tag)
        # Model path is config-only — switch models via config.yaml.
        self.modelPath = DefaultModelPath()
        self.maxNewTokens = maxNewTokens
        self.maxModelLen = maxModelLen
        self.gpuMemoryUtilization = gpuMemoryUtilization
        self.recompRatio = recompRatio
        self.maxLocalCpuSize = maxLocalCpuSize
        self.dtype = dtype
        self.enforceEager = enforceEager

        #: Per-case state accumulated by :meth:`Prepare` and consumed by
        #: :meth:`Run`: each entry is ``{"prepare", "context_tokens"}`` (the
        #: case's warm-up chunks and their salted, sep-joined token stream).
        self._states: List[dict] = []

    def Initialize(self, gpuIds: Sequence[int]) -> None:
        super().Initialize(gpuIds)
        # Order matters: env vars must be set (and patches applied) before vLLM
        # is imported, so the spawned EngineCore inherits the right config and
        # the model runner gets patched in the child.
        SetBlendEnv(
            recompRatio=self.recompRatio,
            maxLocalCpuSize=self.maxLocalCpuSize,
        )
        ApplyPatches()
        self.llm = CreateBlendLlm(
            self.modelPath,
            self.gpuIds,
            gpuMemoryUtilization=self.gpuMemoryUtilization,
            maxModelLen=self.maxModelLen,
            dtype=self.dtype,
            enforceEager=self.enforceEager,
            tensorParallelSize=self.gpuNums,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.sep = SepTokens(self.tokenizer)
        # First request in a fresh EngineCore: throw away its (corrupt) store.
        Warmup(self.llm)

    # ---------------------------------------------------------------- Method
    def _SaltTokens(self, prepare: List[str]) -> List[int]:
        """A per-case token prefix that namespaces this case's cache keys.

        LMCache keys segments by *content hash* (chained through the prefix).
        When a dataset reuses the same text across samples (RULER niah slices
        the same essay), two cases share the same 256-token blocks, so they
        would produce the *same* cache keys and the store keeps the first
        writer's KV (see ``submit_put_task``: a key already in ``hot_cache`` is
        not overwritten). A later case would then retrieve an earlier case's KV
        — computed at different positions/context — and blend repairs the wrong
        thing.

        Prepending a salt derived from this case's prepared context (unique per
        sample, stable across this case's own Prepare/Run) makes every key in
        the chain case-unique, so cross-case reuse never happens while
        within-case reuse is untouched. The salt tokens are a harmless prefix
        the model sees in the run stream exactly as it saw them when storing.
        ``prepare`` is the case's *original-order* warm-up chunks — the salt
        stays tied to those even when ``Run`` rebuilds a reordered stream.
        """
        text = "".join(prepare or [])
        if not text:
            return []
        h = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
        return EncodeText(self.tokenizer, f" [kvbench:{h}] ")

    def _SaltedContextTokens(
        self, chunks: List[str], prepare: List[str]
    ) -> List[int]:
        """Assemble ``chunks`` with the per-case salt before *every* chunk.

        ``SegmentTokenDatabase`` splits the stored/run stream on the sep ids and
        hashes each segment independently by content — only the *first* segment
        is chained through its prefix. Prepending the salt just once (as the
        earlier fix did) therefore namespaces only the first segment; the middle
        essay/needle segments stayed pure content-addressed, so two samples
        slicing the same RULER essay still collided on those keys. Prepending
        the salt to each chunk makes *every* segment ``salt+chunk``, so the whole
        chain is case-unique while within-case reuse (store vs. reordered run)
        still hits the same keys.

        ``chunks`` are the segments in the order they appear in the stream;
        ``prepare`` is the case's original-order warm-up chunks, from which the
        salt is derived (unchanged whether the stream is the original order or a
        re-detected run order).
        """
        salt = self._SaltTokens(prepare)
        if not salt:
            return BuildContextTokens(chunks, self.tokenizer, self.sep)
        ids: List[int] = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                ids += self.sep
            ids += salt
            ids += EncodeText(self.tokenizer, chunk)
        return ids

    def Prepare(self, data: List[List[str]]) -> None:
        """Store the context chunks' KV for a batch of cases.

        For each case, each chunk is encoded separately and joined by the
        literal sep ids at the token level, then prefilled once so lmcache
        stores a KV segment per chunk. A per-case salt is prepended to *every*
        chunk (see :meth:`_SaltedContextTokens`) so this case's segment keys
        cannot collide with another sample's — lmcache hashes the middle
        segments by content alone, so a single stream-level salt would leave the
        essay/needle segments addressable across samples.

        All cases' contexts are prefilled in one concurrent
        :meth:`_GenerateBatch` call, so the V1 scheduler stores their KV in
        parallel.
        """
        self._states = []
        ctxTokensList: List[Optional[List[int]]] = []
        for chunks in data:
            prepare = list(chunks or [])
            contextTokens = self._SaltedContextTokens(prepare, prepare)
            ctxTokensList.append(contextTokens or None)
            self._states.append(
                {"prepare": prepare, "context_tokens": contextTokens}
            )
        validCtx = [ctx for ctx in ctxTokensList if ctx]
        if validCtx:
            self._GenerateBatch(validCtx, maxTokens=1)  # store all KV together

    def Run(self, data: List[str], retainOutput: Optional[List[bool]] = None) -> List[Result]:
        """Generate a batch of prompts, submitting all at once.

        Each case's token stream is assembled from its prepared state (the same
        per-case split / reorder / fuse logic as before), but all streams are
        then submitted to :meth:`_GenerateBatch` in one call so the V1 engine
        generates them concurrently.
        """
        # LMCache owns prefix-cache lifetime; generated-output retention is not
        # currently exposed by the in-process connector.
        _ = retainOutput
        tokenStreams: List[List[int]] = []
        metas: List[dict] = []
        for run_input, state in zip(data, self._states):
            prepare: List[str] = state["prepare"]
            contextTokens: List[int] = state["context_tokens"]
            order, suffix = ComposeReuse(prepare, run_input)
            if order is not None and contextTokens:
                # Found reusable chunks forming a prefix of run_input
                ids = self._SaltedContextTokens(order, prepare)
                if suffix:
                    # The trailing fresh text is salted too, so it can never
                    # collide with another case's segment keys (it is never
                    # stored, only computed fresh).
                    ids = (
                        ids
                        + self.sep
                        + self._SaltTokens(prepare)
                        + EncodeText(self.tokenizer, suffix)
                    )
                tokenStreams.append(ids)
                nInput = len(ids)
                # Check if order differs from original prepare order
                reordered = order != prepare
                metas.append({"reordered": reordered, "n_input": nInput})
                continue
            
            # No reusable content (empty warm-up, or no chunks match the start
            # of run_input): generate the whole prompt.
            ids = EncodeText(self.tokenizer, run_input)
            tokenStreams.append(ids)
            nInput = len(ids)
            metas.append({"n_input": nInput})

        batchOut = self._GenerateBatch(tokenStreams, self.maxNewTokens)
        results = []
        for (text, ttft, nTokens, totalTime, numCached), meta in zip(
            batchOut, metas
        ):
            # Actual cache reuse as reported by the engine per request: the
            # tokens the scheduler served from lmcache (``out.num_cached_tokens``),
            # not the constructed stream's context share. The constructed share
            # lies under memory pressure (reports ~1.0 when the cache could not
            # store anything and zero tokens were actually hit); the engine count
            # reflects what was really hit/scheduled to load.
            nInput = meta["n_input"]
            reuseRatio = min(1.0, numCached / nInput) if nInput else 0.0
            metadata = {
                "reuse_ratio": reuseRatio,
                "recomp_ratio": self.recompRatio,
                "n_input": nInput,
                "num_cached_tokens": numCached,
            }
            if meta.get("reordered"):
                metadata["reordered"] = True
            results.append(
                Result(
                    output=text,
                    performance={
                        TtftKey: ttft,
                        NumOutputTokensKey: nTokens,
                        TotalTimeKey: totalTime,
                    },
                    metadata=metadata,
                )
            )
        return results


    def Reset(self) -> None:
        """Drop the per-case token state.

        Cached KV itself is keyed by content hashes of the token segments, so a
        previous case's segments are simply never hit by the next case's
        (different) context. No cache clearing is needed.
        """
        self._states = []

    def Close(self) -> None:
        llm = getattr(self, "llm", None)
        if llm is not None:
            try:
                del llm
            except Exception:  # noqa: BLE001
                pass
            self.llm = None
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    # --------------------------------------------------------------- generate
    def _GenerateBatch(self, tokenIdsList: List[List[int]], maxTokens: int):
        """Drive the V1 engine over many concurrent requests.

        Every token stream is submitted up front (``add_request``) and all are
        stepped together, so the V1 scheduler batches them natively. Returns one
        ``(text, ttft, numTokens, totalTime, numCached)`` per request, in input
        order — TTFT measured vLLM-benchmark style (submission -> first decoded
        token). Both ``ttft`` and ``totalTime`` are *amortized* over the batch
        (``measured / batchSize``): the raw submission->first-token wall-clock
        includes queueing behind the sibling requests (they share the GPU), so
        amortizing gives the fair per-sample first-token time, consistent with
        ``totalTime`` and the transformers batch path. ``totalTime`` amortized
        means summing per-sample times equals the actual run wall-clock — what
        :class:`~metrics.Throughput.ThroughputMetric` expects.
        """
        from vllm import SamplingParams

        engine = self.llm.llm_engine
        requestIds: List[str] = []
        t0 = time.perf_counter()
        for i, tokenIds in enumerate(tokenIdsList):
            requestId = f"kvbench-{time.time_ns()}-{i}"
            engine.add_request(
                requestId,
                tokenIds,
                SamplingParams(temperature=0, max_tokens=maxTokens),
            )
            requestIds.append(requestId)

        ttfts: Dict[str, float] = {}
        numCached: Dict[str, int] = {}
        texts: Dict[str, str] = {}
        tokenLens: Dict[str, int] = {}
        while engine.has_unfinished_requests():
            for out in engine.step():
                if out.request_id not in requestIds:
                    continue
                rid = out.request_id
                if rid not in ttfts and out.outputs and out.outputs[0].token_ids:
                    ttfts[rid] = time.perf_counter() - t0  # first token: vllm-bench style
                numCached[rid] = max(
                    numCached.get(rid, 0),
                    int(getattr(out, "num_cached_tokens", 0) or 0),
                )
                if out.finished and out.outputs:
                    texts[rid] = out.outputs[0].text
                    tokenLens[rid] = len(out.outputs[0].token_ids)
        totalTime = time.perf_counter() - t0
        n = len(requestIds) or 1
        amortized = totalTime / n
        return [
            (
                texts.get(rid, ""),
                float(ttfts.get(rid, 0.0)) / n,
                tokenLens.get(rid, 0),
                amortized,
                numCached.get(rid, 0),
            )
            for rid in requestIds
        ]
