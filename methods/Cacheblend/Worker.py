"""JSONL worker for the bundled CacheBlend backend.

Collect, reserve, fuse, full, reset and close retain the existing protocol.
Cached KV and reusable assembly buffers stay on the assigned worker GPU.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

if __package__:
    from .backend.cacheblend_ops import RetainedOutputCollector
else:
    from backend.cacheblend_ops import RetainedOutputCollector


def _load_backend():
    """Resolve bundled code and the worker environment's own NCCL library."""
    backend = Path(__file__).resolve().parent / "backend"
    sys.path.insert(0, str(backend))
    os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
    import torch
    nccl = Path(torch.__file__).resolve().parent.parent / "nvidia/nccl/lib/libnccl.so.2"
    if nccl.is_file():
        os.environ.setdefault("VLLM_NCCL_SO_PATH", str(nccl))
    import kvbench_cacheblend
    if not Path(kvbench_cacheblend.__file__).resolve().is_relative_to(backend):
        raise RuntimeError(f"CacheBlend loaded an unexpected backend: {kvbench_cacheblend.__file__}")
    return kvbench_cacheblend.LLM, kvbench_cacheblend.SamplingParams


def _prepare_model_for_cacheblend(model: str) -> str:
    """Return a loader-safe view of a ModelScope/HuggingFace model directory.

    Some ModelScope snapshots contain both the normal HuggingFace sharded
    checkpoint and ``consolidated.safetensors``.  The latter uses Meta's
    ``layers.0.attention.wk.weight`` names, which the old vLLM 0.4.1 fork
    bundled with CacheBlend cannot load.  If an HF index is present, expose a
    small symlink-only directory without that extra file.  The source model
    is never modified and models without this layout keep their original path.
    """
    source = Path(model)
    index = source / "model.safetensors.index.json"
    consolidated = source / "consolidated.safetensors"
    if not source.is_dir() or not index.is_file() or not consolidated.is_file():
        return model

    source = source.resolve()
    digest = hashlib.sha256(
        f"{source}\n{index.stat().st_mtime_ns}\n{index.stat().st_size}".encode()
    ).hexdigest()[:16]
    staged = Path(tempfile.gettempdir()) / "kvbench-cacheblend-models" / (
        f"{source.name}-{digest}"
    )
    staged.mkdir(parents=True, exist_ok=True)

    # Link all snapshot files except the incompatible consolidated checkpoint.
    # Keeping the metadata and tokenizer files in the view lets both vLLM and
    # transformers resolve the model exactly as they would from the source.
    for item in source.iterdir():
        if item.name == "consolidated.safetensors":
            continue
        destination = staged / item.name
        if destination.exists() or destination.is_symlink():
            continue
        os.symlink(item, destination)
    return str(staged)


def _repair_plan(segments: list, prefix_len: int, ratio: float) -> dict:
    """Describe positions the suffix-only check layer must recompute.

    The patched vLLM fork can express one native trailing suffix, but a fresh
    span in the middle must be explicitly selected.  Cached tokens after it
    remain candidates for the fork's normal V-difference top-k selection; this
    is the important distinction between CacheBlend repair and full tail
    recomputation.  Keeping this calculation pure makes the behavior testable
    without loading vLLM.
    """
    first_fresh = next(
        (start for cached, _, _, start, _ in segments
         if not cached and start < prefix_len),
        None,
    )
    forced = {
        pos
        for cached, _, _, start, end in segments
        if not cached
        for pos in range(start, min(end, prefix_len))
    }
    cached_prefix = sum(
        max(0, min(end, prefix_len) - start)
        for cached, _, _, start, end in segments
        if cached and start < prefix_len
    )
    ratio_repair = max(0, min(cached_prefix, int(cached_prefix * float(ratio))))
    topk_count = max(0, min(prefix_len, len(forced) + ratio_repair))
    if prefix_len <= 0 or topk_count <= 0:
        effective_ratio = 0.0
    elif topk_count >= prefix_len:
        effective_ratio = 1.0
    else:
        # The fork executes int(prefix_len * recomp_ratio).  This midpoint
        # maps back to topk_count without floating point boundary surprises.
        effective_ratio = (topk_count + 0.5) / prefix_len
    return {
        "first_fresh": first_fresh,
        "forced": forced,
        "cached_prefix": cached_prefix,
        "topk_count": topk_count,
        "effective_ratio": effective_ratio,
    }


class CacheBlendWorker:
    def __init__(self, args):
        # Heavy deps load only inside this subprocess (see module docstring).
        from transformers import AutoTokenizer
        LLM, SamplingParams = _load_backend()

        self.args = args
        try:
            self.samplingConfig = json.loads(args.sampling_config or "{}")
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid --sampling_config JSON") from exc
        self._torch = __import__("torch")
        print(f"[cacheblend-helper] loading model {args.model} ...", flush=True)
        modelPath = _prepare_model_for_cacheblend(args.model)
        if modelPath != args.model:
            print(
                "[cacheblend-helper] using HF sharded checkpoint view "
                f"{modelPath}",
                flush=True,
            )
        from kvbench_cacheblend.transformers_utils.config import get_config
        config = get_config(modelPath, trust_remote_code=False)
        self._isQwen3 = "Qwen3ForCausalLM" in (config.architectures or [])
        # dtype="auto" loads the model's own config dtype (Mistral-7B-Instruct
        # is bfloat16). enforce_eager is left unset on purpose: CacheBlend's
        # collect reads hack_kv captured during the prefill forward, and the
        # single decode step must NOT re-run the forward (eager would overwrite
        # hack_kv) — CUDA-graph decode replays instead, preserving it.
        # max_num_seqs bounds the KV cache (the stored per-chunk KVs live
        # outside it). max_num_batched_tokens is left to vLLM's default
        # (max(max_model_len, 2048) = max_model_len here), which guarantees the
        # batched collect's groups (<= max_collect_tokens) prefill in a single
        # forward.
        #
        # Keep the original Qwen3 slow BPE tokenizer to preserve tokenization
        # across the environment upgrade; Mistral keeps its fast tokenizer.
        self.llm = LLM(
            model=modelPath,
            dtype="auto",
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            enforce_eager=args.enforce_eager,
            **({"tokenizer_mode": "slow"} if self._isQwen3 else {}),
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            modelPath, **({"use_fast": False} if self._isQwen3 else {}))
        self.llm.set_tokenizer(self.tokenizer)
        self.sampling_params = SamplingParams

        engine = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
        self.engine = engine
        self.layers = engine.model.layers
        self.device = next(engine.parameters()).device
        self.cfm = engine.model.cache_fuse_metadata
        self.cfm["recomp_ratio"] = args.recomp_ratio
        self.cfm["_base_recomp_ratio"] = float(args.recomp_ratio)

        #: Per-chunk cached KV, keyed by the chunk's exact text: chunk text ->
        #: per-layer ``[[k, v], ...]``. Collected once per batch in ``collect``,
        #: consumed in ``fuse`` in whatever (run) order the method needs. The
        #: tensors stay on the worker GPU so serving never stages KV through
        #: host memory.
        self._chunkKv = {}
        #: Per-chunk token ids, keyed the same way.
        self._chunkIds = {}
        #: Per-layer dense K/V buffers used to assemble reordered/interleaved
        #: prompts. They grow before a run batch, then are overwritten in-place
        #: for every request. This avoids both H2D copies and allocator churn.
        self._assembledKv = []
        self._assembledCapacity = 0
        self._retainCollector = None
        print("[cacheblend-helper] ready", flush=True)

    def _SamplingParams(self, maxTokens: int):
        # The parent passes the resolved sampling policy over JSON.
        config = self.samplingConfig
        if (
            not config
            or config.get("mode") == "greedy"
            or not bool(config.get("do_sample", False))
            or float(config.get("temperature", 1.0)) == 0
        ):
            return self.sampling_params(temperature=0, max_tokens=maxTokens)

        kwargs = {"max_tokens": maxTokens}
        for key in (
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "repetition_penalty",
            "presence_penalty",
            "frequency_penalty",
            "seed",
        ):
            if key in config:
                kwargs[key] = config[key]
        return self.sampling_params(**kwargs)

    def _beginRetainOutput(self):
        self._retainCollector = RetainedOutputCollector(self.layers)
        self.cfm["retain_collector"] = self._retainCollector

    def _endRetainOutput(self):
        collector = self._retainCollector
        self._retainCollector = None
        self.cfm.pop("retain_collector", None)
        retained = collector.finish() if collector is not None else None
        self._ClearHackKv()
        return retained

    def _ClearHackKv(self) -> None:
        """Drop attention capture aliases once owned copies/views exist."""
        for layer in self.layers:
            layer.self_attn.hack_kv = None

    def _generateWithRetention(
        self, fullIds, retain_output=False, max_new_tokens=None
    ):
        if not retain_output:
            return self._Generate(fullIds, max_new_tokens=max_new_tokens)
        self._beginRetainOutput()
        try:
            return self._Generate(
                fullIds,
                retain_output=True,
                max_new_tokens=max_new_tokens,
            )
        finally:
            self._retainCollector = None
            self.cfm.pop("retain_collector", None)

    # ------------------------------------------------------------- collect
    def _collectIds(self, ids: list):
        """Run ``ids`` through the model once with collect=True; return per-layer [K,V].

        Used for a single sequence (a fresh suffix). The KV length is checked
        against ``ids`` so a stale ``hack_kv`` (eager decode overwriting it)
        surfaces as an error instead of silently fusing wrong tensors.
        """
        self.cfm["collect"] = True
        self.cfm["check"] = False
        self.llm.generate(
            prompt_token_ids=[ids],
            sampling_params=self._SamplingParams(1),
        )
        out = []
        for layer in self.layers:
            k, v = layer.self_attn.hack_kv
            if k.shape[0] != len(ids):
                # The decode step overwrote hack_kv (e.g. eager mode); the KV
                # no longer matches the collected prompt.
                raise RuntimeError(
                    f"collected KV length {k.shape[0]} != prompt {len(ids)}"
                )
            out.append([k.clone(), v.clone()])
        return out

    def _CollectGroup(self, texts: list, idsList: list, depth: int = 0) -> None:
        """Collect a batch of chunk sequences; store per-chunk KV keyed by text.

        All chunk sequences go in one ``llm.generate`` call — each chunk is its
        own sequence, so attention stays chunk-isolated. vLLM concatenates the
        prefill tokens in sequence (submission) order, so ``hack_kv`` holds the
        whole group's ``[K, V]`` and is split back by the known chunk lengths.

        If ``hack_kv`` does not cover the group's tokens the prefill was
        chunked into several forwards (the last one overwrote ``hack_kv``);
        split the group in half and retry (each half is a fresh collect).
        """
        total = sum(len(ids) for ids in idsList)
        self.cfm["collect"] = True
        self.cfm["check"] = False
        self.llm.generate(
            prompt_token_ids=idsList,
            sampling_params=self._SamplingParams(1),
        )
        # ``hack_kv`` is stable until the next generate. Slice-clone directly
        # from it below; cloning the whole group first would temporarily double
        # the GPU footprint during collection.
        kvBatch = [layer.self_attn.hack_kv for layer in self.layers]
        if kvBatch[0][0].shape[0] != total:
            if len(idsList) == 1 or depth >= 3:
                raise RuntimeError(
                    f"batch collect: captured {kvBatch[0][0].shape[0]} KV tokens "
                    f"for {total} input tokens ({len(idsList)} sequences); "
                    f"the prefill was chunked and hack_kv only holds the last "
                    f"forward"
                )
            mid = len(idsList) // 2
            # Do not retain the oversized/partial capture while recursive
            # collection allocates owned chunks for both halves.
            del kvBatch
            self._ClearHackKv()
            self._CollectGroup(texts[:mid], idsList[:mid], depth + 1)
            self._CollectGroup(texts[mid:], idsList[mid:], depth + 1)
            return
        off = 0
        for text, ids in zip(texts, idsList):
            n = len(ids)
            # Keep owned slices on GPU. The benchmark now uses batch=4; the
            # previous CPU staging existed for the old batch=32 configuration
            # and cost 60-230ms of H2D traffic per request on Qwen3-8B.
            kv = [
                [
                    kvBatch[j][0][off:off + n].clone(),
                    kvBatch[j][1][off:off + n].clone(),
                ]
                for j in range(len(self.layers))
            ]
            self._chunkIds[text] = ids
            self._chunkKv[text] = kv
            off += n
        self._ClearHackKv()

    def Collect(self, chunks: list):
        """Cache every chunk's KV in isolation, batched across the input list.

        Chunks already collected (same text, e.g. two cases sharing the RULER
        head) are skipped — the isolated KV is content-deterministic and the
        fuse re-rotates it to each run's positions, so sharing is safe.
        """
        encoded = []
        seen = set()
        for text in chunks:
            if text in self._chunkIds or text in seen:
                continue
            seen.add(text)
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if not ids:
                continue
            encoded.append((text, ids))
        budget = self.args.max_collect_tokens
        groups = []
        cur, curLen = [], 0
        for text, ids in encoded:
            if cur and curLen + len(ids) > budget:
                groups.append(cur)
                cur, curLen = [], 0
            cur.append((text, ids))
            curLen += len(ids)
        if cur:
            groups.append(cur)
        for group in groups:
            self._CollectGroup([t for t, _ in group], [ids for _, ids in group])
        self.cfm["collect"] = False
        return {
            "ok": True,
            "n_chunks": len(encoded),
            "n_tokens": sum(len(ids) for _, ids in encoded),
        }

    # ---------------------------------------------------------------- fuse
    def _Generate(
        self,
        fullIds: list,
        retain_output=False,
        max_new_tokens=None,
    ):
        """Decode ``fullIds``, returning the standard result dict."""
        maxNewTokens = (
            self.args.max_new_tokens
            if max_new_tokens is None
            else int(max_new_tokens)
        )
        if maxNewTokens < 1:
            raise ValueError("max_new_tokens must be at least 1")
        # This is the boundary immediately before backend generation.  The
        # parent adapter combines it with Method.Run's monotonic timestamp so
        # matching, reserve, IPC, and worker-side assembly are included once.
        backendWallStart = time.time()
        t0 = time.perf_counter()
        out = self.llm.generate(
            prompt_token_ids=[fullIds],
            sampling_params=self._SamplingParams(
                maxNewTokens + (1 if retain_output else 0)
            ),
        )
        r = out[0]
        # The CacheBlend fork records its native request timestamps with
        # ``time.time()``.  Measure from the wall-clock instant immediately
        # before backend submission to the native first-token timestamp. This
        # includes submission-to-arrival and queueing; first_scheduled_time
        # would omit both the initial submission gap and queueing.
        ttft = r.metrics.first_token_time - backendWallStart
        resp = r.outputs[0]
        token_ids = list(resp.token_ids)
        visible_ids = token_ids[:maxNewTokens]
        retained = self._endRetainOutput() if retain_output else None
        retained_tokens = 0
        if retain_output and retained is not None:
            n = min(len(visible_ids), retained[0][0].shape[0])
            retained_tokens = n
            if n:
                text = self.tokenizer.decode(visible_ids, skip_special_tokens=True)
                self._chunkIds[text] = visible_ids[:n]
                self._chunkKv[text] = [[kv[0][:n], kv[1][:n]] for kv in retained]
            else:
                text = self.tokenizer.decode(visible_ids, skip_special_tokens=True)
        else:
            text = resp.text
        return {
            "ok": True,
            "text": text,
            "ttft": round(float(ttft), 6),
            "generation_start": t0,
            "num_tokens": len(visible_ids),
            "retained_tokens": retained_tokens,
            "total_time": round(float(time.perf_counter() - t0), 6),
            "n_input": len(fullIds),
        }

    def _reuseRatio(self, fullLen: int, reusedTokens: int):
        return round(reusedTokens / fullLen, 6) if fullLen else 0.0

    def _ResolveSegments(self, parts: list):
        """Resolve wire-format parts to token ids and resident GPU KV."""
        if not parts:
            raise ValueError("fuse: no prompt parts")

        segments = []
        fullIds = []
        sampleKv = None
        for part in parts:
            if not isinstance(part, (list, tuple)) or len(part) != 2:
                raise ValueError(f"fuse: invalid part {part!r}")

            cached, text = bool(part[0]), part[1]
            if cached:
                if text not in self._chunkKv:
                    raise ValueError(f"fuse: chunk not collected: {text[:60]!r}")
                ids = self._chunkIds[text]
                kv = self._chunkKv[text]
                if sampleKv is None:
                    sampleKv = kv
            else:
                ids = self.tokenizer.encode(text, add_special_tokens=False)
                kv = None

            if not ids:
                continue
            start = len(fullIds)
            fullIds.extend(ids)
            segments.append((cached, ids, kv, start, len(fullIds)))

        if not segments:
            raise ValueError("fuse: empty prompt")
        if sampleKv is None:
            raise ValueError("fuse: no cached span")
        return segments, fullIds, sampleKv

    def _EnsureAssembledCapacity(self, capacity: int, sampleKv: list) -> None:
        """Allocate one reusable dense GPU K/V buffer per model layer."""
        capacity = int(capacity)
        if capacity <= self._assembledCapacity:
            return
        if capacity > self.args.max_model_len:
            raise ValueError(
                f"fuse input {capacity} exceeds max_model_len "
                f"{self.args.max_model_len}"
            )

        # No request is in flight when reserve/growth runs. Remove the model's
        # views before releasing old buffers, and return their cached blocks to
        # the allocator so a larger contiguous allocation does not overlap it.
        self.engine.model.old_kvs = [[None, None]] * len(self.layers)
        self._assembledKv = []
        self._assembledCapacity = 0
        self._torch.cuda.empty_cache()

        buffers = []
        for layerIndex in range(len(self.layers)):
            sampleK, sampleV = sampleKv[layerIndex]
            buffers.append([
                sampleK.new_empty((capacity, *sampleK.shape[1:])),
                sampleV.new_empty((capacity, *sampleV.shape[1:])),
            ])
        self._assembledKv = buffers
        self._assembledCapacity = capacity

    def Reserve(self, partsBatch: list):
        """Reserve assembly space for the largest request in a run batch."""
        maximum = 0
        sampleKv = None
        for parts in partsBatch or []:
            _, fullIds, resolvedSample = self._ResolveSegments(parts)
            if len(fullIds) > maximum:
                maximum = len(fullIds)
                sampleKv = resolvedSample
        if maximum and sampleKv is not None:
            self._EnsureAssembledCapacity(maximum, sampleKv)
        return {
            "ok": True,
            "capacity": self._assembledCapacity,
            "requested": maximum,
        }

    def Fuse(
        self,
        parts: list,
        retain_output: bool = False,
        max_new_tokens=None,
    ):
        """Fuse cached and fresh spans appearing anywhere in one prompt.

        ``parts`` has the form::

            [
                [True,  cachedText],
                [False, freshText],
                [True,  cachedText],
                [False, freshText],
            ]

        Cached spans use KV collected by :meth:`Collect`.

        The original CacheBlend fork only has a native *trailing* ``suffix``.
        A middle-fresh span and every later token therefore receive a repair
        marker in the check layer's top-k set. After that check layer, the fork
        overwrites those positions with freshly computed KV, so later layers
        do not consume cache captured before the inserted span.
        """
        try:
            segments, fullIds, sampleKv = self._ResolveSegments(parts)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}

        fullLen = len(fullIds)

        # The fork's check path assumes suffix_len > 0 and always recomputes
        # those final tokens. Use a real trailing fresh span when one exists.
        # Otherwise sacrifice exactly one final cached token as the suffix.
        lastCached, lastIds, _, _, _ = segments[-1]
        suffixLen = 1 if lastCached else len(lastIds)
        prefixLen = fullLen - suffixLen

        baseRatio = float(self.args.recomp_ratio)
        repairPlan = _repair_plan(segments, prefixLen, baseRatio)
        forcedPositions = repairPlan["forced"]
        topkCount = repairPlan["topk_count"]
        effectiveRatio = repairPlan["effective_ratio"]

        checkLayers = self.cfm.get("check_layers") or [1]
        checkLayer = checkLayers[0]
        self._EnsureAssembledCapacity(fullLen, sampleKv)
        # The previous check pass replaces each old-K view with a rotated K
        # tensor. Drop those per-request tensors before overwriting the shared
        # assembly buffers for this request.
        self.engine.model.old_kvs = [[None, None]] * len(self.layers)
        oldKvs = []

        for layerIndex in range(len(self.layers)):
            keyBuffer, valueBuffer = self._assembledKv[layerIndex]

            for cached, ids, kv, start, end in segments:
                if cached:
                    keyBuffer[start:end].copy_(kv[layerIndex][0])
                    valueBuffer[start:end].copy_(kv[layerIndex][1])
                    continue

                keyBuffer[start:end].zero_()
                valueBuffer[start:end].zero_()

                # At the first check layer the fork ranks tokens by the squared
                # V difference. Infinity guarantees that every middle-fresh
                # token is selected. Cached C is deliberately left finite: the
                # normal CacheBlend V-diff top-k selects its repair set (rather
                # than forcing a full C-tail recomputation).
                if layerIndex == checkLayer:
                    forceEnd = min(end, prefixLen)
                    if forceEnd > start:
                        valueBuffer[start:forceEnd].fill_(float("inf"))

            oldKvs.append([
                keyBuffer[:fullLen],
                valueBuffer[:fullLen],
            ])

        reusedTokens = sum(
            end - start
            for cached, _, _, start, end in segments
            if cached
        )
        if lastCached:
            reusedTokens = max(0, reusedTokens - suffixLen)

        self.engine.model.old_kvs = oldKvs
        self.cfm["collect"] = False
        self.cfm["check"] = True
        self.cfm["suffix_len"] = suffixLen

        oldRatio = self.cfm.get("recomp_ratio", baseRatio)
        self.cfm["recomp_ratio"] = effectiveRatio
        try:
            resp = self._generateWithRetention(
                fullIds,
                retain_output=retain_output,
                max_new_tokens=max_new_tokens,
            )
        finally:
            self.cfm["recomp_ratio"] = oldRatio

        resp["reuse_ratio"] = self._reuseRatio(fullLen, reusedTokens)
        resp["cacheblend_debug"] = {
            "suffix_len": suffixLen,
            "prefix_len": prefixLen,
            "first_fresh": repairPlan["first_fresh"],
            "repair_tokens": topkCount,
            "forced_tokens": len(forcedPositions),
            "effective_recomp_ratio": effectiveRatio,
            "kv_residency": "gpu",
            "assembled_capacity": self._assembledCapacity,
        }
        selected = self.cfm.get("imp_indices")
        if selected is not None:
            resp["cacheblend_debug"]["selected_indices"] = (
                selected.detach().cpu().tolist()
                if hasattr(selected, "detach") else list(selected)
            )
        return resp

    def FuseSuffix(
        self,
        chunks: list,
        suffix: str,
        retain_output: bool = False,
        max_new_tokens=None,
    ):
        """Legacy contiguous-prefix + fresh-suffix fuse path."""
        if not chunks:
            return {"ok": False, "error": "fuse: no context chunks"}

        ctxIds = []
        ctxKv = []
        for text in chunks:
            if text not in self._chunkIds:
                return {
                    "ok": False,
                    "error": f"fuse: chunk not collected: {text[:60]!r}",
                }
            ctxIds.append(self._chunkIds[text])
            ctxKv.append(self._chunkKv[text])

        nLayers = len(self.layers)
        sufIds = (
            self.tokenizer.encode(suffix, add_special_tokens=False)
            if suffix
            else []
        )

        fullIds = [i for ids in ctxIds for i in ids]
        ctxGpu = [
            [
                self._torch.cat([kv[l][0] for kv in ctxKv]),
                self._torch.cat([kv[l][1] for kv in ctxKv]),
            ]
            for l in range(nLayers)
        ]

        if sufIds:
            qKv = self._collectIds(sufIds)
            fullIds += sufIds
            suffixLen = len(sufIds)
            oldKvs = [
                [
                    self._torch.cat([ctxGpu[l][0], qKv[l][0]]),
                    self._torch.cat([ctxGpu[l][1], qKv[l][1]]),
                ]
                for l in range(nLayers)
            ]
        else:
            suffixLen = len(ctxIds[-1])
            oldKvs = ctxGpu

        self.cfm["collect"] = False
        self.cfm["check"] = True
        self.cfm["suffix_len"] = suffixLen
        self.engine.model.old_kvs = oldKvs

        resp = self._generateWithRetention(
            fullIds,
            retain_output=retain_output,
            max_new_tokens=max_new_tokens,
        )
        reusedTokens = max(0, len(fullIds) - suffixLen)
        resp["reuse_ratio"] = self._reuseRatio(len(fullIds), reusedTokens)
        return resp

    def Full(self, text: str, retain_output: bool = False, max_new_tokens=None):
        """Generate the whole prompt from scratch (no reuse of cached KVs)."""
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            return {"ok": False, "error": "empty prompt"}
        self.cfm["collect"] = False
        self.cfm["check"] = False
        self.engine.model.old_kvs = [[None, None]] * len(self.layers)
        resp = self._generateWithRetention(
            ids,
            retain_output=retain_output,
            max_new_tokens=max_new_tokens,
        )
        resp["reuse_ratio"] = 0.0
        return resp

    def Reset(self):
        self.engine.model.old_kvs = [[None, None]] * len(self.layers)
        self._chunkKv = {}
        self._chunkIds = {}
        self._ClearHackKv()
        return {"ok": True}

    # -------------------------------------------------------------- service
    def Serve(self):
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                op = req.get("op")
                if op == "collect":
                    _stdout(self.Collect(req["chunks"]))
                elif op == "reserve":
                    _stdout(self.Reserve(req.get("parts_batch", [])))
                elif op == "fuse":
                    if "parts" in req:
                        _stdout(
                            self.Fuse(
                                req["parts"],
                                bool(req.get("retain_output", False)),
                                req.get("max_new_tokens"),
                            )
                        )
                    else:
                        _stdout(
                            self.FuseSuffix(
                                req["chunks"],
                                req.get("suffix", ""),
                                bool(req.get("retain_output", False)),
                                req.get("max_new_tokens"),
                            )
                        )
                elif op == "full":
                    _stdout(
                        self.Full(
                            req["text"],
                            bool(req.get("retain_output", False)),
                            req.get("max_new_tokens"),
                        )
                    )
                elif op == "reset":
                    _stdout(self.Reset())
                elif op == "close":
                    _stdout({"ok": True})
                    return
                else:
                    _stdout({"ok": False, "error": f"unknown op {op!r}"})
            except Exception as exc:  # noqa: BLE001 - report, keep serving
                _stdout({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def _stdout(msg: dict) -> None:
    sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def Main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--max_model_len", type=int, default=32768)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    ap.add_argument("--recomp_ratio", type=float, default=0.15)
    ap.add_argument("--sampling_config", default="{}")
    ap.add_argument("--max_num_seqs", type=int, default=64)
    ap.add_argument(
        "--enforce_eager",
        action="store_true",
        help="disable CUDA-graph decode (diagnostic; CacheBlend normally needs graphs)",
    )
    ap.add_argument("--max_collect_tokens", type=int, default=3500,
                    help="token budget per batched collect generate")
    args = ap.parse_args()
    CacheBlendWorker(args).Serve()


if __name__ == "__main__":
    Main()
