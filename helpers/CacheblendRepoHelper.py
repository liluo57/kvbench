"""CacheBlend helper: thin subprocess driver for the original CacheBlend repo.

This worker is launched by :class:`~methods.CacheblendRepo.CacheblendRepo` under
the original CacheBlend repo's own venv (patched vLLM 0.4.1). KVBench talks to
it over JSON-lines:

    {"op": "collect", "chunks": [...]}
        Cache every reusable chunk independently. Collection is batched and the
        resulting per-chunk KV is kept on CPU between runs.

    {"op": "fuse", "parts": [[cached, text], ...]}
        Generate an interleaved prompt. ``cached=True`` spans use previously
        collected KV; ``cached=False`` spans are fresh. Fresh spans before the
        trailing suffix are forced into CacheBlend's recomputation set.

    {"op": "full", "text": ...}
        Generate the whole prompt from scratch.

    {"op": "reset"}
        Drop all cached per-chunk KV.

    {"op": "close"}
        Terminate the worker.

The legacy ``fuse/chunks/suffix`` request is still accepted temporarily for
compatibility.

The CacheBlend algorithm itself remains in the authors' patched vLLM. This
helper only orchestrates its hooks: ``cache_fuse_metadata``, per-layer
``hack_kv``, and ``model.old_kvs``.

``reuse_ratio`` is computed from the spans actually supplied by stored cached
KV, divided by the full input length. CacheBlend's selective repair does not
reduce this metric, matching the previous CacheBlend semantics. A final cached
token that is deliberately used as the fork-required native suffix is not
counted as reused because it is fully recomputed.
"""

import argparse
import json
import os
import sys
import time


class CacheBlendWorker:
    def __init__(self, args):
        # Heavy deps load only inside this subprocess (see module docstring).
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        self.args = args
        self._torch = __import__("torch")
        print(f"[cacheblend-helper] loading model {args.model} ...", flush=True)
        # The CacheBlend hooks live only in the xformers attention backend; the
        # fork's default already resolves there (no flash_attn in this venv),
        # but pin it so a future flash_attn install cannot silently switch
        # backends.
        os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
        # Qwen3 is not known to the fork's vLLM 0.4.1 / transformers 4.44.2;
        # the out-of-tree model in Qwen3ForCacheBlendRepo registers itself when
        # the model's config.json declares Qwen3ForCausalLM. Mistral (the fork's
        # native model) goes through untouched.
        self._registerOutOfTreeModel(args.model)
        # dtype="auto" loads the model's own config dtype (Mistral-7B-Instruct
        # v0.2 is bfloat16). enforce_eager is left unset on purpose: CacheBlend's
        # collect reads hack_kv captured during the prefill forward, and the
        # single decode step must NOT re-run the forward (eager would overwrite
        # hack_kv) — CUDA-graph decode replays instead, preserving it.
        # max_num_seqs bounds the KV cache (the stored per-chunk KVs live
        # outside it). max_num_batched_tokens is left to vLLM's default
        # (max(max_model_len, 2048) = max_model_len here), which guarantees the
        # batched collect's groups (<= max_collect_tokens) prefill in a single
        # forward.
        #
        # Qwen3's tokenizer.json was serialized by a newer ``tokenizers`` than
        # this venv (0.19.x) can parse, so Qwen3 runs on the slow BPE tokenizer
        # (vocab.json + merges.txt); Mistral keeps its fast tokenizer.
        self.llm = LLM(
            model=args.model,
            dtype="auto",
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            **({"tokenizer_mode": "slow"} if self._isQwen3 else {}),
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model, **({"use_fast": False} if self._isQwen3 else {}))
        self.llm.set_tokenizer(self.tokenizer)
        self.sampling_params = SamplingParams

        engine = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
        self.engine = engine
        self.layers = engine.model.layers
        self.device = next(engine.parameters()).device
        self.cfm = engine.model.cache_fuse_metadata
        self.cfm["recomp_ratio"] = args.recomp_ratio

        #: Per-chunk cached KV, keyed by the chunk's exact text: chunk text ->
        #: per-layer ``[[k, v], ...]``. Collected once per batch in ``collect``,
        #: consumed in ``fuse`` in whatever (run) order the method needs. Stored
        #: on *CPU* so a whole batch's chunks do not exhaust GPU memory — only
        #: one case's chunks move to GPU per fused run.
        self._chunkKv = {}
        #: Per-chunk token ids, keyed the same way.
        self._chunkIds = {}
        print("[cacheblend-helper] ready", flush=True)

    def _registerOutOfTreeModel(self, model: str) -> None:
        """Register the model's architecture if the fork does not know it.

        Reads ``config.json`` at the model path (a local modelscope / HF
        checkout) and, for architectures the fork's registry lacks, imports and
        runs the matching out-of-tree registration. The fork ships no Qwen3, so
        this is where ``Qwen3ForCausalLM`` is wired in — additively, without
        touching the original repo.
        """
        self._isQwen3 = False
        cfgPath = os.path.join(model, "config.json")
        try:
            with open(cfgPath) as f:
                archs = json.load(f).get("architectures", [])
        except OSError:
            return
        if "Qwen3ForCausalLM" in archs:
            self._isQwen3 = True
            import Qwen3ForCacheBlendRepo
            Qwen3ForCacheBlendRepo.register_qwen3()
            print("[cacheblend-helper] registered out-of-tree "
                  "Qwen3ForCausalLM", flush=True)

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
            sampling_params=self.sampling_params(temperature=0, max_tokens=1),
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
            sampling_params=self.sampling_params(temperature=0, max_tokens=1),
        )
        kvBatch = [
            [layer.self_attn.hack_kv[0].clone(), layer.self_attn.hack_kv[1].clone()]
            for layer in self.layers
        ]
        if kvBatch[0][0].shape[0] != total:
            if len(idsList) == 1 or depth >= 3:
                raise RuntimeError(
                    f"batch collect: captured {kvBatch[0][0].shape[0]} KV tokens "
                    f"for {total} input tokens ({len(idsList)} sequences); "
                    f"the prefill was chunked and hack_kv only holds the last "
                    f"forward"
                )
            mid = len(idsList) // 2
            self._CollectGroup(texts[:mid], idsList[:mid], depth + 1)
            self._CollectGroup(texts[mid:], idsList[mid:], depth + 1)
            return
        off = 0
        for text, ids in zip(texts, idsList):
            n = len(ids)
            # Store on CPU: the whole batch's chunks are collected here, and
            # holding them all on GPU is what OOM'd the 32-case benchmark
            # (~128KB/token for Mistral-7B). Only the current case's chunks
            # move back to the device in ``Fuse``.
            kv = [
                [
                    kvBatch[j][0][off:off + n].clone().cpu(),
                    kvBatch[j][1][off:off + n].clone().cpu(),
                ]
                for j in range(len(self.layers))
            ]
            self._chunkIds[text] = ids
            self._chunkKv[text] = kv
            off += n

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
    def _Generate(self, fullIds: list):
        """Decode ``fullIds``, returning the standard result dict."""
        t0 = time.perf_counter()
        out = self.llm.generate(
            prompt_token_ids=[fullIds],
            sampling_params=self.sampling_params(
                temperature=0, max_tokens=self.args.max_new_tokens
            ),
        )
        r = out[0]
        ttft = r.metrics.first_token_time - r.metrics.first_scheduled_time
        resp = r.outputs[0]
        return {
            "ok": True,
            "text": resp.text,
            "ttft": round(float(ttft), 6),
            "num_tokens": len(resp.token_ids),  # vllm 0.4.1 attribute name
            "total_time": round(float(time.perf_counter() - t0), 6),
            "n_input": len(fullIds),
        }

    def _reuseRatio(self, fullLen: int, reusedTokens: int):
        return round(reusedTokens / fullLen, 6) if fullLen else 0.0

    def Fuse(self, parts: list):
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
        Middle-fresh tokens therefore receive placeholder old KV and are forced
        into the check layer's top-k recomputation set. After that check layer,
        the fork overwrites those positions with their freshly computed KV, so
        later layers see the real fresh tokens.
        """
        if not parts:
            return {"ok": False, "error": "fuse: no prompt parts"}

        segments = []
        fullIds = []
        sampleKv = None

        # Resolve every segment first. Cached KV stays on CPU here; the complete
        # per-layer old_kvs tensor is moved to GPU only after concatenation.
        for part in parts:
            if not isinstance(part, (list, tuple)) or len(part) != 2:
                return {
                    "ok": False,
                    "error": f"fuse: invalid part {part!r}",
                }

            cached, text = part
            cached = bool(cached)

            if cached:
                if text not in self._chunkKv:
                    return {
                        "ok": False,
                        "error": f"fuse: chunk not collected: {text[:60]!r}",
                    }
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
            end = len(fullIds)
            segments.append((cached, ids, kv, start, end))

        if not segments:
            return {"ok": False, "error": "fuse: empty prompt"}

        if sampleKv is None:
            return {"ok": False, "error": "fuse: no cached span"}

        fullLen = len(fullIds)

        # The fork's check path assumes suffix_len > 0 and always recomputes
        # those final tokens. Use a real trailing fresh span when one exists.
        # Otherwise sacrifice exactly one final cached token as the suffix.
        lastCached, lastIds, _, _, _ = segments[-1]
        suffixLen = 1 if lastCached else len(lastIds)
        prefixLen = fullLen - suffixLen

        # Count genuinely fresh tokens that lie before the native suffix. Every
        # one of them must be selected by the check layer.
        freshPrefixTokens = 0
        for cached, _, _, start, end in segments:
            if cached:
                continue
            overlapEnd = min(end, prefixLen)
            if overlapEnd > start:
                freshPrefixTokens += overlapEnd - start

        cachedPrefixTokens = max(0, prefixLen - freshPrefixTokens)

        # Preserve the configured CacheBlend repair ratio for cached positions,
        # then add all middle-fresh positions to the same top-k budget.
        baseRatio = float(self.args.recomp_ratio)
        repairCachedTokens = int(cachedPrefixTokens * baseRatio)
        repairCachedTokens = max(
            0,
            min(cachedPrefixTokens, repairCachedTokens),
        )

        topkCount = freshPrefixTokens + repairCachedTokens
        topkCount = max(0, min(prefixLen, topkCount))

        if prefixLen <= 0 or topkCount <= 0:
            effectiveRatio = 0.0
        elif topkCount >= prefixLen:
            effectiveRatio = 1.0
        else:
            # The fork later executes int(prefixLen * recomp_ratio). Choose a
            # value safely inside the interval that maps back to topkCount.
            effectiveRatio = (topkCount + 0.5) / prefixLen

        checkLayers = self.cfm.get("check_layers") or [1]
        checkLayer = checkLayers[0]
        oldKvs = []

        for layerIndex in range(len(self.layers)):
            sampleK = sampleKv[layerIndex][0]
            sampleV = sampleKv[layerIndex][1]
            keyParts = []
            valueParts = []

            for cached, ids, kv, start, end in segments:
                if cached:
                    keyParts.append(kv[layerIndex][0])
                    valueParts.append(kv[layerIndex][1])
                    continue

                shape = (len(ids), *sampleK.shape[1:])
                keyPart = sampleK.new_zeros(shape)
                valuePart = sampleV.new_zeros(shape)

                # At the first check layer the fork ranks tokens by the squared
                # V difference. Infinity guarantees that every middle-fresh
                # token ranks ahead of ordinary cached repair candidates. The
                # trailing suffix is already selected by suffix_len.
                if layerIndex == checkLayer:
                    forceEnd = min(end, prefixLen)
                    if forceEnd > start:
                        valuePart[:forceEnd - start].fill_(float("inf"))

                keyParts.append(keyPart)
                valueParts.append(valuePart)

            oldKvs.append(
                [
                    self._torch.cat(keyParts).to(self.device),
                    self._torch.cat(valueParts).to(self.device),
                ]
            )

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
            resp = self._Generate(fullIds)
        finally:
            self.cfm["recomp_ratio"] = oldRatio

        resp["reuse_ratio"] = self._reuseRatio(fullLen, reusedTokens)
        return resp

    def FuseSuffix(self, chunks: list, suffix: str):
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
                self._torch.cat([kv[l][0] for kv in ctxKv]).to(self.device),
                self._torch.cat([kv[l][1] for kv in ctxKv]).to(self.device),
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

        resp = self._Generate(fullIds)
        reusedTokens = max(0, len(fullIds) - suffixLen)
        resp["reuse_ratio"] = self._reuseRatio(len(fullIds), reusedTokens)
        return resp

    def Full(self, text: str):
        """Generate the whole prompt from scratch (no reuse of cached KVs)."""
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            return {"ok": False, "error": "empty prompt"}
        self.cfm["collect"] = False
        self.cfm["check"] = False
        self.engine.model.old_kvs = [[None, None]] * len(self.layers)
        resp = self._Generate(ids)
        resp["reuse_ratio"] = 0.0
        return resp

    def Reset(self):
        self.engine.model.old_kvs = [[None, None]] * len(self.layers)
        self._chunkKv = {}
        self._chunkIds = {}
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
                elif op == "fuse":
                    if "parts" in req:
                        _stdout(self.Fuse(req["parts"]))
                    else:
                        _stdout(
                            self.FuseSuffix(
                                req["chunks"],
                                req.get("suffix", ""),
                            )
                        )
                elif op == "full":
                    _stdout(self.Full(req["text"]))
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
    ap.add_argument("--repo_root", default="", help="path to the original CacheBlend repo (informational; the venv's editable vllm resolves vllm_blend)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--max_model_len", type=int, default=32768)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.7)
    ap.add_argument("--recomp_ratio", type=float, default=0.15)
    ap.add_argument("--max_num_seqs", type=int, default=64)
    ap.add_argument("--max_collect_tokens", type=int, default=3500,
                    help="token budget per batched collect generate")
    args = ap.parse_args()
    CacheBlendWorker(args).Serve()


if __name__ == "__main__":
    Main()
