"""CacheBlend helper: a thin driver that *calls* the original CacheBlend code.

This script is a worker subprocess for :class:`~methods.CacheblendRepo.CacheblendRepo`.
It runs under the **original repo's own venv** (``<RepoPath>/.venv/bin/python``),
whose ``vllm==0.4.1`` is an editable install of the repo's patched ``vllm_blend``
— so ``import vllm`` here resolves to the CacheBlend fork with the collect /
check / fusion machinery, without any ``sys.path`` manipulation.

The framework's main process (the bare conda env, vllm 0.25) never imports this
module's heavy deps: torch / vllm / transformers are imported only inside the
worker class constructor, which runs in this subprocess. Importing the module
from the framework process is therefore a no-op.

KVBench's method launches it once per method instance and talks to it over
JSON-lines on stdin/stdout:

    {"op": "collect", "chunks": [...]}  cache each chunk's KV in isolation
                                        (all chunks of the batch, batched)
    {"op": "fuse", "chunks": [...], "suffix": ""}
        generate a prompt whose context is the chunks, concatenated *in the
        given run order*, fused against their cached KV (check phase).
        ``chunks`` are the run-order context segments; ``suffix`` is fresh
        text after them (empty when the whole prompt is explained by the
        chunks — then the last chunk's tail is treated as the fresh suffix).
    {"op": "full", "text": ...}         generate the whole prompt from scratch
                                        (no reuse — shuffle / no warm-up)
    {"op": "reset"}                     drop the cached per-chunk KVs
    {"op": "close"}                     terminate

The CacheBlend *algorithm* (partial attention, check layers, important-token
recomputation) is not reimplemented here — it lives in the original patched
vLLM. This worker only orchestrates the model hooks the same way the original
``example/blend.py`` does: set ``cache_fuse_metadata["collect"]`` / ``["check"]``,
read ``layer.self_attn.hack_kv``, and install ``model.old_kvs``.

``collect`` is batched: each chunk is its own *sequence* in the generate call
(vLLM keeps sequences isolated, so chunk ``i`` never attends to chunk ``j`` —
the chunk-isolated knowledge-base setup CacheBlend's check phase later
repairs), the prefill's concatenated KV is captured from ``hack_kv`` and split
back by the known chunk lengths. The batch is grouped so each generate's
prefill stays a single forward (≤ ``max_collect_tokens`` < ``max_num_batched_tokens``);
a length mismatch (vLLM chunked the prefill) halves the group and retries.

The ``reuse_ratio`` the method reports is returned here, straight from the cache
state: for a fused run it is the share of the input tokens served from the
cached context KV — ``(len(fullIds) - suffix_len) / len(fullIds)`` — the
official CacheBlend semantics (the whole context comes from cache; the check
phase only repairs attention, so nothing is subtracted for recomp_ratio). A
``full`` run has ``reuse_ratio`` 0.
"""

import argparse
import json
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
        self.llm = LLM(
            model=args.model,
            dtype="auto",
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
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

    def _reuseRatio(self, fullLen: int, suffixLen: int):
        return round((fullLen - suffixLen) / fullLen, 6) if fullLen else 0.0

    def Fuse(self, chunks: list, suffix: str):
        """Generate the run prompt from the cached chunk KVs in ``chunks`` order.

        ``chunks`` are the context segments in their *run* order (a prefix
        order for prefix-reuse cases, a re-detected order for shuffled ones);
        their cached KVs are concatenated in that order and fused against the
        fresh suffix (``suffix`` non-empty) — or, when the whole run is
        explained by ``chunks``, the last chunk's tail is the "suffix" that is
        always recomputed (it ends at the prompt's last token, which the check
        run must repair to produce correct logits).
        """
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
        # The stored chunk KVs are on CPU (collected for the whole batch, see
        # ``_CollectGroup``); bring only this case's chunks to the device and
        # concatenate them there.
        ctxGpu = [
            [
                self._torch.cat([kv[l][0] for kv in ctxKv]).to(self.device),
                self._torch.cat([kv[l][1] for kv in ctxKv]).to(self.device),
            ]
            for l in range(nLayers)
        ]
        if sufIds:
            # Fresh suffix KV, collected in isolation and appended so old_kvs
            # spans the whole prompt (the check layer compares value_old[:-len]
            # against the fresh value). qKv is already on the device.
            qKv = self._collectIds(sufIds)
            fullIds += sufIds
            suffixLen = len(sufIds)
            old_kvs = [
                [
                    self._torch.cat([ctxGpu[l][0], qKv[l][0]]),
                    self._torch.cat([ctxGpu[l][1], qKv[l][1]]),
                ]
                for l in range(nLayers)
            ]
        else:
            suffixLen = len(ctxIds[-1])
            old_kvs = ctxGpu

        self.cfm["collect"] = False
        self.cfm["check"] = True
        self.cfm["suffix_len"] = suffixLen
        self.engine.model.old_kvs = old_kvs
        resp = self._Generate(fullIds)
        resp["reuse_ratio"] = self._reuseRatio(len(fullIds), suffixLen)
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
                    _stdout(self.Fuse(req["chunks"], req.get("suffix", "")))
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
