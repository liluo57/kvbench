"""CacheBlend worker: a thin driver that *calls* the original CacheBlend code.

This script runs under the original repo's own Python environment
(``~/cache-blend/CacheBlend/.venv``, Python 3.11 + the patched vLLM 0.4.1 in
which CacheBlend's collect / check / fusion machinery lives). KVBench's
:class:`~methods.Cacheblend.CacheBlendMethod` launches it once per method
instance and talks to it over JSON-lines on stdin/stdout:

    {"op": "collect", "text": ...}          cache the KV of one contiguous prefix
    {"op": "collect_chunks", "chunks": []}  cache each knowledge chunk in
                                            isolation (no cross-chunk attention),
                                            concatenated into one cached prefix
    {"op": "fuse", "suffix": ...}           collect the fresh suffix KV, fuse it
                                            against the cached context (check
                                            phase), generate, return (text, ttft)
    {"op": "full", "text": ...}             generate the whole prompt from scratch
                                            (no reuse — shuffle / no warm-up)
    {"op": "reset"}                         drop the cached KVs
    {"op": "close"}                         terminate

The CacheBlend *algorithm* (partial attention, check layers, important-token
recomputation) is not reimplemented here — it lives in the original patched
vLLM (``vllm_blend``) and in the original ``example/blend_niah.py`` /
``blend_musique.py`` scripts. This worker only orchestrates the model hooks the
same way those examples do: set ``cache_fuse_metadata["collect"]`` / ``["check"]``,
read ``layer.self_attn.hack_kv``, and install ``engine.model.old_kvs``.
"""

import argparse
import json
import sys
import time

# The CacheBlend machinery lives in the original repo's patched vLLM
# (vllm_blend). If --repo_root is given, prefer that copy over any other.
def _argValue(name: str) -> str:
    """Value of ``--name value`` or ``--name=value`` from sys.argv (or "")."""
    for i, a in enumerate(sys.argv[1:], start=1):
        if a == name and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return ""


def _installRepo(repoRoot: str) -> None:
    if not repoRoot:
        return
    blend = __import__("pathlib").Path(repoRoot) / "vllm_blend"
    if blend.is_dir():
        sys.path.insert(0, str(blend))


_installRepo(_argValue("--repo_root"))

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def _stdout(msg: dict) -> None:
    sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
    sys.stdout.flush()


class CacheBlendWorker:
    def __init__(self, args):
        self.args = args
        print(f"[cacheblend-worker] loading model {args.model} ...", flush=True)
        self.llm = LLM(
            model=args.model,
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            enforce_eager=args.enforce_eager,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
        self.llm.set_tokenizer(self.tokenizer)

        engine = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
        self.engine = engine
        self.layers = engine.model.layers
        self.cfm = engine.model.cache_fuse_metadata
        self.cfm["recomp_ratio"] = args.recomp_ratio

        self._ctxKv = None     # per-layer [K, V] of the cached context
        self._ctxIds = None    # prompt token ids of the cached context
        print("[cacheblend-worker] ready", flush=True)

    # ------------------------------------------------------------- collect
    def _collectIds(self, ids: list):
        """Run ``ids`` through the model once with collect=True; return per-layer [K,V]."""
        self.cfm["collect"] = True
        self.cfm["check"] = False
        self.llm.generate(
            prompt_token_ids=[ids],
            sampling_params=SamplingParams(temperature=0, max_tokens=1),
        )
        out = []
        for layer in self.layers:
            k, v = layer.self_attn.hack_kv
            out.append([k.clone(), v.clone()])
        return out

    def _checkLen(self, kv, ids: list):
        if kv[0][0].shape[0] != len(ids):
            # The decode step overwrote hack_kv (e.g. eager mode); the KV no
            # longer matches the collected prompt.
            raise RuntimeError(f"collected KV length {kv[0][0].shape[0]} != prompt {len(ids)}")

    def Collect(self, text: str):
        # No BOS: the tasks build the prompt as chat-template text, matching
        # the transformers backend (FullPrefill / Naive) token-for-token.
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        kv = self._collectIds(ids)
        self._checkLen(kv, ids)
        self._ctxIds = ids
        self._ctxKv = kv
        return {"ok": True, "n_tokens": len(ids)}

    def CollectChunks(self, chunks: list):
        idsAll = []
        kvParts = None
        for chunk in chunks:
            cids = self.tokenizer.encode(chunk, add_special_tokens=False)
            if not cids:
                continue
            # Each chunk is collected with a *fresh* cache, so chunk i does not
            # attend to chunks 0..i-1 — the naive knowledge-base setup that
            # CacheBlend's check phase later repairs.
            kv = self._collectIds(cids)
            self._checkLen(kv, cids)
            if kvParts is None:
                kvParts = kv
            else:
                kvParts = [
                    [torch.cat([kvParts[j][0], kv[j][0]]),
                     torch.cat([kvParts[j][1], kv[j][1]])]
                    for j in range(len(self.layers))
                ]
            idsAll += cids
        if kvParts is None:
            return {"ok": False, "error": "all chunks were empty"}
        self._ctxIds = idsAll
        self._ctxKv = kvParts
        return {"ok": True, "n_tokens": len(idsAll)}

    # ---------------------------------------------------------------- fuse
    def _Generate(self, fullIds: list, suffixLen: Optional[int] = None):
        """Decode ``fullIds``, returning the standard result dict."""
        t0 = time.perf_counter()
        out = self.llm.generate(
            prompt_token_ids=[fullIds],
            sampling_params=SamplingParams(
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

    def Fuse(self, suffix: str):
        sufIds = self.tokenizer.encode(suffix, add_special_tokens=False)
        if not sufIds:
            return {"ok": False, "error": "empty suffix"}

        qKv = self._collectIds(sufIds)
        # old_kvs = cached context KVs ++ fresh suffix KVs
        self.engine.model.old_kvs = [
            [
                torch.cat([self._ctxKv[j][0], qKv[j][0]]),
                torch.cat([self._ctxKv[j][1], qKv[j][1]]),
            ]
            for j in range(len(self.layers))
        ]

        fullIds = self._ctxIds + sufIds
        self.cfm["collect"] = False
        self.cfm["check"] = True
        self.cfm["suffix_len"] = len(sufIds)
        return self._Generate(fullIds)

    def Full(self, text: str):
        """Generate the whole prompt from scratch (no reuse of cached KVs)."""
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            return {"ok": False, "error": "empty prompt"}
        self.cfm["collect"] = False
        self.cfm["check"] = False
        self.engine.model.old_kvs = [[None, None]] * len(self.layers)
        return self._Generate(ids)

    def Reset(self):
        self.engine.model.old_kvs = [[None, None]] * len(self.layers)
        self._ctxKv = None
        self._ctxIds = None
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
                    _stdout(self.Collect(req["text"]))
                elif op == "collect_chunks":
                    _stdout(self.CollectChunks(req["chunks"]))
                elif op == "fuse":
                    _stdout(self.Fuse(req["suffix"]))
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


def Main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", default="", help="path to the original CacheBlend repo (for vllm_blend)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--num_layers", type=int, default=28)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.7)
    ap.add_argument("--max_model_len", type=int, default=40960)
    ap.add_argument("--enforce_eager", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--recomp_ratio", type=float, default=0.0)
    args = ap.parse_args()
    CacheBlendWorker(args).Serve()


if __name__ == "__main__":
    Main()
