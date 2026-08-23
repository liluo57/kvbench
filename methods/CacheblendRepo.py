"""CacheBlend method — *calls* the original CacheBlend repo, not a copy.

The CacheBlend algorithm (collect phase, check phase, important-token
recomputation) is implemented in the original authors' patched vLLM
(``vllm_blend``, v0.4.1) and driven by their ``example/blend.py``. This method
does not reimplement any of it: it launches a persistent worker subprocess
under the original repo's own venv (``helpers/CacheblendRepoHelper.py``,
run with ``<RepoPath>/.venv/bin/python``) and orchestrates it over JSON-lines,
exactly the way ``example/blend.py`` drives the model hooks.

The main process stays on the framework's bare conda env (vllm 0.25): only the
worker subprocess imports the repo's patched vLLM, so this method cannot pollute
the other methods' environment. The worker's ``sys.path`` is its own.

``reuse_ratio`` (the method's metric) is reported by the worker from the cache
state: the share of the input tokens served from the cached context KV
(``(len - suffix_len) / len``) for a fused run, 0 for a full run. This follows
the official CacheBlend semantics — the whole context comes from cache and the
check phase only repairs attention, so nothing is subtracted for ``recompRatio``.

Batch scope: the collect phase (the expensive per-chunk prefill) is truly
batched. The fused *run* is one ``generate`` per case — the fork's check phase
(global ``model.old_kvs`` / ``cache_fuse_metadata`` and the single-sequence
``selected_token_indices[0]`` logits hack) is inherently single-sequence, so a
batch of fused runs cannot be submitted together without rewriting the fork.

The constructor takes the task's available GPUs (``gpu_ids``, equivalent to
``CUDA_VISIBLE_DEVICES``), the recomputation ratio ``recompRatio`` (0.15 default;
>0 repairs cross-chunk attention in a chunk-isolated knowledge base), and
``fullPrefill`` — when True every query is a plain full prefill (no cache, no
fusion), serving as the control group against the fused runs. The repo and model
paths come **only** from ``config.yaml`` (``Cacheblend.Repo.RepoPath``
/ ``Cacheblend.Repo.ModelPath``); the constructor raises if either is missing.
"""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from core.Config import Get
from core.Method import Method
from core.Result import NumOutputTokensKey, Result, TotalTimeKey, TtftKey

from helpers.Prompt import ComposeInterleavedReuse

#: Worker readiness marker printed on the helper's stdout at startup.
_ReadyLine = "[cacheblend-helper] ready"


class CacheblendRepo(Method):
    name = "cacheblend_repo"

    #: The share of the input actually served from the cached context KV, as
    #: reported by the worker (0 for full-prefill runs). Mirrors the framework's
    #: cacheblend method metric; Naive / FullPrefill leave this empty.
    method_metrics = ("reuse_ratio",)

    def __init__(
        self,
        gpuIds: Union[str, List[int]] = "0",
        *,
        maxNewTokens: int = 64,
        maxModelLen: int = 32768,
        gpuMemoryUtilization: float = 0.7,
        recompRatio: float = 0.15,
        fullPrefill: bool = False,
        startTimeout: float = 1800.0,
        tag: Optional[str] = None,
    ):
        super().__init__(tag=tag)
        self.gpuIds = gpuIds
        self.maxNewTokens = maxNewTokens
        self.maxModelLen = maxModelLen
        self.gpuMemoryUtilization = gpuMemoryUtilization
        self.recompRatio = recompRatio
        self.fullPrefill = fullPrefill
        self.startTimeout = startTimeout

        # Repo + model paths come only from config.yaml; no defaults.
        repo = (Get("Cacheblend", {}) or {}).get("Repo", {}) or {}
        repoPath = repo.get("RepoPath")
        modelPath = repo.get("ModelPath")
        if not repoPath or not modelPath:
            raise RuntimeError(
                "CacheblendRepo: Cacheblend.Repo.RepoPath / Cacheblend.Repo."
                "ModelPath are missing in config.yaml"
            )
        self.repoRoot = Path(str(repoPath)).expanduser()
        self.modelPath = str(modelPath)
        self.workerPython = self.repoRoot / ".venv" / "bin" / "python"
        if not self.workerPython.exists():
            raise FileNotFoundError(
                f"CacheblendRepo: worker python not found: {self.workerPython} "
                f"(expected the original repo's venv under Cacheblend.Repo.RepoPath)"
            )

        self._proc: Optional[subprocess.Popen] = None
        self._drainThread = None
        #: Per-case warm-up chunks from the last :meth:`Prepare` call
        #: (empty chunks filtered out; used only to detect reuse in Run).
        self._chunks: List[List[str]] = []
        self._startWorker()

    # ------------------------------------------------------------- worker io
    def _drainStderr(self) -> None:
        """Forward the worker's stderr (vLLM logs) to our own stderr."""
        assert self._proc is not None and self._proc.stderr is not None
        try:
            for line in self._proc.stderr:
                sys.stderr.write(f"[cacheblend-helper] {line}")
                sys.stderr.flush()
        except Exception:  # noqa: BLE001 - pipe closed on shutdown
            pass

    def _startWorker(self) -> None:
        helperScript = Path(__file__).resolve().parent.parent / "helpers" / "CacheblendRepoHelper.py"
        env = dict(os.environ)
        # Keep the subprocess clean: the repo's venv must resolve its own vllm,
        # and a login shell's LD_PRELOAD / PYTHONPATH must not leak in.
        env.pop("LD_PRELOAD", None)
        env.pop("PYTHONPATH", None)
        env["CUDA_VISIBLE_DEVICES"] = (
            str(self.gpuIds)
            if isinstance(self.gpuIds, str)
            else ",".join(str(g) for g in self.gpuIds)
        )
        self._proc = subprocess.Popen(
            [
                str(self.workerPython),
                str(helperScript),
                "--repo_root", str(self.repoRoot),
                "--model", self.modelPath,
                "--max_new_tokens", str(self.maxNewTokens),
                "--max_model_len", str(self.maxModelLen),
                "--gpu_memory_utilization", str(self.gpuMemoryUtilization),
                "--recomp_ratio", str(self.recompRatio),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            bufsize=1,
        )
        # The helper's vLLM logs a lot to stderr (progress bars, INFO). The pipe
        # must be drained or it fills (64KB) and the worker deadlocks on write.
        self._drainThread = threading.Thread(
            target=self._drainStderr, daemon=True, name="cb-helper-stderr"
        )
        self._drainThread.start()
        # Wait until the worker has loaded the model and is ready to serve.
        ready = False
        deadline = time.time() + self.startTimeout
        while time.time() < deadline:
            line = self._proc.stdout.readline()
            if not line:
                if self._proc.poll() is not None:
                    err = self._proc.stderr.read() or "(no stderr)"
                    raise RuntimeError(f"cacheblend helper exited during startup: {err}")
                continue
            line = line.strip()
            if line.startswith(_ReadyLine):
                ready = True
                break
            # forward any other startup logging
            print(line, flush=True)
        if not ready:
            self._proc.kill()
            raise TimeoutError("cacheblend helper did not become ready")

    def _Request(self, payload: Dict[str, Any], timeout: float = 3600.0) -> Dict[str, Any]:
        self._proc.stdin.write(json.dumps(payload) + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            err = (self._proc.stderr.read() or "")[-2000:]
            raise RuntimeError(f"cacheblend helper closed stdout: {err}")
        resp = json.loads(line)
        if not resp.get("ok"):
            raise RuntimeError(f"cacheblend helper error: {resp.get('error')}")
        return resp

    # ---------------------------------------------------------------- Method
    def Prepare(self, data: List[List[str]]) -> None:
        """Collect every chunk's KV in the worker, batched across the batch.

        The chunks of all cases are sent in one ``collect`` request; the worker
        groups them by token budget and prefills each group as one batched
        generate (each chunk = its own isolated sequence), then splits the
        captured KV back per chunk. ``Run`` later fuses them in each prompt's
        own (possibly shuffled) order — the per-chunk prefill is setup, not
        per-query work. Skipped entirely for ``fullPrefill`` control runs.
        """
        self._chunks = [[c for c in (chunks or []) if c] for chunks in data]
        if self.fullPrefill:
            return
        flat: List[str] = []
        seen = set()
        for chunks in self._chunks:
            for c in chunks:
                if c not in seen:
                    seen.add(c)
                    flat.append(c)
        if flat:
            self._Request({"op": "collect", "chunks": flat})

    def Run(self, data: List[str]) -> List[Result]:
        """Run a batch of prompts, fusing cached and fresh spans in prompt order."""
        results: List[Result] = []

        for prompt, chunks in zip(data, self._chunks):
            if self.fullPrefill:
                results.append(self._runFull(prompt))
                continue

            parts, reordered = self._splitForFuse(chunks, prompt)

            if parts is None:
                results.append(self._runFull(prompt))
                continue

            resp = self._Request(
                {
                    "op": "fuse",
                    "parts": [
                        [prepareIndex is not None, text]
                        for prepareIndex, text in parts
                    ],
                }
            )

            results.append(
                self._Result(
                    resp,
                    full=False,
                    reordered=reordered,
                )
            )

        return results


    def _splitForFuse(self, chunks: List[str], prompt: str):
        """Return interleaved cached/fresh parts for a fused run.

        Returns ``(parts, reordered)`` when at least one prepared chunk occurs in
        the prompt. ``parts`` is the result of ``ComposeInterleavedReuse``.

        Returns ``(None, False)`` when nothing can be reused, in which case the
        caller performs a full prefill.
        """
        parts = ComposeInterleavedReuse(chunks, prompt)

        reuseOrder = [
            chunks[prepareIndex]
            for prepareIndex, _ in parts
            if prepareIndex is not None
        ]

        if not reuseOrder:
            return None, False

        return parts, reuseOrder != chunks

    def _runFull(self, prompt: str) -> Result:
        resp = self._Request({"op": "full", "text": prompt})
        return self._Result(resp, full=True)

    def _Result(self, resp: Dict[str, Any], *, full: bool, reordered: bool = False) -> Result:
        metadata: Dict[str, Any] = {
            "reuse_ratio": resp.get("reuse_ratio", 0.0),
            "recomp_ratio": self.recompRatio,
            "n_input": resp.get("n_input"),
        }
        if full:
            metadata["full_prefill"] = True
        if reordered:
            metadata["reordered"] = True
        return Result(
            output=resp["text"],
            performance={
                TtftKey: resp["ttft"],
                NumOutputTokensKey: resp["num_tokens"],
                TotalTimeKey: resp.get("total_time", resp["ttft"]),
            },
            metadata=metadata,
        )

    def Reset(self) -> None:
        self._chunks = []
        try:
            self._Request({"op": "reset"})
        except (BrokenPipeError, RuntimeError):
            self._restartWorker()

    def Close(self) -> None:
        if self._proc is not None:
            try:
                self._proc.stdin.write(json.dumps({"op": "close"}) + "\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=30)
            except Exception:  # noqa: BLE001
                self._proc.kill()
            self._proc = None

    def _restartWorker(self) -> None:
        self.Close()
        self._startWorker()

    def __del__(self):
        try:
            self.Close()
        except Exception:  # noqa: BLE001
            pass
