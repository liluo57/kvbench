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
spans supplied to the fuse request: cached input tokens divided by full input
tokens, with the deliberately recomputed native suffix excluded. This remains
meaningful for interleaved parts; the old ``(len - suffix_len) / len`` formula
was suffix-only.

Batch scope: the collect phase (the expensive per-chunk prefill) is truly
batched. The fused *run* is one ``generate`` per case — the fork's check phase
(global ``model.old_kvs`` / ``cache_fuse_metadata`` and the single-sequence
``selected_token_indices[0]`` logits hack) is inherently single-sequence, so a
batch of fused runs cannot be submitted together without rewriting the fork.

The constructor declares a strict GPU count, the relative scheduling weight,
and the recomputation ratio ``recompRatio`` (0.15 default;
>0 repairs cross-chunk attention in a chunk-isolated knowledge base), and
``fullPrefill`` — when True every query is a plain full prefill (no cache, no
fusion), serving as the control group against the fused runs. The repo path
comes **only** from ``config.yaml`` (``Cacheblend.Repo.RepoPath``); the model
path comes from the framework-wide top-level ``ModelPath`` (same as every other
method). The constructor raises if the repo path is missing.
"""

import json
import os
import subprocess
import sys
import threading
import time
from collections import deque
from contextlib import suppress
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from core.Config import Get, ModelPath as DefaultModelPath
from core.Method import Method
from core.Result import NumOutputTokensKey, Result, TotalTimeKey, TtftKey

from helpers.backends.Prompt import ComposeInterleavedReuse

#: Worker readiness marker printed on the helper's stdout at startup.
_ReadyLine = "[cacheblend-helper] ready"


class CacheblendRepo(Method):
    name = "cacheblend_repo"

    # The patched fork's fusion/check state is global to one sequence. Keeping
    # several Cases in one Engine batch only retains all of their chunk KV on
    # GPU while Run still serves them sequentially.
    maxCaseBatchSize = 1

    #: The share of the input actually served from the cached context KV, as
    #: reported by the worker (0 for full-prefill runs). Mirrors the framework's
    #: cacheblend method metric. Naive records the same metadata diagnostically
    #: but does not declare it as a method metric; FullPrefill does not record it.
    method_metrics = ("reuse_ratio",)

    def __init__(
        self,
        gpuNums: int = 1,
        perfWeight: float = 1.0,
        *,
        maxNewTokens: int = 64,
        maxModelLen: int = 32768,
        gpuMemoryUtilization: float = 0.7,
        recompRatio: float = 0.15,
        fullPrefill: bool = False,
        startTimeout: float = 1800.0,
        tag: Optional[str] = None,
    ):
        super().__init__(
            gpuNums=gpuNums,
            perfWeight=perfWeight,
            maxGpuNums=1,
            tag=tag,
        )
        self.maxNewTokens = maxNewTokens
        self.maxModelLen = maxModelLen
        self.gpuMemoryUtilization = gpuMemoryUtilization
        self.recompRatio = recompRatio
        self.fullPrefill = fullPrefill
        self.startTimeout = startTimeout

        # Repo path comes from config.yaml; model path from the framework-wide
        # ``ModelPath`` (the same source every other method uses).
        repo = (Get("Cacheblend", {}) or {}).get("Repo", {}) or {}
        repoPath = repo.get("RepoPath")
        if not repoPath:
            raise RuntimeError(
                "CacheblendRepo: Cacheblend.Repo.RepoPath is missing in config.yaml"
            )
        self.repoRoot = Path(str(repoPath)).expanduser()
        self.modelPath = DefaultModelPath()
        self.workerPython = self.repoRoot / ".venv" / "bin" / "python"
        if not self.workerPython.exists():
            raise FileNotFoundError(
                f"CacheblendRepo: worker python not found: {self.workerPython} "
                f"(expected the original repo's venv under Cacheblend.Repo.RepoPath)"
            )

        self._proc: Optional[subprocess.Popen] = None
        self._drainThread = None
        self._stderrTail = deque(maxlen=200)
        #: Per-case warm-up chunks from the last :meth:`Prepare` call
        #: (empty chunks filtered out; used only to detect reuse in Run).
        self._chunks: List[List[str]] = []

    def Initialize(self, gpuIds: Sequence[int]) -> None:
        super().Initialize(gpuIds)
        self._startWorker()

    # ------------------------------------------------------------- worker io
    def _drainStderr(self) -> None:
        """Forward the worker's stderr (vLLM logs) to our own stderr."""
        proc = self._proc
        assert proc is not None and proc.stderr is not None
        try:
            for line in proc.stderr:
                self._stderrTail.append(line)
                sys.stderr.write(f"[cacheblend-helper] {line}")
                sys.stderr.flush()
        except Exception:  # noqa: BLE001 - pipe closed on shutdown
            pass

    def _startWorker(self) -> None:
        self._stderrTail.clear()
        helperScript = Path(__file__).resolve().parent.parent / "helpers" / "cacheblend_repo" / "CacheblendRepoHelper.py"
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
        print(f"[cacheblend-method] helper pid={self._proc.pid}", flush=True)
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
                    if self._drainThread is not None:
                        self._drainThread.join(timeout=0.5)
                    err = self._StderrTail() or "(no stderr)"
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
            if self._drainThread is not None:
                self._drainThread.join(timeout=0.5)
            err = self._StderrTail()[-2000:]
            raise RuntimeError(f"cacheblend helper closed stdout: {err}")
        resp = json.loads(line)
        if not resp.get("ok"):
            raise RuntimeError(f"cacheblend helper error: {resp.get('error')}")
        return resp

    def _StderrTail(self) -> str:
        """Return stderr already consumed by the sole pipe-drain thread."""
        return "".join(self._stderrTail)

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

    def Run(self, data: List[str], retainOutput: Optional[List[bool]] = None) -> List[Result]:
        """Run a batch of prompts, fusing cached and fresh spans in prompt order."""
        if len(self._chunks) != len(data):
            self._chunks = [[] for _ in data]
        results: List[Result] = []

        # Resolve every prompt before serving the batch. The helper uses these
        # plans to allocate one right-sized GPU assembly buffer up front, so a
        # later longer request never grows the buffer inside its measured TTFT.
        plans = []
        reserveParts = []
        if not self.fullPrefill:
            for prompt, chunks in zip(data, self._chunks):
                parts, reordered = self._splitForFuse(chunks, prompt)
                wireParts = (
                    [
                        [prepareIndex is not None, text]
                        for prepareIndex, text in parts
                    ]
                    if parts is not None
                    else None
                )
                plans.append((wireParts, reordered))
                if wireParts is not None:
                    reserveParts.append(wireParts)
            if reserveParts:
                self._Request({"op": "reserve", "parts_batch": reserveParts})

        for i, (prompt, chunks) in enumerate(zip(data, self._chunks)):
            retain = bool(retainOutput[i]) if retainOutput is not None and i < len(retainOutput) else False
            if self.fullPrefill:
                result = self._runFull(prompt, retain_output=retain)
                results.append(result)
                if retain and result.output and result.output not in chunks:
                    chunks.append(result.output)
                continue

            wireParts, reordered = plans[i]

            if wireParts is None:
                result = self._runFull(prompt, retain_output=retain)
                results.append(result)
                if retain and result.output and result.output not in chunks:
                    chunks.append(result.output)
                continue

            resp = self._Request(
                {
                    "op": "fuse",
                    "parts": wireParts,
                    "retain_output": retain,
                }
            )

            result = self._Result(resp, full=False, reordered=reordered)
            results.append(result)
            if retain and result.output and result.output not in chunks:
                chunks.append(result.output)

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

    def _runFull(self, prompt: str, *, retain_output: bool = False) -> Result:
        resp = self._Request({"op": "full", "text": prompt, "retain_output": retain_output})
        return self._Result(resp, full=True)

    def _Result(self, resp: Dict[str, Any], *, full: bool, reordered: bool = False) -> Result:
        metadata: Dict[str, Any] = {
            "reuse_ratio": resp.get("reuse_ratio", 0.0),
            "recomp_ratio": self.recompRatio,
            "n_input": resp.get("n_input"),
        }
        if "cacheblend_debug" in resp:
            metadata["cacheblend_debug"] = resp["cacheblend_debug"]
        if "retained_tokens" in resp:
            metadata["retained_tokens"] = resp["retained_tokens"]
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
        proc = self._proc
        self._proc = None
        if proc is not None:
            try:
                if proc.poll() is None and proc.stdin is not None:
                    proc.stdin.write(json.dumps({"op": "close"}) + "\n")
                    proc.stdin.flush()
                proc.wait(timeout=30)
            except Exception:  # noqa: BLE001
                with suppress(Exception):
                    proc.kill()
                with suppress(Exception):
                    proc.wait(timeout=5)
            finally:
                for streamName in ("stdin", "stdout", "stderr"):
                    stream = getattr(proc, streamName, None)
                    if stream is not None:
                        with suppress(BrokenPipeError, OSError, ValueError):
                            stream.close()
                        setattr(proc, streamName, None)
                if self._drainThread is not None:
                    self._drainThread.join(timeout=1)
                self._drainThread = None

    def _restartWorker(self) -> None:
        self.Close()
        self._startWorker()

    def __del__(self):
        try:
            self.Close()
        except Exception:  # noqa: BLE001
            pass


class NaiveCacheblendRepo(CacheblendRepo):
    """Naive KV stitching on the same patched-vLLM backend as CacheBlend.

    A zero recomputation ratio disables CacheBlend's cached-token repair while
    preserving the mandatory computation of genuinely fresh prompt spans.
    This is the apples-to-apples naive baseline; ``NaiveTransformer`` remains
    available for experiments specifically targeting the HF backend.
    """

    name = "naive_repo"

    def __init__(self, *args, recompRatio: float = 0.0, **kwargs):
        if recompRatio != 0.0:
            raise ValueError("NaiveCacheblendRepo requires recompRatio=0")
        super().__init__(*args, recompRatio=0.0, **kwargs)
