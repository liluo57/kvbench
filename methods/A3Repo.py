"""A^3 adapter for the authors' official ``ragkv`` repository.

This module intentionally contains no A^3 algorithm implementation.  It owns
the KVBench ``Method`` lifecycle and starts a worker with the official ragkv
checkout's Python environment.  The worker lives in
``helpers/a3_repo/A3RepoHelper.py`` and imports the official model,
precompute, attention-selection, and decoding code from ``ragkv``.

The repository path is read from ``config.yaml``::

    A3:
      Repo:
        RepoPath: /path/to/ragkv
        Python: /path/to/ragkv/.venv/bin/python

``Python`` is optional; when omitted, ``<RepoPath>/.venv/bin/python`` is
used.  Keeping the official implementation in a subprocess is important:
ragkv monkey-patches HuggingFace model classes and depends on flashinfer.
Those changes must not leak into other KVBench methods.

The initial adapter follows the official A^3 prompt contract: prepared
chunks form a contiguous prefix and the query is a fresh suffix.  Prompts
that do not satisfy this contract fall back to full prefill instead of
silently changing the algorithm.
"""

from __future__ import annotations

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
from core.Sampling import ResolveSamplingConfig

from helpers.backends.Prompt import ComposeReuse


_ReadyLine = "[a3-repo-helper] ready"


class A3Repo(Method):
    """Run official A^3 through an isolated ragkv worker."""

    name = "a3_repo"
    maxCaseBatchSize = 1
    method_metrics = ("reuse_ratio",)

    def __init__(
        self,
        gpuNums: int = 1,
        perfWeight: float = 1.0,
        *,
        maxNewTokens: int = 64,
        maxModelLen: int = 32768,
        recompRatio: float = 0.15,
        reuseMethod: str = "debug",
        repoPath: Optional[str] = None,
        pythonPath: Optional[str] = None,
        startTimeout: float = 1800.0,
        tag: Optional[str] = None,
    ):
        super().__init__(gpuNums=gpuNums, perfWeight=perfWeight, maxGpuNums=1, tag=tag)
        self.maxNewTokens = int(maxNewTokens)
        self.maxModelLen = int(maxModelLen)
        self.recompRatio = float(recompRatio)
        self.reuseMethod = str(reuseMethod)
        self.startTimeout = float(startTimeout)

        section = Get("A3", {}) or {}
        repo = section.get("Repo", {}) or {}
        configuredRoot = repoPath or repo.get("RepoPath") or os.environ.get("A3_REPO_PATH")
        if not configuredRoot:
            raise RuntimeError(
                "A3Repo: set A3.Repo.RepoPath in config.yaml or A3_REPO_PATH"
            )
        self.repoRoot = Path(str(configuredRoot)).expanduser().resolve()
        if not self.repoRoot.is_dir():
            raise FileNotFoundError(f"A3Repo: official ragkv repo not found: {self.repoRoot}")

        configuredPython = pythonPath or repo.get("Python")
        self.workerPython = (
            Path(str(configuredPython)).expanduser()
            if configuredPython
            else self.repoRoot / ".venv" / "bin" / "python"
        )
        if not self.workerPython.exists():
            raise FileNotFoundError(
                f"A3Repo: worker Python not found: {self.workerPython}; "
                "install ragkv requirements or set A3.Repo.Python"
            )

        self.modelPath = DefaultModelPath()
        self.samplingConfig = ResolveSamplingConfig(self.modelPath)
        self._proc: Optional[subprocess.Popen[str]] = None
        self._drainThread: Optional[threading.Thread] = None
        self._stderrTail = deque(maxlen=200)
        self._chunks: List[List[str]] = []

    def Initialize(self, gpuIds: Sequence[int]) -> None:
        super().Initialize(gpuIds)
        self._startWorker()

    def _drainStderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        try:
            for line in proc.stderr:
                self._stderrTail.append(line)
                sys.stderr.write(f"[a3-repo-helper] {line}")
                sys.stderr.flush()
        except Exception:
            pass

    def _startWorker(self) -> None:
        helper = Path(__file__).resolve().parent.parent / "helpers" / "a3_repo" / "A3RepoHelper.py"
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env.pop("LD_PRELOAD", None)
        # Official ragkv invokes ``ninja`` through the subprocess PATH when
        # FlashInfer compiles its first kernel.  Use the selected worker's
        # environment explicitly instead of depending on the caller's shell.
        workerBin = str(self.workerPython.parent)
        env["PATH"] = workerBin + os.pathsep + env.get("PATH", "")
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in self.gpuIds)
        self._proc = subprocess.Popen(
            [
                str(self.workerPython),
                str(helper),
                "--repo_root", str(self.repoRoot),
                "--model", self.modelPath,
                "--max_new_tokens", str(self.maxNewTokens),
                "--max_model_len", str(self.maxModelLen),
                "--recomp_ratio", str(self.recompRatio),
                "--reuse_method", self.reuseMethod,
                "--sampling_config", json.dumps(self.samplingConfig),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
            cwd=str(self.repoRoot),
        )
        assert self._proc.stdout is not None
        self._drainThread = threading.Thread(
            target=self._drainStderr, daemon=True, name="a3-repo-stderr"
        )
        self._drainThread.start()
        deadline = time.time() + self.startTimeout
        while time.time() < deadline:
            line = self._proc.stdout.readline()
            if not line:
                if self._proc.poll() is not None:
                    raise RuntimeError(
                        "A3 official worker exited during startup: "
                        + ("".join(self._stderrTail) or "(no stderr)")
                    )
                continue
            line = line.strip()
            if line.startswith(_ReadyLine):
                return
            print(line, flush=True)
        self._proc.kill()
        raise TimeoutError("A3 official worker did not become ready")

    def _request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self._proc is None or self._proc.stdin is None or self._proc.stdout is None:
            raise RuntimeError("A3Repo worker is not running")
        self._proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()
        # The official repository may emit compiler/progress text on stdout
        # during the first FlashInfer kernel build.  The worker's actual
        # protocol messages are JSON lines; ignore and forward any incidental
        # text instead of treating it as a protocol failure.
        while True:
            line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError(
                    "A3 official worker closed stdout: "
                    + ("".join(self._stderrTail) or "(no stderr)")
                )
            try:
                response = json.loads(line)
                break
            except json.JSONDecodeError:
                print(f"[a3-repo-helper stdout] {line.rstrip()}", file=sys.stderr, flush=True)
        if not response.get("ok"):
            raise RuntimeError(f"A3 official worker error: {response.get('error')}")
        return response

    def Prepare(self, data: List[List[str]]) -> None:
        self._chunks = [[str(chunk) for chunk in (chunks or []) if str(chunk)] for chunks in data]
        flat: List[str] = []
        seen = set()
        for chunks in self._chunks:
            for chunk in chunks:
                if chunk not in seen:
                    seen.add(chunk)
                    flat.append(chunk)
        if flat:
            self._request({"op": "collect", "chunks": flat})

    def Run(
        self,
        data: List[str],
        retainOutput: Optional[List[bool]] = None,
    ) -> List[Result]:
        if len(self._chunks) != len(data):
            self._chunks = [[] for _ in data]
        results: List[Result] = []
        for index, prompt in enumerate(data):
            chunks = self._chunks[index]
            order, suffix = ComposeReuse(chunks, prompt)
            retain = bool(retainOutput[index]) if retainOutput and index < len(retainOutput) else False
            if order:
                response = self._request({
                    "op": "reuse",
                    "chunks": order,
                    "suffix": suffix,
                    "retain_output": retain,
                })
                full = False
            else:
                response = self._request({"op": "full", "text": prompt, "retain_output": retain})
                full = True
            results.append(self._result(response, full=full))
        return results

    def _result(self, response: Dict[str, Any], *, full: bool) -> Result:
        metadata = {
            "reuse_ratio": float(response.get("reuse_ratio", 0.0)),
            "recomp_ratio": self.recompRatio,
            "n_input": response.get("n_input"),
            "official_repo": str(self.repoRoot),
        }
        if response.get("a3_debug") is not None:
            metadata["a3_debug"] = response["a3_debug"]
        if full:
            metadata["full_prefill"] = True
        return Result(
            output=response.get("text", ""),
            performance={
                TtftKey: float(response.get("ttft", 0.0)),
                NumOutputTokensKey: int(response.get("num_tokens", 0)),
                TotalTimeKey: float(response.get("total_time", response.get("ttft", 0.0))),
            },
            metadata=metadata,
        )

    def Reset(self) -> None:
        self._chunks = []
        try:
            self._request({"op": "reset"})
        except (BrokenPipeError, RuntimeError):
            self._restartWorker()

    def _restartWorker(self) -> None:
        self.Close()
        self._startWorker()

    def Close(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            if proc.poll() is None and proc.stdin is not None:
                proc.stdin.write(json.dumps({"op": "close"}) + "\n")
                proc.stdin.flush()
            proc.wait(timeout=30)
        except Exception:
            with suppress(Exception):
                proc.kill()
            with suppress(Exception):
                proc.wait(timeout=5)
        finally:
            for stream_name in ("stdin", "stdout", "stderr"):
                stream = getattr(proc, stream_name, None)
                if stream is not None:
                    with suppress(Exception):
                        stream.close()
            if self._drainThread is not None:
                self._drainThread.join(timeout=1)
            self._drainThread = None

    def __del__(self):
        with suppress(Exception):
            self.Close()
