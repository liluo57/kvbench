"""A^3 over the bundled ragkv backend and an isolated JSONL worker.

The worker uses the project Python. Prepared chunks form a contiguous prefix
and the fresh query is a suffix, following the upstream prompt contract.
Other prompts fall back to full prefill through the same bundled runtime.
"""

from __future__ import annotations

import json
import hashlib
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
from core.Method import Method, ResolveMaxNewTokens
from core.Result import NumOutputTokensKey, Result, TotalTimeKey, TtftKey
from core.Sampling import ResolveSamplingConfig

from helpers.backends.Prompt import ComposeReuse
from helpers.backends.ModelAdapter import user_turn_prefix
from helpers.backends.Tokenization import EncodeText


_ReadyLine = "[a3-repo-helper] ready"


class A3Repo(Method):
    """Run A^3 through an isolated worker with the bundled runtime."""

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
        pythonPath: Optional[str] = None,
        startTimeout: float = 1800.0,
        tag: Optional[str] = None,
    ):
        super().__init__(gpuNums=gpuNums, perfWeight=perfWeight, maxGpuNums=1, tag=tag)
        self.maxNewTokens = ResolveMaxNewTokens(maxNewTokens)
        self.maxModelLen = int(maxModelLen)
        self.recompRatio = float(recompRatio)
        self.reuseMethod = str(reuseMethod)
        self.startTimeout = float(startTimeout)

        section = Get("A3", {}) or {}
        self.repoRoot = Path(__file__).resolve().parent / "backend"
        configuredPython = pythonPath or section.get("Python")
        self.workerPython = (
            Path(str(configuredPython)).expanduser()
            if configuredPython else Path(sys.executable)
        )
        if not self.workerPython.is_absolute():
            self.workerPython = Path(__file__).resolve().parents[2] / self.workerPython

        self.modelPath = DefaultModelPath()
        self.samplingConfig = ResolveSamplingConfig(self.modelPath)
        self._proc: Optional[subprocess.Popen[str]] = None
        self._drainThread: Optional[threading.Thread] = None
        self._stderrTail = deque(maxlen=200)
        self._chunks: List[List[List[int]]] = []
        # A KVBench KB task renders the complete user message through the
        # model chat template.  Keep a second, prepared representation whose
        # first chunk includes that literal user-turn prefix, so the official
        # reuse path can match the rendered Run prompt without changing the
        # raw-chunk path used by non-chat callers.
        self._wire_chunks: List[List[List[int]]] = []

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
        self._stderrTail.clear()
        if not self.workerPython.is_file():
            raise FileNotFoundError(
                f"A3 worker Python not found: {self.workerPython}; run ./setup.py --a3"
            )
        helper = Path(__file__).resolve().parent / "Worker.py"
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
            cwd=str(Path(__file__).resolve().parents[2]),
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
                        "A3 worker exited during startup: "
                        + ("".join(self._stderrTail) or "(no stderr)")
                    )
                continue
            line = line.strip()
            if line.startswith(_ReadyLine):
                return
            print(line, flush=True)
        self._proc.kill()
        raise TimeoutError("A3 worker did not become ready")

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
                    "A3 worker closed stdout: "
                    + ("".join(self._stderrTail) or "(no stderr)")
                )
            try:
                response = json.loads(line)
                break
            except json.JSONDecodeError:
                print(f"[a3-repo-helper stdout] {line.rstrip()}", file=sys.stderr, flush=True)
        if not response.get("ok"):
            raise RuntimeError(f"A3 worker error: {response.get('error')}")
        return response

    def Prepare(self, data: List[List[List[int]]]) -> None:
        self._chunks = [
            [[int(token) for token in chunk] for chunk in (chunks or []) if chunk]
            for chunks in data
        ]
        prefix = user_turn_prefix(self.modelPath, thinking=False)
        prefix_ids = EncodeText(prefix, self.modelPath) if prefix else []
        self._wire_chunks = []
        flat: List[str] = []
        flat_ids: List[List[int]] = []
        seen = set()
        for chunks in self._chunks:
            wire = [list(chunk) for chunk in chunks]
            if wire and prefix_ids:
                wire[0] = prefix_ids + wire[0]
            self._wire_chunks.append(wire)
            for chunk in [*chunks, *wire]:
                key = tuple(chunk)
                if key not in seen:
                    seen.add(key)
                    flat.append("ids:" + hashlib.sha256(
                        b"".join(t.to_bytes(4, "little", signed=True) for t in chunk)
                    ).hexdigest())
                    flat_ids.append(chunk)
        if flat:
            self._request({"op": "collect", "chunks": flat, "chunk_ids": flat_ids})

    def Run(
        self,
        data: List[List[int]],
        retainOutput: Optional[List[bool]] = None,
        maxNewTokens: Optional[int] = None,
    ) -> List[Result]:
        runStart = time.perf_counter()
        budget = self.maxNewTokens if maxNewTokens is None else ResolveMaxNewTokens(maxNewTokens)
        if len(self._chunks) != len(data):
            self._chunks = [[] for _ in data]
            self._wire_chunks = [[] for _ in data]
        results: List[Result] = []
        for index, prompt in enumerate(data):
            chunks = self._chunks[index]
            order, suffix = ComposeReuse(chunks, prompt)
            variant = "raw"
            if not order and index < len(self._wire_chunks):
                wire_chunks = self._wire_chunks[index]
                order, suffix = ComposeReuse(wire_chunks, prompt)
                if order:
                    variant = "chat_prefixed"
            # RULER shuffle tasks prepare the question as their final chunk.
            # A3's attention selector requires a non-empty fresh query; keep
            # that last chunk fresh when the cache covers the entire prompt.
            if order and not suffix:
                suffix = order[-1]
                order = order[:-1]
            retain = bool(retainOutput[index]) if retainOutput and index < len(retainOutput) else False
            if order:
                response = self._request({
                    "op": "reuse",
                    "chunks": ["ids:" + hashlib.sha256(
                        b"".join(t.to_bytes(4, "little", signed=True) for t in chunk)
                    ).hexdigest() for chunk in order],
                    "suffix_ids": suffix,
                    "retain_output": retain,
                    "max_new_tokens": budget,
                })
                full = False
            else:
                response = self._request({"op": "full", "input_ids": prompt,
                                          "retain_output": retain, "max_new_tokens": budget})
                full = True
            results.append(
                self._result(
                    response,
                    full=full,
                    prompt_variant=variant,
                    onlineStart=runStart,
                )
            )
        return results

    def _result(
        self,
        response: Dict[str, Any],
        *,
        full: bool,
        prompt_variant: str = "raw",
        onlineStart: Optional[float] = None,
    ) -> Result:
        metadata = {
            "reuse_ratio": float(response.get("reuse_ratio", 0.0)),
            "recomp_ratio": self.recompRatio,
            "n_input": response.get("n_input"),
            "official_repo": str(self.repoRoot),
            "prompt_variant": prompt_variant,
        }
        if response.get("a3_debug") is not None:
            metadata["a3_debug"] = response["a3_debug"]
        if full:
            metadata["full_prefill"] = True
        ttft = float(response.get("ttft", 0.0))
        generationStart = response.get("generation_start")
        if onlineStart is not None and generationStart is not None:
            ttft += float(generationStart) - onlineStart
        return Result(
            output=response.get("text", ""),
            performance={
                TtftKey: ttft,
                NumOutputTokensKey: int(response.get("num_tokens", 0)),
                TotalTimeKey: float(response.get("total_time", response.get("ttft", 0.0))),
            },
            metadata=metadata,
        )

    def Reset(self) -> None:
        self._chunks = []
        self._wire_chunks = []
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
