"""Thin process bridge to the installed BenchFlow CLI.

BenchFlow owns task semantics, sandbox setup, skill provisioning, agent
lifecycle, tool execution, verification, and result writing. This module only
starts the official ``bench eval run`` process and locates the official result
artifact after it exits.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from helpers.endpoint import KVBenchEndpoint, OpenAIRequest


class BenchflowRunner:
    """Run one real BenchFlow evaluation around a KVBench endpoint."""

    def __init__(
        self,
        *,
        taskId: str,
        modelPath: str,
        sourceMode: str = "dataset",
        dataset: Optional[str] = "skillsbench@1.1",
        skillsbenchDir: Optional[str | Path] = None,
        agent: str = "pi-acp",
        sandbox: str = "docker",
        skillMode: str = "with-skill",
        providerHost: str = "127.0.0.1",
        endpointHost: str = "0.0.0.0",
        port: int = 0,
        modelId: Optional[str] = None,
        jobsDir: Optional[str | Path] = None,
        resultJsonTimeout: float = 3600.0,
        thinking: Optional[bool] = None,
        providerApiKey: Optional[str] = None,
        providerApiKeyEnv: str = "KVBENCH_PROVIDER_API_KEY",
        benchCommand: str | Sequence[str] = "bench",
        extraArgs: Sequence[str] = (),
        endpoint: Optional[KVBenchEndpoint] = None,
        endpointApiKey: Optional[str] = None,
        popenFactory: Callable[..., subprocess.Popen] = subprocess.Popen,
    ):
        if sourceMode not in {"dataset", "local"}:
            raise ValueError("sourceMode must be 'dataset' or 'local'")
        if sourceMode == "dataset" and not dataset:
            raise ValueError("dataset is required when sourceMode='dataset'")
        if sourceMode == "local" and not skillsbenchDir:
            raise ValueError("skillsbenchDir is required when sourceMode='local'")
        if skillMode not in {"with-skill", "no-skill"}:
            raise ValueError("skillMode must be 'with-skill' or 'no-skill'")
        if not providerHost:
            raise ValueError("providerHost must not be empty")

        self.taskId = taskId
        self.modelPath = str(modelPath)
        self.sourceMode = sourceMode
        self.dataset = dataset
        self.skillsbenchDir = Path(skillsbenchDir) if skillsbenchDir else None
        self.agent = agent
        self.sandbox = sandbox
        self.skillMode = skillMode
        self.providerHost = providerHost
        self.endpointHost = endpointHost
        self.port = int(port)
        self.modelId = modelId or Path(self.modelPath).name or "model"
        # ``bench eval run --jobs-dir <dir>`` reuses any pre-existing
        # ``result.json`` under ``<dir>`` ("resuming" the job), so a fresh
        # subdirectory is required for every KVBench attempt. Without this
        # the runner silently inherits a prior run's reward (e.g. a
        # completed citation-check from an earlier KVBench invocation
        # scores 1.0 on every subsequent attempt because bench finds no
        # remaining work and reports the old summary).
        self.jobsDir = Path(jobsDir) if jobsDir else Path.cwd() / "jobs" / taskId
        self.jobsDir = self.jobsDir / f"run-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
        self.resultJsonTimeout = float(resultJsonTimeout)
        self.thinking = thinking
        self.providerApiKey = providerApiKey
        self.providerApiKeyEnv = providerApiKeyEnv
        self.benchCommand = benchCommand
        self.extraArgs = list(extraArgs)
        self.endpoint = endpoint or KVBenchEndpoint(
            modelPath=self.modelPath,
            host=self.endpointHost,
            port=self.port,
            thinking=self.thinking,
            debugLogPath=self.jobsDir / "kvbench_llm_io.jsonl",
            apiKey=endpointApiKey,
        )
        self.popenFactory = popenFactory
        self.process: Optional[subprocess.Popen] = None
        self.processReturnCode: Optional[int] = None
        self.officialResultPath: Optional[Path] = None
        self.officialResult: Optional[Dict[str, Any]] = None
        self.benchflowError: Optional[str] = None
        self._monitorThread: Optional[threading.Thread] = None
        self._stopLock = threading.Lock()
        self._stopped = False

    @property
    def endpointUrl(self) -> str:
        return self.endpoint.endpointUrl

    @property
    def is_done(self) -> bool:
        return self.processReturnCode is not None or self.benchflowError is not None

    @property
    def providerUrl(self) -> str:
        """Host-side URL BenchFlow is told to use for the KVBench provider."""
        return f"http://{self.providerHost}:{self.endpoint.port}/v1"

    def start(self) -> "BenchflowRunner":
        if self.process is not None:
            return self
        self.jobsDir.mkdir(parents=True, exist_ok=True)
        try:
            self.endpoint.start()
            command = self.BuildCommand()
            env = dict(os.environ)
            # LiteLLM exposes its boolean CLI debug flag through the generic
            # DEBUG environment variable. KVBench's shell setup uses values
            # such as DEBUG=release, which makes every BenchFlow LiteLLM
            # proxy fail during argument parsing before the agent starts.
            debugValue = env.get("DEBUG")
            if debugValue is not None and debugValue.strip().lower() not in {
                "0",
                "1",
                "false",
                "true",
                "no",
                "yes",
                "off",
                "on",
            }:
                env.pop("DEBUG", None)
            logPath = self.jobsDir / "benchflow.log"
            log = logPath.open("ab")
            try:
                self.process = self.popenFactory(
                    command,
                    cwd=str(Path.cwd()),
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            finally:
                log.close()
        except BaseException:
            self.benchflowError = "BenchFlow failed to start"
            self.endpoint.stop(self.benchflowError)
            raise

        self._monitorThread = threading.Thread(
            target=self._Monitor,
            name=f"benchflow-runner-{self.taskId}",
            daemon=True,
        )
        self._monitorThread.start()
        return self

    def wait_for_request(
        self, timeout: Optional[float] = None
    ) -> Optional[OpenAIRequest]:
        return self.endpoint.wait_for_request(timeout=timeout)

    def respond(self, request: OpenAIRequest, output: str) -> None:
        self.endpoint.respond(request, output)

    def BuildCommand(self) -> List[str]:
        """Build the current installed BenchFlow command without executing it."""
        command = (
            shlex.split(self.benchCommand)
            if isinstance(self.benchCommand, str)
            else list(self.benchCommand)
        )
        command.extend(["eval", "run"])
        if self.sourceMode == "dataset":
            command.extend(["--dataset", str(self.dataset)])
        else:
            command.extend(["--tasks-dir", str(self.skillsbenchDir / "tasks")])
        command.extend(
            [
                "--include",
                self.taskId,
                "--agent",
                self.agent,
                "--model",
                self._ProviderModel(),
                "--sandbox",
                self.sandbox,
                "--skill-mode",
                self.skillMode,
                "--usage-tracking",
                "off",
                "--jobs-dir",
                str(self.jobsDir),
                "--concurrency",
                "1",
                "--agent-env",
                f"BENCHFLOW_PROVIDER_BASE_URL={self.providerUrl}",
                "--agent-env",
                f"BENCHFLOW_PROVIDER_API_KEY={self._ApiKey()}",
            ]
        )
        command.extend(self.extraArgs)
        return command

    build_command = BuildCommand

    def ReadOfficialResult(self) -> Optional[Dict[str, Any]]:
        """Read the latest official-shaped ``result.json`` under ``jobsDir``."""
        candidates = sorted(
            self.jobsDir.rglob("result.json"),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        malformed: Optional[str] = None
        for path in candidates:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                malformed = f"could not read official result {path}: {exc}"
                continue
            if not isinstance(payload, dict):
                malformed = f"official result is not an object: {path}"
                continue
            if payload.get("task_name") not in (None, self.taskId):
                continue
            self.officialResultPath = path
            self.officialResult = payload
            return payload
        if malformed:
            self.benchflowError = malformed
        return None

    read_official_result = ReadOfficialResult

    def Diagnostics(self) -> Dict[str, Any]:
        payload = self.officialResult or {}
        diagnostics: Dict[str, Any] = {
            "jobs_dir": str(self.jobsDir),
            "official_result_path": (
                str(self.officialResultPath) if self.officialResultPath else None
            ),
            "task_name": self.taskId,
            "source_mode": self.sourceMode,
            "dataset": self.dataset,
            "skill_mode": self.skillMode,
            "agent": self.agent,
            "sandbox": self.sandbox,
            "provider_url": self.providerUrl,
            "endpoint_url": self.endpointUrl,
            "benchflow_returncode": self.processReturnCode,
            "benchflow_error": self.benchflowError,
        }
        for key in (
            "task_name",
            "rewards",
            "error",
            "error_category",
            "verifier_error",
            "verifier_error_category",
            "n_tool_calls",
            "n_skill_invocations",
            "agent_result",
            "final_metrics",
        ):
            if key in payload:
                diagnostics[f"benchflow_{key}"] = payload[key]
        return diagnostics

    diagnostics = Diagnostics

    def stop(self) -> None:
        with self._stopLock:
            if self._stopped:
                return
            self._stopped = True
            self.endpoint.stop()
            process = self.process
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5.0)
            if process is not None and self.processReturnCode is None:
                self.processReturnCode = process.returncode
            thread = self._monitorThread
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=2.0)

    def __enter__(self) -> "BenchflowRunner":
        return self.start()

    def __exit__(self, excType, excValue, excTraceback) -> None:
        self.stop()

    def __del__(self) -> None:
        try:
            self.stop()
        except Exception:  # noqa: BLE001 - best effort only
            pass

    # -------------------------------------------------------------- internals
    def _ProviderModel(self) -> str:
        return (
            self.modelId
            if self.modelId.startswith("vllm/")
            else f"vllm/{self.modelId}"
        )

    def _ApiKey(self) -> str:
        # The default is deliberately a non-secret placeholder. A real key,
        # if an agent/provider requires one, must come from process env or an
        # explicit caller value, never from source code.
        return (
            self.providerApiKey
            or os.environ.get(self.providerApiKeyEnv)
            or "dummy"
        )

    def _Monitor(self) -> None:
        process = self.process
        if process is None:
            return
        try:
            self.processReturnCode = process.wait(timeout=self.resultJsonTimeout)
        except subprocess.TimeoutExpired:
            self.benchflowError = (
                f"BenchFlow did not finish within {self.resultJsonTimeout:.1f}s"
            )
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)
            self.processReturnCode = process.returncode
        except BaseException as exc:
            self.benchflowError = (
                f"BenchFlow process failed: {type(exc).__name__}: {exc}"
            )
        self.ReadOfficialResult()
        if self.officialResult is None and self.benchflowError is None:
            if self.processReturnCode:
                self.benchflowError = (
                    f"BenchFlow exited with code {self.processReturnCode} "
                    "without an official result.json"
                )
            else:
                self.benchflowError = "BenchFlow exited without an official result.json"
        self.endpoint.finish(self.benchflowError)


__all__ = ["BenchflowRunner"]
