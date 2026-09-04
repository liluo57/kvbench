#!/usr/bin/env python3
"""Run BenchFlow Docker jobs for a remote KVBench coordinator.

This server deliberately does not read KVBench's ``config.yaml``.  Edit the
small deployment constants below (or use their command-line overrides), copy
this script to the Docker host, and run it there.  Per-run model endpoints are
sent by the KVBench coordinator because their ports are allocated dynamically.
"""

listen_endpoint = "127.0.0.1:8765"
work_root = "/var/tmp/kvbench-remote-docker"
bench_command = "bench"
auth_token_env = "KVBENCH_REMOTE_TOKEN"
max_concurrent_runs = 8

import argparse
import hashlib
import hmac
import importlib.metadata
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Mapping, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen


PROTOCOL_VERSION = 1
MAX_JSON_BYTES = 1024 * 1024
MAX_SOURCE_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_SOURCE_UNPACKED_BYTES = 2 * 1024 * 1024 * 1024
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_TERMINAL_STATES = frozenset({"succeeded", "failed", "cancelled"})
_RESERVED_EXTRA_OPTIONS = frozenset(
    {
        "--dataset",
        "--tasks-dir",
        "--include",
        "--agent",
        "--model",
        "--sandbox",
        "--skill-mode",
        "--usage-tracking",
        "--jobs-dir",
        "--concurrency",
        # ``--agent-env`` is intentionally NOT reserved. The server accepts
        # ``--agent-env KEY=VALUE`` pairs from the client and routes them into
        # ``bench`` alongside the server's own reserved ``--agent-env`` lines
        # (see ``_BuildCommand``). Keys that the server itself injects to wire
        # the agent to KVBench (``_AGENT_ENV_RESERVED_KEYS``) are still
        # rejected so the control plane cannot be repointed by the client.
    }
)
_AGENT_ENV_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.+$")
_AGENT_ENV_RESERVED_KEYS = frozenset(
    {
        "BENCHFLOW_PROVIDER_BASE_URL",
        "BENCHFLOW_PROVIDER_API_KEY",
    }
)


class ApiError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = int(status)


@dataclass
class RunRecord:
    runId: str
    clientRunId: str
    spec: dict[str, Any]
    runDir: Path
    jobsDir: Path
    state: str
    createdAt: float = field(default_factory=time.time)
    startedAt: Optional[float] = None
    finishedAt: Optional[float] = None
    process: Optional[subprocess.Popen] = field(default=None, repr=False)
    processReturnCode: Optional[int] = None
    error: Optional[str] = None
    officialResult: Optional[dict[str, Any]] = None
    officialResultPath: Optional[Path] = None
    artifactPath: Optional[Path] = None
    artifactSha256: Optional[str] = None
    artifactBytes: Optional[int] = None
    cancelRequested: bool = False
    monitorThread: Optional[threading.Thread] = field(default=None, repr=False)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def PublicStatus(self) -> dict[str, Any]:
        with self.lock:
            resultPath = None
            if self.officialResultPath is not None:
                try:
                    resultPath = str(self.officialResultPath.relative_to(self.jobsDir))
                except ValueError:
                    resultPath = str(self.officialResultPath)
            artifact = None
            if self.artifactPath is not None:
                artifact = {
                    "sha256": self.artifactSha256,
                    "bytes": self.artifactBytes,
                    "url": f"/v1/runs/{self.runId}/artifacts",
                }
            return {
                "protocol_version": PROTOCOL_VERSION,
                "run_id": self.runId,
                "client_run_id": self.clientRunId,
                "state": self.state,
                "task_id": self.spec["task_id"],
                "created_at": self.createdAt,
                "started_at": self.startedAt,
                "finished_at": self.finishedAt,
                "process_returncode": self.processReturnCode,
                "error": self.error,
                "official_result_path": resultPath,
                "official_result": self.officialResult,
                "artifact": artifact,
            }


class RemoteRunManager:
    """Own subprocesses and artifacts independently of the HTTP transport."""

    def __init__(
        self,
        *,
        workRoot: str | Path,
        benchCommand: str | Sequence[str] = "bench",
        maxConcurrentRuns: int = 8,
        validateDockerImages: bool = True,
    ):
        self.workRoot = Path(workRoot).expanduser().resolve()
        self.workRoot.mkdir(parents=True, exist_ok=True)
        self.benchCommand = benchCommand
        self.maxConcurrentRuns = max(1, int(maxConcurrentRuns))
        self.validateDockerImages = bool(validateDockerImages)
        self._runs: dict[str, RunRecord] = {}
        self._clientRuns: dict[str, RunRecord] = {}
        self._lock = threading.RLock()

    def CreateRun(self, rawSpec: Mapping[str, Any]) -> RunRecord:
        spec = self._ValidateSpec(rawSpec)
        clientRunId = spec["client_run_id"]
        with self._lock:
            existing = self._clientRuns.get(clientRunId)
            if existing is not None:
                if existing.spec != spec:
                    raise ApiError(
                        409, "client_run_id already exists with a different spec"
                    )
                return existing

            runId = uuid.uuid4().hex
            runDir = self.workRoot / runId
            jobsDir = runDir / "jobs"
            jobsDir.mkdir(parents=True, exist_ok=False)
            state = "ready" if spec["source_mode"] == "dataset" else "created"
            record = RunRecord(
                runId=runId,
                clientRunId=clientRunId,
                spec=spec,
                runDir=runDir,
                jobsDir=jobsDir,
                state=state,
            )
            self._runs[runId] = record
            self._clientRuns[clientRunId] = record
            return record

    def GetRun(self, runId: str) -> RunRecord:
        with self._lock:
            record = self._runs.get(runId)
        if record is None:
            raise ApiError(404, f"unknown run: {runId}")
        return record

    def UploadSource(self, runId: str, stream: BinaryIO, length: int) -> RunRecord:
        record = self.GetRun(runId)
        with record.lock:
            if record.spec["source_mode"] != "local":
                raise ApiError(409, "source upload is only valid for source_mode=local")
            if record.state == "ready":
                return record
            if record.state != "created":
                raise ApiError(409, f"cannot upload source while run is {record.state}")
            record.state = "uploading"

        archivePath = record.runDir / "source.tar.gz.part"
        sourceRoot = record.runDir / "source"
        try:
            remaining = length
            with archivePath.open("wb") as output:
                while remaining:
                    chunk = stream.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ApiError(400, "source upload ended before Content-Length")
                    output.write(chunk)
                    remaining -= len(chunk)
            self._ExtractSourceArchive(
                archivePath,
                sourceRoot,
                taskId=record.spec["task_id"],
            )
        except BaseException:
            shutil.rmtree(sourceRoot, ignore_errors=True)
            with record.lock:
                record.state = "created"
            raise
        finally:
            archivePath.unlink(missing_ok=True)

        with record.lock:
            record.state = "ready"
        return record

    def StartRun(self, runId: str) -> RunRecord:
        record = self.GetRun(runId)
        with self._lock, record.lock:
            if record.state == "running" or record.state in _TERMINAL_STATES:
                return record
            if record.state != "ready":
                raise ApiError(409, f"cannot start run while it is {record.state}")
            running = sum(item.state == "running" for item in self._runs.values())
            if running >= self.maxConcurrentRuns:
                raise ApiError(429, "remote runtime has reached its concurrency limit")

            self._CheckProvider(record)
            if record.spec["source_mode"] == "local" and self.validateDockerImages:
                self._CheckLocalDockerImage(record)
            command = self._BuildCommand(record)
            logPath = record.jobsDir / "benchflow.log"
            log = logPath.open("ab")
            try:
                record.process = subprocess.Popen(
                    command,
                    cwd=str(record.runDir),
                    env=dict(os.environ),
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=(os.name != "nt"),
                )
            except BaseException as exc:
                record.state = "failed"
                record.error = f"BenchFlow failed to start: {type(exc).__name__}: {exc}"
                record.finishedAt = time.time()
                raise ApiError(500, record.error) from exc
            finally:
                log.close()
            record.state = "running"
            record.startedAt = time.time()
            record.monitorThread = threading.Thread(
                target=self._Monitor,
                args=(record,),
                name=f"remote-benchflow-{record.runId}",
                daemon=True,
            )
            record.monitorThread.start()
            return record

    def CancelRun(self, runId: str) -> RunRecord:
        record = self.GetRun(runId)
        process = None
        with record.lock:
            if record.state in _TERMINAL_STATES:
                return record
            record.cancelRequested = True
            process = record.process
            if process is None:
                record.state = "cancelled"
                record.error = "run cancelled before start"
                record.finishedAt = time.time()
        if process is not None and process.poll() is None:
            self._TerminateProcess(process)
        return record

    def Close(self) -> None:
        with self._lock:
            runIds = list(self._runs)
        for runId in runIds:
            try:
                self.CancelRun(runId)
            except Exception:
                pass

    def _ValidateSpec(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(raw, Mapping):
            raise ApiError(400, "run spec must be a JSON object")
        protocolVersion = raw.get("protocol_version")
        if protocolVersion != PROTOCOL_VERSION:
            raise ApiError(400, f"unsupported protocol_version: {protocolVersion!r}")

        def identifier(name: str) -> str:
            value = raw.get(name)
            if not isinstance(value, str) or not _ID_PATTERN.fullmatch(value):
                raise ApiError(400, f"{name} must match {_ID_PATTERN.pattern}")
            return value

        clientRunId = identifier("client_run_id")
        taskId = identifier("task_id")
        sourceMode = raw.get("source_mode")
        if sourceMode not in {"dataset", "local"}:
            raise ApiError(400, "source_mode must be 'dataset' or 'local'")
        dataset = raw.get("dataset")
        if sourceMode == "dataset" and (not isinstance(dataset, str) or not dataset):
            raise ApiError(400, "dataset is required for source_mode=dataset")

        providerBaseUrl = raw.get("provider_base_url")
        if not isinstance(providerBaseUrl, str):
            raise ApiError(400, "provider_base_url must be a URL")
        parsedProvider = urlsplit(providerBaseUrl)
        if (
            parsedProvider.scheme not in {"http", "https"}
            or not parsedProvider.hostname
        ):
            raise ApiError(400, "provider_base_url must be an HTTP(S) URL with a host")

        providerApiKey = raw.get("provider_api_key")
        if not isinstance(providerApiKey, str) or not providerApiKey:
            raise ApiError(400, "provider_api_key must not be empty")

        extraArgs = raw.get("bench_extra_args", [])
        if not isinstance(extraArgs, list) or any(
            not isinstance(arg, str) for arg in extraArgs
        ):
            raise ApiError(400, "bench_extra_args must be a list of strings")
        if len(extraArgs) > 100 or any(len(arg) > 4096 for arg in extraArgs):
            raise ApiError(400, "bench_extra_args exceeds the protocol limit")
        for index, arg in enumerate(extraArgs):
            if arg == "--agent-env" or arg.startswith("--agent-env="):
                pair = (
                    arg[len("--agent-env="):]
                    if arg.startswith("--agent-env=")
                    else (
                        extraArgs[index + 1]
                        if index + 1 < len(extraArgs)
                        else None
                    )
                )
                if pair is None:
                    raise ApiError(400, "--agent-env requires KEY=VALUE")
                if not _AGENT_ENV_PATTERN.match(pair):
                    raise ApiError(
                        400, f"--agent-env value must match KEY=VALUE: {pair!r}"
                    )
                key = pair.split("=", 1)[0]
                if key in _AGENT_ENV_RESERVED_KEYS:
                    raise ApiError(
                        400, f"--agent-env may not override reserved key {key}"
                    )
                continue
            option = arg.split("=", 1)[0]
            if option in _RESERVED_EXTRA_OPTIONS:
                raise ApiError(400, f"bench_extra_args may not override {option}")

        timeout = raw.get("result_json_timeout", 3600.0)
        if not isinstance(timeout, (int, float)) or not 0 < float(timeout) <= 7 * 86400:
            raise ApiError(
                400, "result_json_timeout must be between 0 and 604800 seconds"
            )

        def nonempty(name: str) -> str:
            value = raw.get(name)
            if not isinstance(value, str) or not value:
                raise ApiError(400, f"{name} must not be empty")
            return value

        skillMode = nonempty("skill_mode")
        if skillMode not in {"with-skill", "no-skill"}:
            raise ApiError(400, "skill_mode must be 'with-skill' or 'no-skill'")

        return {
            "protocol_version": PROTOCOL_VERSION,
            "client_run_id": clientRunId,
            "task_id": taskId,
            "source_mode": sourceMode,
            "dataset": dataset if isinstance(dataset, str) else None,
            "agent": nonempty("agent"),
            "skill_mode": skillMode,
            "model_id": nonempty("model_id"),
            "provider_base_url": providerBaseUrl.rstrip("/"),
            "provider_api_key": providerApiKey,
            "result_json_timeout": float(timeout),
            "bench_extra_args": list(extraArgs),
        }

    def _ExtractSourceArchive(
        self, archivePath: Path, target: Path, *, taskId: str
    ) -> None:
        try:
            archive = tarfile.open(archivePath, "r:gz")
        except (OSError, tarfile.TarError) as exc:
            raise ApiError(400, f"invalid source archive: {exc}") from exc
        with archive:
            members = archive.getmembers()
            total = 0
            expectedPrefix = ("tasks", taskId)
            for member in members:
                parts = PurePosixPath(member.name).parts
                if (
                    not parts
                    or PurePosixPath(member.name).is_absolute()
                    or ".." in parts
                    or tuple(parts[:2]) != expectedPrefix
                ):
                    raise ApiError(
                        400, f"source archive has an invalid path: {member.name}"
                    )
                if (
                    member.issym()
                    or member.islnk()
                    or not (member.isdir() or member.isfile())
                ):
                    raise ApiError(
                        400, f"source archive has an unsupported entry: {member.name}"
                    )
                total += max(0, member.size)
                if total > MAX_SOURCE_UNPACKED_BYTES:
                    raise ApiError(413, "unpacked source exceeds the size limit")
            target.mkdir(parents=True, exist_ok=False)
            archive.extractall(target, members=members, filter="data")
        taskFile = target / "tasks" / taskId / "task.md"
        if not taskFile.is_file():
            raise ApiError(
                400, f"source archive does not contain tasks/{taskId}/task.md"
            )

    def _CheckProvider(self, record: RunRecord) -> None:
        parsed = urlsplit(record.spec["provider_base_url"])
        healthUrl = urlunsplit((parsed.scheme, parsed.netloc, "/health", "", ""))
        request = Request(
            healthUrl,
            headers={"Authorization": f"Bearer {record.spec['provider_api_key']}"},
        )
        try:
            with urlopen(request, timeout=10) as response:
                if response.status != 200:
                    raise ApiError(
                        502, f"KVBench endpoint health returned {response.status}"
                    )
        except HTTPError as exc:
            raise ApiError(502, f"KVBench endpoint health returned {exc.code}") from exc
        except (OSError, URLError) as exc:
            raise ApiError(
                502, f"could not reach KVBench endpoint {healthUrl}: {exc}"
            ) from exc

    def _CheckLocalDockerImage(self, record: RunRecord) -> None:
        taskFile = (
            record.runDir / "source" / "tasks" / record.spec["task_id"] / "task.md"
        )
        try:
            from benchflow.task.document import TaskDocument

            image = TaskDocument.from_path(taskFile).config.sandbox.docker_image
        except Exception as exc:
            raise ApiError(
                400, f"could not read remote task configuration: {exc}"
            ) from exc
        if not image:
            # task.md 没有钉死镜像，让 bench 自己按任务/skill 的默认规则
            # 解析。SkillsBench 的任务普遍不带 docker_image；只有当任务
            # 显式声明了镜像、并且 B 上确实缺失时才报错。
            return
        if not shutil.which("docker"):
            raise ApiError(500, "docker is not on PATH on the remote runtime host")
        try:
            inspected = subprocess.run(
                ["docker", "image", "inspect", image],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=10,
            )
        except subprocess.TimeoutExpired as exc:
            raise ApiError(
                504, "timed out while checking the remote Docker image"
            ) from exc
        if inspected.returncode != 0:
            raise ApiError(
                409,
                f"Docker image {image!r} is missing on the remote runtime host; "
                "prepare the SkillsBench images on that host first",
            )

    def _BuildCommand(self, record: RunRecord) -> list[str]:
        command = (
            shlex.split(self.benchCommand)
            if isinstance(self.benchCommand, str)
            else list(self.benchCommand)
        )
        if not command:
            raise ApiError(500, "bench_command is empty")
        command.extend(["eval", "run"])
        if record.spec["source_mode"] == "dataset":
            command.extend(["--dataset", record.spec["dataset"]])
        else:
            command.extend(["--tasks-dir", str(record.runDir / "source" / "tasks")])
        modelId = record.spec["model_id"]
        providerModel = modelId if modelId.startswith("vllm/") else f"vllm/{modelId}"
        command.extend(
            [
                "--include",
                record.spec["task_id"],
                "--agent",
                record.spec["agent"],
                "--model",
                providerModel,
                "--sandbox",
                "docker",
                "--skill-mode",
                record.spec["skill_mode"],
                "--usage-tracking",
                "off",
                "--jobs-dir",
                str(record.jobsDir),
                "--concurrency",
                "1",
                "--agent-env",
                f"BENCHFLOW_PROVIDER_BASE_URL={record.spec['provider_base_url']}",
                "--agent-env",
                f"BENCHFLOW_PROVIDER_API_KEY={record.spec['provider_api_key']}",
            ]
        )
        command.extend(record.spec["bench_extra_args"])
        return command

    def _Monitor(self, record: RunRecord) -> None:
        process = record.process
        if process is None:
            return
        timeoutError = None
        try:
            returnCode = process.wait(timeout=record.spec["result_json_timeout"])
        except subprocess.TimeoutExpired:
            timeoutError = (
                "BenchFlow did not finish within "
                f"{record.spec['result_json_timeout']:.1f}s"
            )
            self._TerminateProcess(process)
            returnCode = process.returncode
        except BaseException as exc:
            timeoutError = f"BenchFlow process failed: {type(exc).__name__}: {exc}"
            returnCode = process.returncode

        result, resultPath, readError = self._ReadOfficialResult(record)
        with record.lock:
            record.processReturnCode = returnCode
            record.officialResult = result
            record.officialResultPath = resultPath
            record.finishedAt = time.time()
        archiveError = None
        try:
            self._BuildArtifact(record)
        except BaseException as exc:
            archiveError = (
                f"could not build output archive: {type(exc).__name__}: {exc}"
            )

        with record.lock:
            if record.cancelRequested:
                record.state = "cancelled"
                record.error = "run cancelled"
            elif timeoutError is not None:
                record.state = "failed"
                record.error = timeoutError
            elif archiveError is not None:
                record.state = "failed"
                record.error = archiveError
            elif result is None:
                record.state = "failed"
                record.error = readError or (
                    f"BenchFlow exited with code {returnCode} without an official result.json"
                    if returnCode
                    else "BenchFlow exited without an official result.json"
                )
            else:
                # Match the local runner: an official result is authoritative even
                # if an outer CLI wrapper happened to return a non-zero code.
                record.state = "succeeded"
                record.error = None

    def _ReadOfficialResult(
        self, record: RunRecord
    ) -> tuple[Optional[dict[str, Any]], Optional[Path], Optional[str]]:
        candidates = sorted(
            record.jobsDir.rglob("result.json"),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        malformed = None
        for path in candidates:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                malformed = f"could not read official result {path}: {exc}"
                continue
            if not isinstance(payload, dict):
                malformed = f"official result is not an object: {path}"
                continue
            if payload.get("task_name") not in (None, record.spec["task_id"]):
                continue
            return payload, path, None
        return None, None, malformed

    def _BuildArtifact(self, record: RunRecord) -> None:
        metadata = {
            "protocol_version": PROTOCOL_VERSION,
            "run_id": record.runId,
            "client_run_id": record.clientRunId,
            "task_id": record.spec["task_id"],
            "source_mode": record.spec["source_mode"],
            "created_at": record.createdAt,
            "started_at": record.startedAt,
            "process_returncode": record.processReturnCode,
        }
        (record.jobsDir / "remote-runtime.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporaryPath = record.runDir / "artifacts.tar.gz.part"
        finalPath = record.runDir / "artifacts.tar.gz"
        with tarfile.open(temporaryPath, "w:gz") as archive:
            archive.add(record.jobsDir, arcname="remote-runtime", recursive=True)
        os.replace(temporaryPath, finalPath)
        digest = hashlib.sha256()
        with finalPath.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        with record.lock:
            record.artifactPath = finalPath
            record.artifactSha256 = digest.hexdigest()
            record.artifactBytes = finalPath.stat().st_size

    @staticmethod
    def _TerminateProcess(process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return
        try:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            process.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            if process.poll() is None:
                try:
                    if os.name != "nt":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass


class _RuntimeHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class RemoteRuntimeRequestHandler(BaseHTTPRequestHandler):
    server_version = "KVBenchRemoteDocker/1"

    @property
    def manager(self) -> RemoteRunManager:
        return self.server.manager  # type: ignore[attr-defined]

    @property
    def authToken(self) -> Optional[str]:
        return self.server.authToken  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        print(
            f"[remote-docker] {self.address_string()} - {format % args}",
            file=sys.stderr,
        )

    def _Authorized(self) -> bool:
        if not self.authToken:
            return True
        value = self.headers.get("Authorization") or ""
        prefix = "Bearer "
        return value.startswith(prefix) and hmac.compare_digest(
            value[len(prefix) :], self.authToken
        )

    def _BeforeRequest(self) -> bool:
        if self._Authorized():
            return True
        self._WriteError(401, "invalid or missing bearer token")
        return False

    def _WriteJSON(self, status: int, payload: Mapping[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _WriteError(self, status: int, message: str) -> None:
        self._WriteJSON(status, {"error": message})

    def _ReadJSON(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ApiError(400, "invalid Content-Length") from exc
        if length <= 0 or length > MAX_JSON_BYTES:
            raise ApiError(
                413 if length > MAX_JSON_BYTES else 400, "invalid JSON body size"
            )
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ApiError(400, f"invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ApiError(400, "JSON body must be an object")
        return payload

    def _Dispatch(self, operation) -> None:
        try:
            operation()
        except ApiError as exc:
            self._WriteError(exc.status, str(exc))
        except (BrokenPipeError, ConnectionResetError):
            return
        except BaseException as exc:
            self._WriteError(500, f"internal server error: {type(exc).__name__}: {exc}")

    def do_GET(self) -> None:  # noqa: N802
        if not self._BeforeRequest():
            return
        self._Dispatch(self._DoGET)

    def _DoGET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/health":
            try:
                benchflowVersion = importlib.metadata.version("benchflow")
            except importlib.metadata.PackageNotFoundError:
                benchflowVersion = None
            self._WriteJSON(
                200,
                {
                    "status": "ok",
                    "protocol_version": PROTOCOL_VERSION,
                    "benchflow_version": benchflowVersion,
                },
            )
            return
        matched = re.fullmatch(r"/v1/runs/([A-Za-z0-9]+)/status", path)
        if matched:
            self._WriteJSON(200, self.manager.GetRun(matched.group(1)).PublicStatus())
            return
        matched = re.fullmatch(r"/v1/runs/([A-Za-z0-9]+)/artifacts", path)
        if matched:
            record = self.manager.GetRun(matched.group(1))
            with record.lock:
                artifactPath = record.artifactPath
                sha256 = record.artifactSha256
            if artifactPath is None or not artifactPath.is_file():
                raise ApiError(409, "output artifact is not ready")
            size = artifactPath.stat().st_size
            self.send_response(200)
            self.send_header("Content-Type", "application/gzip")
            self.send_header("Content-Length", str(size))
            self.send_header("X-Artifact-SHA256", sha256 or "")
            self.end_headers()
            with artifactPath.open("rb") as stream:
                shutil.copyfileobj(stream, self.wfile, length=1024 * 1024)
            return
        raise ApiError(404, f"not found: {path}")

    def do_POST(self) -> None:  # noqa: N802
        if not self._BeforeRequest():
            return
        self._Dispatch(self._DoPOST)

    def _DoPOST(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/v1/runs":
            record = self.manager.CreateRun(self._ReadJSON())
            self._WriteJSON(200, record.PublicStatus())
            return
        matched = re.fullmatch(r"/v1/runs/([A-Za-z0-9]+)/start", path)
        if matched:
            record = self.manager.StartRun(matched.group(1))
            self._WriteJSON(200, record.PublicStatus())
            return
        raise ApiError(404, f"not found: {path}")

    def do_PUT(self) -> None:  # noqa: N802
        if not self._BeforeRequest():
            return
        self._Dispatch(self._DoPUT)

    def _DoPUT(self) -> None:
        path = self.path.split("?", 1)[0]
        matched = re.fullmatch(r"/v1/runs/([A-Za-z0-9]+)/source", path)
        if not matched:
            raise ApiError(404, f"not found: {path}")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ApiError(400, "invalid Content-Length") from exc
        if length <= 0:
            raise ApiError(400, "empty source archive")
        if length > MAX_SOURCE_ARCHIVE_BYTES:
            raise ApiError(413, "source archive exceeds the size limit")
        record = self.manager.UploadSource(matched.group(1), self.rfile, length)
        self._WriteJSON(200, record.PublicStatus())

    def do_DELETE(self) -> None:  # noqa: N802
        if not self._BeforeRequest():
            return
        self._Dispatch(self._DoDELETE)

    def _DoDELETE(self) -> None:
        path = self.path.split("?", 1)[0]
        matched = re.fullmatch(r"/v1/runs/([A-Za-z0-9]+)", path)
        if not matched:
            raise ApiError(404, f"not found: {path}")
        record = self.manager.CancelRun(matched.group(1))
        self._WriteJSON(200, record.PublicStatus())


def CreateServer(
    *,
    host: str,
    port: int,
    manager: RemoteRunManager,
    authToken: Optional[str] = None,
) -> _RuntimeHTTPServer:
    server = _RuntimeHTTPServer((host, int(port)), RemoteRuntimeRequestHandler)
    server.manager = manager  # type: ignore[attr-defined]
    server.authToken = authToken or None  # type: ignore[attr-defined]
    return server


def _ParseListen(value: str) -> tuple[str, int]:
    parsed = urlsplit(value if "://" in value else f"//{value}")
    if not parsed.hostname or parsed.port is None:
        raise ValueError("listen endpoint must look like 0.0.0.0:8765")
    return parsed.hostname, parsed.port


def Main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen", default=listen_endpoint)
    parser.add_argument("--work-root", default=work_root)
    parser.add_argument("--bench-command", default=bench_command)
    parser.add_argument("--max-concurrent-runs", type=int, default=max_concurrent_runs)
    parser.add_argument(
        "--skip-image-check",
        action="store_true",
        help="skip the local-source prebuilt-image check (intended for protocol tests)",
    )
    args = parser.parse_args(argv)
    host, port = _ParseListen(args.listen)
    manager = RemoteRunManager(
        workRoot=args.work_root,
        benchCommand=args.bench_command,
        maxConcurrentRuns=args.max_concurrent_runs,
        validateDockerImages=not args.skip_image_check,
    )
    token = os.environ.get(auth_token_env)
    if not token:
        print(
            f"[remote-docker] warning: {auth_token_env} is unset; control API has no authentication",
            file=sys.stderr,
        )
    server = CreateServer(host=host, port=port, manager=manager, authToken=token)
    print(
        f"[remote-docker] listening on http://{host}:{server.server_address[1]} "
        f"work_root={manager.workRoot}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        manager.Close()
    return 0


if __name__ == "__main__":
    raise SystemExit(Main())
