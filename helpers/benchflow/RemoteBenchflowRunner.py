"""BenchFlow runner whose CLI and Docker sandbox live on another host."""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import secrets
import shutil
import socket
import tarfile
import tempfile
import threading
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Mapping, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from helpers.benchflow.BenchflowRunner import BenchflowRunner


PROTOCOL_VERSION = 1
_TERMINAL_STATES = frozenset({"succeeded", "failed", "cancelled"})


class RemoteBenchflowError(RuntimeError):
    """A control-plane, transfer, or remote runtime failure."""


class RemoteBenchflowRunner(BenchflowRunner):
    """Preserve the local runner interface while executing BenchFlow remotely."""

    def __init__(
        self,
        *,
        remoteEndpoint: str,
        advertiseHost: Optional[str] = None,
        remoteAuthToken: Optional[str] = None,
        remoteAuthTokenEnv: str = "KVBENCH_REMOTE_TOKEN",
        remoteConnectTimeout: float = 10.0,
        remotePollInterval: float = 1.0,
        artifactDownloadRetries: int = 3,
        **kwargs: Any,
    ):
        parsed = urlsplit(str(remoteEndpoint))
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("remoteEndpoint must be an HTTP(S) URL with a host")
        if parsed.query or parsed.fragment:
            raise ValueError("remoteEndpoint must not contain a query or fragment")

        providerApiKeyEnv = str(
            kwargs.get("providerApiKeyEnv", "KVBENCH_PROVIDER_API_KEY")
        )
        providerApiKey = (
            kwargs.get("providerApiKey")
            or os.environ.get(providerApiKeyEnv)
            or secrets.token_urlsafe(32)
        )
        kwargs["providerApiKey"] = providerApiKey
        kwargs["endpointApiKey"] = providerApiKey
        # ``remote-docker`` is a KVBench transport choice.  The remote
        # BenchFlow process itself always receives ``--sandbox docker``.
        kwargs["sandbox"] = "docker"
        super().__init__(**kwargs)

        self.sandbox = "remote-docker"
        self.remoteEndpoint = str(remoteEndpoint).rstrip("/")
        self.advertiseHost = advertiseHost
        self.remoteAuthToken = remoteAuthToken or os.environ.get(remoteAuthTokenEnv)
        self.remoteAuthTokenEnv = remoteAuthTokenEnv
        self.remoteConnectTimeout = float(remoteConnectTimeout)
        self.remotePollInterval = max(0.05, float(remotePollInterval))
        self.artifactDownloadRetries = max(1, int(artifactDownloadRetries))
        self.remoteRunId: Optional[str] = None
        self.remoteState: Optional[str] = None
        self.remoteArtifactPath: Optional[Path] = None
        self._clientRunId = uuid.uuid4().hex
        self._resolvedAdvertiseHost: Optional[str] = None
        self._monitorStop = threading.Event()

    @property
    def providerUrl(self) -> str:
        host = self._resolvedAdvertiseHost or self.advertiseHost or self.providerHost
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        return f"http://{host}:{self.endpoint.port}/v1"

    def start(self) -> "RemoteBenchflowRunner":
        if self.remoteRunId is not None:
            return self
        self.jobsDir.mkdir(parents=True, exist_ok=True)
        try:
            self.endpoint.start()
            self._resolvedAdvertiseHost = self._ResolveAdvertiseHost()
            health = self._RequestJSON("GET", "/health")
            if health.get("protocol_version") != PROTOCOL_VERSION:
                raise RemoteBenchflowError(
                    "remote runtime protocol mismatch: "
                    f"client={PROTOCOL_VERSION}, server={health.get('protocol_version')!r}"
                )
            created = self._RequestJSON("POST", "/v1/runs", self._RunSpec())
            self.remoteRunId = self._StatusRunId(created)
            self.remoteState = str(created.get("state") or "")
            if self.sourceMode == "local":
                sourceArchive = self._BuildSourceArchive()
                try:
                    uploaded = self._UploadSource(sourceArchive)
                finally:
                    sourceArchive.unlink(missing_ok=True)
                self.remoteState = str(uploaded.get("state") or "")
            started = self._RequestJSON(
                "POST",
                f"/v1/runs/{self.remoteRunId}/start",
                timeout=max(30.0, self.remoteConnectTimeout),
            )
            self.remoteState = str(started.get("state") or "")
        except BaseException as exc:
            self.benchflowError = (
                f"Remote BenchFlow failed to start: {type(exc).__name__}: {exc}"
            )
            self._CancelRemote()
            self.endpoint.stop(self.benchflowError)
            raise

        self._monitorThread = threading.Thread(
            target=self._MonitorRemote,
            name=f"remote-benchflow-runner-{self.taskId}",
            daemon=True,
        )
        self._monitorThread.start()
        return self

    def stop(self) -> None:
        with self._stopLock:
            if self._stopped:
                return
            self._stopped = True
            self._monitorStop.set()
            self.endpoint.stop()
            self._CancelRemote()
            thread = self._monitorThread
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=max(2.0, self.remoteConnectTimeout + 1.0))

    def Diagnostics(self) -> Dict[str, Any]:
        diagnostics = super().Diagnostics()
        diagnostics.update(
            {
                "remote_runtime_endpoint": self.remoteEndpoint,
                "remote_run_id": self.remoteRunId,
                "remote_state": self.remoteState,
                "remote_artifact_path": (
                    str(self.remoteArtifactPath) if self.remoteArtifactPath else None
                ),
                "kvbench_advertise_host": self._resolvedAdvertiseHost,
            }
        )
        return diagnostics

    diagnostics = Diagnostics

    def _RunSpec(self) -> dict[str, Any]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "client_run_id": self._clientRunId,
            "task_id": self.taskId,
            "source_mode": self.sourceMode,
            "dataset": self.dataset,
            "agent": self.agent,
            "skill_mode": self.skillMode,
            "model_id": self.modelId,
            "provider_base_url": self.providerUrl,
            "provider_api_key": self._ApiKey(),
            "result_json_timeout": self.resultJsonTimeout,
            "bench_extra_args": list(self.extraArgs),
        }

    def _ResolveAdvertiseHost(self) -> str:
        if self.advertiseHost:
            return self.advertiseHost
        parsed = urlsplit(self.remoteEndpoint)
        remoteHost = parsed.hostname
        if remoteHost is None:
            raise RemoteBenchflowError("remote runtime URL has no host")
        if remoteHost in {"localhost", "127.0.0.1", "::1"}:
            return "::1" if ":" in remoteHost else "127.0.0.1"
        remotePort = parsed.port or (443 if parsed.scheme == "https" else 80)
        lastError: Optional[BaseException] = None
        for family, socktype, protocol, _, address in socket.getaddrinfo(
            remoteHost, remotePort, type=socket.SOCK_DGRAM
        ):
            sock = socket.socket(family, socktype, protocol)
            try:
                sock.connect(address)
                value = sock.getsockname()[0]
                if value:
                    return str(value)
            except OSError as exc:
                lastError = exc
            finally:
                sock.close()
        raise RemoteBenchflowError(
            f"could not determine the A-side address used to reach {remoteHost}: {lastError}"
        )

    def _BuildSourceArchive(self) -> Path:
        if self.skillsbenchDir is None:
            raise RemoteBenchflowError(
                "skillsbenchDir is required for local source mode"
            )
        taskDir = self.skillsbenchDir / "tasks" / self.taskId
        if not (taskDir / "task.md").is_file():
            raise RemoteBenchflowError(f"local task directory is invalid: {taskDir}")
        for path in taskDir.rglob("*"):
            if path.is_symlink():
                raise RemoteBenchflowError(
                    f"local task source contains an unsupported symlink: {path}"
                )
        temporary = tempfile.NamedTemporaryFile(
            prefix=f"kvbench-{self.taskId}-",
            suffix=".tar.gz",
            delete=False,
        )
        archivePath = Path(temporary.name)
        temporary.close()
        try:
            with tarfile.open(archivePath, "w:gz") as archive:
                archive.add(taskDir, arcname=f"tasks/{self.taskId}", recursive=True)
        except BaseException:
            archivePath.unlink(missing_ok=True)
            raise
        return archivePath

    def _UploadSource(self, archivePath: Path) -> dict[str, Any]:
        if self.remoteRunId is None:
            raise RemoteBenchflowError("remote run has not been created")
        parsed = urlsplit(self.remoteEndpoint)
        connectionClass = (
            http.client.HTTPSConnection
            if parsed.scheme == "https"
            else http.client.HTTPConnection
        )
        connection = connectionClass(
            parsed.hostname,
            parsed.port,
            timeout=self.remoteConnectTimeout,
        )
        basePath = parsed.path.rstrip("/")
        path = f"{basePath}/v1/runs/{self.remoteRunId}/source"
        size = archivePath.stat().st_size
        connection.putrequest("PUT", path)
        connection.putheader("Content-Type", "application/gzip")
        connection.putheader("Content-Length", str(size))
        if self.remoteAuthToken:
            connection.putheader("Authorization", f"Bearer {self.remoteAuthToken}")
        connection.endheaders()
        try:
            with archivePath.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    connection.send(chunk)
            response = connection.getresponse()
            body = response.read()
        finally:
            connection.close()
        if response.status >= 400:
            raise RemoteBenchflowError(
                f"remote runtime PUT {path} returned {response.status}: "
                f"{self._ErrorMessage(body)}"
            )
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RemoteBenchflowError(
                f"remote runtime returned invalid JSON: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise RemoteBenchflowError("remote runtime response is not a JSON object")
        return payload

    def _MonitorRemote(self) -> None:
        if self.remoteRunId is None:
            return
        deadline = time.monotonic() + self.resultJsonTimeout + 30.0
        consecutiveErrors = 0
        try:
            while not self._monitorStop.is_set():
                if time.monotonic() >= deadline:
                    raise RemoteBenchflowError(
                        f"remote runtime did not finish within {self.resultJsonTimeout + 30.0:.1f}s"
                    )
                try:
                    status = self._RequestJSON(
                        "GET", f"/v1/runs/{self.remoteRunId}/status"
                    )
                    consecutiveErrors = 0
                except BaseException:
                    consecutiveErrors += 1
                    if consecutiveErrors >= 3:
                        raise
                    self._monitorStop.wait(self.remotePollInterval)
                    continue
                self.remoteState = str(status.get("state") or "")
                if self.remoteState in _TERMINAL_STATES:
                    self._CollectTerminalStatus(status)
                    return
                self._monitorStop.wait(self.remotePollInterval)
        except BaseException as exc:
            if self._stopped:
                return
            self.benchflowError = (
                f"Remote BenchFlow monitor failed: {type(exc).__name__}: {exc}"
            )
            self._CancelRemote()
            self.endpoint.finish(self.benchflowError)

    def _CollectTerminalStatus(self, status: Mapping[str, Any]) -> None:
        returnCode = status.get("process_returncode")
        self.processReturnCode = int(returnCode) if isinstance(returnCode, int) else -1
        artifact = status.get("artifact")
        if not isinstance(artifact, Mapping):
            raise RemoteBenchflowError("remote run ended without an output artifact")
        expectedSha = artifact.get("sha256")
        expectedBytes = artifact.get("bytes")
        if not isinstance(expectedSha, str) or not expectedSha:
            raise RemoteBenchflowError("remote output artifact has no SHA-256")
        if not isinstance(expectedBytes, int) or expectedBytes < 0:
            raise RemoteBenchflowError("remote output artifact has no valid size")
        self._DownloadArtifact(expectedSha, expectedBytes)
        self.ReadOfficialResult()
        remoteError = status.get("error")
        if self.officialResult is None:
            self.benchflowError = str(
                remoteError or "remote BenchFlow output has no official result.json"
            )
        elif remoteError:
            self.benchflowError = str(remoteError)
        self.endpoint.finish(self.benchflowError)

    def _DownloadArtifact(self, expectedSha: str, expectedBytes: int) -> None:
        if self.remoteRunId is None:
            raise RemoteBenchflowError("remote run has not been created")
        partPath = self.jobsDir / ".remote-artifacts.tar.gz.part"
        lastError: Optional[BaseException] = None
        for _ in range(self.artifactDownloadRetries):
            partPath.unlink(missing_ok=True)
            try:
                request = Request(
                    self._URL(f"/v1/runs/{self.remoteRunId}/artifacts"),
                    headers=self._Headers(),
                )
                digest = hashlib.sha256()
                received = 0
                with urlopen(
                    request, timeout=max(30.0, self.remoteConnectTimeout)
                ) as response:
                    with partPath.open("wb") as output:
                        while True:
                            chunk = response.read(1024 * 1024)
                            if not chunk:
                                break
                            output.write(chunk)
                            digest.update(chunk)
                            received += len(chunk)
                if received != expectedBytes:
                    raise RemoteBenchflowError(
                        f"artifact size mismatch: expected {expectedBytes}, received {received}"
                    )
                if digest.hexdigest() != expectedSha:
                    raise RemoteBenchflowError("artifact SHA-256 mismatch")
                self._ExtractArtifact(partPath)
                self.remoteArtifactPath = self.jobsDir / "remote-runtime"
                return
            except BaseException as exc:
                lastError = exc
        raise RemoteBenchflowError(
            f"could not download a valid remote output artifact: {lastError}"
        ) from lastError

    def _ExtractArtifact(self, archivePath: Path) -> None:
        staging = self.jobsDir / ".remote-runtime-staging"
        target = self.jobsDir / "remote-runtime"
        shutil.rmtree(staging, ignore_errors=True)
        if target.exists():
            raise RemoteBenchflowError(f"remote output target already exists: {target}")
        try:
            with tarfile.open(archivePath, "r:gz") as archive:
                members = archive.getmembers()
                for member in members:
                    parts = PurePosixPath(member.name).parts
                    if (
                        not parts
                        or parts[0] != "remote-runtime"
                        or PurePosixPath(member.name).is_absolute()
                        or ".." in parts
                    ):
                        raise RemoteBenchflowError(
                            f"remote artifact has an invalid path: {member.name}"
                        )
                    if (
                        member.issym()
                        or member.islnk()
                        or not (member.isdir() or member.isfile())
                    ):
                        raise RemoteBenchflowError(
                            f"remote artifact has an unsupported entry: {member.name}"
                        )
                staging.mkdir(parents=True, exist_ok=False)
                archive.extractall(staging, members=members, filter="data")
            os.replace(staging / "remote-runtime", target)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
            archivePath.unlink(missing_ok=True)

    def _RequestJSON(
        self,
        method: str,
        path: str,
        payload: Optional[Mapping[str, Any]] = None,
        *,
        timeout: Optional[float] = None,
    ) -> dict[str, Any]:
        data = None
        headers = self._Headers()
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(self._URL(path), data=data, headers=headers, method=method)
        try:
            with urlopen(
                request,
                timeout=self.remoteConnectTimeout if timeout is None else timeout,
            ) as response:
                body = response.read()
        except HTTPError as exc:
            body = exc.read()
            raise RemoteBenchflowError(
                f"remote runtime {method} {path} returned {exc.code}: "
                f"{self._ErrorMessage(body)}"
            ) from exc
        except (OSError, URLError) as exc:
            raise RemoteBenchflowError(
                f"remote runtime {method} {path} failed: {exc}"
            ) from exc
        try:
            decoded = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RemoteBenchflowError(
                f"remote runtime returned invalid JSON: {exc}"
            ) from exc
        if not isinstance(decoded, dict):
            raise RemoteBenchflowError("remote runtime response is not a JSON object")
        return decoded

    def _CancelRemote(self) -> None:
        if self.remoteRunId is None:
            return
        try:
            self._RequestJSON("DELETE", f"/v1/runs/{self.remoteRunId}")
        except BaseException:
            pass

    def _URL(self, path: str) -> str:
        return f"{self.remoteEndpoint}{path}"

    def _Headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.remoteAuthToken:
            headers["Authorization"] = f"Bearer {self.remoteAuthToken}"
        return headers

    @staticmethod
    def _ErrorMessage(body: bytes) -> str:
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return body.decode("utf-8", errors="replace")[:1000]
        if isinstance(payload, Mapping) and payload.get("error"):
            return str(payload["error"])
        return str(payload)[:1000]

    @staticmethod
    def _StatusRunId(payload: Mapping[str, Any]) -> str:
        runId = payload.get("run_id")
        if not isinstance(runId, str) or not runId:
            raise RemoteBenchflowError("remote runtime did not return a run_id")
        return runId


__all__ = ["RemoteBenchflowError", "RemoteBenchflowRunner"]
