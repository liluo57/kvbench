"""Tests for remote runtime."""

import importlib
import io
import json
import sys
import tarfile
import threading
from http.client import HTTPConnection

import pytest

from scripts.RemoteDockerRuntimeServer import (
    PROTOCOL_VERSION,
    ApiError,
    CreateServer,
    RemoteRunManager,
)

remoteDockerModule = importlib.import_module(
    "scripts.RemoteDockerRuntimeServer"
)


def _spec(**overrides):
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "client_run_id": "client-run",
        "task_id": "demo-task",
        "source_mode": "local",
        "dataset": None,
        "agent": "pi-acp",
        "skill_mode": "with-skill",
        "model_id": "test-model",
        "provider_base_url": "http://127.0.0.1:1/v1",
        "provider_api_key": "provider-secret",
        "result_json_timeout": 10,
        "retry_attempts": 0,
        "bench_extra_args": [],
    }
    payload.update(overrides)
    return payload


def test_remote_server_rejects_source_path_traversal(tmp_path):
    manager = RemoteRunManager(
        workRoot=tmp_path / "runtime",
        benchCommand="bench",
        validateDockerImages=False,
    )
    record = manager.CreateRun(_spec())
    archiveBuffer = io.BytesIO()
    with tarfile.open(fileobj=archiveBuffer, mode="w:gz") as archive:
        content = b"escape"
        member = tarfile.TarInfo("tasks/demo-task/../../escaped.txt")
        member.size = len(content)
        archive.addfile(member, io.BytesIO(content))
    archiveBuffer.seek(0)

    with pytest.raises(ApiError, match="invalid path"):
        manager.UploadSource(record.runId, archiveBuffer, len(archiveBuffer.getvalue()))
    assert not (tmp_path / "runtime" / "escaped.txt").exists()


def test_remote_server_rejects_reserved_bench_arguments(tmp_path):
    manager = RemoteRunManager(workRoot=tmp_path / "runtime")
    other_jobs = str(tmp_path / "other-jobs")
    with pytest.raises(ApiError, match="may not override --jobs-dir"):
        manager.CreateRun(_spec(bench_extra_args=["--jobs-dir", other_jobs]))


def test_remote_server_forwards_client_agent_env_into_command(tmp_path):
    manager = RemoteRunManager(
        workRoot=tmp_path / "runtime",
        benchCommand="bench",
        validateDockerImages=False,
    )
    record = manager.CreateRun(
        _spec(
            bench_extra_args=[
                "--agent-env",
                "REQUEST_TIMEOUT=18000",
                "--some-flag",
                "value",
            ]
        )
    )
    command = manager._BuildCommand(record)
    assert "--agent-env" in command
    assert "REQUEST_TIMEOUT=18000" in command
    assert "BENCHFLOW_PROVIDER_BASE_URL=http://127.0.0.1:1/v1" in command
    assert "BENCHFLOW_PROVIDER_API_KEY=provider-secret" in command
    # Server-controlled --agent-env lines precede the client's.
    baseIdx = command.index("BENCHFLOW_PROVIDER_BASE_URL=http://127.0.0.1:1/v1")
    clientIdx = command.index("REQUEST_TIMEOUT=18000")
    assert baseIdx < clientIdx
    # Non-agent-env extras still flow through verbatim.
    assert command[command.index("--some-flag") + 1] == "value"


def test_remote_server_forwards_benchflow_retry_attempts(tmp_path):
    manager = RemoteRunManager(
        workRoot=tmp_path / "runtime",
        benchCommand="bench",
        validateDockerImages=False,
    )
    record = manager.CreateRun(_spec(retry_attempts=2))
    command = manager._BuildCommand(record)
    index = command.index("--retry-attempts")
    assert command[index:index + 2] == ["--retry-attempts", "2"]


def test_remote_server_normalizes_without_skill_mode(tmp_path):
    manager = RemoteRunManager(workRoot=tmp_path / "runtime")
    record = manager.CreateRun(_spec(skill_mode="WithoutSkill"))

    command = manager._BuildCommand(record)

    assert record.spec["skill_mode"] == "no-skill"
    assert command[command.index("--skill-mode") + 1] == "no-skill"


def test_remote_server_rejects_client_agent_env_override_attempts(tmp_path):
    manager = RemoteRunManager(workRoot=tmp_path / "runtime")
    with pytest.raises(ApiError, match="reserved key BENCHFLOW_PROVIDER_BASE_URL"):
        manager.CreateRun(
            _spec(
                bench_extra_args=[
                    "--agent-env",
                    "BENCHFLOW_PROVIDER_BASE_URL=http://evil",
                ]
            )
        )
    with pytest.raises(ApiError, match="requires KEY=VALUE"):
        manager.CreateRun(_spec(bench_extra_args=["--agent-env"]))
    with pytest.raises(ApiError, match="must match KEY=VALUE"):
        manager.CreateRun(
            _spec(bench_extra_args=["--agent-env", "REQUEST_TIMEOUT"])
        )


def _MakeLocalSource(taskDir, taskMdText):
    taskDir.mkdir(parents=True)
    (taskDir / "task.md").write_text(taskMdText, encoding="utf-8")
    (taskDir / "input.txt").write_text("input\n", encoding="utf-8")


def test_remote_server_skips_docker_check_when_image_unpinned(tmp_path):
    """SkillsBench 的 task.md 普遍不带 docker_image；该情况下跳过预检查。"""
    skillsbench = tmp_path / "skillsbench"
    taskDir = skillsbench / "tasks" / "demo-task"
    _MakeLocalSource(
        taskDir,
        "---\n"
        "sandbox:\n"
        "  cpus: 1\n"
        "  memory_mb: 1024\n"
        "---\n"
        "# demo without docker_image\n",
    )
    manager = RemoteRunManager(
        workRoot=tmp_path / "runtime",
        benchCommand="bench",
        validateDockerImages=True,
    )
    record = manager.CreateRun(
        _spec(source_mode="local", dataset="skillsbench@1.1")
    )
    archiveBuffer = io.BytesIO()
    with tarfile.open(fileobj=archiveBuffer, mode="w:gz") as archive:
        archive.add(taskDir, arcname="tasks/demo-task", recursive=True)
    archiveBuffer.seek(0)
    manager.UploadSource(
        record.runId,
        archiveBuffer,
        len(archiveBuffer.getvalue()),
    )
    # Re-parse image from real task.md and confirm it's empty, so the check
    # would have to 400 under the old behaviour.
    from benchflow.task.document import TaskDocument

    parsedImage = TaskDocument.from_path(taskDir / "task.md").config.sandbox.docker_image
    assert not parsedImage
    # Should NOT raise: empty docker_image means skip the check.
    manager._CheckLocalDockerImage(record)


def test_remote_server_uses_existing_conventional_prebuilt_image(
    monkeypatch, tmp_path
):
    """An unmodified local task should still reuse a prepared image."""

    skillsbench = tmp_path / "skillsbench"
    taskDir = skillsbench / "tasks" / "exoplanet-detection-period"
    _MakeLocalSource(
        taskDir,
        "---\n"
        "sandbox:\n"
        "  cpus: 1\n"
        "  memory_mb: 1024\n"
        "---\n"
        "# demo\n",
    )
    manager = RemoteRunManager(
        workRoot=tmp_path / "runtime",
        benchCommand="bench",
        validateDockerImages=True,
    )
    record = manager.CreateRun(
        _spec(task_id="exoplanet-detection-period", source_mode="local")
    )
    archiveBuffer = io.BytesIO()
    with tarfile.open(fileobj=archiveBuffer, mode="w:gz") as archive:
        archive.add(taskDir, arcname="tasks/exoplanet-detection-period", recursive=True)
    archiveBuffer.seek(0)
    manager.UploadSource(
        record.runId,
        archiveBuffer,
        len(archiveBuffer.getvalue()),
    )

    monkeypatch.setattr(
        remoteDockerModule.shutil,
        "which",
        lambda _: str(tmp_path / "docker"),
    )

    class InspectResult:
        returncode = 0

    monkeypatch.setattr(
        remoteDockerModule.subprocess,
        "run",
        lambda *args, **kwargs: InspectResult(),
    )
    manager._UsePrebuiltTaskImage(record)

    taskFile = record.runDir / "source" / "tasks" / "exoplanet-detection-period" / "task.md"
    assert "image: kvbench-skillsbench/exoplanet-detection-period:latest" in (
        taskFile.read_text(encoding="utf-8")
    )
    manager._CheckLocalDockerImage(record)


def test_remote_server_times_out_process_and_still_packages_logs(monkeypatch, tmp_path):
    manager = RemoteRunManager(
        workRoot=tmp_path / "runtime",
        benchCommand=[sys.executable, "-c", "import time; time.sleep(30)"],
        validateDockerImages=False,
    )
    monkeypatch.setattr(manager, "_CheckProvider", lambda record: None)
    record = manager.CreateRun(
        _spec(
            source_mode="dataset",
            dataset="skillsbench@1.1",
            result_json_timeout=0.1,
        )
    )
    manager.StartRun(record.runId)
    record.monitorThread.join(timeout=5)
    status = record.PublicStatus()
    assert status["state"] == "failed"
    assert "did not finish within" in status["error"]
    assert status["artifact"]["sha256"]
    assert record.process.poll() is not None


def test_remote_control_endpoint_requires_bearer_token(tmp_path):
    manager = RemoteRunManager(workRoot=tmp_path / "runtime")
    server = CreateServer(
        host="127.0.0.1",
        port=0,
        manager=manager,
        authToken="control-secret",
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = HTTPConnection("127.0.0.1", server.server_address[1], timeout=3)
        connection.request("GET", "/health")
        response = connection.getresponse()
        assert response.status == 401
        response.read()
        connection.close()

        connection = HTTPConnection("127.0.0.1", server.server_address[1], timeout=3)
        connection.request(
            "GET",
            "/health",
            headers={"Authorization": "Bearer control-secret"},
        )
        response = connection.getresponse()
        assert response.status == 200
        assert json.loads(response.read())["protocol_version"] == PROTOCOL_VERSION
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
