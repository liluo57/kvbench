"""Remote BenchFlow control-plane and same-host A/B integration tests."""

import io
import json
import sys
import tarfile
import threading
from http.client import HTTPConnection

import pytest

from helpers.backends import ModelAdapter
from helpers.benchflow import RemoteBenchflowRunner
from scripts.RemoteDockerRuntimeServer import (
    ApiError,
    CreateServer,
    PROTOCOL_VERSION,
    RemoteRunManager,
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
    with pytest.raises(ApiError, match="may not override --jobs-dir"):
        manager.CreateRun(_spec(bench_extra_args=["--jobs-dir", "/tmp/other"]))


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


def test_remote_runner_same_host_round_trip(monkeypatch, tmp_path):
    monkeypatch.setattr(
        ModelAdapter,
        "render_chat",
        lambda messages, *, modelPath, tools=None, thinking=None: "rendered remotely",
    )
    monkeypatch.setattr(
        ModelAdapter,
        "parse_tool_calls",
        lambda output, payload, *, modelPath: (str(output), None, []),
    )

    taskId = "demo-task"
    skillsbench = tmp_path / "skillsbench"
    taskDir = skillsbench / "tasks" / taskId
    taskDir.mkdir(parents=True)
    (taskDir / "task.md").write_text("# remote demo\n", encoding="utf-8")
    (taskDir / "input.txt").write_text("uploaded from A\n", encoding="utf-8")

    fakeBench = tmp_path / "fake_bench.py"
    fakeBench.write_text(
        """
import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen

args = sys.argv[1:]
jobs_dir = Path(args[args.index('--jobs-dir') + 1])
tasks_dir = Path(args[args.index('--tasks-dir') + 1])
task_id = args[args.index('--include') + 1]
agent_env = [args[i + 1] for i, value in enumerate(args) if value == '--agent-env']
env = dict(value.split('=', 1) for value in agent_env)
payload = json.dumps({
    'model': 'vllm/test-model',
    'messages': [{'role': 'user', 'content': 'hello from B'}],
}).encode('utf-8')
request = Request(
    env['BENCHFLOW_PROVIDER_BASE_URL'].rstrip('/') + '/chat/completions',
    data=payload,
    headers={
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + env['BENCHFLOW_PROVIDER_API_KEY'],
    },
)
with urlopen(request, timeout=10) as response:
    model_response = json.loads(response.read())
rollout = jobs_dir / 'fake-job' / (task_id + '__remote')
(rollout / 'artifacts').mkdir(parents=True)
(rollout / 'artifacts' / 'source.txt').write_text(
    (tasks_dir / task_id / 'input.txt').read_text(encoding='utf-8'),
    encoding='utf-8',
)
(rollout / 'result.json').write_text(json.dumps({
    'task_name': task_id,
    'rewards': {'reward': 1.0},
    'agent_result': model_response['choices'][0]['message']['content'],
}), encoding='utf-8')
""",
        encoding="utf-8",
    )

    manager = RemoteRunManager(
        workRoot=tmp_path / "runtime",
        benchCommand=[sys.executable, str(fakeBench)],
        validateDockerImages=False,
    )
    server = CreateServer(
        host="127.0.0.1",
        port=0,
        manager=manager,
        authToken="control-secret",
    )
    serverThread = threading.Thread(target=server.serve_forever, daemon=True)
    serverThread.start()

    runner = RemoteBenchflowRunner(
        taskId=taskId,
        modelPath="/models/test-model",
        sourceMode="local",
        skillsbenchDir=skillsbench,
        jobsDir=tmp_path / "outputs",
        endpointHost="127.0.0.1",
        remoteEndpoint=f"http://127.0.0.1:{server.server_address[1]}",
        advertiseHost="127.0.0.1",
        remoteAuthToken="control-secret",
        remotePollInterval=0.05,
        remoteConnectTimeout=3,
        resultJsonTimeout=10,
    )
    try:
        runner.start()
        request = runner.wait_for_request(timeout=5)
        assert request is not None
        assert request.prompt == "rendered remotely"
        runner.respond(request, "answer from A")
        runner._monitorThread.join(timeout=10)

        assert runner.is_done
        assert runner.benchflowError is None
        assert runner.officialResult == {
            "task_name": taskId,
            "rewards": {"reward": 1.0},
            "agent_result": "answer from A",
        }
        assert runner.officialResultPath is not None
        assert "remote-runtime" in runner.officialResultPath.parts
        returnedSource = next(
            (runner.jobsDir / "remote-runtime").rglob("artifacts/source.txt")
        )
        assert returnedSource.read_text(encoding="utf-8") == "uploaded from A\n"
        diagnostics = runner.Diagnostics()
        assert diagnostics["remote_state"] == "succeeded"
        assert diagnostics["remote_run_id"]
    finally:
        runner.stop()
        server.shutdown()
        server.server_close()
        manager.Close()
        serverThread.join(timeout=2)


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
