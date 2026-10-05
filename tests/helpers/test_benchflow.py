"""Tests for benchflow."""

import importlib
import json
import sys
import threading

import pytest

from helpers.backends import ModelAdapter
from helpers.benchflow import BenchflowRunner, RemoteBenchflowRunner
from scripts.RemoteDockerRuntimeServer import CreateServer, RemoteRunManager

pytestmark = pytest.mark.usefixtures("fakeTaskTokenizer")


def test_runner_builds_real_benchflow_dataset_command(tmp_path):
    runner = BenchflowRunner(
        taskId="citation-check",
        modelPath="/models/Qwen3.8-27B",
        sourceMode="dataset",
        dataset="skillsbench@1.1",
        agent="pi-acp",
        sandbox="docker",
        skillMode="with-skill",
        providerHost="host.docker.internal",
        port=43123,
        jobsDir=tmp_path / "case",
        providerApiKey="dummy-from-test",
    )
    command = runner.BuildCommand()
    assert command[:3] == ["bench", "eval", "run"]
    assert ["--dataset", "skillsbench@1.1"] == command[3:5]
    assert "--include" in command and "citation-check" in command
    assert "--sandbox" in command and command[command.index("--sandbox") + 1] == "docker"
    assert "--skill-mode" in command and command[command.index("--skill-mode") + 1] == "with-skill"
    assert ["--usage-tracking", "off"] == command[command.index("--usage-tracking"):command.index("--usage-tracking") + 2]
    assert command[command.index("--retry-attempts"):command.index("--retry-attempts") + 2] == [
        "--retry-attempts",
        "0",
    ]
    assert "BENCHFLOW_PROVIDER_BASE_URL=http://host.docker.internal:43123/v1" in command
    assert "BENCHFLOW_PROVIDER_API_KEY=dummy-from-test" in command
    assert str(runner.jobsDir) in command
    assert runner.jobsDir.parent == tmp_path / "case"
    assert "vllm/Qwen3.8-27B" in command


def test_runner_normalizes_without_skill_alias_for_benchflow(tmp_path):
    runner = BenchflowRunner(
        taskId="citation-check",
        modelPath="/models/model",
        sourceMode="dataset",
        dataset="skillsbench@1.1",
        skillMode="WithoutSkill",
        jobsDir=tmp_path / "case",
    )

    command = runner.BuildCommand()

    assert runner.skillMode == "no-skill"
    assert command[command.index("--skill-mode") + 1] == "no-skill"


def test_runner_builds_local_tasks_dir_command(tmp_path):
    repo = tmp_path / "skillsbench"
    runner = BenchflowRunner(
        taskId="citation-check",
        modelPath="/models/model",
        sourceMode="local",
        skillsbenchDir=repo,
        jobsDir=tmp_path / "case",
    )
    command = runner.BuildCommand()
    assert command[3:5] == ["--tasks-dir", str(repo / "tasks")]
    assert command[command.index("--include") + 1] == "citation-check"


def test_runner_configures_benchflow_retry_attempts(tmp_path):
    runner = BenchflowRunner(
        taskId="citation-check",
        modelPath="/models/model",
        jobsDir=tmp_path / "case",
        retryAttempts=2,
    )

    command = runner.BuildCommand()
    index = command.index("--retry-attempts")
    assert command[index:index + 2] == ["--retry-attempts", "2"]


def test_runner_rejects_negative_benchflow_retry_attempts(tmp_path):
    with pytest.raises(ValueError, match="retryAttempts"):
        BenchflowRunner(
            taskId="citation-check",
            modelPath="/models/model",
            jobsDir=tmp_path / "case",
            retryAttempts=-1,
        )


def test_runner_reads_official_result_shape(tmp_path):
    runner = BenchflowRunner(
        taskId="citation-check",
        modelPath="/models/model",
        jobsDir=tmp_path / "case",
    )
    resultPath = runner.jobsDir / "job" / "citation-check" / "rollout" / "result.json"
    resultPath.parent.mkdir(parents=True)
    payload = {
        "task_name": "citation-check",
        "rewards": {"reward": 1.0},
        "error": None,
        "verifier_error": None,
        "n_tool_calls": 4,
        "n_skill_invocations": 1,
        "agent_result": {"n_prompts": 3},
        "final_metrics": {"reward": 1.0},
    }
    resultPath.write_text(json.dumps(payload), encoding="utf-8")
    assert runner.ReadOfficialResult() == payload
    assert runner.officialResultPath == resultPath
    assert runner.Diagnostics()["benchflow_n_tool_calls"] == 4
    assert runner.Diagnostics()["benchflow_retry_attempts"] == 0


def test_runner_attempt_directories_are_unique_with_same_pid_and_second(tmp_path, monkeypatch):
    module = importlib.import_module("helpers.benchflow.BenchflowRunner")
    monkeypatch.setattr(module.time, "strftime", lambda *args: "20261004-010101")
    first = BenchflowRunner(taskId="demo", modelPath="model", jobsDir=tmp_path, endpoint=object())
    second = BenchflowRunner(taskId="demo", modelPath="model", jobsDir=tmp_path, endpoint=object())
    assert first.jobsDir != second.jobsDir
    assert first.jobsDir.parent == second.jobsDir.parent == tmp_path


def test_runner_subprocess_is_mockable_and_lifecycle_is_explicit(tmp_path):
    class FakeProcess:
        returncode = 0

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

        def terminate(self):
            pass

        def kill(self):
            pass

    calls = []

    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return FakeProcess()

    runner = BenchflowRunner(
        taskId="citation-check",
        modelPath="/models/model",
        jobsDir=tmp_path / "case",
        popenFactory=popen,
        port=43124,
    )
    runner.start()
    runner._monitorThread.join(timeout=2)
    assert calls and calls[0][0][0:3] == ["bench", "eval", "run"]
    assert calls[0][1]["env"]["REQUEST_TIMEOUT"] == "3600.000"
    assert runner.is_done
    runner.stop()


remoteBenchflowModule = importlib.import_module(
    "helpers.benchflow.RemoteBenchflowRunner"
)


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
import os
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
    'provider_request_timeout': os.environ.get('REQUEST_TIMEOUT'),
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
            "provider_request_timeout": "10.000",
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


def test_remote_runner_uses_dedicated_source_upload_timeout(monkeypatch, tmp_path):
    seen = {}

    class Response:
        status = 200

        @staticmethod
        def read():
            return b'{"state":"ready"}'

    class Connection:
        def __init__(self, host, port, timeout):
            seen["timeout"] = timeout

        def putrequest(self, method, path):
            pass

        def putheader(self, name, value):
            pass

        def endheaders(self):
            pass

        def send(self, chunk):
            pass

        def getresponse(self):
            return Response()

        def close(self):
            pass

    monkeypatch.setattr(remoteBenchflowModule.http.client, "HTTPConnection", Connection)
    archivePath = tmp_path / "source.tar.gz"
    archivePath.write_bytes(b"source archive")
    runner = RemoteBenchflowRunner(
        taskId="demo-task",
        modelPath="/models/test-model",
        sourceMode="local",
        skillsbenchDir=tmp_path,
        remoteEndpoint="http://127.0.0.1:9000",
        remoteConnectTimeout=3,
        remoteUploadTimeout=123,
    )
    runner.remoteRunId = "run-id"

    assert runner._UploadSource(archivePath) == {"state": "ready"}
    assert seen["timeout"] == pytest.approx(123)

