"""Boundary tests for the KVBench endpoint and real BenchFlow bridge."""

import json
from concurrent.futures import Future
import threading
from http.client import HTTPConnection
from pathlib import Path

import pytest

from core.Result import Result
from core.Workload import ActionResult
from helpers.backends import ModelAdapter
from helpers.benchflow import BenchflowRunner
from helpers.endpoint import KVBenchEndpoint, OpenAIRequest
from tasks.AgentBenchFlowTask import AgentBenchFlowTask
from workload.AgentBenchFlowWorkload import AgentBenchFlowInput, AgentBenchFlowWorkload


@pytest.fixture
def fakeSkillsbench(tmp_path):
    for taskId in ("alpha", "beta", "citation-check"):
        taskDir = tmp_path / "tasks" / taskId
        taskDir.mkdir(parents=True)
        (taskDir / "task.md").write_text(f"# {taskId}\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def endpoint(monkeypatch, tmp_path):
    renderCalls = []

    def render(messages, *, modelPath, tools=None, thinking=None):
        renderCalls.append(
            {
                "messages": messages,
                "modelPath": modelPath,
                "tools": tools,
                "thinking": thinking,
            }
        )
        return "rendered prompt"

    def parse(output, payload, *, modelPath):
        if output == "tool-output":
            return "", "internal reasoning", [{
                "id": "call-1",
                "type": "function",
                "function": {"name": "bash", "arguments": '{"command":"ls"}'},
            }]
        return output, None, []

    monkeypatch.setattr(ModelAdapter, "render_chat", render)
    monkeypatch.setattr(ModelAdapter, "parse_tool_calls", parse)
    server = KVBenchEndpoint(
        modelPath="/models/test",
        host="127.0.0.1",
        debugLogPath=tmp_path / "llm.jsonl",
    ).start()
    yield server, renderCalls
    server.stop()


def _post(endpoint, payload):
    connection = HTTPConnection("127.0.0.1", endpoint.port, timeout=3)
    connection.request(
        "POST",
        "/v1/chat/completions",
        body=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )
    response = connection.getresponse()
    body = response.read()
    connection.close()
    return response.status, response.getheader("Content-Type"), body


def _serve_one(endpoint, payload):
    result = {}

    def call():
        result["value"] = _post(endpoint, payload)

    thread = threading.Thread(target=call)
    thread.start()
    request = endpoint.wait_for_request(timeout=2)
    assert request is not None
    return thread, request, result


def test_endpoint_health(endpoint):
    server, _calls = endpoint
    connection = HTTPConnection("127.0.0.1", server.port, timeout=3)
    connection.request("GET", "/health")
    response = connection.getresponse()
    assert response.status == 200
    assert json.loads(response.read()) == {"status": "ok"}
    connection.close()


def test_endpoint_renders_and_queues_without_skill_prefix(endpoint):
    server, renderCalls = endpoint
    payload = {
        "model": "vllm/test",
        "messages": [{"role": "system", "content": "system"}, {"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "bash"}}],
    }
    thread, request, result = _serve_one(server, payload)
    assert request.prompt == "rendered prompt"
    assert renderCalls[0]["tools"] == payload["tools"]
    assert "system_prefix" not in renderCalls[0]
    assert request.payload == payload
    server.respond(request, "hello")
    thread.join(timeout=2)
    assert result["value"][0] == 200
    body = json.loads(result["value"][2])
    assert body["choices"][0]["message"]["content"] == "hello"
    assert body["choices"][0]["finish_reason"] == "stop"


def test_endpoint_tool_calls_and_reasoning(endpoint):
    server, _calls = endpoint
    thread, request, result = _serve_one(
        server,
        {"model": "vllm/test", "messages": [{"role": "user", "content": "run ls"}]},
    )
    server.respond(request, "tool-output")
    thread.join(timeout=2)
    body = json.loads(result["value"][2])
    message = body["choices"][0]["message"]
    assert message["content"] == ""
    assert message["reasoning_content"] == "internal reasoning"
    assert message["tool_calls"][0]["function"]["name"] == "bash"
    assert body["choices"][0]["finish_reason"] == "tool_calls"


def test_endpoint_stream_true_returns_sse(endpoint):
    server, _calls = endpoint
    thread, request, result = _serve_one(
        server,
        {
            "model": "vllm/test",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    server.respond(request, "hello")
    thread.join(timeout=2)
    status, contentType, body = result["value"]
    assert status == 200
    assert contentType == "text/event-stream"
    assert b"chat.completion.chunk" in body
    assert body.endswith(b"data: [DONE]\n\n")


def test_endpoint_malformed_request_is_4xx(endpoint):
    server, _calls = endpoint
    status, _contentType, body = _post(server, {"model": "vllm/test"})
    assert status == 400
    assert "messages must be a list" in json.loads(body)["error"]["message"]


def test_endpoint_logs_raw_request_and_response(endpoint, tmp_path):
    server, _calls = endpoint
    thread, request, result = _serve_one(
        server,
        {
            "model": "vllm/test",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.2,
        },
    )
    server.respond(request, "raw output")
    thread.join(timeout=2)
    assert result["value"][0] == 200
    records = [json.loads(line) for line in (tmp_path / "llm.jsonl").read_text().splitlines()]
    assert [record["phase"] for record in records] == ["request", "response"]
    assert records[0]["unsupported_generation_fields"] == ["temperature"]
    assert records[1]["raw_output"] == "raw output"


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
    assert f"BENCHFLOW_PROVIDER_BASE_URL=http://host.docker.internal:43123/v1" in command
    assert "BENCHFLOW_PROVIDER_API_KEY=dummy-from-test" in command
    assert str(tmp_path / "case") in command
    assert "vllm/Qwen3.8-27B" in command


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


def test_runner_reads_official_result_shape(tmp_path):
    resultPath = tmp_path / "case" / "job" / "citation-check" / "rollout" / "result.json"
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
    runner = BenchflowRunner(
        taskId="citation-check",
        modelPath="/models/model",
        jobsDir=tmp_path / "case",
    )
    assert runner.ReadOfficialResult() == payload
    assert runner.officialResultPath == resultPath
    assert runner.Diagnostics()["benchflow_n_tool_calls"] == 4


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
    assert runner.is_done
    runner.stop()


class _FakeRunner:
    def __init__(self, requests):
        self.requests = list(requests)
        self.responses = []
        self.officialResult = {
            "task_name": "citation-check",
            "rewards": {"reward": 0.75},
            "error": "",
        }
        self.started = False

    def start(self):
        self.started = True
        return self

    def wait_for_request(self):
        return self.requests.pop(0) if self.requests else None

    def respond(self, request, output):
        self.responses.append((request, output))

    def Diagnostics(self):
        return {"official_result_path": "/jobs/result.json", "benchflow_error": None}

    def stop(self):
        pass


class _FailingRunner(_FakeRunner):
    def __init__(self):
        super().__init__([])
        self.stopped = False

    def start(self):
        raise RuntimeError("BenchFlow task failed to start")

    def stop(self):
        self.stopped = True


def _request(prompt):
    return OpenAIRequest(
        requestId="req",
        payload={"model": "vllm/model", "messages": []},
        messages=[],
        tools=None,
        prompt=prompt,
        stream=False,
        responseFuture=Future(),
    )


def test_workload_bridges_multiple_turns_and_retains_output():
    first = _request("first rendered prompt")
    second = _request("second rendered prompt")
    runner = _FakeRunner([first, second])
    workload = AgentBenchFlowWorkload(
        case_id=3,
        data=AgentBenchFlowInput(task_id="citation-check"),
        runner=runner,
    )
    action = workload.next()[0]
    assert runner.started
    assert action.kind.value == "run"
    assert action.data == "first rendered prompt"
    assert action.retainOutput is True
    workload.observe([ActionResult(3, Result(output="first output"))])
    assert runner.responses == [(first, "first output")]
    assert workload.next()[0].data == "second rendered prompt"
    workload.observe([ActionResult(3, Result(output="second output"))])
    assert workload.next() is None
    assert workload.finished
    assert workload.final_result.output["rewards"]["reward"] == 0.75


def test_workload_converts_runner_failure_to_zero_score():
    runner = _FailingRunner()
    workload = AgentBenchFlowWorkload(
        case_id=3,
        data=AgentBenchFlowInput(task_id="broken-task"),
        runner=runner,
    )

    assert workload.next() is None
    assert workload.finished
    assert runner.stopped
    assert workload.final_result.output["reward"] == 0.0


def test_task_filters_and_propagates_benchflow_configuration(fakeSkillsbench):
    task = AgentBenchFlowTask(
        skillsbench_dir=fakeSkillsbench,
        source_mode="local",
        task_ids=["citation-check", "alpha"],
        exclude_task_ids=["alpha"],
        agent="opencode",
        skill_mode="no-skill",
        provider_host="host.docker.internal",
    )
    case = next(iter(task.Cases()))
    assert case.metadata["task_id"] == "citation-check"
    assert case.input.agent == "opencode"
    assert case.input.skill_mode == "no-skill"
    assert case.input.source_mode == "local"
    assert case.input.provider_host == "host.docker.internal"


@pytest.mark.parametrize(
    "payload,expected",
    [
        ({"rewards": {"reward": 1.0}}, 1.0),
        ({"rewards": {"criteria_a": 1.0, "criteria_b": 0.0}}, 0.5),
        ({"rewards": [0.0, 1.0]}, 0.5),
        ({"final_metrics": {"pass_rate": 0.25}}, 0.25),
        ({"error": "verifier failed", "rewards": None}, 0.0),
    ],
)
def test_task_extracts_official_rewards(payload, expected):
    assert AgentBenchFlowTask._ExtractReward(payload) == pytest.approx(expected)


def test_task_evaluate_keeps_infrastructure_diagnostics_available():
    task = AgentBenchFlowTask(skillsbench_dir=Path("/tmp"), task_ids=["citation-check"])
    result = Result(
        output={"rewards": {"reward": 0.0}, "error": "Docker environment failed"},
        metadata={"benchflow_error": "Docker environment failed"},
    )
    assert task.Evaluate(result, {}) == {"reward": 0.0, "accuracy": 0.0}
    assert result.metadata["benchflow_error"] == "Docker environment failed"
