"""Boundary tests for the KVBench endpoint and real BenchFlow bridge."""

import json
from concurrent.futures import Future
import threading
from http.client import HTTPConnection
from pathlib import Path

import pytest

from core.Result import Result, TtftKey
from core.Workload import ActionKind, ActionResult
from helpers.backends import ModelAdapter
from helpers.benchflow import BenchflowRunner
from helpers.endpoint import KVBenchEndpoint, OpenAIRequest
from tasks.AgentBenchFlowTask import AgentBenchFlowTask
from workload.AgentBenchFlowWorkload import (
    AgentBenchFlowInput,
    AgentBenchFlowWorkload,
    _ExtractSkillDocuments,
)


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


def test_endpoint_optional_bearer_authentication(tmp_path):
    server = KVBenchEndpoint(
        modelPath="/models/test",
        host="127.0.0.1",
        apiKey="provider-secret",
    ).start()
    try:
        connection = HTTPConnection("127.0.0.1", server.port, timeout=3)
        connection.request("GET", "/health")
        response = connection.getresponse()
        assert response.status == 401
        response.read()
        connection.close()

        connection = HTTPConnection("127.0.0.1", server.port, timeout=3)
        connection.request(
            "GET",
            "/health",
            headers={"Authorization": "Bearer provider-secret"},
        )
        response = connection.getresponse()
        assert response.status == 200
        assert json.loads(response.read()) == {"status": "ok"}
        connection.close()
    finally:
        server.stop()


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


def test_endpoint_uses_native_finish_and_stop_reason(endpoint):
    server, _calls = endpoint
    thread, request, result = _serve_one(
        server,
        {"model": "vllm/test", "messages": [{"role": "user", "content": "hi"}]},
    )
    server.respond(
        request,
        "partial output",
        finishReason="length",
        stopReason=123,
    )
    thread.join(timeout=2)
    body = json.loads(result["value"][2])
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["choices"][0]["stop_reason"] == 123


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


def test_endpoint_stream_sends_heartbeat_while_generation_is_pending(endpoint):
    server, _calls = endpoint
    server.sseHeartbeatSec = 0.05
    connection = HTTPConnection("127.0.0.1", server.port, timeout=3)
    connection.request(
        "POST",
        "/v1/chat/completions",
        body=json.dumps(
            {
                "model": "vllm/test",
                "messages": [{"role": "user", "content": "wait"}],
                "stream": True,
            }
        ),
        headers={"Content-Type": "application/json"},
    )
    response = connection.getresponse()
    assert response.status == 200
    assert response.getheader("X-Accel-Buffering") == "no"
    assert response.fp.readline().startswith(b": kvbench keep-alive")
    assert response.fp.readline() == b"\n"

    request = server.wait_for_request(timeout=2)
    assert request is not None
    server.respond(request, "hello")
    body = response.read()
    connection.close()
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
    assert [record["phase"] for record in records] == [
        "request",
        "response_wait_started",
        "response_wait_finished",
        "response",
    ]
    assert records[0]["unsupported_generation_fields"] == ["temperature"]
    assert records[3]["raw_output"] == "raw output"


def test_endpoint_finish_drains_queued_but_not_inflight(endpoint):
    server, _calls = endpoint
    inflight = server._MakeRequest({"model": "vllm/test", "messages": []})
    queued = server._MakeRequest({"model": "vllm/test", "messages": []})
    server._Enqueue(inflight)
    server._Enqueue(queued)

    assert server.wait_for_request(timeout=1) is inflight
    server.finish("BenchFlow exited")

    assert not inflight.responseFuture.done()
    assert queued.responseFuture.done()

    server.respond(inflight, "raw output")
    assert server.wait_for_request(timeout=1) is None


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
    assert str(runner.jobsDir) in command
    assert runner.jobsDir.parent == tmp_path / "case"
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


def _request(prompt, messages=None):
    messages = [] if messages is None else messages
    return OpenAIRequest(
        requestId="req",
        payload={"model": "vllm/model", "messages": messages},
        messages=messages,
        tools=None,
        prompt=prompt,
        stream=False,
        responseFuture=Future(),
    )


def _skill_messages(skill_path, content, call_id="skill-call"):
    return [
        {
            "role": "system",
            "content": "available skills: " + skill_path,
        },
        {
            "role": "user",
            "content": "TASK: do not prepare this task description",
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "read",
                        "arguments": json.dumps({"path": skill_path}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": content,
        },
    ]


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


def test_skill_document_extraction_excludes_system_task_and_other_tool_output():
    skill = "---\nname: demo-skill\n---\n\nUse the skill.\n"
    messages = _skill_messages("/home/agent/.pi/agent/skills/demo/SKILL.md", skill)
    messages[1]["content"] += " /tmp/fake/SKILL.md"
    messages.extend(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "other-call",
                        "type": "function",
                        "function": {
                            "name": "read",
                            "arguments": '{"path":"/tmp/notes.txt"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "other-call",
                "content": "not a skill document",
            },
        ]
    )

    assert _ExtractSkillDocuments(messages) == [skill]


def test_workload_prepares_only_skill_documents_before_the_original_run_prompt():
    skill = "---\nname: demo-skill\n---\n\nUse the skill.\n"
    first = _request("first rendered prompt")
    second = _request(
        "second rendered prompt",
        _skill_messages("/home/agent/.pi/agent/skills/demo/SKILL.md", skill),
    )
    runner = _FakeRunner([first, second])
    workload = AgentBenchFlowWorkload(
        case_id=3,
        data=AgentBenchFlowInput(task_id="citation-check"),
        runner=runner,
    )

    firstAction = workload.next()[0]
    assert firstAction.kind.value == "run"
    workload.observe([ActionResult(3, Result(output="first output"))])

    prepare = workload.next()[0]
    assert prepare.kind == ActionKind.PREPARE
    assert prepare.data == [skill]
    assert "available skills" not in prepare.data[0]
    assert "TASK:" not in prepare.data[0]

    workload.observe([ActionResult(3, Result())])
    secondAction = workload.next()[0]
    assert secondAction.kind == ActionKind.RUN
    assert secondAction.data == "second rendered prompt"
    workload.observe([ActionResult(3, Result(output="second output"))])

    assert runner.responses == [
        (first, "first output"),
        (second, "second output"),
    ]


def test_local_task_skills_are_prepared_once_before_first_run(tmp_path):
    skillPath = (
        tmp_path
        / "tasks"
        / "demo-task"
        / "environment"
        / "skills"
        / "demo"
        / "SKILL.md"
    )
    skillPath.parent.mkdir(parents=True)
    skill = "---\nname: demo\n---\n\nLocal skill body\n"
    skillPath.write_text(skill, encoding="utf-8")

    request = _request("rendered prompt without a skill tool result")
    runner = _FakeRunner([request])
    workload = AgentBenchFlowWorkload(
        case_id=3,
        data=AgentBenchFlowInput(
            task_id="demo-task",
            source_mode="local",
            skillsbench_dir=str(tmp_path),
        ),
        runner=runner,
    )

    prepare = workload.next()[0]
    assert prepare.kind == ActionKind.PREPARE
    assert prepare.data == [skill]
    workload.observe([ActionResult(3, Result())])
    run = workload.next()[0]
    assert run.kind == ActionKind.RUN
    assert skill in run.data
    assert run.data.endswith("rendered prompt without a skill tool result")


def test_first_run_skill_injection_uses_chat_template_when_request_has_metadata(
    monkeypatch, tmp_path
):
    skillPath = (
        tmp_path
        / "tasks"
        / "demo-task"
        / "environment"
        / "skills"
        / "demo"
        / "SKILL.md"
    )
    skillPath.parent.mkdir(parents=True)
    skill = "---\nname: demo\n---\n\nUse the skill.\n"
    skillPath.write_text(skill, encoding="utf-8")

    renderCalls = []

    def render(messages, *, modelPath, tools=None, thinking=None):
        renderCalls.append((messages, modelPath, tools, thinking))
        return "rendered with " + messages[0]["content"]

    monkeypatch.setattr(ModelAdapter, "render_chat", render)
    request = _request("original rendered prompt")
    request.modelPath = "/models/test"
    request.thinking = True
    runner = _FakeRunner([request])
    workload = AgentBenchFlowWorkload(
        case_id=3,
        data=AgentBenchFlowInput(
            task_id="demo-task",
            source_mode="local",
            skillsbench_dir=str(tmp_path),
        ),
        runner=runner,
    )

    workload.next()
    workload.observe([ActionResult(3, Result())])
    run = workload.next()[0]

    assert run.data.startswith("rendered with")
    assert skill in renderCalls[0][0][0]["content"]
    assert renderCalls[0][1:] == ("/models/test", None, True)
    assert "original rendered prompt" not in run.data


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


def test_task_filters_and_propagates_benchflow_configuration(
    monkeypatch, fakeSkillsbench
):
    monkeypatch.setattr(AgentBenchFlowTask, "_EnsureLocalImages", lambda *args: None)
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


def test_task_selects_remote_runtime_without_local_docker_validation(fakeSkillsbench):
    task = AgentBenchFlowTask(
        skillsbench_dir=fakeSkillsbench,
        source_mode="local",
        task_ids=["citation-check"],
        sandbox="remote-docker",
    )
    case = next(iter(task.Cases()))
    assert case.input.sandbox == "remote-docker"
    assert case.input.remote_endpoint == "http://127.0.0.1:8765"
    assert case.input.remote_advertise_host is None
    assert case.input.remote_poll_interval == pytest.approx(1.0)


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


def test_workload_captures_first_run_ttft_and_reuse_ratio_only_once():
    first = _request("first rendered prompt")
    second = _request("second rendered prompt")
    runner = _FakeRunner([first, second])
    workload = AgentBenchFlowWorkload(
        case_id=3,
        data=AgentBenchFlowInput(task_id="citation-check"),
        runner=runner,
    )

    # First RUN: capture ttft + reuse_ratio.
    workload.next()
    workload.observe(
        [
            ActionResult(
                3,
                Result(
                    output="first output",
                    performance={TtftKey: 0.42},
                    metadata={"reuse_ratio": 0.85},
                ),
            )
        ]
    )

    # Second RUN: very different numbers — must NOT overwrite the first-run
    # capture, since the report is per-case first-RUN.
    workload.next()
    workload.observe(
        [
            ActionResult(
                3,
                Result(
                    output="second output",
                    performance={TtftKey: 0.99},
                    metadata={"reuse_ratio": 0.10},
                ),
            )
        ]
    )
    workload.next()  # returns None, finishes the workload

    final = workload.final_result
    assert final.metadata["first_run_ttft"] == pytest.approx(0.42)
    assert final.metadata["first_run_reuse_ratio"] == pytest.approx(0.85)


def test_workload_first_run_capture_is_absent_when_no_run_completes():
    runner = _FailingRunner()
    workload = AgentBenchFlowWorkload(
        case_id=3,
        data=AgentBenchFlowInput(task_id="citation-check"),
        runner=runner,
    )

    workload.next()  # runner.start() raises -> fail() runs

    final = workload.final_result
    assert final.metadata.get("first_run_ttft") is None
    assert final.metadata.get("first_run_reuse_ratio") is None
    assert final.output["error"]


def test_workload_first_run_metadata_is_attached_even_after_a_mid_run_failure():
    first = _request("first rendered prompt")
    runner = _FakeRunner([first])

    def boom(_request, _output):
        raise RuntimeError("model crashed mid-rollout")

    runner.respond = boom
    workload = AgentBenchFlowWorkload(
        case_id=3,
        data=AgentBenchFlowInput(task_id="citation-check"),
        runner=runner,
    )

    workload.next()
    workload.observe(
        [
            ActionResult(
                3,
                Result(
                    output="first output",
                    performance={TtftKey: 0.21},
                    metadata={"reuse_ratio": 0.73},
                ),
            )
        ]
    )
    # observe()'s respond() raises -> fail() is called, which must keep the
    # already-captured first-RUN stats.
    final = workload.final_result
    assert final.metadata["first_run_ttft"] == pytest.approx(0.21)
    assert final.metadata["first_run_reuse_ratio"] == pytest.approx(0.73)


def test_task_evaluate_surfaces_first_run_task_scores_when_present():
    task = AgentBenchFlowTask(skillsbench_dir=Path("/tmp"), task_ids=["citation-check"])
    result = Result(
        output={"rewards": {"reward": 1.0}},
        metadata={
            "first_run_ttft": 0.38,
            "first_run_reuse_ratio": 0.83,
        },
    )
    assert task.Evaluate(result, {}) == {
        "reward": 1.0,
        "accuracy": 1.0,
        "first_run_ttft": pytest.approx(0.38),
        "first_run_reuse_ratio": pytest.approx(0.83),
    }


def test_task_evaluate_omits_first_run_scores_when_unavailable():
    task = AgentBenchFlowTask(skillsbench_dir=Path("/tmp"), task_ids=["citation-check"])
    result = Result(
        output={"rewards": {"reward": 0.0}, "error": "no provider reachable"},
        metadata={"benchflow_error": "no provider reachable"},
    )
    scores = task.Evaluate(result, {})
    assert "first_run_ttft" not in scores
    assert "first_run_reuse_ratio" not in scores
