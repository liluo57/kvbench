"""Tests for agent benchflow."""

import json
from concurrent.futures import Future

import pytest

from core.Result import Result, TtftKey
from core.Workflow import ActionKind, ActionResult
from helpers.backends import ModelAdapter
from helpers.backends.Prompt import ComposeInterleavedReuseText
from helpers.endpoint import OpenAIRequest
from workflow.AgentBenchFlowWorkflow import (
    AgentBenchFlowInput,
    AgentBenchFlowPreRunError,
    AgentBenchFlowWorkflow,
    _CanonicalSkillDocument,
    _ExtractSkillDocuments,
)

pytestmark = pytest.mark.usefixtures("fakeTaskTokenizer")


def _decodeTokenPrompt(prompt):
    return "".join(chr(token) for token in prompt)


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
        self.stopped = False

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
        self.stopped = True


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


def test_workflow_bridges_multiple_turns_and_retains_output():
    first = _request("first rendered prompt")
    second = _request("second rendered prompt")
    runner = _FakeRunner([first, second])
    workflow = AgentBenchFlowWorkflow(
        case_id=3,
        data=AgentBenchFlowInput(task_id="citation-check"),
        runner=runner,
    )
    action = workflow.next()[0]
    assert runner.started
    assert action.kind.value == "run"
    assert _decodeTokenPrompt(action.data) == "first rendered prompt"
    assert action.retainOutput is False
    workflow.observe([ActionResult(3, Result(output="first output"))])
    assert runner.responses == [(first, "first output")]
    assert _decodeTokenPrompt(workflow.next()[0].data) == "second rendered prompt"
    workflow.observe([ActionResult(3, Result(output="second output"))])
    assert workflow.next() is None
    assert workflow.finished
    assert workflow.final_result.output["rewards"]["reward"] == 0.75


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


def test_workflow_prepares_only_skill_documents_before_the_original_run_prompt():
    skill = "---\nname: demo-skill\n---\n\nUse the skill.\n"
    first = _request("first rendered prompt")
    second = _request(
        "second rendered prompt",
        _skill_messages("/home/agent/.pi/agent/skills/demo/SKILL.md", skill),
    )
    runner = _FakeRunner([first, second])
    workflow = AgentBenchFlowWorkflow(
        case_id=3,
        data=AgentBenchFlowInput(task_id="citation-check"),
        runner=runner,
    )

    firstAction = workflow.next()[0]
    assert firstAction.kind.value == "run"
    workflow.observe([ActionResult(3, Result(output="first output"))])

    prepare = workflow.next()[0]
    assert prepare.kind == ActionKind.PREPARE
    assert [_decodeTokenPrompt(chunk) for chunk in prepare.data] == [
        _CanonicalSkillDocument(skill)
    ]
    assert "available skills" not in _decodeTokenPrompt(prepare.data[0])
    assert "TASK:" not in _decodeTokenPrompt(prepare.data[0])

    workflow.observe([ActionResult(3, Result())])
    secondAction = workflow.next()[0]
    assert secondAction.kind == ActionKind.RUN
    assert _decodeTokenPrompt(secondAction.data) == "second rendered prompt"
    workflow.observe([ActionResult(3, Result(output="second output"))])

    assert runner.responses == [
        (first, "first output"),
        (second, "second output"),
    ]


def test_without_skill_does_not_read_or_prepare_skill_documents():
    skill = "---\nname: demo-skill\n---\n\nUse the skill.\n"
    first = _request("first rendered prompt")
    second = _request(
        "second rendered prompt",
        _skill_messages("/home/agent/.pi/agent/skills/demo/SKILL.md", skill),
    )
    runner = _FakeRunner([first, second])
    workflow = AgentBenchFlowWorkflow(
        case_id=3,
        data=AgentBenchFlowInput(
            task_id="citation-check",
            skill_mode="WithoutSkill",
        ),
        runner=runner,
    )

    assert workflow.next()[0].kind == ActionKind.RUN
    workflow.observe([ActionResult(3, Result(output="first output"))])

    secondAction = workflow.next()[0]
    assert secondAction.kind == ActionKind.RUN
    assert _decodeTokenPrompt(secondAction.data) == "second rendered prompt"
    assert workflow._skillDocuments == []


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
    workflow = AgentBenchFlowWorkflow(
        case_id=3,
        data=AgentBenchFlowInput(
            task_id="demo-task",
            source_mode="local",
            skillsbench_dir=str(tmp_path),
        ),
        runner=runner,
    )

    prepare = workflow.next()[0]
    assert prepare.kind == ActionKind.PREPARE
    assert [_decodeTokenPrompt(chunk) for chunk in prepare.data] == [
        _CanonicalSkillDocument(skill)
    ]
    workflow.observe([ActionResult(3, Result())])
    run = workflow.next()[0]
    assert run.kind == ActionKind.RUN
    runText = _decodeTokenPrompt(run.data)
    assert _CanonicalSkillDocument(skill) in runText
    assert runText.endswith("rendered prompt without a skill tool result")


def test_dataset_source_does_not_inject_skills_from_local_checkout(tmp_path):
    skill_path = tmp_path / "tasks/demo-task/environment/skills/demo/SKILL.md"
    skill_path.parent.mkdir(parents=True)
    skill_path.write_text("LOCAL VERSION INSTRUCTION", encoding="utf-8")
    request = _request("registry dataset prompt")
    workflow = AgentBenchFlowWorkflow(
        case_id=3,
        data=AgentBenchFlowInput(task_id="demo-task", source_mode="dataset",
                                dataset="skillsbench@1.1", skillsbench_dir=str(tmp_path)),
        runner=_FakeRunner([request]),
    )
    action = workflow.next()[0]
    assert action.kind == ActionKind.RUN
    assert _decodeTokenPrompt(action.data) == "registry dataset prompt"
    assert workflow._skillDocuments == []


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
    workflow = AgentBenchFlowWorkflow(
        case_id=3,
        data=AgentBenchFlowInput(
            task_id="demo-task",
            source_mode="local",
            skillsbench_dir=str(tmp_path),
        ),
        runner=runner,
    )

    workflow.next()
    workflow.observe([ActionResult(3, Result())])
    run = workflow.next()[0]

    assert _decodeTokenPrompt(run.data).startswith("rendered with")
    assert _CanonicalSkillDocument(skill) in renderCalls[0][0][0]["content"]
    assert renderCalls[0][1:] == ("/models/test", None, True)
    assert "original rendered prompt" not in _decodeTokenPrompt(run.data)


def test_first_run_skill_segment_matches_after_chat_template_moves_terminal_newline(
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
    canonicalSkill = _CanonicalSkillDocument(skill)
    skillPath.write_text(skill, encoding="utf-8")

    def render(messages, *, modelPath, tools=None, thinking=None):
        systemContent = messages[0]["content"].rstrip("\r\n")
        # Match the relevant Qwen chat-template boundary: the final newline
        # belongs after the system message terminator, not inside its content.
        return systemContent + "<|im_end|>\n<|im_start|>user\noriginal prompt"

    monkeypatch.setattr(ModelAdapter, "render_chat", render)
    request = _request("original prompt")
    request.modelPath = "/models/test"
    runner = _FakeRunner([request])
    workflow = AgentBenchFlowWorkflow(
        case_id=3,
        data=AgentBenchFlowInput(
            task_id="demo-task",
            source_mode="local",
            skillsbench_dir=str(tmp_path),
        ),
        runner=runner,
    )

    prepare = workflow.next()[0]
    assert [_decodeTokenPrompt(chunk) for chunk in prepare.data] == [canonicalSkill]
    workflow.observe([ActionResult(3, Result())])
    run = workflow.next()[0]

    assert (0, canonicalSkill) in ComposeInterleavedReuseText(
        [_decodeTokenPrompt(chunk) for chunk in prepare.data],
        _decodeTokenPrompt(run.data),
    )


def test_workflow_surfaces_runner_failure_before_first_run():
    runner = _FailingRunner()
    workflow = AgentBenchFlowWorkflow(
        case_id=3,
        data=AgentBenchFlowInput(task_id="broken-task"),
        runner=runner,
    )

    with pytest.raises(AgentBenchFlowPreRunError, match="before its first RUN"):
        workflow.next()
    assert workflow.finished
    assert runner.stopped
    assert workflow.final_result.output["reward"] == 0.0


def test_workflow_captures_first_run_stats_only_once():
    first = _request("first rendered prompt")
    second = _request("second rendered prompt")
    runner = _FakeRunner([first, second])
    workflow = AgentBenchFlowWorkflow(
        case_id=3,
        data=AgentBenchFlowInput(task_id="citation-check"),
        runner=runner,
    )

    # First RUN: capture ttft + reuse_ratio + prompt length.
    workflow.next()
    workflow.observe(
        [
            ActionResult(
                3,
                Result(
                    output="first output",
                    performance={TtftKey: 0.42},
                    metadata={"reuse_ratio": 0.85, "n_input": 12000},
                ),
            )
        ]
    )

    # Second RUN: very different numbers — must NOT overwrite the first-run
    # capture, since the report is per-case first-RUN.
    workflow.next()
    workflow.observe(
        [
            ActionResult(
                3,
                Result(
                    output="second output",
                    performance={TtftKey: 0.99},
                    metadata={"reuse_ratio": 0.10, "n_input": 400},
                ),
            )
        ]
    )
    workflow.next()  # returns None, finishes the workflow

    final = workflow.final_result
    assert final.metadata["first_run_ttft"] == pytest.approx(0.42)
    assert final.metadata["first_run_reuse_ratio"] == pytest.approx(0.85)
    assert final.metadata["first_run_prompt_length"] == 12000


def test_workflow_first_run_only_stops_after_first_run():
    first = _request("first rendered prompt")
    second = _request("second rendered prompt")
    runner = _FakeRunner([first, second])
    workflow = AgentBenchFlowWorkflow(
        case_id=3,
        data=AgentBenchFlowInput(
            task_id="citation-check",
            first_run_only=True,
        ),
        runner=runner,
    )

    workflow.next()
    workflow.observe(
        [
            ActionResult(
                3,
                Result(
                    output="first output",
                    performance={TtftKey: 0.42},
                    metadata={"reuse_ratio": 0.85, "n_input": 12000},
                ),
            )
        ]
    )

    assert workflow.finished
    assert workflow.next() is None
    assert runner.responses == []
    assert runner.stopped
    assert workflow.final_result.metadata["first_run_ttft"] == pytest.approx(0.42)
    assert workflow.final_result.metadata["first_run_reuse_ratio"] == pytest.approx(0.85)
    assert workflow.final_result.metadata["first_run_prompt_length"] == 12000


def test_workflow_first_run_prompt_length_is_absent_without_n_input():
    first = _request("first rendered prompt")
    runner = _FakeRunner([first])
    workflow = AgentBenchFlowWorkflow(
        case_id=3,
        data=AgentBenchFlowInput(task_id="citation-check"),
        runner=runner,
    )

    # A Method that does not report n_input must still contribute the other
    # two first-RUN readings.
    workflow.next()
    workflow.observe(
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
    workflow.next()

    final = workflow.final_result
    assert final.metadata["first_run_ttft"] == pytest.approx(0.42)
    assert final.metadata["first_run_reuse_ratio"] == pytest.approx(0.85)
    assert "first_run_prompt_length" not in final.metadata


def test_workflow_first_run_capture_is_absent_when_no_run_completes():
    runner = _FailingRunner()
    workflow = AgentBenchFlowWorkflow(
        case_id=3,
        data=AgentBenchFlowInput(task_id="citation-check"),
        runner=runner,
    )

    with pytest.raises(AgentBenchFlowPreRunError, match="before its first RUN"):
        workflow.next()  # runner.start() raises -> fail() runs

    final = workflow.final_result
    assert final.metadata.get("first_run_ttft") is None
    assert final.metadata.get("first_run_reuse_ratio") is None
    assert final.metadata.get("first_run_prompt_length") is None
    assert final.output["error"]


def test_workflow_first_run_metadata_is_attached_even_after_a_mid_run_failure():
    first = _request("first rendered prompt")
    runner = _FakeRunner([first])

    def boom(_request, _output):
        raise RuntimeError("model crashed mid-rollout")

    runner.respond = boom
    workflow = AgentBenchFlowWorkflow(
        case_id=3,
        data=AgentBenchFlowInput(task_id="citation-check"),
        runner=runner,
    )

    workflow.next()
    workflow.observe(
        [
            ActionResult(
                3,
                Result(
                    output="first output",
                    performance={TtftKey: 0.21},
                    metadata={"reuse_ratio": 0.73, "n_input": 8192},
                ),
            )
        ]
    )
    # observe()'s respond() raises -> fail() is called, which must keep the
    # already-captured first-RUN stats.
    final = workflow.final_result
    assert final.metadata["first_run_ttft"] == pytest.approx(0.21)
    assert final.metadata["first_run_reuse_ratio"] == pytest.approx(0.73)
    assert final.metadata["first_run_prompt_length"] == 8192

