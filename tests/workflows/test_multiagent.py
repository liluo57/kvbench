"""Tests for multiagent."""

import pytest

from core.Result import Result
from core.Workflow import ActionKind, ActionResult
from helpers.backends import ModelAdapter
from workflow import (
    AgentSpec,
    MultiAgentFullConnectionInput,
    MultiAgentFullConnectionWorkflow,
)


@pytest.mark.parametrize("with_decision, expected_retention", [(True, [True, True, True, False]), (False, [True, True, False])])
def test_retention_stops_at_final_agent_or_decision(modelPath, with_decision, expected_retention):
    decision = AgentSpec("D", "decide {task}") if with_decision else None
    workflow = MultiAgentFullConnectionWorkflow(1, MultiAgentFullConnectionInput(
        "TASK", [AgentSpec(role, f"{role} {{task}}") for role in ("A", "B", "C")],
        decision, chatTemplate=False, modelPath=modelPath,
    ))
    prepare = workflow.next()[0]
    assert prepare.kind == ActionKind.PREPARE
    assert prepare.data == [[ord(c) for c in "TASK"]]
    workflow.observe([ActionResult(1, Result(), prepare.tag)])
    retention = []
    for index in range(len(expected_retention)):
        action = workflow.next()[0]
        retention.append(action.retainOutput)
        for prior in range(index):
            assert 900 + prior in action.data
        result = Result(output=f"OUT{index + 1}", output_ids=[900 + index])
        workflow.observe([ActionResult(1, result, action.tag)])
    assert retention == expected_retention
    assert workflow.finished and workflow.next() is None
    assert workflow.final_result is result


def test_equal_or_empty_display_outputs_keep_distinct_native_fragments(modelPath):
    workflow = MultiAgentFullConnectionWorkflow(7, MultiAgentFullConnectionInput(
        "TASK", [AgentSpec(role, f"{role} {{task}}") for role in ("A", "B", "C")],
        AgentSpec("D", "D {task}"), chatTemplate=False, modelPath=modelPath,
    ))
    prepare = workflow.next()[0]
    workflow.observe([ActionResult(7, Result(), prepare.tag)])
    fragments = [[900, 999], [901, 999], [902, 999]]
    for display, fragment in zip(["same", "same", ""], fragments, strict=True):
        action = workflow.next()[0]
        workflow.observe([ActionResult(7, Result(output=display, output_ids=fragment), action.tag)])
    prompt = workflow.next()[0].data
    positions = [prompt.index(fragment[0]) for fragment in fragments]
    assert positions == sorted(positions)
    assert all(prompt[pos:pos + 2] == fragment for pos, fragment in zip(positions, fragments, strict=True))


def test_multiagent_plain_prompts_without_shared_prepare(modelPath):
    workflow = MultiAgentFullConnectionWorkflow(
        3,
        MultiAgentFullConnectionInput(
            task="TASK",
            agents=[AgentSpec("A", "A says {task}"), AgentSpec("B", "B says {task}")],
            prepareSharedTask=False,
            chatTemplate=False,
            modelPath=modelPath,
        ),
    )

    first = workflow.next()[0]
    assert first.kind == ActionKind.RUN
    assert first.data == [ord(character) for character in "A says TASK"]
    assert first.retainOutput
    workflow.observe([
        ActionResult(
            3,
            Result(output="first answer", output_ids=[900, 999, 901]),
            first.tag,
        )
    ])

    second = workflow.next()[0]
    assert second.kind == ActionKind.RUN
    prompt = "".join(chr(token) for token in second.data)
    assert prompt.startswith("B says TASK")
    assert "Agent 0, role is A" in prompt
    assert any(
        second.data[index : index + 3] == [900, 999, 901]
        for index in range(len(second.data) - 2)
    )
    assert not second.retainOutput
    workflow.observe([ActionResult(3, Result(output="final answer"), second.tag)])

    assert workflow.finished
    assert workflow.final_result.output == "final answer"


@pytest.mark.parametrize(("userContent", "systemContent"), [("same", "same"), ("", "system")])
def test_multiagent_chat_boundaries_accept_repeated_or_empty_content(
    modelPath, userContent, systemContent
):
    workflow = MultiAgentFullConnectionWorkflow(
        3,
        MultiAgentFullConnectionInput(
            task="unused",
            agents=[AgentSpec("A", userContent, systemContent)],
            prepareSharedTask=False,
            chatTemplate=True,
            modelPath=modelPath,
        ),
    )

    action = workflow.next()[0]
    rendered = ModelAdapter.render_chat(
        [{"role": "user", "content": userContent}],
        modelPath=modelPath,
        system_prefix=systemContent,
        thinking=False,
    )
    assert action.data == [ord(character) for character in rendered]


def test_multiagent_observe_requires_one_result(modelPath):
    workflow = MultiAgentFullConnectionWorkflow(
        1,
        MultiAgentFullConnectionInput(
            "task", [AgentSpec("A", "{task}")],
            chatTemplate=False,
            modelPath=modelPath,
        ),
    )
    workflow.next()

    with pytest.raises(ValueError, match="exactly one"):
        workflow.observe([])
