"""Tests for rag."""

import pytest

from core.Result import Result
from core.Workflow import ActionKind, ActionResult
from workflow import RAGInput, RAGWorkflow


def test_rag_workflow_prepare_run_and_final_result(modelPath):
    workflow = RAGWorkflow(7, RAGInput(["a", "b"], "prompt", modelPath))

    prepare = workflow.next()[0]
    assert prepare.kind == ActionKind.PREPARE
    assert prepare.case_id == 7
    assert prepare.data == [[ord("a")], [ord("b")]]
    workflow.observe([ActionResult(7, Result(), prepare.tag)])
    assert workflow.final_result is None

    run = workflow.next()[0]
    assert run.kind == ActionKind.RUN
    assert run.data == [ord(character) for character in "prompt"]
    assert not run.retainOutput
    workflow.observe([ActionResult(7, Result(output="answer"), run.tag)])

    assert workflow.finished
    assert workflow.next() is None
    assert workflow.final_result.output == "answer"


def test_rag_workflow_skips_empty_prepare(modelPath):
    workflow = RAGWorkflow(1, RAGInput([], "prompt", modelPath))

    assert workflow.next()[0].kind == ActionKind.RUN


def test_rag_workflow_keeps_a_final_result_with_none_output(modelPath):
    workflow = RAGWorkflow(1, RAGInput([], "prompt", modelPath))
    run = workflow.next()[0]
    result = Result(output=None, metadata={"diagnostic": True})

    workflow.observe([ActionResult(1, result, run.tag)])

    assert workflow.final_result is result


def test_rag_workflow_observe_requires_one_result(modelPath):
    workflow = RAGWorkflow(1, RAGInput([], "prompt", modelPath))
    workflow.next()

    with pytest.raises(ValueError, match="exactly one"):
        workflow.observe([])
