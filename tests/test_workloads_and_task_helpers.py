import json

import pytest

from core.Config import ModelPath as _ModelPath
from core.Result import Result
from core.Workload import ActionKind, ActionResult
from helpers import ModelAdapter
from tasks.bases.KBBase import (
    NormalizeAnswer,
    ParseGeneration,
    RougeL,
    TokenEm,
    TokenF1,
    _FlattenAnswers,
)
from tasks.bases.RulerBase import (
    ExactMatch,
    RulerBase,
    StringMatchAll,
    _LengthFromName,
    _findNeedleSentence,
)
from workload import (
    AgentSpec,
    MultiAgentFullConnectionInput,
    MultiAgentFullConnectionWorkload,
    RAGInput,
    RAGWorkload,
)


@pytest.fixture
def modelPath():
    """The model path used by ``render_chat`` / ``_thinking_kwargs`` tests.

    Mirrors :class:`core.Config.ModelPath` so tests use the real configured
    tokenizer (Qwen3 in this environment).
    """
    return _ModelPath()


def test_rag_workload_prepare_run_and_final_result():
    workload = RAGWorkload(7, RAGInput(["a", "b"], "prompt"))

    prepare = workload.next()[0]
    assert prepare.kind == ActionKind.PREPARE
    assert prepare.case_id == 7
    assert prepare.data == ["a", "b"]
    workload.observe([ActionResult(7, Result(), prepare.tag)])
    assert workload.final_result is None

    run = workload.next()[0]
    assert run.kind == ActionKind.RUN
    assert run.data == "prompt"
    assert not run.retainOutput
    workload.observe([ActionResult(7, Result(output="answer"), run.tag)])

    assert workload.finished
    assert workload.next() is None
    assert workload.final_result.output == "answer"


def test_rag_workload_skips_empty_prepare():
    workload = RAGWorkload(1, RAGInput([], "prompt"))

    assert workload.next()[0].kind == ActionKind.RUN


def test_rag_workload_keeps_a_final_result_with_none_output():
    workload = RAGWorkload(1, RAGInput([], "prompt"))
    run = workload.next()[0]
    result = Result(output=None, metadata={"diagnostic": True})

    workload.observe([ActionResult(1, result, run.tag)])

    assert workload.final_result is result


def test_rag_workload_observe_requires_one_result():
    workload = RAGWorkload(1, RAGInput([], "prompt"))
    workload.next()

    with pytest.raises(ValueError, match="exactly one"):
        workload.observe([])


def test_multiagent_plain_prompts_without_shared_prepare():
    workload = MultiAgentFullConnectionWorkload(
        3,
        MultiAgentFullConnectionInput(
            task="TASK",
            agents=[AgentSpec("A", "A says {task}"), AgentSpec("B", "B says {task}")],
            prepareSharedTask=False,
            chatTemplate=False,
        ),
    )

    first = workload.next()[0]
    assert first.kind == ActionKind.RUN
    assert first.data == "A says TASK"
    assert first.retainOutput
    workload.observe([ActionResult(3, Result(output="first answer"), first.tag)])

    second = workload.next()[0]
    assert second.kind == ActionKind.RUN
    assert second.data.startswith("B says TASK")
    assert "Agent 0, role is A" in second.data
    assert "first answer" in second.data
    assert not second.retainOutput
    workload.observe([ActionResult(3, Result(output="final answer"), second.tag)])

    assert workload.finished
    assert workload.final_result.output == "final answer"


def test_multiagent_observe_requires_one_result():
    workload = MultiAgentFullConnectionWorkload(
        1,
        MultiAgentFullConnectionInput("task", [AgentSpec("A", "{task}")]),
    )
    workload.next()

    with pytest.raises(ValueError, match="exactly one"):
        workload.observe([])


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("The Quick, brown fox!", "quick brown fox"),
        ("  AN apple  ", "apple"),
    ],
)
def test_kb_normalization(raw, expected):
    assert NormalizeAnswer(raw) == expected


def test_kb_generation_and_overlap_metrics():
    assert ParseGeneration(" \n\nExeter College\nextra") == "Exeter College"
    assert ParseGeneration(" no, that is false\nexplanation") == "No"
    assert TokenEm("The answer!", "answer") == 1.0
    assert TokenF1("red blue", "blue green") == pytest.approx(0.5)
    assert RougeL("red blue green", "red x green") == pytest.approx(2 / 3)
    assert _FlattenAnswers([["a", ""], "b"]) == ["a", "b"]


def test_flatten_answers_handles_scalars_nesting_and_empty_values():
    assert _FlattenAnswers("answer") == ["answer"]
    assert _FlattenAnswers([None, "answer", "", 0, [["nested"]]]) == [
        "answer",
        "0",
        "nested",
    ]


def test_ruler_metric_and_parsing_helpers(tmp_path):
    assert StringMatchAll("Alpha and beta", ["alpha", "missing"]) == 0.5
    assert ExactMatch(" Answer \n", ["answer", "other"]) == 1.0
    assert _LengthFromName(tmp_path / "niah_len8192.jsonl") == 8192
    assert _LengthFromName(tmp_path / "niah.jsonl") is None
    body = "First sentence. The key is 42. Final sentence."
    assert _findNeedleSentence(body, "42") == "The key is 42."
    with pytest.raises(RuntimeError, match="not found"):
        _findNeedleSentence(body, "missing")


@pytest.mark.parametrize("refs", [None, [], [""], [None]])
def test_ruler_metrics_reject_empty_references(refs):
    with pytest.raises(ValueError, match="reference"):
        StringMatchAll("prediction", refs)
    with pytest.raises(ValueError, match="reference"):
        ExactMatch("prediction", refs)


class _RulerTask(RulerBase):
    taskName = "sample"

    def Cases(self):
        return iter(())

    def Evaluate(self, result, metadata):
        return {}


def test_ruler_loader_filters_length_and_slices(tmp_path):
    (tmp_path / "sample_len8.jsonl").write_text(
        json.dumps({"input": "eight", "outputs": ["8"]}) + "\n\n",
        encoding="utf-8",
    )
    (tmp_path / "sample_len16.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"input": "first", "outputs": ["1"]}),
                json.dumps({"input": "second", "outputs": ["2"]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    lengthFiltered = _RulerTask(dataDir=str(tmp_path), maxSeqLength=8)
    assert [sample["input"] for sample in lengthFiltered._LoadSamples()] == ["eight"]

    sliced = _RulerTask(dataDir=str(tmp_path), startIdx=1, maxSamples=1)
    samples = sliced._LoadSamples()
    assert [sample["input"] for sample in samples] == ["second"]
    assert samples[0]["file"] == "sample_len16.jsonl"


@pytest.mark.parametrize("outputs", [None, [], [""], [None]])
def test_ruler_loader_rejects_missing_or_blank_outputs(tmp_path, outputs):
    sample = {"input": "prompt"}
    if outputs is not None:
        sample["outputs"] = outputs
    path = tmp_path / "sample_len8.jsonl"
    path.write_text(json.dumps(sample) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"sample_len8\.jsonl:1"):
        _RulerTask(dataDir=str(tmp_path))._LoadSamples()


def test_model_adapter_arch_family_reads_config_json(tmp_path):
    """``arch_family`` inspects ``<modelPath>/config.json`` ``architectures``."""
    modelDir = tmp_path / "model"
    modelDir.mkdir()
    (modelDir / "config.json").write_text(
        json.dumps({"architectures": ["Qwen3ForCausalLM"]}),
        encoding="utf-8",
    )
    ModelAdapter.arch_family.cache_clear()
    try:
        assert ModelAdapter.arch_family(str(modelDir)) == "qwen3"
    finally:
        ModelAdapter.arch_family.cache_clear()

    (modelDir / "config.json").write_text(
        json.dumps({"architectures": ["MuseGlimmerForCausalLM"]}),
        encoding="utf-8",
    )
    ModelAdapter.arch_family.cache_clear()
    try:
        assert ModelAdapter.arch_family(str(modelDir)) == "muse_glimmer"
    finally:
        ModelAdapter.arch_family.cache_clear()


def test_model_adapter_thinking_kwargs_translate_per_arch():
    """The CoT toggle is translated to the kwarg name the model's jinja reads."""
    ModelAdapter.set_arch_for_testing("qwen3")
    assert ModelAdapter._thinking_kwargs(False, "") == {"enable_thinking": False}
    assert ModelAdapter._thinking_kwargs(True, "") == {"enable_thinking": True}
    assert ModelAdapter._thinking_kwargs(None, "") == {}

    ModelAdapter.set_arch_for_testing("muse_glimmer")
    assert ModelAdapter._thinking_kwargs(False, "") == {"reasoning_strength": "low"}
    assert ModelAdapter._thinking_kwargs(True, "") == {"reasoning_strength": "high"}
    assert ModelAdapter._thinking_kwargs(None, "") == {}

    # Unknown archs ignore the kwarg entirely.
    ModelAdapter.set_arch_for_testing("other")
    assert ModelAdapter._thinking_kwargs(False, "") == {}
    assert ModelAdapter._thinking_kwargs(True, "") == {}

    ModelAdapter.set_arch_for_testing(None)


def test_model_adapter_render_chat_includes_assistant_generation_prompt(modelPath):
    """render_chat appends the assistant generation prompt regardless of body."""
    ModelAdapter.set_arch_for_testing("qwen3")
    out = ModelAdapter.render_chat([{"role": "user", "content": "hi"}], modelPath=modelPath)
    assert out.startswith("<|im_start|>user\n")
    assert out.endswith("<|im_start|>assistant\n")
    ModelAdapter.set_arch_for_testing(None)
