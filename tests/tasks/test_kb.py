"""Tests for kb."""

import json

import pytest

from helpers.backends import ModelAdapter
from tasks.bases.KBBase import (
    NormalizeAnswer,
    ParagraphChunks,
    ParseGeneration,
    RougeL,
    TokenEm,
    TokenF1,
    _FlattenAnswers,
)


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


def test_paragraph_chunks_preserve_text_and_prefer_paragraph_boundaries():
    text = "Paragraph one.\n\nParagraph two.\n\nParagraph three.\n\nParagraph four."
    chunks = ParagraphChunks(text, 2, prefix="Prefix: ")

    assert len(chunks) == 2
    assert "".join(chunks) == "Prefix: " + text
    assert chunks[0].endswith("Paragraph two.")
    assert chunks[1].lstrip().startswith("Paragraph three.")


def test_paragraph_chunks_use_sentence_boundaries_for_flattened_text():
    text = "First sentence. Second sentence! Third sentence? Fourth sentence."
    chunks = ParagraphChunks(text, 3)

    assert len(chunks) == 3
    assert "".join(chunks) == text
    assert all(not chunk or chunk[-1] in ".!?" for chunk in chunks[:-1])


def test_govreport_default_and_requested_chunk_counts(tmp_path):
    from tasks.GovReport import GovReportTask

    context = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    task = GovReportTask(dataDir=str(tmp_path), nChunks=1)
    one, suffix = task._Build({"context": context})
    assert one == [task.prefixPrompt + context]
    assert suffix == task.suffixPrompt

    splitTask = GovReportTask(dataDir=str(tmp_path), nChunks=3)
    chunks, _ = splitTask._Build({"context": context})
    assert len(chunks) == 3
    assert "".join(chunks) == task.prefixPrompt + context


def test_govreport_max_sample_length_filters_complete_prompts(tmp_path, monkeypatch):
    from tasks.GovReport import GovReportTask

    (tmp_path / "govreport.jsonl").write_text(
        "\n".join(
            json.dumps({"context": context, "answers": ["summary"]})
            for context in ("short report.", "oversized report.")
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        ModelAdapter,
        "render_user_prompt",
        lambda userContent, **kwargs: userContent,
    )

    class FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            if "Report:" in text and "Summary:" in text:
                # Whole-prompt encoding would undercount both examples.
                return [0]
            if "oversized report" in text:
                return [0] * 5
            if "Write a one-page" in text:
                return [0] * 2
            return [0]

    monkeypatch.setattr(
        ModelAdapter, "_tokenizer", lambda modelPath: FakeTokenizer()
    )

    for nChunks in (1, 4, 8, 16):
        filtered = GovReportTask(
            dataDir=str(tmp_path),
            nChunks=nChunks,
            maxSampleLength=4,
        )
        cases = list(filtered.Cases())
        assert [case.metadata["case_id"] for case in cases] == [0]

    unfiltered = GovReportTask(dataDir=str(tmp_path), maxSampleLength=0)
    assert len(list(unfiltered.Cases())) == 2
