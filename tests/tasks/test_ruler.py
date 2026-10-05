"""Tests for ruler."""

import json

import pytest

from tasks.bases.RulerBase import (
    ExactMatch,
    RulerBase,
    StringMatchAll,
    _findNeedleSentence,
    _LengthFromName,
)


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


def test_ruler_shuffle_guard_handles_empty_parts(tmp_path):
    assert _RulerTask(dataDir=str(tmp_path))._Shuffled(
        ["", "needle"], seed=7
    ) == ["", "needle"]


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
