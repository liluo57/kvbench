"""Paired quality reporting supports both current and historical arm sets."""
import json

import pytest

from scripts.SummarizeHybridCacheRepair import PRIMARY, summarize, write_summary


@pytest.mark.parametrize("arms", [["full", "cacherepair"], ["full", "trr", "cacherepair"], ["full", "trr", "stage_b"]])
def test_paired_summary_uses_declared_arms_and_checks_inputs(tmp_path, arms):
    selection = {"frozen_pickle_sha256": "fixed", "arms": arms, "tasks": []}
    for task, metric in PRIMARY.items():
        selection["tasks"].append({"task": task, "count": 2, "case_ids": [0, 1],
            "input_sha256": ["a", "b"], "input_lengths": [30, 40], "max_new_tokens": 4})
        for arm in arms:
            directory = tmp_path / "predictions" / arm
            directory.mkdir(parents=True, exist_ok=True)
            score = {"full": .8, "trr": .6, "cacherepair": .7, "stage_b": .7}[arm]
            rows = [{"case_id": i, "input_sha256": value, "output": "answer", "output_ids": [1],
                "scores": {metric: score}, "performance": {"ttft": .2, "total_time": 1., "num_output_tokens": 1},
                "result_metadata": {"num_cached_tokens": 10, "repairer_invoked": True, "attention_kv_source": "cacherepair"}}
                for i, value in enumerate(["a", "b"])]
            (directory / f"{task}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "selection.json").write_text(json.dumps(selection))
    (tmp_path / "engine_report.json").write_text('{"status":"completed","failures":[]}')
    result = summarize(tmp_path)
    assert result["complete"] and len(result["tasks"]) == 10
    assert result["arms"] == arms
    comparison = "hybrid_minus_trr" if "trr" in arms else "cacherepair_minus_full"
    assert result["tasks"][0][comparison]["mean"] == pytest.approx(.1 if "trr" in arms else -.1)
    assert all(row["arms"][arm]["n"] == 2 for row in result["tasks"] for arm in arms)
    write_summary(tmp_path, result)
    assert (tmp_path / "comparison.csv").exists()
    assert "Full" in (tmp_path / "comparison.md").read_text()
    path = tmp_path / "predictions/full/govreport.jsonl"
    path.write_text(path.read_text().replace('"input_sha256": "a"', '"input_sha256": "changed"'))
    with pytest.raises(ValueError, match="changed input"):
        summarize(tmp_path)
