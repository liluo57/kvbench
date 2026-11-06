"""Validate and summarize the paired outputs of EvaluateHybridCacheRepair.py."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import statistics


ARMS = ("full", "trr", "stage_b")
PRIMARY = {
    "govreport": "rougeL", "multinews": "rougeL", "samsum": "rougeL",
    "triviaqa": "f1", "musique": "f1", "2wikimultihopqa": "f1", "hotpotqa": "f1",
    "niah_shuffle": "accuracy", "cwe_shuffle": "accuracy", "vt_shuffle": "accuracy",
}


def identity(case_id):
    return json.dumps(case_id, sort_keys=True)


def paired_interval(differences, seed=38, repetitions=2000):
    """Percentile bootstrap over paired cases, returned in score units (0..1)."""
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(differences, k=len(differences)))
                   for _ in range(repetitions))
    return [means[int(.025 * repetitions)], means[int(.975 * repetitions)]]


def summarize(root, allow_partial=False):
    selection = json.loads((root / "selection.json").read_text())
    arm_names = tuple(selection.get("arms", ARMS))
    if not arm_names or len(set(arm_names)) != len(arm_names) or not set(arm_names) <= {"full", "trr", "stage_b", "cacherepair"}:
        raise ValueError("Invalid evaluation arms")
    if {entry["task"] for entry in selection["tasks"]} != set(PRIMARY):
        raise ValueError("Expected the ten-workload selection")
    report_path = root / "engine_report.json"
    engine = json.loads(report_path.read_text()) if report_path.exists() else {}
    if not allow_partial and (engine.get("status") != "completed" or engine.get("failures")):
        raise ValueError("Engine has not completed successfully")
    result = {"complete": engine.get("status") == "completed" and not engine.get("failures"),
              "primary_score_scale": "0..1", "tasks": [], "arms": list(arm_names),
              "frozen_pickle_sha256": selection["frozen_pickle_sha256"],
              "bootstrap": {"method": "paired percentile", "repetitions": 2000, "seed": 38}}
    pooled = {arm: [] for arm in arm_names}
    for entry in selection["tasks"]:
        task = entry["task"]
        expected = dict(zip(map(identity, entry["case_ids"]), entry["input_sha256"], strict=True))
        if len(expected) != entry["count"]:
            raise ValueError(f"Duplicate IDs or inconsistent selection count in {task}")
        arms = {}
        for arm in arm_names:
            path = root / "predictions" / arm / (task + ".jsonl")
            rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
            indexed = {}
            for row in rows:
                key = identity(row["case_id"])
                if key in indexed or key not in expected or row["input_sha256"] != expected[key]:
                    raise ValueError(f"Duplicate, unexpected or changed input: {arm}/{task}/{key}")
                performance = row["performance"]
                if any(not math.isfinite(performance[k]) or performance[k] <= 0
                       for k in ("ttft", "total_time", "num_output_tokens")):
                    raise ValueError(f"Invalid performance: {arm}/{task}/{key}")
                if performance["ttft"] > performance["total_time"]:
                    raise ValueError(f"TTFT exceeds total time: {arm}/{task}/{key}")
                if performance["num_output_tokens"] > entry["max_new_tokens"]:
                    raise ValueError(f"Generation budget exceeded: {arm}/{task}/{key}")
                if any(not math.isfinite(value) or not 0 <= value <= 1 for value in row["scores"].values()):
                    raise ValueError(f"Invalid quality score: {arm}/{task}/{key}")
                if PRIMARY[task] not in row["scores"]:
                    raise ValueError(f"Missing primary score: {arm}/{task}/{key}")
                metadata = row["result_metadata"]
                if arm in ("stage_b", "cacherepair"):
                    hit = metadata["num_cached_tokens"] > 0
                    if metadata.get("repairer_invoked") is not hit or metadata.get("attention_kv_source") != ("cacherepair" if hit else "full_prefill"):
                        raise ValueError(f"Incorrect hybrid route metadata: {task}/{key}")
                indexed[key] = row
            if len(indexed) != len(expected):
                result["complete"] = False
                if not allow_partial:
                    raise ValueError(f"Incomplete {arm}/{task}: {len(indexed)}/{len(expected)}")
            arms[arm] = indexed
            pooled[arm].extend(rows)
        common = set.intersection(*(set(arms[arm]) for arm in arm_names))
        task_result = {"task": task, "expected": entry["count"], "primary": PRIMARY[task],
                       "max_new_tokens": entry["max_new_tokens"],
                       "input_tokens_mean": statistics.mean(entry["input_lengths"]),
                       "input_tokens_max": max(entry["input_lengths"]), "arms": {}}
        for arm, rows in arms.items():
            values = list(rows.values())
            metrics = sorted(set().union(*(row["scores"] for row in values)))
            task_result["arms"][arm] = {
                "n": len(values),
                "scores": {metric: statistics.mean(row["scores"][metric] for row in values) for metric in metrics},
                "ttft_mean_s": statistics.mean(row["performance"]["ttft"] for row in values) if values else None,
                "ttft_median_s": statistics.median(row["performance"]["ttft"] for row in values) if values else None,
                "total_mean_s": statistics.mean(row["performance"]["total_time"] for row in values) if values else None,
                "output_tokens_mean": statistics.mean(row["performance"]["num_output_tokens"] for row in values) if values else None,
                "repaired_cases": sum(row["result_metadata"].get("repairer_invoked", False) for row in values),
                "budget_hit_cases": sum(row["performance"]["num_output_tokens"] == entry["max_new_tokens"] for row in values),
                "empty_visible_outputs": sum(not row["output_ids"] for row in values),
            }
        repair_arm = "cacherepair" if "cacherepair" in arms else "stage_b"
        reference_arm = "trr" if "trr" in arms else "full"
        if common and repair_arm in arms and reference_arm in arms:
            differences = [arms[repair_arm][key]["scores"][PRIMARY[task]] - arms[reference_arm][key]["scores"][PRIMARY[task]]
                           for key in sorted(common)]
            difference_name = "hybrid_minus_trr" if reference_arm == "trr" else "cacherepair_minus_full"
            task_result[difference_name] = {"paired_n": len(common), "mean": statistics.mean(differences),
                "bootstrap_95ci": paired_interval(differences),
                "hybrid_wins": sum(d > 0 for d in differences), "ties": sum(d == 0 for d in differences),
                "hybrid_losses": sum(d < 0 for d in differences)}
        result["tasks"].append(task_result)
    result["pooled_timing"] = {
        arm: {"n": len(rows),
              "ttft_mean_s": statistics.mean(row["performance"]["ttft"] for row in rows) if rows else None,
              "ttft_median_s": statistics.median(row["performance"]["ttft"] for row in rows) if rows else None}
        for arm, rows in pooled.items()}
    return result


def write_summary(root, summary):
    stem = "comparison" if summary["complete"] else "progress"
    arms = tuple(summary.get("arms", ARMS))
    labels = {"full": "Full", "trr": "TRR", "stage_b": "Hybrid", "cacherepair": "CacheRepair"}
    (root / (stem + ".json")).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    lines = ["# HybridCacheRepair ten-workload evaluation", "",
             "Status: " + ("complete" if summary["complete"] else "partial; pending results are not final conclusions"), "",
             "Quality scores are shown on a 0–100 scale.", "",
             "| Task | Metric | n | " + " | ".join(labels[a] for a in arms) + " |",
             "|---|---|---:|" + "---:|" * len(arms)]
    def fmt(value, scale=1):
        return f"{value*scale:.2f}" if value is not None else "—"
    with (root / (stem + ".csv")).open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["task", "metric", "arm", "n", "quality_0_to_1", "ttft_mean_s", "ttft_median_s", "total_mean_s", "output_tokens_mean"])
        for task in summary["tasks"]:
            scores = [task["arms"][arm]["scores"].get(task["primary"]) for arm in arms]
            counts = "/".join(str(task["arms"][arm]["n"]) for arm in arms)
            lines.append(f'| {task["task"]} | {task["primary"]} | {counts} | ' + " | ".join(fmt(s, 100) for s in scores) + " |")
            for arm in arms:
                row = task["arms"][arm]
                writer.writerow([task["task"], task["primary"], arm, row["n"], row["scores"].get(task["primary"]),
                    row["ttft_mean_s"], row["ttft_median_s"], row["total_mean_s"], row["output_tokens_mean"]])
    lines += ["", "Mean TTFT in seconds, including all online work. Preparation is outside TTFT.", "",
              "| Task | " + " | ".join(labels[a] for a in arms) + " |", "|---|" + "---:|" * len(arms)]
    for task in summary["tasks"]:
        lines.append("| " + task["task"] + " | " + " | ".join(fmt(task["arms"][a]["ttft_mean_s"]) for a in arms) + " |")
    lines += ["", "No overall quality average is computed across different task metrics.",
              "The JSON includes all metrics, paired bootstrap intervals, output lengths, and median timing.",
              "The bootstrap interval reflects case sampling only, not training-seed or model uncertainty.", ""]
    (root / (stem + ".md")).write_text("\n".join(lines))
    if summary["complete"]:
        # Keep links used while monitoring accurate after the run finishes.
        for extension in ("json", "csv", "md"):
            progress = root / ("progress." + extension)
            if progress.exists():
                progress.write_bytes((root / ("comparison." + extension)).read_bytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    result = summarize(args.output, args.allow_partial)
    write_summary(args.output, result)
    print(json.dumps({"complete": result["complete"], "counts": {a: v["n"] for a, v in result["pooled_timing"].items()}}))


if __name__ == "__main__":
    main()
