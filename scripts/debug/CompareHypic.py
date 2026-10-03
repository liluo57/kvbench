"""Compare the bundled HYPIC variants on paired RULER shuffle cases.

Pass a separate KVBench YAML config so the main config.yaml is not changed.
Prepare matching RULER data with scripts/PrepareDataset.py first. Example::

    python scripts/debug/CompareHypic.py --config /path/to/eval.yaml \
        --gpus 0,3 --limit 16 --output outputs/hypic-comparison

Initialization and a discarded request for each task are outside measurement.
Every measured case uses its original token IDs and an isolated cache lifecycle.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.Config import LoadConfig


def _Configure(path: str) -> None:
    LoadConfig().update(LoadConfig(Path(path)))


# The scheduler uses multiprocessing spawn and reimports this file.
if configPath := os.environ.get("KVBENCH_HYPIC_EVAL_CONFIG"):
    _Configure(configPath)


def Main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpus", required=True, help="comma-separated physical GPU IDs")
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int,
                        help="override the Tasks' default generation limit (64)")
    parser.add_argument("--tasks", nargs="+", choices=("cwe", "vt"), default=["cwe", "vt"])
    parser.add_argument("--modes", nargs="+", default=[
        "full", "addition", "transition", "transition_rope",
        "transition_rope_recompute", "linearkv",
    ], choices=("full", "addition", "transition", "transition_rope",
                "transition_rope_recompute", "linearkv"))
    parser.add_argument("--mem-fraction-static", type=float, default=0.80)
    parser.add_argument("--seam-tokens", type=float, default=8)
    parser.add_argument("--recomp-ratio", type=float, default=0.2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.limit < 1 or args.start < 0:
        parser.error("limit must be positive and start must be non-negative")
    if args.max_new_tokens is not None and args.max_new_tokens < 1:
        parser.error("max-new-tokens must be positive")

    path = str(Path(args.config).resolve())
    os.environ["KVBENCH_HYPIC_EVAL_CONFIG"] = path
    _Configure(path)
    gpuIds = [int(value) for value in args.gpus.split(",")]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    from core.engine.Gpu import QueryGpus
    from core.Metrics import AggregateStats
    from methods import HypicMethod, LinearKVMethod
    from tasks import CWEShuffleTask, VTShuffleTask

    tasks = {
        name: (CWEShuffleTask if name == "cwe" else VTShuffleTask)(
            maxSamples=args.limit, startIdx=args.start, maxSeqLength=args.length,
            maxNewTokens=args.max_new_tokens,
        ) for name in args.tasks
    }
    cases = {name: list(task.Cases()) for name, task in tasks.items()}
    if any(len(items) != args.limit for items in cases.values()):
        raise ValueError("The requested paired sample range is missing from the datasets")

    manifest = dict(
        config=LoadConfig(), gpu_ids=gpuIds,
        gpu_snapshot=[gpu.AsDict() for gpu in QueryGpus()],
        limit=args.limit, start=args.start, length=args.length,
        generation_limits={name: task.maxNewTokens for name, task in tasks.items()},
        mem_fraction_static=args.mem_fraction_static,
        seam_tokens=args.seam_tokens, recomp_ratio=args.recomp_ratio,
        upstream_commit=(ROOT / "methods/hypic/UPSTREAM").read_text(),
        case_inputs={name: [{
            "case_id": case.workflow.case_id,
            "file": case.metadata["file"],
            "input_tokens": len(case.input.run_input),
            "prepared_tokens": [len(chunk) for chunk in case.input.prepare_input],
            "sha256": hashlib.sha256(json.dumps(case.input.run_input).encode()).hexdigest(),
        } for case in items] for name, items in cases.items()},
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    summaries = []

    for mode in args.modes:
        kwargs = dict(gpuNums=len(gpuIds), memFractionStatic=args.mem_fraction_static)
        method = (LinearKVMethod(recompRatio=args.recomp_ratio, **kwargs)
                  if mode == "linearkv" else HypicMethod(
                      picMode="transition_rope_recompute" if mode == "full" else mode,
                      seamTokens=args.seam_tokens, fullPrefill=mode == "full", **kwargs))
        print(f"[compare] initialize {mode} on GPUs {gpuIds}", flush=True)
        try:
            method.Initialize(gpuIds)
            for name, task in tasks.items():
                warmup = cases[name][0]
                method.Prepare([warmup.input.prepare_input])
                method.Run([warmup.input.run_input], maxNewTokens=1)
                method.Reset()
                rows = []
                with (output / f"{mode}-{name}.jsonl").open("w") as file:
                    for index, case in enumerate(cases[name]):
                        begin = time.perf_counter()
                        method.Prepare([case.input.prepare_input])
                        prepareSeconds = time.perf_counter() - begin
                        result = method.Run(
                            [case.input.run_input], maxNewTokens=task.maxNewTokens,
                        )[0]
                        scores = task.Evaluate(result, case.metadata)
                        row = dict(mode=mode, task=name, sample_index=args.start + index,
                                   references=case.metadata["outputs"], scores=scores,
                                   prepare_seconds=prepareSeconds, **asdict(result))
                        file.write(json.dumps(row, ensure_ascii=False) + "\n")
                        file.flush()
                        rows.append(row)
                        method.Reset()
                        print(f"[compare] {mode}/{name} {index + 1}/{args.limit}: "
                              f"Q={scores['accuracy']:.3f}, "
                              f"TTFT={result.performance['ttft']:.3f}s, "
                              f"reuse={result.metadata['reuse_ratio']:.3f}", flush=True)
                summary = dict(mode=mode, task=name, samples=len(rows),
                    quality=sum(row["scores"]["accuracy"] for row in rows) / len(rows),
                    **AggregateStats([row["performance"]["ttft"] for row in rows], name="ttft"),
                    reuse_ratio=sum(row["metadata"]["reuse_ratio"] for row in rows) / len(rows))
                summaries.append(summary)
                (output / "summary.json").write_text(json.dumps(summaries, indent=2))
                print(f"[compare] summary {json.dumps(summary)}", flush=True)
        finally:
            method.Close()

    baselines = {row["task"]: row for row in summaries if row["mode"] == "full"}
    for row in summaries:
        if baseline := baselines.get(row["task"]):
            row["RQ"] = row["quality"] / baseline["quality"] if baseline["quality"] else None
            row["RT_mean"] = row["ttft_mean"] / baseline["ttft_mean"]
            row["RT_p50"] = row["ttft_p50"] / baseline["ttft_p50"]
    (output / "summary.json").write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    Main()
