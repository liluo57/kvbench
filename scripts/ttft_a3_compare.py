#!/usr/bin/env python3
"""Compare KVBench TTFT for Vanilla, Naive, and official A3.

This is a speed-ratio experiment, not a paper Table 1 reproduction.  It
reuses the existing KVBench task/data path and the A3 Mistral driver, then
reports per-task TTFT summaries and the ratios:

    Full/A3   (A3 speedup over full recomputation)
    A3/Naive  (A3 overhead relative to the cheapest reuse baseline)

Only the Run-stage TTFT recorded by each Method is compared.  Prepare/cache
construction remains outside TTFT, matching the paper's offline-cache setup.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The driver sits beside this file in KVBench's ``scripts/`` directory.  Use a
# sibling import so this standalone script works even though ``scripts/`` is
# intentionally not a Python package.
import run_a3_kvbench_mistral as driver


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--a3-repo-root", required=True)
    parser.add_argument("--a3-python", required=True)
    parser.add_argument("--cacheblend-root", required=True,
                        help="Existing path needed by the shared KVBench preflight")
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-root", type=Path,
                        default=Path("outputs/ttft_a3_compare"))
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"),
                        default="float16")
    parser.add_argument("--recomp-ratio", type=float, default=0.15)
    parser.add_argument("--task-timeout", type=float, default=18000.0)
    parser.add_argument("--initialize-timeout", type=float, default=1800.0)
    parser.add_argument("--tasks", nargs="+", choices=tuple(driver.TASK_SPECS),
                        default=list(driver.TASK_SPECS))
    parser.add_argument("--pair-retries", type=int, default=0)
    return parser.parse_args()


def _driver_args(args: argparse.Namespace) -> argparse.Namespace:
    """Build the shared driver's argument surface without duplicating logic."""
    values = vars(args).copy()
    values["methods"] = ["vanilla", "fullreuse", "a3"]
    return argparse.Namespace(**values)


def _ttft_summary(run: Dict[str, Any]) -> Dict[str, Any]:
    summary = dict((run.get("system_metrics") or {}).get("ttft") or {})
    return {
        "method": run.get("method"),
        "task": run.get("task"),
        "cases": run.get("cases", 0),
        "ttft_mean": summary.get("ttft_mean"),
        "ttft_p50": summary.get("ttft_p50"),
        "ttft_p90": summary.get("ttft_p90"),
        "ttft_count": summary.get("ttft_count", 0),
        "task_metrics": run.get("task_metrics", {}),
    }


def _ratio(numerator: Any, denominator: Any) -> Any:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def make_report(engine_report: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    rows = [_ttft_summary(run) for run in engine_report.get("runs", [])]
    by_task: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(str(row["task"]), {})[str(row["method"])] = row

    comparisons: List[Dict[str, Any]] = []
    full_over_a3: List[float] = []
    a3_over_naive: List[float] = []
    for task, methods in sorted(by_task.items()):
        full = methods.get("full_prefill(vanilla)") or methods.get("full_prefill")
        naive = methods.get("naive(fullreuse)") or methods.get("naive")
        a3 = methods.get("a3_repo(a3)") or methods.get("a3_repo")
        if not full or not naive or not a3:
            # Labels are included in the report even if a future Method tag
            # changes; do not silently invent a ratio for an incomplete pair.
            comparisons.append({"task": task, "complete": False, "methods": methods})
            continue
        full_a3 = _ratio(full["ttft_p50"], a3["ttft_p50"])
        a3_naive = _ratio(a3["ttft_p50"], naive["ttft_p50"])
        if full_a3 is not None:
            full_over_a3.append(full_a3)
        if a3_naive is not None:
            a3_over_naive.append(a3_naive)
        comparisons.append({
            "task": task,
            "complete": True,
            "full_ttft_p50": full["ttft_p50"],
            "naive_ttft_p50": naive["ttft_p50"],
            "a3_ttft_p50": a3["ttft_p50"],
            "full_over_a3_speedup": full_a3,
            "a3_over_naive_overhead": a3_naive,
        })

    def mean(values: List[float]) -> Any:
        return sum(values) / len(values) if values else None

    return {
        "protocol": "kvbench_ttft_ratio_v1",
        "goal": "Compare Run-stage TTFT ratios; not a full paper reproduction",
        "config": {
            "model": str(Path(args.model).expanduser().resolve()),
            "gpu": args.gpu,
            "tasks": list(args.tasks),
            "methods": ["vanilla", "fullreuse", "a3"],
            "max_samples": args.max_samples,
            "start_index": args.start_index,
            "chunk_size_tokens": args.chunk_size,
            "recomp_ratio": args.recomp_ratio,
            "baseline_dtype": args.dtype,
        },
        "runs": rows,
        "comparisons": comparisons,
        "macro_mean": {
            "full_over_a3_speedup": mean(full_over_a3),
            "a3_over_naive_overhead": mean(a3_over_naive),
        },
        "engine_report": engine_report,
    }


def main() -> int:
    raw_args = parse_args()
    args = _driver_args(raw_args)
    driver.preflight(args)
    config = driver.configure_runtime(args)
    tasks = driver.build_tasks(args)
    methods = driver.build_methods(args)

    from core.engine import Engine
    from metrics import TTFTMetric

    print("[ttft] tasks:", ", ".join(task.Label for task in tasks), flush=True)
    print("[ttft] methods:", ", ".join(method.Label for method in methods), flush=True)
    engine_report = Engine().Evaluate(
        tasks=tasks,
        methods=methods,
        metrics=[TTFTMetric()],
    )

    output_root = raw_args.output_root.expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    report = make_report(engine_report, raw_args)
    report["runtime_config"] = config.get("Engine", {})
    path = output_root / "ttft_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n")
    print(json.dumps({
        "ok": True,
        "output": str(path.resolve()),
        "full_over_a3_speedup": report["macro_mean"]["full_over_a3_speedup"],
        "a3_over_naive_overhead": report["macro_mean"]["a3_over_naive_overhead"],
        "engine_output": engine_report.get("output_dir"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
