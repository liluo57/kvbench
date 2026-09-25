#!/usr/bin/env python3
"""KVBench TTFT-only comparison for ProphetKV.

This is intentionally separate from the accuracy runner.  It compares only
FullPrefill, NaiveReuse, and ProphetKV@20%, and reports each speedup relative
to FullPrefill.
The default tasks are the locally available RULER CWE/VT tasks; use
``--ruler-length 4096`` and ``--ruler-length 8192`` for the available
Table-2-style context lengths.  KVBench currently has no 16K RULER files.
"""

from __future__ import annotations

import json
from pathlib import Path

if __package__:
    from .prophetkv_kvbench_common import (
        build_methods, build_tasks, configure, parser_for, preflight
    )
else:
    from prophetkv_kvbench_common import (
        build_methods, build_tasks, configure, parser_for, preflight
    )


def _speedup(report):
    values = {}
    for row in report.get("cores", []):
        task = str(row.get("task", ""))
        method = str(row.get("method", ""))
        ttft = row.get("ttft")
        if task and ttft is not None:
            values.setdefault(task, {})[method] = float(ttft)
    rows = []
    for task in sorted(values):
        full = values[task].get("full_prefill(full)")
        naive = values[task].get("naive(naive)")
        prophet = values[task].get("prophetkv(r0.20)")
        if full is None or prophet is None or full <= 0 or prophet <= 0:
            continue
        result = {
            "task": task,
            "full_ttft_sec": full,
            "prophetkv_ttft_sec": prophet,
            "speedup_x": full / prophet,
            "ttft_reduction": 1.0 - prophet / full,
        }
        if naive is not None and naive > 0:
            result.update({
                "naive_ttft_sec": naive,
                "naive_speedup_x": full / naive,
                "naive_ttft_reduction": 1.0 - naive / full,
            })
        rows.append(result)
    if rows:
        full = sum(row["full_ttft_sec"] for row in rows) / len(rows)
        prophet = sum(row["prophetkv_ttft_sec"] for row in rows) / len(rows)
        mean = {
            "task": "__mean_over_tasks__",
            "full_ttft_sec": full,
            "prophetkv_ttft_sec": prophet,
            "speedup_x": full / prophet,
            "ttft_reduction": 1.0 - prophet / full,
        }
        naive_values = [row["naive_ttft_sec"] for row in rows if "naive_ttft_sec" in row]
        if len(naive_values) == len(rows):
            naive = sum(naive_values) / len(naive_values)
            mean.update({
                "naive_ttft_sec": naive,
                "naive_speedup_x": full / naive,
                "naive_ttft_reduction": 1.0 - naive / full,
            })
        rows.append(mean)
    return rows


def main():
    parser = parser_for("full")
    parser.description = __doc__
    parser.set_defaults(
        tasks=["cwe", "vt"],
        methods=["full", "naive", "prophet20"],
        output_root=str(Path(__file__).resolve().parents[2] / "outputs" / "prophetkv_ttft"),
    )
    args = parser.parse_args()
    if set(args.methods) != {"full", "naive", "prophet20"}:
        raise ValueError("TTFT runner only accepts --methods full naive prophet20")
    preflight(args)
    configure(args)
    from core.engine import Engine
    from metrics import ThroughputMetric, TTFTMetric

    report = Engine().Evaluate(
        tasks=build_tasks(args),
        methods=build_methods(args),
        metrics=[TTFTMetric(), ThroughputMetric()],
    )
    payload = {
        "protocol": "kvbench_prophetkv_ttft_vs_full_v1",
        "model": str(Path(args.model).expanduser().resolve()),
        "tasks": args.tasks,
        "ruler_length": args.ruler_length,
        "chunk_size_tokens": args.chunk_size,
        "recompute_ratio": args.recomp_ratio,
        "max_samples": args.max_samples,
        "methods": ["FullPrefill", "NaiveReuse", "ProphetKV@20%"],
        "speedup": _speedup(report),
    }
    output_dir = Path(report["output_dir"])
    (output_dir / "results" / "ttft_speedup.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
