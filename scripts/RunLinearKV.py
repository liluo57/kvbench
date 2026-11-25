"""Compare bundled LinearKV selectors with the existing KVBench tasks.

Set ModelPath and the GPU pool in config.yaml. Start with --dry-run, then
run the dataset-free fresh-gap smoke before selecting a prepared QA task.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def ratio(value):
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("must be a finite ratio between 0 and 1")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("fresh-gap", "hotpotqa", "musique", "2wiki"),
                        default="fresh-gap")
    parser.add_argument("--samples", type=positive_int, default=1,
                        help="maximum cases per method (default: 1)")
    parser.add_argument("--selectors", nargs="+", choices=("epic", "cacheblend", "prophetkv"),
                        default=["epic", "cacheblend", "prophetkv"])
    parser.add_argument("--ratios", nargs="+", type=ratio, default=[0.2])
    parser.add_argument("--gpu-nums", type=positive_int, default=1,
                        help="GPUs per method; physical IDs come from config.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="print configurations without loading tasks, models or CUDA")
    return parser.parse_args(argv)


def build_methods(args):
    from methods import LinearKVMethod

    methods = [
        LinearKVMethod(gpuNums=args.gpu_nums, selector=selector, recompRatio=recompute_ratio)
        for selector in dict.fromkeys(args.selectors)
        for recompute_ratio in dict.fromkeys(args.ratios)
    ]
    methods.extend([
        LinearKVMethod(gpuNums=args.gpu_nums, recompRatio=0, tag="naive"),
        LinearKVMethod(gpuNums=args.gpu_nums, fullPrefill=True, tag="full_prefill"),
    ])
    return methods


def build_task(args):
    from tasks import FreshGapTask, HotpotQATask, MusiqueTask, TwoWikiMultiHopQATask

    if args.task == "fresh-gap":
        return FreshGapTask(nCases=args.samples)
    task_class = {"hotpotqa": HotpotQATask, "musique": MusiqueTask,
                  "2wiki": TwoWikiMultiHopQATask}[args.task]
    return task_class(maxSamples=args.samples)


def main(argv=None):
    args = parse_args(argv)
    from core.Config import DefaultConfigPath, Get, ModelPath

    methods = build_methods(args)
    plan = dict(config=str(DefaultConfigPath), model=ModelPath(), task=args.task,
                max_samples=args.samples, gpu_pool=(Get("Engine", {}) or {}).get("AvailableGpuIds", "auto"),
                methods=[dict(label=method.Label, gpu_nums=method.gpuNums,
                              selector=method.selector, recompute_ratio=method.recompRatio,
                              full_prefill=method.fullPrefill) for method in methods])
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if args.dry_run:
        return 0

    from core.engine import Engine
    from metrics import ThroughputMetric, TTFTMetric

    report = Engine().Evaluate(tasks=[build_task(args)], methods=methods,
                               metrics=[TTFTMetric(), ThroughputMetric()])
    print(json.dumps(report["cores"], indent=2, ensure_ascii=False))
    print(f"Full results: {report['output_dir']}")
    return 0 if report.get("status") == "completed" and not report.get("failures") else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[linearkv] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
