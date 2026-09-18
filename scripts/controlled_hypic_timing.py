"""Run the existing direct HotpotQA harness while retaining measured TTFT.

The quality artifacts in this workspace predate TTFT being persisted in the
direct harness.  This wrapper reuses the exact same prompt builder, evaluator,
sampling parameters, cache flush, and engine configuration, and only records
the measured request's wall time / engine e2e latency.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.direct_hypic_hotpotqa as harness

MODEL = os.environ.get("PIC_MODEL", "/root/autodl-tmp/models/Qwen3.5-35b")


def run_one(mode: str, cases, protocol: str, priming: str) -> dict:
    if mode == "full":
        os.environ.pop("DIRECT_WARMUP_ALL", None)
        os.environ.pop("DIRECT_WARMUP_SEGMENT_ONLY", None)
    elif priming == "joint":
        os.environ["DIRECT_WARMUP_ALL"] = "1"
        os.environ.pop("DIRECT_WARMUP_SEGMENT_ONLY", None)
    else:
        os.environ.pop("DIRECT_WARMUP_ALL", None)
        os.environ["DIRECT_WARMUP_SEGMENT_ONLY"] = "1"

    measured_texts = set()
    for chunks, suffix, _answers in cases:
        measured_texts.add(
            harness.prompt(
                chunks,
                suffix,
                segmented=(mode != "full"),
                protocol=protocol,
            )
        )
    timings = []
    original = harness.generate

    def timed_generate(engine, text, *, max_new_tokens=512):
        started = time.perf_counter()
        output, meta = original(engine, text, max_new_tokens=max_new_tokens)
        if text in measured_texts:
            timings.append(
                {
                    "wall_sec": time.perf_counter() - started,
                    "e2e_latency": meta.get("e2e_latency"),
                }
            )
        return output, meta

    harness.generate = timed_generate
    try:
        summary = harness.run_mode(mode, MODEL, cases, protocol)
    finally:
        harness.generate = original
    summary["ttft_wall_mean"] = sum(x["wall_sec"] for x in timings) / len(timings)
    summary["ttft_wall_median"] = sorted(x["wall_sec"] for x in timings)[len(timings) // 2]
    e2e = [x["e2e_latency"] for x in timings if x["e2e_latency"] is not None]
    summary["e2e_latency_mean"] = sum(e2e) / len(e2e) if e2e else None
    summary["timed_cases"] = len(timings)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/root/kvbench/data/hotpotqa/hotpotqa.jsonl")
    ap.add_argument("--limit", type=int, default=64)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--modes", default="full,addition,transition_rope_recompute")
    ap.add_argument("--priming", choices=("isolated", "joint"), default="isolated")
    ap.add_argument("--protocol", choices=("official", "kvbench"), default="official")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    data = [
        json.loads(line)
        for line in Path(args.data).read_text().splitlines()
        if line.strip()
    ]
    cases = [
        case
        for sample in data[args.start:]
        for case in [harness.build_case(sample)]
        if case
    ][: args.limit]
    started = time.time()
    results = {
        mode: run_one(mode, cases, args.protocol, args.priming)
        for mode in [x.strip() for x in args.modes.split(",") if x.strip()]
    }
    result = {
        "model": MODEL,
        "data": args.data,
        "cases": len(cases),
        "priming": args.priming,
        "protocol": args.protocol,
        "elapsed_sec": time.time() - started,
        "results": results,
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
