#!/usr/bin/env python3
"""Multi-case validation driver for ``methods.A3Repo`` on Qwen3.

The cases are intentionally small and deterministic.  They validate the
adapter contract (exact chunk concatenation, whitespace preservation, several
chunks, and mismatch-to-full fallback) rather than claiming a paper-level
benchmark result.  ``--with-full-baseline`` additionally runs every prompt
with an empty prepared context, which is useful for checking output and
latency fields but doubles the number of generations.

The report is JSON, so it can be archived and compared across revisions::

    python scripts/TestQwen3A3.py \
      --model /data1/ly/models/Qwen3-8b \
      --repo-root /data1/ly/Projects/ragkv \
      --python /data1/ly/envs/ragkv/bin/python \
      --gpu 3 --output outputs/test_qwen3_a3.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--python", dest="python_path", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/test_qwen3_a3.json"))
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--recomp-ratio", type=float, default=0.15)
    parser.add_argument("--reuse-method", default="debug",
                        choices=("debug", "sample"))
    parser.add_argument("--with-full-baseline", action="store_true",
                        help="Also run each prompt with no prepared chunks")
    parser.add_argument("--quick", action="store_true",
                        help="Run only the smallest reuse case as a smoke test")
    return parser.parse_args()


def cases(*, quick: bool = False) -> List[Dict[str, Any]]:
    first = "Document one. Key fact: alpha."
    second = " Document two contains a distractor."
    all_cases = [
        {
            "name": "two_chunks_contiguous",
            "chunks": [first, second],
            "prompt": first + second + "\nQuestion: What is the key fact?\nAnswer:",
            "expect_reuse": True,
        },
        {
            "name": "whitespace_preserved",
            "chunks": ["Title:\n", "  The answer is blue.\n"],
            "prompt": "Title:\n  The answer is blue.\nQuestion: answer?\nAnswer:",
            "expect_reuse": True,
        },
        {
            "name": "three_chunks",
            "chunks": ["A=1; ", "B=2; ", "C=3."],
            "prompt": "A=1; B=2; C=3.\nQuestion: What is C?\nAnswer:",
            "expect_reuse": True,
        },
        {
            "name": "prefix_mismatch_falls_back",
            "chunks": ["Cached context that is not present."],
            "prompt": "A completely different context.\nQuestion: answer?\nAnswer:",
            "expect_reuse": False,
        },
    ]
    return all_cases[:1] if quick else all_cases


def result_record(result: Any, elapsed: float, *, baseline: bool = False) -> Dict[str, Any]:
    metadata = dict(result.metadata)
    return {
        "output": result.output,
        "performance": dict(result.performance),
        "metadata": metadata,
        "elapsed_wall": elapsed,
        "baseline": baseline,
    }


def main() -> int:
    args = parse_args()
    selected_cases = cases(quick=args.quick)
    from core import Config as C
    C.LoadConfig()["ModelPath"] = str(Path(args.model).expanduser().resolve())
    from methods.A3Repo import A3Repo

    method = A3Repo(
        gpuNums=1,
        maxNewTokens=args.max_new_tokens,
        maxModelLen=args.max_model_len,
        recompRatio=args.recomp_ratio,
        reuseMethod=args.reuse_method,
        repoPath=args.repo_root,
        pythonPath=args.python_path,
        tag="qwen3-full",
    )
    report: Dict[str, Any] = {
        "model": str(Path(args.model).expanduser().resolve()),
        "gpu": args.gpu,
        "recomp_ratio": args.recomp_ratio,
        "reuse_method": args.reuse_method,
        "quick": args.quick,
        "cases": [],
    }
    failures: List[str] = []
    try:
        method.Initialize([args.gpu])
        for case in selected_cases:
            method.Reset()
            started = time.perf_counter()
            method.Prepare([case["chunks"]])
            result = method.Run([case["prompt"]])[0]
            reuse = result_record(result, time.perf_counter() - started)
            full_prefill = bool(reuse["metadata"].get("full_prefill"))
            observed_reuse = not full_prefill
            case_report: Dict[str, Any] = {
                "name": case["name"],
                "chunks": case["chunks"],
                "prompt": case["prompt"],
                "reuse": reuse,
                "checks": {
                    "expected_path_matches": observed_reuse == case["expect_reuse"],
                    "has_output": bool(result.output),
                    "has_input_count": reuse["metadata"].get("n_input") is not None,
                },
            }
            if args.with_full_baseline:
                method.Reset()
                started = time.perf_counter()
                method.Prepare([[]])
                baseline_result = method.Run([case["prompt"]])[0]
                case_report["full_baseline"] = result_record(
                    baseline_result, time.perf_counter() - started, baseline=True
                )
                case_report["checks"]["baseline_is_full"] = bool(
                    case_report["full_baseline"]["metadata"].get("full_prefill")
                )
            case_report["ok"] = all(case_report["checks"].values())
            if not case_report["ok"]:
                failures.append(case["name"])
            report["cases"].append(case_report)
    finally:
        method.Close()

    report["ok"] = not failures and len(report["cases"]) == len(selected_cases)
    report["failures"] = failures
    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n")
    print(json.dumps({"ok": report["ok"], "output": str(output), "failures": failures},
                     ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
