#!/usr/bin/env python3
"""Small end-to-end smoke test for the official A^3 adapter on Qwen3.

This script deliberately exercises the public KVBench Method lifecycle only:
``Initialize -> Prepare -> Run -> Reset -> Close``.  It does not import or
modify the official ragkv checkout in the coordinator process.

Example (run with KVBench's coordinator environment)::

    python scripts/qwen3_a3_smoke.py \
      --model /data1/ly/models/Qwen3-8b \
      --repo-root /data1/ly/Projects/ragkv \
      --python /data1/ly/envs/ragkv/bin/python \
      --gpu 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local Qwen3 checkpoint")
    parser.add_argument("--repo-root", required=True, help="Official ragkv checkout")
    parser.add_argument("--python", dest="python_path", required=True,
                        help="Python executable for the ragkv worker")
    parser.add_argument("--gpu", type=int, required=True, help="Physical GPU id")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--recomp-ratio", type=float, default=0.15)
    parser.add_argument("--reuse-method", default="debug",
                        choices=("debug", "sample"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # A3Repo intentionally reads the global model path at construction time.
    # Override it in memory so this diagnostic does not edit config.yaml.
    from core import Config as C
    C.LoadConfig()["ModelPath"] = str(Path(args.model).expanduser().resolve())
    from methods.A3Repo import A3Repo

    chunks = [
        "Document one. Key fact: alpha.",
        " Document two contains a distractor.",
    ]
    prompt = "".join(chunks) + "\nQuestion: What is the key fact?\nAnswer:"

    method = A3Repo(
        gpuNums=1,
        maxNewTokens=args.max_new_tokens,
        maxModelLen=args.max_model_len,
        recompRatio=args.recomp_ratio,
        reuseMethod=args.reuse_method,
        repoPath=args.repo_root,
        pythonPath=args.python_path,
        tag="qwen3-smoke",
    )
    try:
        method.Initialize([args.gpu])
        method.Prepare([chunks])
        result = method.Run([prompt])[0]
        metadata = dict(result.metadata)
        report = {
            "ok": True,
            "model": str(Path(args.model).expanduser().resolve()),
            "gpu": args.gpu,
            "output": result.output,
            "performance": result.performance,
            "metadata": metadata,
            "checks": {
                "has_output": bool(result.output),
                "has_input_count": metadata.get("n_input") is not None,
                "used_reuse_path": metadata.get("full_prefill") is not True,
                "has_a3_debug": metadata.get("a3_debug") is not None,
            },
        }
        report["ok"] = all(report["checks"].values())
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        return 0 if report["ok"] else 2
    finally:
        method.Close()


if __name__ == "__main__":
    raise SystemExit(main())
