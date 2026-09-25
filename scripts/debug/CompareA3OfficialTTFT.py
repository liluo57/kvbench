#!/usr/bin/env python3
"""Compare ragkv official vanilla() and A3 reuse on one identical prompt.

The script is intentionally independent of KVBench core/workload code.  It is
an adapter/runtime sanity check: the full baseline is ragkv ``vanilla()`` and
the reuse path is ragkv ``decode(..., reuse_config)`` through A3RepoHelper.
Both workers use the same model checkpoint, official checkout, tokenizer,
dtype, GPU and timing boundary; they differ only in the official reuse mode.

Input is a JSON file containing one case or a list of cases.  Each case has
``{"chunks": [...], "suffix": "..."}``; the full prompt is reconstructed as
``''.join(chunks) + suffix``.  Every case is checked against the tokenizer so
the two requests cannot silently measure different token sequences.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from pathlib import Path


def _wait_ready(proc: subprocess.Popen[str], marker: str) -> None:
    assert proc.stdout is not None
    while True:
        line = proc.stdout.readline()
        if not line:
            error = proc.stderr.read() if proc.stderr else ""
            raise RuntimeError(f"worker exited before ready: {error[-2000:]}")
        line = line.rstrip()
        if line.startswith(marker):
            return
        print(line, file=sys.stderr, flush=True)


def _drain_stderr(proc: subprocess.Popen[str]) -> None:
    if proc.stderr is None:
        return
    for line in proc.stderr:
        print(line.rstrip(), file=sys.stderr, flush=True)


def _start(python: str, helper: str, marker: str, args: argparse.Namespace) -> subprocess.Popen[str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    # The ragkv environment contains its own ninja executable, required by
    # FlashInfer's first-use CUDA extension build.  Do not rely on the parent
    # SSH/login shell having activated that environment.
    env["PATH"] = str(Path(python).resolve().parent) + os.pathsep + env.get("PATH", "")
    proc = subprocess.Popen(
        [
            python,
            helper,
            "--repo_root", args.repo_root,
            "--model", args.model,
            "--max_new_tokens", str(args.max_new_tokens),
            "--recomp_ratio", str(args.recomp_ratio),
            "--reuse_method", args.reuse_method,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        cwd=args.repo_root,
        env=env,
    )
    threading.Thread(target=_drain_stderr, args=(proc,), daemon=True).start()
    _wait_ready(proc, marker)
    return proc


def _request(proc: subprocess.Popen[str], payload: dict) -> dict:
    if proc.stdin is None or proc.stdout is None:
        raise RuntimeError("worker pipes are unavailable")
    proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
    proc.stdin.flush()
    while True:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("worker closed stdout")
        try:
            response = json.loads(line)
        except json.JSONDecodeError:
            print(line.rstrip(), file=sys.stderr, flush=True)
            continue
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "worker request failed"))
        return response


def _close(proc: subprocess.Popen[str]) -> None:
    try:
        _request(proc, {"op": "close"})
    except Exception:
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _validate_cases(raw_case, tokenizer):
    cases = raw_case if isinstance(raw_case, list) else [raw_case]
    validated = []
    for index, case in enumerate(cases):
        chunks = [str(item) for item in case.get("chunks", [])]
        suffix = str(case.get("suffix", ""))
        prompt = "".join(chunks) + suffix
        if not chunks or not suffix:
            raise ValueError(f"case {index} must contain non-empty chunks and suffix")
        explicit_ids = case.get("chunk_ids") is not None and case.get("suffix_ids") is not None
        if explicit_ids:
            chunk_ids = [list(map(int, ids)) for ids in case["chunk_ids"]]
            suffix_ids = list(map(int, case["suffix_ids"]))
            full_ids = [token_id for ids in chunk_ids for token_id in ids] + suffix_ids
            if case.get("input_ids") is not None and list(map(int, case["input_ids"])) != full_ids:
                raise ValueError(f"case {index} input_ids do not equal chunk_ids + suffix_ids")
        else:
            full_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
            chunk_ids = [list(tokenizer.encode(text, add_special_tokens=False)) for text in chunks]
            suffix_ids = list(tokenizer.encode(suffix, add_special_tokens=False))
            split_ids = [token_id for ids in chunk_ids + [suffix_ids] for token_id in ids]
            if full_ids != split_ids:
                raise ValueError(
                    f"case {index} is not token-additive: full prompt has "
                    f"{len(full_ids)} tokens but independent chunks have {len(split_ids)}"
                )
        validated.append({
            "id": case.get("id", index),
            "chunks": chunks,
            "suffix": suffix,
            "prompt": prompt,
            "input_tokens": len(full_ids),
            "input_ids": full_ids,
            "chunk_ids": chunk_ids,
            "suffix_ids": suffix_ids,
            "tokenization_mode": "explicit_ids" if explicit_ids else "text_verified",
        })
    return validated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-json", required=True, help="JSON with chunks and suffix")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--python", required=True, help="official ragkv environment Python")
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--recomp-ratio", type=float, default=0.15)
    parser.add_argument("--reuse-method", default="debug")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    raw_case = json.loads(Path(args.case_json).read_text(encoding="utf-8"))
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    cases = _validate_cases(raw_case, tokenizer)
    all_chunks = []
    seen_chunks = set()
    for case in cases:
        for chunk in case["chunks"]:
            if chunk not in seen_chunks:
                seen_chunks.add(chunk)
                all_chunks.append(chunk)

    helper_dir = Path(__file__).resolve().parents[2] / "helpers" / "a3_repo"
    # Run the workers sequentially.  A3 keeps both its precompute and runtime
    # model resident; starting Vanilla concurrently would unnecessarily double
    # peak GPU memory and can turn a valid comparison into an OOM.
    vanilla = None
    a3 = None
    try:
        vanilla = _start(
            args.python,
            str(helper_dir / "OfficialVanillaHelper.py"),
            "[official-vanilla-helper] ready",
            args,
        )
        for _ in range(max(0, args.warmup)):
            _request(vanilla, {"op": "run", "text": cases[0]["prompt"], "input_ids": cases[0]["input_ids"]})
        full_results = [
            _request(vanilla, {"op": "run", "text": case["prompt"], "input_ids": case["input_ids"]})
            for case in cases
        ]
        _close(vanilla)
        vanilla = None

        a3 = _start(
            args.python,
            str(helper_dir / "A3RepoHelper.py"),
            "[a3-repo-helper] ready",
            args,
        )
        all_chunk_ids = []
        for chunk in all_chunks:
            owner = next(case for case in cases if chunk in case["chunks"])
            all_chunk_ids.append(owner["chunk_ids"][owner["chunks"].index(chunk)])
        _request(a3, {"op": "collect", "chunks": all_chunks, "chunk_ids": all_chunk_ids})
        for _ in range(max(0, args.warmup)):
            _request(a3, {"op": "reuse", "chunks": cases[0]["chunks"], "suffix": cases[0]["suffix"], "suffix_ids": cases[0]["suffix_ids"]})
        reuse_results = [
            _request(a3, {"op": "reuse", "chunks": case["chunks"], "suffix": case["suffix"], "suffix_ids": case["suffix_ids"]})
            for case in cases
        ]
        runtime_match = all(
            full.get("runtime_mode") == reuse.get("runtime_mode")
            and full.get("runtime_mode") == "ragkv_official_patched"
            for full, reuse in zip(full_results, reuse_results)
        )
        per_case = []
        for case, full, reuse in zip(cases, full_results, reuse_results):
            per_case.append({
                "id": case["id"],
                "input_tokens": case["input_tokens"],
                "num_chunks": len(case["chunks"]),
                "tokenization_mode": case["tokenization_mode"],
                "full": full,
                "a3_reuse": reuse,
                "ttft_speedup": (float(full["ttft"]) / float(reuse["ttft"])) if reuse["ttft"] else None,
            })
        full_mean = sum(float(item["full"]["ttft"]) for item in per_case) / len(per_case)
        reuse_mean = sum(float(item["a3_reuse"]["ttft"]) for item in per_case) / len(per_case)
        report = {
            "config": {
                "model": args.model,
                "gpu_id": str(args.gpu_id),
                "max_new_tokens": args.max_new_tokens,
                "recomp_ratio": args.recomp_ratio,
                "reuse_method": args.reuse_method,
                "warmup": args.warmup,
                "runtime": "official ragkv; separate vanilla/reuse workers",
                "runtime_kernel_match": runtime_match,
                "num_cases": len(cases),
            },
            "per_case": per_case,
            "summary": {
                "mean_full_ttft": full_mean,
                "mean_a3_ttft": reuse_mean,
                "ttft_speedup": full_mean / reuse_mean if reuse_mean else None,
            },
        }
        rendered = json.dumps(report, indent=2, ensure_ascii=False)
        print(rendered)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    finally:
        if vanilla is not None:
            _close(vanilla)
        if a3 is not None:
            _close(a3)


if __name__ == "__main__":
    main()
