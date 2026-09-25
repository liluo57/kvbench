"""Small four-workload HYPIC reproduction harness.

This research script uses the local LongBench snapshots and the same raw
Qwen3.5 prompt shape as HYPIC's official quick test.  It separates:

* ``isolated``: one ``SYSTEM + SEP + chunk + SEP + query`` request per chunk;
* ``joint``: one request containing all chunks plus a disposable final tail;
* ``none``: no PIC priming.

The point is protocol comparison, not a replacement for KVBench's benchmark
entry point.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import string
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import sglang as sgl
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.GovReport import GovReportTask
from tasks.HotpotQA import HotpotQATask
from tasks.MultiNews import MultiNewsTask
from tasks.TriviaQA import TriviaQATask
from tasks.bases.KBBase import RougeL, TokenEm, TokenF1


SEP = "<<PIC_SEP>>"
SYSTEM = os.environ.get("DIRECT_SYSTEM", "You are a helpful assistant.")
POST = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
WARMUP_TAIL = "\n[KVBench HYPIC cache warmup]\n"
WARMUP_MAX_NEW_TOKENS = int(os.environ.get("DIRECT_WARMUP_MAX_NEW_TOKENS", "4"))
MAX_NEW_TOKENS = int(os.environ.get("DIRECT_MAX_NEW_TOKENS", "512"))
MODEL_DEFAULT = os.environ.get("PIC_MODEL")


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _task(task_name: str, data_path: str, n_chunks: int):
    # _Build is pure with respect to a sample, so a task instance is enough;
    # loading is done here to allow selecting the checked-in Hotpot snapshots.
    common = dict(dataDir=str(Path(data_path).parent), maxSamples=-1)
    if task_name == "hotpotqa":
        return HotpotQATask(**common)
    if task_name == "triviaqa":
        return TriviaQATask(**common)
    if task_name == "multinews":
        return MultiNewsTask(**common)
    if task_name == "govreport":
        return GovReportTask(**common, nChunks=n_chunks)
    raise ValueError(f"unknown task: {task_name}")


def _answers(sample: dict[str, Any]) -> list[str]:
    value = sample.get("answers") or []
    if not isinstance(value, (list, tuple)):
        value = [value]
    out: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        elif item is not None and str(item).strip():
            out.append(str(item))

    visit(value)
    return out


def _build_cases(
    task_name: str, data_path: str, start: int, limit: int, n_chunks: int
) -> list[tuple[list[str], str, list[str]]]:
    builder = _task(task_name, data_path, n_chunks)
    cases = []
    for sample in _read_jsonl(data_path)[start : start + limit]:
        chunks, suffix = builder._Build(sample)
        answers = _answers(sample)
        if chunks and suffix and answers:
            cases.append((chunks, suffix, answers))
    return cases


def _join(parts: Iterable[str]) -> str:
    values = [part for part in parts if part]
    if any(SEP in value for value in values):
        raise ValueError("input contains PIC separator")
    return SEP.join(values)


def _segmented(chunks: list[str], suffix: str) -> str:
    return _join([SYSTEM, *chunks, suffix]) + POST


def _full(chunks: list[str], suffix: str) -> str:
    return SYSTEM + "".join(chunks) + suffix + POST


def _normalize(text: str) -> str:
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _first_line(text: str) -> str:
    text = (text or "").lstrip().split("\n")[0]
    if text.startswith(("Yes", "yes")):
        return "Yes"
    words = text.split()
    if words and words[0].startswith(("No", "no")):
        return "No"
    return text


def _score(task_name: str, text: str, answers: list[str]) -> tuple[float, float, str]:
    if task_name in ("multinews", "govreport"):
        pred = str(text or "").lstrip()
        value = max((RougeL(pred, answer) for answer in answers), default=0.0)
        return value, value, pred
    pred = _first_line(text)
    f1 = max(TokenF1(pred, answer) for answer in answers)
    em = max(TokenEm(pred, answer) for answer in answers)
    return f1, em, pred


def _engine_kwargs(model: str, mode: str) -> dict[str, Any]:
    common = dict(
        model_path=model,
        tp_size=int(os.environ.get("PIC_TP", "1")),
        dtype="bfloat16",
        context_length=32768,
        max_prefill_tokens=32768,
        max_running_requests=1,
        mem_fraction_static=float(os.environ.get("PIC_MEM", "0.80")),
        trust_remote_code=True,
        enable_multimodal=False,
        page_size=1,
        chunked_prefill_size=-1,
        cuda_graph_backend_prefill="disabled",
        log_level="error",
    )
    if mode == "full":
        common.update(
            pic_enable=False,
            mamba_radix_cache_strategy="no_buffer",
            disable_radix_cache=True,
            disable_overlap_schedule=True,
        )
    else:
        common.update(
            pic_enable=True,
            pic_mode=mode,
            pic_separator_str=SEP,
            max_mamba_cache_size=int(os.environ.get("PIC_MAMBA", "32")),
        )
    return common


def _generate(engine: Any, prompt: str, max_new_tokens: int) -> tuple[str, dict[str, Any]]:
    out = engine.generate(
        prompt,
        sampling_params={"temperature": 0.0, "max_new_tokens": max_new_tokens},
    )
    return out.get("text", ""), out.get("meta_info", {})


def _run_mode(
    task_name: str,
    mode: str,
    priming: str,
    model: str,
    cases: list[tuple[list[str], str, list[str]]],
    tokenizer: Any,
) -> dict[str, Any]:
    print(f"\n=== {task_name} {mode} priming={priming} ({len(cases)} cases) ===", flush=True)
    engine = sgl.Engine(**_engine_kwargs(model, mode))
    rows: list[dict[str, Any]] = []
    try:
        for index, (chunks, suffix, answers) in enumerate(cases):
            engine.flush_cache()
            if mode == "full":
                request = _full(chunks, suffix)
            else:
                if priming == "isolated":
                    for chunk in chunks:
                        _generate(
                            engine,
                            _segmented([chunk], suffix),
                            WARMUP_MAX_NEW_TOKENS,
                        )
                elif priming == "joint":
                    _generate(
                        engine,
                        _segmented([*chunks, WARMUP_TAIL], ""),
                        WARMUP_MAX_NEW_TOKENS,
                    )
                elif priming != "none":
                    raise ValueError(f"unknown priming policy: {priming}")
                request = _segmented(chunks, suffix)

            text, meta = _generate(engine, request, MAX_NEW_TOKENS)
            value, secondary, prediction = _score(task_name, text, answers)
            chunk_tokens = [
                len(tokenizer.encode(chunk, add_special_tokens=False))
                for chunk in chunks
            ]
            system_tokens = len(tokenizer.encode(SYSTEM, add_special_tokens=False))
            row = {
                "index": index,
                "score": value,
                "f1_or_rougeL": value,
                "em_or_duplicate": secondary,
                "prediction": prediction,
                "answers": answers,
                "chunk_count": len(chunks),
                "chunk_token_counts": chunk_tokens,
                "expected_reusable_tokens": system_tokens + sum(chunk_tokens),
                "prompt_tokens": meta.get("prompt_tokens", 0),
                "cached_tokens": meta.get("cached_tokens", 0),
            }
            rows.append(row)
            print(
                f"{index + 1:02d}/{len(cases)} score={value:.3f} "
                f"chunks={len(chunks)} prompt={row['prompt_tokens']} "
                f"cache={row['cached_tokens']}/{row['expected_reusable_tokens']} "
                f"pred={prediction[:80]!r}",
                flush=True,
            )
    finally:
        engine.shutdown()

    summary = {
        "task": task_name,
        "mode": mode,
        "priming": priming,
        "cases": len(rows),
        "score": sum(row["score"] for row in rows) / len(rows),
        "secondary": sum(row["em_or_duplicate"] for row in rows) / len(rows),
        "cached_tokens_mean": sum(row["cached_tokens"] for row in rows) / len(rows),
        "expected_reusable_tokens_mean": sum(
            row["expected_reusable_tokens"] for row in rows
        ) / len(rows),
        "prompt_tokens_mean": sum(row["prompt_tokens"] for row in rows) / len(rows),
        "chunk_count_mean": sum(row["chunk_count"] for row in rows) / len(rows),
        "rows": rows,
    }
    print(
        f"SUMMARY {task_name} {mode} {priming}: score={summary['score']:.6f} "
        f"secondary={summary['secondary']:.6f} "
        f"cache={summary['cached_tokens_mean']:.1f}/"
        f"{summary['expected_reusable_tokens_mean']:.1f} "
        f"prompt={summary['prompt_tokens_mean']:.1f}",
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", choices=("hotpotqa", "triviaqa", "multinews", "govreport"), required=True
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n-chunks", type=int, default=1)
    parser.add_argument(
        "--modes", default="full,addition,transition_rope_recompute"
    )
    parser.add_argument(
        "--priming", default="isolated", choices=("isolated", "joint", "none")
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not args.model:
        parser.error("--model or PIC_MODEL is required")

    cases = _build_cases(
        args.task, args.data, args.start, args.limit, args.n_chunks
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(
        f"task={args.task} data={args.data} cases={len(cases)} "
        f"modes={args.modes} priming={args.priming} n_chunks={args.n_chunks}",
        flush=True,
    )
    started = time.time()
    results = {
        mode: _run_mode(
            args.task, mode, args.priming, args.model, cases, tokenizer
        )
        for mode in [item.strip() for item in args.modes.split(",") if item.strip()]
    }
    output = {
        "task": args.task,
        "data": args.data,
        "cases": len(cases),
        "priming": args.priming,
        "elapsed_sec": time.time() - started,
        "results": results,
    }
    Path(args.output).write_text(
        json.dumps(output, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
