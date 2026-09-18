"""Run HotpotQA directly against the official HYPIC/SGLang Engine.

This intentionally does not import KVBench.  It uses the Qwen3.5 raw prompt
protocol from HYPIC's quick tests and evaluates the first 64 local HotpotQA
records with the LongBench token F1/EM metrics.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import string
import time
from pathlib import Path
from typing import Any, Iterable

import sglang as sgl


SEP = "<<PIC_SEP>>"
# The official HYPIC quick test uses the first spelling.  The second spelling
# is retained as the default comparison because KVBench's native chat path
# has an explicit whitespace boundary after the system text.
SYSTEM = os.environ.get("DIRECT_SYSTEM", "You are a helpful assistant.\n\n")
POST = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
KVBENCH_USER_PREFIX = "<|im_start|>user\n"
KVBENCH_USER_POST = "<|im_end|>\n" + POST
WARMUP_TAIL = "\n[KVBench HYPIC cache warmup]\n"
WARMUP_MAX_NEW_TOKENS = int(os.environ.get("DIRECT_WARMUP_MAX_NEW_TOKENS", "4"))
PREFIX = (
    "Answer the question based on the given passages. Only give me the "
    "answer and do not output any other words.\n\nThe following are given "
    "passages.\n"
)
QUERY_PREFIX = (
    "\n\nAnswer the question based on the given passages. Only give me the "
    "answer and do not output any other words.\n\nQuestion: "
)


def passage_chunks(context: str, group_size: int = 0) -> list[str]:
    parts = [
        part
        for part in re.split(
            r"(?m)(?=^Passage(?: \d+)?:[ \t]*$)", str(context)
        )
        if part
    ]
    if len(parts) <= 1:
        return [PREFIX + str(context)] if context else []
    parts[0] = PREFIX + parts[0]
    if group_size > 1:
        parts = ["".join(parts[index : index + group_size]) for index in range(0, len(parts), group_size)]
    return parts


def build_case(sample: dict[str, Any], group_size: int = 0) -> tuple[list[str], str, list[str]] | None:
    context = str(sample.get("context") or "")
    question = str(sample.get("input") or "").strip()
    answers = sample.get("answers") or []
    if not isinstance(answers, list):
        answers = [answers]
    answers = [str(answer) for answer in answers if answer is not None and str(answer).strip()]
    chunks = passage_chunks(context, group_size=group_size)
    if not chunks or not question or not answers:
        return None
    return chunks, f"{QUERY_PREFIX}{question}\nAnswer:", answers


def normalize(text: str) -> str:
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def first_line(text: str) -> str:
    text = (text or "").lstrip().split("\n")[0]
    if text.startswith(("Yes", "yes")):
        return "Yes"
    words = text.split()
    if words and words[0].startswith(("No", "no")):
        return "No"
    return text


def f1(prediction: str, answer: str) -> float:
    pred = normalize(prediction).split()
    gold = normalize(answer).split()
    if not pred or not gold:
        return float(pred == gold)
    common = collections.Counter(pred) & collections.Counter(gold)
    overlap = sum(common.values())
    if not overlap:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def score(text: str, answers: Iterable[str]) -> tuple[float, float, str]:
    prediction = first_line(text)
    return (
        max(f1(prediction, answer) for answer in answers),
        max(float(normalize(prediction) == normalize(answer)) for answer in answers),
        prediction,
    )


def prompt(
    chunks: list[str],
    suffix: str,
    *,
    segmented: bool,
    protocol: str = "official",
) -> str:
    if protocol == "kvbench":
        body = SEP.join(chunks) if segmented else "".join(chunks)
        if segmented:
            body = SEP + body + SEP
        return KVBENCH_USER_PREFIX + body + suffix + KVBENCH_USER_POST
    if segmented:
        return SYSTEM + SEP + SEP.join(chunks) + SEP + suffix + POST
    return SYSTEM + "".join(chunks) + suffix + POST


def engine_kwargs(model: str, mode: str) -> dict[str, Any]:
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


def generate(
    engine: Any, text: str, *, max_new_tokens: int = 512
) -> tuple[str, dict[str, Any]]:
    result = engine.generate(
        text,
        sampling_params={"temperature": 0.0, "max_new_tokens": max_new_tokens},
    )
    return result.get("text", ""), result.get("meta_info", {})


def run_mode(
    mode: str,
    model: str,
    cases: list[tuple[list[str], str, list[str]]],
    protocol: str,
) -> dict[str, Any]:
    print(f"\n=== {mode} ({len(cases)} cases) ===", flush=True)
    engine = sgl.Engine(**engine_kwargs(model, mode))
    rows: list[dict[str, Any]] = []
    try:
        for index, (chunks, suffix, answers) in enumerate(cases):
            engine.flush_cache()
            if mode == "full":
                text, meta = generate(
                    engine, prompt(chunks, suffix, segmented=False, protocol=protocol)
                )
            else:
                # HYPIC's official warmup pattern makes each document a
                # reusable non-final segment and leaves the question fresh.
                if os.environ.get("DIRECT_NO_WARMUP", "0") == "1":
                    warmup_chunks = []
                elif os.environ.get("DIRECT_WARMUP_ALL", "0") == "1":
                    # KVBench Prepare submits all reusable chunks in one
                    # request, then appends a disposable tail because HYPIC
                    # never indexes the final segment of a request.
                    warmup_chunks = None
                else:
                    selected = os.environ.get("DIRECT_WARMUP_INDICES", "").strip()
                    if selected:
                        indices = {
                            int(item.strip())
                            for item in selected.split(",")
                            if item.strip()
                        }
                        warmup_chunks = [
                            chunk for index, chunk in enumerate(chunks)
                            if index in indices
                        ]
                    else:
                        warmup_count = int(os.environ.get("DIRECT_WARMUP_CHUNKS", str(len(chunks))))
                        warmup_chunks = chunks[:warmup_count]
                if warmup_chunks is None:
                    if protocol == "kvbench":
                        warmup = SEP.join([*chunks, WARMUP_TAIL])
                    else:
                        warmup = SYSTEM + SEP + SEP.join(chunks) + SEP + WARMUP_TAIL
                    generate(
                        engine, warmup, max_new_tokens=WARMUP_MAX_NEW_TOKENS
                    )
                else:
                    for chunk in warmup_chunks:
                        if os.environ.get("DIRECT_WARMUP_BARE_SEGMENT", "0") == "1":
                            # Paper protocol: a reusable segment is wrapped
                            # directly by two PIC separators, with no system
                            # or query tokens in the warmup request.
                            warmup = SEP + chunk + SEP
                        elif os.environ.get("DIRECT_WARMUP_SINGLE_SEGMENT", "0") == "1":
                            warmup = chunk
                        elif os.environ.get("DIRECT_WARMUP_SEGMENT_ONLY", "0") == "1":
                            if protocol == "kvbench":
                                warmup = SEP + chunk + SEP + WARMUP_TAIL
                            else:
                                warmup = SYSTEM + SEP + chunk + SEP + WARMUP_TAIL
                        else:
                            warmup = prompt(
                                [chunk], suffix, segmented=True, protocol=protocol
                            )
                        generate(
                            engine,
                            warmup,
                            max_new_tokens=WARMUP_MAX_NEW_TOKENS,
                        )
                text, meta = generate(
                    engine, prompt(chunks, suffix, segmented=True, protocol=protocol)
                )
            case_f1, case_em, prediction = score(text, answers)
            row = {
                "index": index,
                "f1": case_f1,
                "em": case_em,
                "prediction": prediction,
                "answers": answers,
                "cached_tokens": meta.get("cached_tokens", 0),
                "prompt_tokens": meta.get("prompt_tokens", 0),
            }
            rows.append(row)
            print(
                f"{index + 1:02d}/{len(cases)} f1={case_f1:.3f} em={case_em:.0f} "
                f"cache={row['cached_tokens']}/{row['prompt_tokens']} "
                f"pred={prediction[:80]!r}",
                flush=True,
            )
    finally:
        engine.shutdown()

    summary = {
        "mode": mode,
        "cases": len(rows),
        "f1": sum(row["f1"] for row in rows) / len(rows),
        "em": sum(row["em"] for row in rows) / len(rows),
        "cached_tokens_mean": sum(row["cached_tokens"] for row in rows) / len(rows),
        "prompt_tokens_mean": sum(row["prompt_tokens"] for row in rows) / len(rows),
        "rows": rows,
    }
    print(
        f"SUMMARY {mode}: f1={summary['f1']:.6f} em={summary['em']:.6f} "
        f"cached={summary['cached_tokens_mean']:.1f}/{summary['prompt_tokens_mean']:.1f}",
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/autodl-tmp/models/Qwen3.5-35b")
    parser.add_argument("--data", default="/root/kvbench/data/hotpotqa/hotpotqa.jsonl")
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument(
        "--group-size",
        type=int,
        default=0,
        help="combine this many adjacent passages into one PIC segment (0 keeps 10 passages)",
    )
    parser.add_argument(
        "--modes",
        default="full,addition,transition_rope_recompute",
        help="comma-separated: full, addition, transition_rope_recompute",
    )
    parser.add_argument(
        "--protocol",
        choices=("official", "kvbench"),
        default="official",
        help="official raw prompt or KVBench's Qwen ChatML boundary protocol",
    )
    parser.add_argument("--output", default="/root/kvbench/outputs/direct-hypic-hotpotqa-64.json")
    args = parser.parse_args()

    data = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    cases = [
        case
        for sample in data[args.start :]
        for case in [build_case(sample, group_size=args.group_size)]
        if case
    ][: args.limit]
    print(f"model={args.model} data={args.data} cases={len(cases)}", flush=True)
    print(
        f"modes={args.modes} group_size={args.group_size} protocol={args.protocol}",
        flush=True,
    )
    started = time.time()
    results = {
        mode: run_mode(mode, args.model, cases, args.protocol)
        for mode in [item.strip() for item in args.modes.split(",") if item.strip()]
    }
    output = {"model": args.model, "data": args.data, "cases": len(cases), "elapsed_sec": time.time() - started, "results": results}
    Path(args.output).write_text(json.dumps(output, indent=2, ensure_ascii=True) + "\n")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
