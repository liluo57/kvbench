#!/usr/bin/env python3
"""Reproducible HotpotQA / TriviaQA evaluation client for HYPIC + SGLang.

Workflow:
  1) prepare: stream a deterministic slice from Hugging Face into one JSONL file.
  2) run: read that exact JSONL, warm the cache according to the selected mode,
     call SGLang's /generate endpoint with streaming, measure client-side TTFT,
     compute QA EM/F1, and write raw + summary results.

This is an independent evaluation harness. The public HYPIC repository does not
ship the paper's exact dataset preprocessing templates or raw Figure 10 data.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import math
import re
import statistics
import string
import sys
import time
from pathlib import Path
from typing import Any, Iterable

SEP = "<<PIC_SEP>>"
INSTRUCTION = (
    "Answer the question using only the evidence passages. "
    "Return only the shortest answer, without explanation."
)


def normalize_answer(text: str) -> str:
    def remove_articles(s: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def remove_punc(s: str) -> str:
        return "".join(ch for ch in s if ch not in string.punctuation)

    return " ".join(remove_articles(remove_punc(text.lower())).split())


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)
    overlap = collections.Counter(pred_tokens) & collections.Counter(ref_tokens)
    common = sum(overlap.values())
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def best_score(prediction: str, references: list[str]) -> tuple[float, float]:
    refs = [x for x in references if isinstance(x, str) and x.strip()] or [""]
    return (
        max(exact_match(prediction, ref) for ref in refs),
        max(token_f1(prediction, ref) for ref in refs),
    )


def shorten_prediction(text: str) -> str:
    # Qwen-family models may emit a hidden/reasoning block despite the prompt.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.I | re.S).strip()
    text = re.sub(r"^(final\s+)?answer\s*:\s*", "", text, flags=re.I).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    answer = lines[0]
    answer = answer.strip("*` ")
    # Avoid a trailing explanatory sentence while retaining abbreviations/numbers.
    answer = re.split(r"(?<=[.!?])\s+(?=[A-Z])", answer, maxsplit=1)[0]
    return answer.strip()


def as_parallel_lists(obj: Any, keys: list[str]) -> Iterable[dict[str, Any]]:
    """Accept HF dict-of-lists, list-of-dicts, or empty values."""
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                yield item
        return
    if not isinstance(obj, dict):
        return
    lengths = [len(obj.get(k, [])) for k in keys if isinstance(obj.get(k), list)]
    for i in range(max(lengths, default=0)):
        yield {
            k: (obj.get(k, [])[i] if isinstance(obj.get(k), list) and i < len(obj[k]) else "")
            for k in keys
        }


def truncate_text(text: str, max_chars: int) -> str:
    text = " ".join(str(text).split())
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    cut = text.rfind(" ", 0, max_chars)
    return text[: cut if cut > max_chars // 2 else max_chars].rstrip() + " …"


def dedupe_segments(segments: list[str], max_segments: int) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for segment in segments:
        key = normalize_answer(segment)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(segment)
        if max_segments > 0 and len(result) >= max_segments:
            break
    return result


def prepare_hotpot(row: dict[str, Any], max_segments: int, max_chars: int) -> dict[str, Any]:
    context = row.get("context") or {}
    segments: list[str] = []
    if isinstance(context, dict):
        titles = context.get("title") or []
        sentence_groups = context.get("sentences") or []
        for title, sentences in itertools.zip_longest(titles, sentence_groups, fillvalue=""):
            body = " ".join(sentences) if isinstance(sentences, list) else str(sentences)
            body = truncate_text(body, max_chars)
            if body:
                segments.append(f"Title: {title}\n{body}")
    elif isinstance(context, list):
        for item in context:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                body = " ".join(item[1]) if isinstance(item[1], list) else str(item[1])
                segments.append(f"Title: {item[0]}\n{truncate_text(body, max_chars)}")
    answer = str(row.get("answer", ""))
    return {
        "id": str(row.get("id", "")),
        "dataset": "hotpotqa",
        "question": str(row.get("question", "")),
        "segments": dedupe_segments(segments, max_segments),
        "references": [answer],
        "metadata": {"type": row.get("type"), "level": row.get("level")},
    }


def prepare_trivia(row: dict[str, Any], max_segments: int, max_chars: int) -> dict[str, Any]:
    segments: list[str] = []
    entity_keys = ["title", "wiki_context", "filename", "doc_source"]
    for doc in as_parallel_lists(row.get("entity_pages"), entity_keys):
        body = truncate_text(doc.get("wiki_context", ""), max_chars)
        if body:
            segments.append(f"Title: {doc.get('title') or doc.get('filename') or 'Evidence'}\n{body}")
    search_keys = ["title", "search_context", "description", "filename", "rank", "url"]
    for doc in as_parallel_lists(row.get("search_results"), search_keys):
        body = doc.get("search_context") or doc.get("description") or ""
        body = truncate_text(body, max_chars)
        if body:
            segments.append(f"Title: {doc.get('title') or doc.get('filename') or 'Evidence'}\n{body}")

    answer = row.get("answer") or {}
    if isinstance(answer, dict):
        refs = list(answer.get("aliases") or [])
        value = answer.get("value")
        if value and value not in refs:
            refs.append(value)
    else:
        refs = [str(answer)]
    return {
        "id": str(row.get("question_id", row.get("id", ""))),
        "dataset": "triviaqa",
        "question": str(row.get("question", "")),
        "segments": dedupe_segments(segments, max_segments),
        "references": refs,
        "metadata": {"question_source": row.get("question_source")},
    }


def answer_is_in_context(item: dict[str, Any]) -> bool:
    context = normalize_answer(" ".join(item["segments"]))
    return any(normalize_answer(ref) in context for ref in item["references"] if normalize_answer(ref))


def cmd_prepare(args: argparse.Namespace) -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Missing dependency: pip install 'datasets>=2.19' requests") from exc

    if args.dataset == "hotpotqa":
        repo, config, converter = "hotpotqa/hotpot_qa", "distractor", prepare_hotpot
    else:
        repo, config, converter = "mandarjoshi/trivia_qa", "rc", prepare_trivia

    ds = load_dataset(repo, config, split=args.split, streaming=True)
    rows = itertools.islice(ds, args.offset, args.offset + args.limit)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output.open("w", encoding="utf-8") as fh:
        for row in rows:
            item = converter(dict(row), args.max_segments, args.max_segment_chars)
            if not item["segments"]:
                continue
            item["answer_in_retained_context"] = answer_is_in_context(item)
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
            written += 1
    print(f"Prepared {written} samples -> {output}")
    if written == 0:
        raise SystemExit("No samples were written; check dataset access and split/config.")


def build_prompt(item: dict[str, Any], separator: str) -> str:
    query = f"Question: {item['question']}\nAnswer:"
    return separator.join([INSTRUCTION, *item["segments"], query])


class SGLangClient:
    def __init__(self, base_url: str, timeout: float):
        try:
            import requests
        except ImportError as exc:
            raise SystemExit("Missing dependency: pip install requests") from exc
        self.requests = requests
        self.url = base_url.rstrip("/") + "/generate"
        self.timeout = timeout

    def generate(self, text: str, max_new_tokens: int, stream: bool) -> dict[str, Any]:
        payload = {
            "text": text,
            "sampling_params": {
                "temperature": 0.6,
                "top_p": 1.0,
                "max_new_tokens": max_new_tokens,
            },
            "stream": stream,
        }
        start = time.perf_counter()
        response = self.requests.post(
            self.url, json=payload, stream=stream, timeout=self.timeout
        )
        response.raise_for_status()
        if not stream:
            data = response.json()
            return {
                "text": data.get("text", ""),
                "meta_info": data.get("meta_info") or {},
                "ttft_s": None,
                "elapsed_s": time.perf_counter() - start,
            }

        last_text = ""
        meta: dict[str, Any] = {}
        ttft: float | None = None
        for raw in response.iter_lines(decode_unicode=True):
            if not raw:
                continue
            line = raw.strip()
            if line.startswith("data:"):
                line = line[5:].strip()
            if line == "[DONE]":
                break
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                current = event.get("text")
                if isinstance(current, str):
                    last_text = current
                    if ttft is None and current:
                        ttft = time.perf_counter() - start
                if isinstance(event.get("meta_info"), dict):
                    meta.update(event["meta_info"])
        elapsed = time.perf_counter() - start
        return {"text": last_text, "meta_info": meta, "ttft_s": ttft or elapsed, "elapsed_s": elapsed}


def warm_sample(client: SGLangClient, item: dict[str, Any], mode: str, separator: str) -> None:
    if mode == "full_recompute":
        return
    if mode == "prefix_cache":
        # Prewarm every document but in prefix form. Only the request's first
        # evidence segment can be reused by an ordinary radix/prefix cache.
        for segment in item["segments"]:
            warm_prompt = separator.join([INSTRUCTION, segment, "Warmup: OK"])
            client.generate(warm_prompt, max_new_tokens=1, stream=False)
        return
    # Addition and HYPIC: each evidence segment is an independent PIC unit.
    for segment in item["segments"]:
        warm_prompt = separator.join([INSTRUCTION, segment, "Warmup: OK"])
        client.generate(warm_prompt, max_new_tokens=1, stream=False)


def first_number(mapping: dict[str, Any], names: list[str]) -> int | None:
    for name in names:
        value = mapping.get(name)
        if isinstance(value, (int, float)):
            return int(value)
    return None


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    rank = (len(xs) - 1) * p
    lo, hi = math.floor(rank), math.ceil(rank)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - rank) + xs[hi] * (rank - lo)


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    good = [r for r in records if not r.get("error")]
    ttfts = [float(r["ttft_s"]) for r in good if r.get("ttft_s") is not None]
    ratios = [float(r["cache_ratio"]) for r in good if r.get("cache_ratio") is not None]
    return {
        "dataset": good[0]["dataset"] if good else None,
        "mode": good[0]["mode"] if good else None,
        "samples_total": len(records),
        "samples_ok": len(good),
        "mean_exact_match": statistics.fmean(r["exact_match"] for r in good) if good else None,
        "mean_f1": statistics.fmean(r["f1"] for r in good) if good else None,
        "p50_ttft_s": percentile(ttfts, 0.50),
        "p95_ttft_s": percentile(ttfts, 0.95),
        "mean_ttft_s": statistics.fmean(ttfts) if ttfts else None,
        "mean_cache_ratio": statistics.fmean(ratios) if ratios else None,
        "answer_in_context_rate": (
            statistics.fmean(float(r["answer_in_retained_context"]) for r in good) if good else None
        ),
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def cmd_run(args: argparse.Namespace) -> None:
    items = load_jsonl(Path(args.input))
    if args.limit > 0:
        items = items[: args.limit]
    client = SGLangClient(args.base_url, args.timeout)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    with output.open("w", encoding="utf-8") as fh:
        for index, item in enumerate(items, 1):
            record: dict[str, Any] = {
                "id": item["id"],
                "dataset": item["dataset"],
                "mode": args.mode,
                "num_segments": len(item["segments"]),
                "answer_in_retained_context": item.get("answer_in_retained_context"),
                "references": item["references"],
            }
            try:
                warm_sample(client, item, args.mode, args.separator)
                result = client.generate(
                    build_prompt(item, args.separator),
                    max_new_tokens=args.max_new_tokens,
                    stream=True,
                )
                prediction = shorten_prediction(result["text"])
                em, f1 = best_score(prediction, item["references"])
                meta = result["meta_info"]
                cached = first_number(meta, ["cached_tokens", "num_cached_tokens"])
                prompt_tokens = first_number(
                    meta, ["prompt_tokens", "input_tokens", "num_prompt_tokens"]
                )
                record.update(
                    {
                        "prediction": prediction,
                        "raw_output": result["text"],
                        "exact_match": em,
                        "f1": f1,
                        "ttft_s": result["ttft_s"],
                        "elapsed_s": result["elapsed_s"],
                        "cached_tokens": cached,
                        "prompt_tokens": prompt_tokens,
                        "cache_ratio": (
                            cached / prompt_tokens if cached is not None and prompt_tokens else None
                        ),
                        "meta_info": meta,
                    }
                )
                print(
                    f"[{index}/{len(items)}] {item['id']} "
                    f"F1={f1:.3f} TTFT={result['ttft_s']:.3f}s "
                    f"cache={cached}/{prompt_tokens}",
                    flush=True,
                )
            except Exception as exc:  # Keep partial results for long benchmark runs.
                record["error"] = f"{type(exc).__name__}: {exc}"
                print(f"[{index}/{len(items)}] {item['id']} ERROR {record['error']}", file=sys.stderr)
                if args.fail_fast:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fh.flush()
                    raise
            records.append(record)
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()

    summary = summarize(records)
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Raw results: {output}")
    print(f"Summary:     {summary_path}")


def cmd_summarize(args: argparse.Namespace) -> None:
    all_summaries = []
    for name in args.inputs:
        records = load_jsonl(Path(name))
        summary = summarize(records)
        summary["file"] = name
        all_summaries.append(summary)
    print(json.dumps(all_summaries, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare", help="Create one deterministic local evaluation set")
    prep.add_argument("--dataset", choices=["hotpotqa", "triviaqa"], required=True)
    prep.add_argument("--split", default="validation")
    prep.add_argument("--limit", type=int, default=200)
    prep.add_argument("--offset", type=int, default=0)
    prep.add_argument("--max-segments", type=int, default=10)
    prep.add_argument("--max-segment-chars", type=int, default=8000)
    prep.add_argument("--output", required=True)
    prep.set_defaults(func=cmd_prepare)

    run = sub.add_parser("run", help="Evaluate one already-running SGLang mode")
    run.add_argument("--input", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--mode", choices=["full_recompute", "prefix_cache", "addition", "hypic"], required=True)
    run.add_argument("--base-url", default="http://127.0.0.1:30000")
    run.add_argument("--separator", default=SEP)
    run.add_argument("--max-new-tokens", type=int, default=64)
    run.add_argument("--limit", type=int, default=0, help="0 means all prepared rows")
    run.add_argument("--timeout", type=float, default=600)
    run.add_argument("--fail-fast", action="store_true")
    run.set_defaults(func=cmd_run)

    summary = sub.add_parser("summarize", help="Print comparable summaries for result JSONL files")
    summary.add_argument("inputs", nargs="+")
    summary.set_defaults(func=cmd_summarize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
