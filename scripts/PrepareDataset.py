"""Prepare every dataset used by KVBench.

The repository intentionally does not contain ``data/``.  This command reads
the ``DatasetPreparation`` section of ``config.yaml`` and builds the declared
dataset tree in a temporary directory.  The directory is installed only after
all downloads, conversions, and basic schema checks succeed.

Run from the repository root with::

    python scripts/PrepareDataset.py

The command has no dataset-size or path overrides on purpose.  Changing the
source split, RULER lengths, or sample count is a configuration change, so the
generated data cannot silently drift from ``config.yaml``.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import html
import io
import json
import random
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.request import Request, urlopen

import yaml


ProjectRoot = Path(__file__).resolve().parent.parent
DefaultConfigPath = ProjectRoot / "config.yaml"


# ---------------------------------------------------------------------------
# General preparation helpers
# ---------------------------------------------------------------------------


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _relative_path(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must stay below DatasetPath: {value!r}")
    return path


def _download(url: str, timeout: int = 180) -> bytes:
    request = Request(url, headers={"User-Agent": "KVBench/PrepareDataset"})
    try:
        with urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", 200)
            if status != 200:
                raise RuntimeError(f"HTTP {status} while downloading {url}")
            body = response.read()
    except Exception as exc:  # noqa: BLE001 - add the source URL to the error
        raise RuntimeError(f"could not download {url}: {exc}") from exc
    if not body:
        raise RuntimeError(f"empty download: {url}")
    return body


def _sha256(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _write_bytes(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(body)
    temporary.replace(path)


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            if not isinstance(record, Mapping):
                raise ValueError(f"non-object record while writing {path}")
            stream.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
            count += 1
    if count == 0:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"source produced no records: {path}")
    temporary.replace(path)
    return count


def _record_length(*parts: Any) -> int:
    """A stable length metadata value; task execution does not depend on it."""
    return len(" ".join(str(part or "") for part in parts).split())


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _row_dict(row: Any) -> Dict[str, Any]:
    if isinstance(row, Mapping):
        return dict(row)
    try:
        return dict(row)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Hugging Face row is not mapping-like: {type(row)!r}") from exc


def _dataset_root(config: Mapping[str, Any], config_path: Path) -> Path:
    value = config.get("DatasetPath", "data")
    if not isinstance(value, str) or not value:
        raise ValueError("DatasetPath must be a non-empty string")
    root = Path(value)
    return (config_path.parent / root).resolve() if not root.is_absolute() else root


def _load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    return dict(_mapping(config, "config.yaml"))


def _top_level_directory(path: Path) -> str:
    if not path.parts:
        raise ValueError("an output path may not be empty")
    return path.parts[0]


def _prepare_direct(entry: Mapping[str, Any], stage: Path) -> None:
    output = _relative_path(entry.get("Output"), "Direct.Output")
    fmt = entry.get("Format", "bytes")
    if fmt == "govreport_jsonl":
        urls = entry.get("Urls")
        if not isinstance(urls, list) or not urls:
            raise ValueError("Direct.Urls must be non-empty for govreport_jsonl")
        raw_files = [_download(str(url)).decode("utf-8") for url in urls]
        count = _write_jsonl(stage / output, _adapt_govreport_raw(raw_files))
        print(f"[dataset] materialized {count} govreport rows -> {output}")
        return
    url = entry.get("Url")
    if not isinstance(url, str) or not url:
        raise ValueError("Direct.Url must be a non-empty URL")
    body = _download(url)
    expected = entry.get("Sha256")
    if expected is not None and _sha256(body).lower() != str(expected).lower():
        raise ValueError(
            f"checksum mismatch for {url}: expected {expected}, got {_sha256(body)}"
        )

    if fmt == "json_list":
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{url} is not valid JSON") from exc
        if not isinstance(payload, list) or not payload:
            raise ValueError(f"{url} must contain a non-empty JSON list")
    elif fmt == "jsonl":
        lines = [line for line in body.decode("utf-8").splitlines() if line.strip()]
        if not lines:
            raise ValueError(f"{url} contains no JSONL records")
        for line in lines:
            json.loads(line)
    elif fmt != "bytes":
        raise ValueError(f"unsupported Direct.Format: {fmt!r}")
    _write_bytes(stage / output, body)
    print(f"[dataset] downloaded {url} -> {output}")


# ---------------------------------------------------------------------------
# Hugging Face adapters
# ---------------------------------------------------------------------------


def _load_huggingface(entry: Mapping[str, Any]):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "PrepareDataset.py needs the `datasets` package for Hugging Face "
            "sources; install requirement.txt first"
        ) from exc

    dataset = entry.get("Dataset")
    split = entry.get("Split")
    config = entry.get("Config")
    revision = entry.get("Revision")
    if not isinstance(dataset, str) or not dataset:
        raise ValueError("HuggingFace.Dataset must be a non-empty string")
    if not isinstance(split, str) or not split:
        raise ValueError("HuggingFace.Split must be a non-empty string")
    kwargs: Dict[str, Any] = {"split": split}
    if revision:
        kwargs["revision"] = str(revision)
    # All configured sources are parquet-backed snapshots.  Keeping this false
    # prevents an upstream Python dataset script from changing what is run.
    kwargs["trust_remote_code"] = False
    try:
        if config is None:
            return load_dataset(dataset, **kwargs)
        return load_dataset(dataset, str(config), **kwargs)
    except Exception as exc:  # noqa: BLE001
        source = f"{dataset}/{config}" if config else str(dataset)
        raise RuntimeError(f"could not load {source} split {split}: {exc}") from exc


def _hotpot_context(value: Any) -> str:
    """Render both the original list form and HF's dict-of-lists form."""
    passages: List[Tuple[str, str]] = []
    if isinstance(value, Mapping):
        titles = list(value.get("title", []))
        contents = list(value.get("content", value.get("sentences", [])))
        for title, content in zip(titles, contents):
            if isinstance(content, (list, tuple)):
                text = " ".join(_text(sentence) for sentence in content)
            else:
                text = _text(content)
            passages.append((_text(title), text))
    elif isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, Mapping):
                title = item.get("title", "")
                content = item.get("content", item.get("sentences", ""))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                title, content = item[0], item[1]
            else:
                continue
            if isinstance(content, (list, tuple)):
                content = " ".join(_text(sentence) for sentence in content)
            passages.append((_text(title), _text(content)))
    return "\n\n".join(
        f"Passage {index}:\n{title}\n{content}"
        for index, (title, content) in enumerate(passages, 1)
    )


def _answers_from_trivia(row: Mapping[str, Any]) -> List[str]:
    answer = row.get("answer", {})
    values: List[str] = []
    if isinstance(answer, Mapping):
        candidates = [answer.get("value"), *(answer.get("aliases") or [])]
    else:
        candidates = [answer]
    for candidate in candidates:
        value = _text(candidate).strip()
        if value and value not in values:
            values.append(value)
    if not values:
        raise ValueError("TriviaQA row has no answer")
    return values


def _trivia_context(row: Mapping[str, Any]) -> str:
    sections: List[str] = []
    pages = row.get("entity_pages") or []
    if isinstance(pages, Mapping):
        # Accommodate a column-oriented representation from older builders.
        pages = [
            {key: values[index] for key, values in pages.items()}
            for index in range(len(next(iter(pages.values()), [])))
        ]
    for page in pages:
        if not isinstance(page, Mapping):
            continue
        title = _text(page.get("title")).strip()
        text = _text(page.get("wiki_context")).strip()
        if text:
            sections.append(f"Passage {len(sections) + 1}:\n{title}\n{text}")
    if not sections:
        for page in row.get("search_results") or []:
            if not isinstance(page, Mapping):
                continue
            title = _text(page.get("title")).strip()
            text = _text(page.get("search_context")).strip()
            if text:
                sections.append(f"Passage {len(sections) + 1}:\n{title}\n{text}")
    if not sections:
        raise ValueError(f"TriviaQA row has no evidence: {row.get('question')!r}")
    return "\n\n".join(sections)


def _adapt_hotpot(row: Mapping[str, Any]) -> Dict[str, Any]:
    question = _text(row.get("question")).strip()
    context = _hotpot_context(row.get("context"))
    answer = _text(row.get("answer")).strip()
    if not question or not context or not answer:
        raise ValueError("HotpotQA row is missing question, context, or answer")
    return {
        "input": question,
        "context": context,
        "answers": [answer],
        "length": _record_length(question, context),
        "dataset": "hotpotqa",
        "language": "en",
        "all_classes": None,
        "_id": _text(row.get("id", row.get("_id", ""))),
    }


def _adapt_govreport(row: Mapping[str, Any]) -> Dict[str, Any]:
    document, summary = _text(row.get("document")), _text(row.get("summary"))
    if not document.strip() or not summary.strip():
        raise ValueError("GovReport row is missing document or summary")
    return {
        "input": "",
        "context": document,
        "answers": [summary],
        "length": _record_length(document),
        "dataset": "gov_report",
        "language": "en",
        "all_classes": None,
        "_id": _text(row.get("id", "")),
    }


def _govreport_sections(
    section: Mapping[str, Any], *, keep_letter: bool, depth: int = 0
) -> List[Dict[str, Any]]:
    title = _text(section.get("section_title")).strip()
    paragraphs = section.get("paragraphs") or []
    current: List[Dict[str, Any]] = []
    if title != "Letter" or keep_letter:
        current.append({
            "title": " ".join(title.split()),
            "paragraphs": "\n".join(" ".join(_text(p).strip().split()) for p in paragraphs),
            "depth": depth,
        })
        for child in section.get("subsections") or []:
            current.extend(_govreport_sections(child, keep_letter=keep_letter, depth=depth + 1))
    else:
        for child in section.get("subsections") or []:
            current.extend(_govreport_sections(child, keep_letter=keep_letter, depth=depth))
    return current


def _adapt_govreport_raw(raw_files: Sequence[str]) -> Iterable[Dict[str, Any]]:
    """Convert the publisher's GAO/CRS JSONL test files to KVBench rows.

    This is the same plain-text projection as the pinned ``gov_report`` data
    builder, kept here so preparation does not execute an unpinned remote
    Python dataset script.
    """
    for raw_text in raw_files:
        for line_number, line in enumerate(raw_text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid GovReport JSONL at line {line_number}") from exc
            if "report" in raw:
                sections: List[Dict[str, Any]] = []
                for section in raw["report"]:
                    sections.extend(_govreport_sections(section, keep_letter=False, depth=1))
                highlights = [
                    {
                        "title": " ".join(_text(section.get("section_title")).strip().split()),
                        "paragraphs": " ".join(
                            " ".join(_text(paragraph).strip().split())
                            for paragraph in section.get("paragraphs") or []
                        ),
                    }
                    for section in raw.get("highlight") or []
                ]
                document = " ".join(
                    f"{section['title']} {section['paragraphs']}".strip()
                    for section in sections
                ).replace("\n", " ").strip()
                summary = " ".join(
                    section["paragraphs"]
                    for section in highlights
                    if section["title"] != "What GAO Recommends"
                ).replace("\n", " ").strip()
                identifier = "GAO_" + _text(raw.get("id"))
            elif "reports" in raw:
                sections = _govreport_sections(raw["reports"], keep_letter=True, depth=0)
                document = " ".join(
                    f"{section['title']} {section['paragraphs']}".strip()
                    for section in sections
                ).replace("\n", " ").strip()
                summary = " ".join(
                    " ".join(_text(paragraph).strip().split())
                    for paragraph in raw.get("summary") or []
                ).replace("\n", " ").strip()
                identifier = "CRS_" + _text(raw.get("id"))
            else:
                raise ValueError("GovReport row has neither report nor reports")
            yield _adapt_govreport({"document": document, "summary": summary, "id": identifier})


def _adapt_multinews(row: Mapping[str, Any]) -> Dict[str, Any]:
    document, summary = _text(row.get("document")), _text(row.get("summary"))
    documents = [piece.strip() for piece in document.split("|||||") if piece.strip()]
    if not documents or not summary.strip():
        raise ValueError("MultiNews row is missing document or summary")

    def normalize_document(text: str) -> str:
        return text.replace("NEWLINE_CHAR", "\n")

    context = "\n\n".join(
        f"Passage {index}:\n{normalize_document(text)}"
        for index, text in enumerate(documents, 1)
    )
    return {
        "input": "",
        "context": context,
        "answers": [summary],
        "length": _record_length(context),
        "dataset": "multi_news",
        "language": "en",
        "all_classes": None,
        "_id": _text(row.get("id", "")),
    }


def _adapt_triviaqa(row: Mapping[str, Any]) -> Dict[str, Any]:
    question = _text(row.get("question")).strip()
    context = _trivia_context(row)
    if not question:
        raise ValueError("TriviaQA row is missing question")
    return {
        "input": question,
        "context": context,
        "answers": _answers_from_trivia(row),
        "length": _record_length(question, context),
        "dataset": "triviaqa",
        "language": "en",
        "all_classes": None,
        "_id": _text(row.get("question_id", row.get("id", ""))),
    }


def _adapt_gsm8k(row: Mapping[str, Any]) -> Dict[str, Any]:
    question, answer = _text(row.get("question")), _text(row.get("answer"))
    if not question.strip() or not answer.strip():
        raise ValueError("GSM8K row is missing question or answer")
    return {"question": question, "answer": answer}


def _adapt_humaneval(row: Mapping[str, Any]) -> Dict[str, Any]:
    required = ("prompt", "test", "entry_point")
    if any(not _text(row.get(key)).strip() for key in required):
        raise ValueError("HumanEval row is missing prompt, test, or entry_point")
    # Keep all upstream fields.  KVBench only consumes the three fields above,
    # but retaining the task id and canonical solution makes the prepared data
    # useful for independent validation too.
    return row


def _mmlu_answer(value: Any) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        if 0 <= value <= 3:
            return "ABCD"[value]
    text = _text(value).strip().upper()
    if text in {"A", "B", "C", "D"}:
        return text
    raise ValueError(f"invalid MMLU answer: {value!r}")


def _prepare_huggingface(entry: Mapping[str, Any], stage: Path) -> None:
    adapter = entry.get("Adapter")
    output = _relative_path(entry.get("Output"), "HuggingFace.Output")
    dataset = _load_huggingface(entry)
    rows = (_row_dict(row) for row in dataset)

    if adapter == "mmlu":
        output_dir = stage / output
        output_dir.mkdir(parents=True, exist_ok=True)
        writers: Dict[str, Tuple[io.TextIOWrapper, csv.writer]] = {}
        count = 0
        try:
            for row in rows:
                subject = _text(row.get("subject")).strip().replace(" ", "_")
                choices = list(row.get("choices") or [])
                if not subject or len(choices) != 4:
                    raise ValueError("MMLU row must contain subject and four choices")
                if subject not in writers:
                    stream = (output_dir / f"{subject}_val.csv").open(
                        "w", encoding="utf-8", newline=""
                    )
                    writers[subject] = (stream, csv.writer(stream, lineterminator="\n"))
                writers[subject][1].writerow(
                    [_text(row.get("question")), *[_text(choice) for choice in choices], _mmlu_answer(row.get("answer"))]
                )
                count += 1
        finally:
            for stream, _ in writers.values():
                stream.close()
        if count == 0:
            raise ValueError("MMLU source produced no rows")
        print(f"[dataset] materialized {count} MMLU rows -> {output}")
        return

    adapters: Dict[str, Callable[[Mapping[str, Any]], Dict[str, Any]]] = {
        "hotpotqa": _adapt_hotpot,
        "govreport": _adapt_govreport,
        "multinews": _adapt_multinews,
        "triviaqa": _adapt_triviaqa,
        "gsm8k": _adapt_gsm8k,
        "humaneval": _adapt_humaneval,
    }
    if adapter not in adapters:
        raise ValueError(f"unsupported HuggingFace.Adapter: {adapter!r}")
    count = _write_jsonl(stage / output, (adapters[adapter](row) for row in rows))
    print(f"[dataset] materialized {count} {adapter} rows -> {output}")


# ---------------------------------------------------------------------------
# RULER generator
# ---------------------------------------------------------------------------


_NiahTemplate = (
    "[INST] A special magic number is hidden within the following text. "
    "Make sure to memorize it. I will quiz you about the number afterwards.\n"
    "{context}\n"
    "What is the special magic number for {query} mentioned in the provided text? [/INST]"
)
_NiahAnswerPrefix = " The special magic number for {query} mentioned in the provided text is"
_NiahNeedle = "One of the special magic numbers for {key} is: {value}."
_NiahPrefixMarker = " The speci"
_VtTemplate = (
    "[INST] Memorize and track the chain(s) of variable assignment hidden in "
    "the following text.\n\n{context}\nQuestion: Find all variables that are "
    "assigned the value {query} in the text above. [/INST]"
)
_VtAnswerPrefix = (
    " Answer: According to the chain(s) of variable assignment in the text "
    "above, {num_v} variables are assigned the value {query}, they are: "
)
_VtPrefixMarker = " Answer: "
_CweTemplate = (
    "[INST] Below is a numbered list of words. In these words, some appear "
    "more often than others. Memorize the ones that appear most often.\n"
    "{context}\nQuestion: What are the 10 most common words in the above list? [/INST]"
)
_CweAnswerPrefix = " Answer: The top 10 words that appear most often in the list are:"
_CwePrefixMarker = " Answer: "
_NoiseSentences = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."


def _num_tokens(tokenizer: Any, text: str) -> int:
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def _split_answer_prefix(full_prompt: str, marker: str) -> Tuple[str, str]:
    index = full_prompt.rfind(marker)
    if index < 0:
        raise RuntimeError(f"RULER answer-prefix marker not found: {marker!r}")
    return full_prompt[:index], full_prompt[index:]


def _sentences(text: str) -> List[str]:
    return [part for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part]


def _largest_fit(
    build: Callable[[int], Tuple[str, Any]],
    tokenizer: Any,
    maximum: int,
    tokens_to_generate: int,
    incremental: int,
) -> int:
    initial, _ = build(incremental)
    tokens_per_unit = max(_num_tokens(tokenizer, initial) / incremental, 1.0)
    lower, upper = incremental, max(int(maximum / tokens_per_unit * 3), incremental * 2)
    best: Optional[int] = None
    while lower <= upper:
        middle = (lower + upper) // 2
        full, _ = build(middle)
        if _num_tokens(tokenizer, full) + tokens_to_generate <= maximum:
            best = middle
            lower = middle + 1
        else:
            upper = middle - 1
    return best if best is not None else incremental


def _load_ruler_words(url: str) -> List[str]:
    body = _download(url).decode("utf-8")
    try:
        payload = json.loads(body)
        values = payload.values() if isinstance(payload, Mapping) else payload
        words = [_text(word).strip().lower() for word in values]
    except json.JSONDecodeError:
        words = [line.strip().lower() for line in body.splitlines()]
    words = sorted({word for word in words if re.fullmatch(r"[a-z][a-z-]{2,30}", word)})
    if len(words) < 20:
        raise ValueError(f"RULER word source has too few usable words: {url}")
    return words


def _load_ruler_corpus(urls_url: str) -> str:
    urls = [line.strip() for line in _download(urls_url).decode("utf-8").splitlines() if line.strip()]
    chunks: List[str] = []
    for url in urls:
        raw = _download(url).decode("utf-8", "replace")
        if ".html" in url:
            body = re.search(r"<font[^>]*>(.*?)</font>", raw, re.IGNORECASE | re.DOTALL)
            text = body.group(1) if body else raw
            text = html.unescape(re.sub(r"<[^>]+>", " ", text))
        else:
            text = raw
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > 2000:
            chunks.append(text)
    if not chunks:
        raise RuntimeError("RULER corpus download produced no usable essays")
    return "\n".join(chunks)


def _write_ruler_records(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    _write_jsonl(path, records)


def _generate_niah(tokenizer: Any, corpus: str, words: Sequence[str], spec: Mapping[str, Any], path: Path) -> None:
    samples = _positive_int(spec.get("Samples"), "Ruler.Tasks[].Samples")
    maximum = _positive_int(spec.get("Length"), "Ruler task length")
    budget = _positive_int(spec.get("TokensToGenerate"), "Ruler.Tasks[].TokensToGenerate")
    seed = int(spec.get("Seed", 42))
    random.seed(seed)
    haystack = re.sub(r"\s+", " ", corpus).strip().split(" ")

    def build(count: int) -> Tuple[str, Dict[str, Any]]:
        text = " ".join((haystack * ((count + len(haystack) - 1) // len(haystack)))[:count])
        sentence_list = _sentences(text)
        depth = random.randint(0, 100)
        key = random.choice(words)
        value = str(random.randint(10**6, 10**7 - 1))
        needle = _NiahNeedle.format(key=key, value=value)
        position = int(len(sentence_list) * depth / 100)
        context = " ".join(sentence_list[:position] + [needle] + sentence_list[position:])
        full = _NiahTemplate.format(context=context, query=key) + _NiahAnswerPrefix.format(query=key)
        return full, {"needle": needle, "value": value, "depth": depth}

    count = _largest_fit(build, tokenizer, maximum, budget, 500)
    records: List[Dict[str, Any]] = []
    for index in range(samples):
        full, meta = build(count)
        input_text, answer_prefix = _split_answer_prefix(full, _NiahPrefixMarker)
        answer_offset = input_text.find(meta["value"])
        records.append({
            "index": answer_offset,
            "input": input_text,
            "outputs": [meta["value"]],
            "length": _num_tokens(tokenizer, full) + budget,
            "length_w_model_temp": _num_tokens(tokenizer, full) + budget,
            "answer_prefix": answer_prefix,
            "token_position_answer": _num_tokens(tokenizer, input_text[:answer_offset]),
            "needle": meta["needle"],
            "depth_percent": meta["depth"],
        })
    _write_ruler_records(path, records)


def _generate_vt(tokenizer: Any, spec: Mapping[str, Any], path: Path) -> None:
    samples = _positive_int(spec.get("Samples"), "Ruler.Tasks[].Samples")
    maximum = _positive_int(spec.get("Length"), "Ruler task length")
    budget = _positive_int(spec.get("TokensToGenerate"), "Ruler.Tasks[].TokensToGenerate")
    seed = int(spec.get("Seed", 42))
    random.seed(seed)

    def build(count: int) -> Tuple[str, Dict[str, Any]]:
        variables = ["".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=5)) for _ in range(5)]
        while len(set(variables)) != len(variables):
            variables = ["".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=5)) for _ in range(5)]
        value = str(random.randint(10000, 99999))
        chain = [f"VAR {variables[0]} = {value}"] + [
            f"VAR {variables[index]} = VAR {variables[index - 1]} " for index in range(1, 5)
        ]
        sentences = [_NoiseSentences] * count
        positions = sorted(random.sample(range(len(sentences)), len(chain)))
        for offset, sentence in enumerate(chain):
            sentences.insert(positions[offset] + offset, sentence)
        context = "\n".join(sentences).replace(". \n", ".\n")
        full = _VtTemplate.format(context=context, query=value) + _VtAnswerPrefix.format(query=value, num_v=5)
        return full, {"variables": variables}

    count = _largest_fit(build, tokenizer, maximum, budget, 5)
    records: List[Dict[str, Any]] = []
    for index in range(samples):
        full, meta = build(count)
        input_text, answer_prefix = _split_answer_prefix(full, _VtPrefixMarker)
        length = _num_tokens(tokenizer, full) + budget
        records.append({
            "index": index,
            "input": input_text,
            "outputs": meta["variables"],
            "length": length,
            "length_w_model_temp": length,
            "answer_prefix": answer_prefix,
        })
    _write_ruler_records(path, records)


def _generate_cwe(tokenizer: Any, words: Sequence[str], spec: Mapping[str, Any], path: Path) -> None:
    samples = _positive_int(spec.get("Samples"), "Ruler.Tasks[].Samples")
    maximum = _positive_int(spec.get("Length"), "Ruler task length")
    budget = _positive_int(spec.get("TokensToGenerate"), "Ruler.Tasks[].TokensToGenerate")
    seed = int(spec.get("Seed", 42))
    random.seed(seed)

    def build(count: int) -> Tuple[str, Dict[str, Any]]:
        if count > len(words):
            raise ValueError("RULER CWE needs more configured words than the source provides")
        selected = random.sample(list(words), count)
        common, uncommon = selected[:10], selected[10:]
        items = list(common) * 30 + list(uncommon) * 3
        random.shuffle(items)
        context = " ".join(f"{index}. {word}" for index, word in enumerate(items, 1))
        return _CweTemplate.format(context=context) + _CweAnswerPrefix, {"common": common}

    count = _largest_fit(build, tokenizer, maximum, budget, 50)
    records: List[Dict[str, Any]] = []
    for index in range(samples):
        full, meta = build(count)
        input_text, answer_prefix = _split_answer_prefix(full, _CwePrefixMarker)
        length = _num_tokens(tokenizer, full) + budget
        records.append({
            "index": index,
            "input": input_text,
            "outputs": meta["common"],
            "length": length,
            "length_w_model_temp": length,
            "answer_prefix": answer_prefix,
        })
    _write_ruler_records(path, records)


def _prepare_ruler(config: Mapping[str, Any], stage: Path, full_config: Mapping[str, Any]) -> None:
    tasks = config.get("Tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Ruler.Tasks must be a non-empty list")
    tokenizer_name = config.get("Tokenizer")
    if tokenizer_name == "ModelPath":
        tokenizer_name = full_config.get("ModelPath")
    if not isinstance(tokenizer_name, str) or not tokenizer_name:
        raise ValueError("Ruler.Tokenizer must be a path or ModelPath")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("RULER preparation needs transformers") from exc
    print(f"[dataset] loading RULER tokenizer {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    words = _load_ruler_words(str(config.get("WordList")))
    corpus = _load_ruler_corpus(str(config.get("CorpusUrls")))

    for raw_spec in tasks:
        task = _mapping(raw_spec, "Ruler.Tasks[]")
        name = task.get("Task")
        lengths = task.get("Lengths")
        if name not in {"niah", "vt", "cwe"} or not isinstance(lengths, list) or not lengths:
            raise ValueError("Ruler task must declare Task=niah/vt/cwe and non-empty Lengths")
        for length_value in lengths:
            length = _positive_int(length_value, "Ruler.Tasks[].Lengths[]")
            spec = dict(task)
            spec["Length"] = length
            output = stage / "ruler" / f"{name}_len{length}.jsonl"
            if name == "niah":
                _generate_niah(tokenizer, corpus, words, spec, output)
            elif name == "vt":
                _generate_vt(tokenizer, spec, output)
            else:
                _generate_cwe(tokenizer, words, spec, output)
            print(f"[dataset] generated RULER {output.relative_to(stage)}")


# ---------------------------------------------------------------------------
# Orchestration and CLI
# ---------------------------------------------------------------------------


def _validate_preparation_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    preparation = _mapping(config.get("DatasetPreparation"), "DatasetPreparation")
    directories = preparation.get("DatasetDirectories")
    if not isinstance(directories, list) or not directories:
        raise ValueError("DatasetPreparation.DatasetDirectories must be non-empty")
    for directory in directories:
        _relative_path(directory, "DatasetDirectories[]")
        if "/" in str(directory) or "\\" in str(directory):
            raise ValueError("DatasetDirectories[] must contain directory names")
    if not isinstance(preparation.get("Direct", []), list):
        raise ValueError("DatasetPreparation.Direct must be a list")
    if not isinstance(preparation.get("HuggingFace", []), list):
        raise ValueError("DatasetPreparation.HuggingFace must be a list")
    _mapping(preparation.get("Ruler"), "DatasetPreparation.Ruler")
    return preparation


def _planned_outputs(preparation: Mapping[str, Any]) -> List[str]:
    outputs = list(preparation.get("DatasetDirectories", []))
    for group_name in ("Direct", "HuggingFace"):
        for raw in preparation.get(group_name, []):
            entry = _mapping(raw, f"DatasetPreparation.{group_name}[]")
            output = _relative_path(entry.get("Output"), f"{group_name}.Output")
            top = _top_level_directory(output)
            if top not in outputs:
                raise ValueError(f"{top!r} is not listed in DatasetDirectories")
    return [str(item) for item in outputs]


def prepare(config_path: Path, dry_run: bool = False) -> Path:
    config = _load_config(config_path)
    preparation = _validate_preparation_config(config)
    root = _dataset_root(config, config_path)
    directories = _planned_outputs(preparation)

    if dry_run:
        print(f"[dataset] DatasetPath: {root}")
        for entry in preparation.get("Direct", []):
            source = entry.get("Url", entry.get("Urls"))
            print(f"[dataset] direct: {source} -> {entry.get('Output')}")
        for entry in preparation.get("HuggingFace", []):
            print(
                f"[dataset] hf: {entry.get('Dataset')}"
                f"[{entry.get('Config', '')}]/{entry.get('Split')} -> {entry.get('Output')}"
            )
        for directory in directories:
            print(f"[dataset] replaces: {directory}/")
        print("[dataset] RULER generation is configured")
        return root

    root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{root.name}.prepare-", dir=root.parent))
    try:
        for raw in preparation.get("Direct", []):
            _prepare_direct(_mapping(raw, "DatasetPreparation.Direct[]"), stage)
        for raw in preparation.get("HuggingFace", []):
            _prepare_huggingface(_mapping(raw, "DatasetPreparation.HuggingFace[]"), stage)
        _prepare_ruler(_mapping(preparation.get("Ruler"), "DatasetPreparation.Ruler"), stage, config)

        for directory in directories:
            staged = stage / directory
            if not staged.is_dir():
                raise RuntimeError(f"configured dataset directory was not generated: {directory}")
        root.mkdir(parents=True, exist_ok=True)
        for directory in directories:
            destination = root / directory
            if destination.exists():
                shutil.rmtree(destination)
            shutil.move(str(stage / directory), str(destination))
            print(f"[dataset] installed {destination}")
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DefaultConfigPath,
        help="configuration file; all dataset choices come from this file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the configured plan without downloading or writing",
    )
    args = parser.parse_args()
    root = prepare(args.config.resolve(), dry_run=args.dry_run)
    print(f"[dataset] ready: {root}")


if __name__ == "__main__":
    main()
