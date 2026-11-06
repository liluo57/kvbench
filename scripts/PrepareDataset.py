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
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import io
import json
import os
import re
import runpy
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.request import Request, urlopen

import yaml


ProjectRoot = Path(__file__).resolve().parent.parent
DefaultConfigPath = ProjectRoot / "config.yaml"
if __package__ in {None, ""}:
    sys.path.insert(0, str(ProjectRoot))


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_cache(url: str) -> Path:
    return ProjectRoot / ".cache/dataset_sources" / _sha256(url.encode())


def _google_drive_url(identifier: str) -> str:
    return f"https://drive.usercontent.google.com/download?id={identifier}&export=download&confirm=t"


def _download_file(url: str, timeout: int = 180) -> Path:
    cache = _source_cache(url)
    payload, metadata = cache / "payload", cache / "source.json"
    if payload.is_file() and metadata.is_file():
        saved = json.loads(metadata.read_text())
        if saved["url"] != url or _file_sha256(payload) != saved["sha256"]:
            raise ValueError(f"cached source was modified: {payload}")
        return payload
    cache.mkdir(parents=True, exist_ok=True)
    partial = cache / "payload.part"
    error = None
    for attempt in range(4):
        try:
            if attempt == 0 and not partial.exists():
                with urlopen(Request(url, headers={"User-Agent": "KVBench/PrepareDataset"}), timeout=timeout) as response:
                    with partial.open("wb") as stream:
                        shutil.copyfileobj(response, stream, 1024 * 1024)
            else:
                subprocess.run(
                    ["curl", "--fail", "--location", "--silent", "--show-error",
                     "--continue-at", "-", "--connect-timeout", "15", "--max-time", str(timeout),
                     url, "--output", str(partial)], check=True,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout + 10,
                )
            if not partial.is_file() or not partial.stat().st_size:
                raise ValueError("source download was empty")
            partial.replace(payload)
            _write_json(metadata, {"url": url, "sha256": _file_sha256(payload), "bytes": payload.stat().st_size})
            return payload
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            error = str(exc)
            if isinstance(exc, subprocess.CalledProcessError):
                error += ": " + exc.stderr.decode("utf-8", "replace")
                if "byte ranges" in error.lower():
                    partial.unlink(missing_ok=True)
            if attempt < 3:
                time.sleep(attempt + 1)
    raise RuntimeError(f"could not download {url}: {error}")


def _download(url: str, timeout: int = 180) -> bytes:
    return _download_file(url, timeout).read_bytes()


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
    root = Path(value).expanduser()
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


def _adapt_2wikimultihopqa(payload: Any) -> List[Dict[str, Any]]:
    """Add the KBBase fields to the official, complete 2Wiki dev split."""
    if not isinstance(payload, list) or not payload:
        raise ValueError("official 2Wiki development split must be a non-empty list")

    records: List[Dict[str, Any]] = []
    for index, raw in enumerate(payload):
        if not isinstance(raw, Mapping):
            raise ValueError(f"2Wiki sample {index} is not an object")
        question = _text(raw.get("question")).strip()
        answer = _text(raw.get("answer")).strip()
        context = raw.get("context")
        if not question or not answer:
            raise ValueError(f"2Wiki sample {index} is missing question or answer")
        if not isinstance(context, (list, tuple)) or not context:
            raise ValueError(f"2Wiki sample {index} has no context passages")

        ctxs: List[Dict[str, str]] = []
        for passage_index, passage in enumerate(context):
            if not isinstance(passage, (list, tuple)) or len(passage) < 2:
                raise ValueError(
                    f"2Wiki sample {index} context passage {passage_index} "
                    "must be [title, sentences]"
                )
            title, sentences = passage[0], passage[1]
            if isinstance(sentences, (list, tuple)):
                text = " ".join(_text(sentence) for sentence in sentences)
            else:
                text = _text(sentences)
            ctxs.append({"title": _text(title), "text": text})

        record = dict(raw)
        record["ctxs"] = ctxs
        # Match the nested answer shape used by CacheBlend's evaluator.
        record["answers"] = [[answer]]
        records.append(record)
    return records


def _adapt_musique_ans(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Adapt one official answerable MuSiQue row to the KBBase schema."""
    question = _text(row.get("question")).strip()
    answer = _text(row.get("answer")).strip()
    if not question or not answer:
        raise ValueError(
            "MuSiQue-Ans row is missing question or answer; the official test "
            "has private labels, so local F1/EM requires the labeled dev split"
        )
    if row.get("answerable") is False:
        raise ValueError("MuSiQue-Ans unexpectedly contains an unanswerable row")

    paragraphs = row.get("paragraphs")
    if not isinstance(paragraphs, list) or not paragraphs:
        raise ValueError("MuSiQue-Ans row has no paragraphs")
    ctxs: List[Dict[str, str]] = []
    for index, paragraph in enumerate(paragraphs):
        if not isinstance(paragraph, Mapping):
            raise ValueError(f"MuSiQue paragraph {index} is not an object")
        title = _text(paragraph.get("title")).strip()
        text = _text(paragraph.get("paragraph_text") or paragraph.get("text")).strip()
        if not text:
            raise ValueError(f"MuSiQue paragraph {index} has no text")
        ctxs.append({"title": title, "text": text})

    answers: List[str] = []
    aliases = row.get("answer_aliases") or []
    if not isinstance(aliases, (list, tuple)):
        aliases = [aliases]
    for candidate in [answer, *aliases]:
        candidate = _text(candidate).strip()
        if candidate and candidate not in answers:
            answers.append(candidate)
    if not answers:
        raise ValueError("MuSiQue-Ans row has no usable answer aliases")
    return {
        "question": question,
        "ctxs": ctxs,
        "answers": answers,
        "length": _record_length(question, *(passage["text"] for passage in ctxs)),
        "dataset": "musique",
        "language": "en",
        "all_classes": None,
        "_id": _text(row.get("id", "")),
    }


def _adapt_samsum(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Adapt one official SAMSum row to the local dialogue-summary prompt."""
    dialogue = _text(row.get("dialogue")).strip()
    summary = _text(row.get("summary")).strip()
    if not dialogue or not summary:
        raise ValueError("SAMSum row is missing dialogue or summary")
    context = f"Dialogue:\n{dialogue}"
    return {
        "ctxs": [{"title": "", "text": context}],
        "question": "Summary:",
        "answers": [summary],
        "length": _record_length(dialogue),
        "dataset": "samsum",
        "language": "en",
        "all_classes": None,
        "_id": _text(row.get("id", "")),
    }


def _read_json_records(path: Path) -> List[Mapping[str, Any]]:
    """Read either a JSON list or JSONL file containing object records."""
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{path} must contain a non-empty list of records")
    if any(not isinstance(row, Mapping) for row in payload):
        raise ValueError(f"{path} contains a non-object record")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _prepare_direct(entry: Mapping[str, Any], stage: Path) -> None:
    output = _relative_path(entry.get("Output"), "Direct.Output")
    fmt = entry.get("Format", "bytes")
    if fmt == "govreport_jsonl":
        if entry.get("Split") != "test":
            raise ValueError("GovReport's GAO/CRS test adapter requires Split=test")
        urls = entry.get("Urls")
        if not isinstance(urls, list) or not urls:
            raise ValueError("Direct.Urls must be non-empty for govreport_jsonl")
        raw_files = [_download(str(url)).decode("utf-8") for url in urls]
        count = _write_jsonl(stage / output, _adapt_govreport_raw(raw_files))
        expected_rows = entry.get("ExpectedRows")
        if expected_rows is not None:
            expected_rows = _positive_int(expected_rows, "Direct.ExpectedRows")
            if count != expected_rows:
                raise ValueError(
                    f"GovReport test has {count} rows; expected {expected_rows}"
                )
        print(f"[dataset] materialized {count} govreport rows -> {output}")
        return
    if fmt == "musique_ans_google_drive_zip":
        drive_id = entry.get("GoogleDriveId")
        split = entry.get("Split")
        if split not in {"dev", "test"}:
            raise ValueError("MuSiQue Split must be dev or test")
        archive_file = entry.get("ArchiveFile")
        if archive_file != f"data/musique_ans_v1.0_{split}.jsonl":
            raise ValueError("MuSiQue ArchiveFile must match its declared Split")
        if not isinstance(drive_id, str) or not drive_id:
            raise ValueError("Direct.GoogleDriveId must be non-empty for MuSiQue")
        if not isinstance(archive_file, str) or not archive_file:
            raise ValueError("Direct.ArchiveFile must be a non-empty string")
        url = _google_drive_url(drive_id)
        archive_path = _download_file(url, timeout=900)
        try:
            with zipfile.ZipFile(archive_path) as archive:
                raw = archive.read(archive_file)
        except KeyError as exc:
            raise ValueError(f"official MuSiQue archive does not contain {archive_file!r}") from exc
        except zipfile.BadZipFile as exc:
            raise ValueError("official MuSiQue download is not a valid ZIP") from exc
        try:
            rows = [
                json.loads(line)
                for line in raw.decode("utf-8").splitlines()
                if line.strip()
            ]
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"official MuSiQue member {archive_file!r} is not valid JSONL"
            ) from exc
        count = _write_jsonl(
            stage / output,
            (_adapt_musique_ans(_mapping(row, "MuSiQue row")) for row in rows),
        )
        expected_rows = entry.get("ExpectedRows")
        if expected_rows is not None:
            expected_rows = _positive_int(expected_rows, "Direct.ExpectedRows")
            if count != expected_rows:
                raise ValueError(
                    f"MuSiQue-Ans {split} has {count} rows; expected {expected_rows}"
                )
        print(f"[dataset] materialized all {count} official MuSiQue-Ans {split} rows -> {output}")
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

    if fmt == "samsum_test_7z":
        if entry.get("Split") != "test":
            raise ValueError("SAMSum test adapter requires Split=test")
        archive_file = entry.get("ArchiveFile", "test.json")
        if not isinstance(archive_file, str) or not archive_file:
            raise ValueError("Direct.ArchiveFile must be a non-empty string")
        try:
            import py7zr
        except ImportError as exc:
            raise RuntimeError(
                "SAMSum preparation needs py7zr; install requirement.txt first"
            ) from exc
        with tempfile.TemporaryDirectory(prefix="kvbench-samsum-") as temporary:
            temporary_dir = Path(temporary)
            archive_path = temporary_dir / "samsum-corpus.7z"
            archive_path.write_bytes(body)
            extracted_dir = temporary_dir / "extracted"
            extracted_dir.mkdir()
            try:
                with py7zr.SevenZipFile(archive_path, mode="r") as archive:
                    matches = [
                        name
                        for name in archive.getnames()
                        if Path(name).name == archive_file
                    ]
                    if len(matches) != 1:
                        raise ValueError(
                            f"SAMSum archive must contain exactly one {archive_file!r}"
                        )
                    archive_member = Path(matches[0])
                    if archive_member.is_absolute() or ".." in archive_member.parts:
                        raise ValueError("SAMSum archive member path is unsafe")
                    targets = [matches[0]]
                    if archive_member.parent != Path("."):
                        targets.insert(0, str(archive_member.parent))
                    # py7zr 0.20.8 exposes ``extract(path, targets)`` without
                    # a ``recursive`` keyword. The exact target list above
                    # limits extraction to this member and its parent path.
                    archive.extract(path=extracted_dir, targets=targets)
                    raw_path = extracted_dir / archive_member
            except Exception as exc:
                raise ValueError(f"could not extract SAMSum {archive_file!r}") from exc
            rows = _read_json_records(raw_path)
        count = _write_jsonl(
            stage / output,
            (_adapt_samsum(_mapping(row, "SAMSum row")) for row in rows),
        )
        expected_rows = entry.get("ExpectedRows")
        if expected_rows is not None:
            expected_rows = _positive_int(expected_rows, "Direct.ExpectedRows")
            if count != expected_rows:
                raise ValueError(
                    f"SAMSum test has {count} rows; expected {expected_rows}"
                )
        print(f"[dataset] materialized all {count} official SAMSum test rows -> {output}")
        return

    if fmt == "2wikimultihopqa_dev_zip":
        archive_file = entry.get("ArchiveFile", "data_ids/dev.json")
        if entry.get("Split") != "dev" or archive_file != "data_ids/dev.json":
            raise ValueError("2Wiki labeled dev adapter requires Split=dev and data_ids/dev.json")
        if not isinstance(archive_file, str) or not archive_file:
            raise ValueError("Direct.ArchiveFile must be a non-empty string")
        try:
            with zipfile.ZipFile(io.BytesIO(body)) as archive:
                try:
                    raw = archive.read(archive_file)
                except KeyError as exc:
                    raise ValueError(
                        f"{url} does not contain {archive_file!r}"
                    ) from exc
        except zipfile.BadZipFile as exc:
            raise ValueError(f"{url} is not a valid ZIP archive") from exc
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{url} archive member {archive_file!r} is not valid JSON") from exc
        records = _adapt_2wikimultihopqa(payload)
        expected_rows = entry.get("ExpectedRows")
        if expected_rows is not None:
            expected_rows = _positive_int(expected_rows, "Direct.ExpectedRows")
            if len(records) != expected_rows:
                raise ValueError(
                    f"{archive_file} has {len(records)} rows; "
                    f"expected the complete split of {expected_rows}"
                )
        _write_json(stage / output, records)
        print(
            f"[dataset] materialized all {len(records)} official 2Wiki dev rows "
            f"from {archive_file} -> {output}"
        )
        return

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
        if value and value.lower() != "<unk>" and value not in values:
            values.append(value)
    if not values:
        raise ValueError(
            "TriviaQA row has no public answer; local F1/EM requires the labeled validation split"
        )
    return values


def _trivia_context(row: Mapping[str, Any]) -> str:
    sections: List[str] = []
    pages = row.get("entity_pages") or []
    if isinstance(pages, Mapping):
        # Accommodate a column-oriented representation from older builders.
        page_count = max(
            (len(values) for values in pages.values() if isinstance(values, (list, tuple))),
            default=0,
        )
        pages = [
            {
                key: values[index]
                for key, values in pages.items()
                if isinstance(values, (list, tuple)) and index < len(values)
            }
            for index in range(page_count)
        ]
    for page in pages:
        if not isinstance(page, Mapping):
            continue
        title = _text(page.get("title")).strip()
        text = _text(page.get("wiki_context")).strip()
        if text:
            sections.append(f"Passage {len(sections) + 1}:\n{title}\n{text}")
    if not sections:
        search_results = row.get("search_results") or []
        if isinstance(search_results, Mapping):
            # The HF rows endpoint represents nested lists of structs as
            # structs of lists; normalize that shape just like entity_pages.
            result_count = max(
                (
                    len(values)
                    for values in search_results.values()
                    if isinstance(values, (list, tuple))
                ),
                default=0,
            )
            search_results = [
                {
                    key: values[index]
                    for key, values in search_results.items()
                    if isinstance(values, (list, tuple)) and index < len(values)
                }
                for index in range(result_count)
            ]
        for page in search_results:
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
    if not summary.strip():
        raise ValueError("MultiNews row is missing summary")

    def normalize_document(text: str) -> str:
        return text.replace("NEWLINE_CHAR", "\n")

    context = "\n\n".join(
        f"Passage {index}:\n{normalize_document(text)}"
        for index, text in enumerate(documents, 1)
    )
    # Keep source rows with an empty document so the converted split retains
    # its declared row count. KBBase.Cases skips them because they have no
    # usable context; never substitute the reference summary as model input.
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
    source_files = []
    for cached in dataset.cache_files:
        path = Path(cached["filename"])
        revision = entry.get("Revision")
        if revision and revision not in path.parts:
            raise ValueError(f"cannot verify pinned HF revision {revision} from cache path {path}")
        source_files.append({"file": str(path), "sha256": _file_sha256(path), "bytes": path.stat().st_size})
    _write_json(stage / _top_level_directory(output) / "_metadata/huggingface.json", {
        "dataset": entry["Dataset"], "revision": entry.get("Revision"),
        "split": entry["Split"], "fingerprint": dataset._fingerprint, "cache_files": source_files,
    })
    rows = (_row_dict(row) for row in dataset)

    if adapter == "mmlu":
        output_dir = stage / output
        output_dir.mkdir(parents=True, exist_ok=True)
        source_split = _text(entry.get("Split")).strip().lower()
        output_split = (
            "val"
            if source_split in {"validation", "valid", "val", "dev"}
            else source_split
        )
        writers: Dict[str, Tuple[io.TextIOWrapper, csv.writer]] = {}
        count = 0
        try:
            for row in rows:
                subject = _text(row.get("subject")).strip().replace(" ", "_")
                choices = list(row.get("choices") or [])
                if not subject or len(choices) != 4:
                    raise ValueError("MMLU row must contain subject and four choices")
                if subject not in writers:
                    stream = (output_dir / f"{subject}_{output_split}.csv").open(
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
        expected_rows = entry.get("ExpectedRows")
        if expected_rows is not None:
            expected_rows = _positive_int(
                expected_rows, "HuggingFace.ExpectedRows"
            )
            if count != expected_rows:
                raise ValueError(f"MMLU split has {count} rows; expected {expected_rows}")
        print(f"[dataset] materialized {count} MMLU rows -> {output}")
        return

    adapters: Dict[str, Callable[[Mapping[str, Any]], Dict[str, Any]]] = {
        "hotpotqa": _adapt_hotpot,
        "govreport": _adapt_govreport,
        "multinews": _adapt_multinews,
        "samsum": _adapt_samsum,
        "triviaqa": _adapt_triviaqa,
        "gsm8k": _adapt_gsm8k,
        "humaneval": _adapt_humaneval,
    }
    if adapter not in adapters:
        raise ValueError(f"unsupported HuggingFace.Adapter: {adapter!r}")
    count = _write_jsonl(stage / output, (adapters[adapter](row) for row in rows))
    expected_rows = entry.get("ExpectedRows")
    if expected_rows is not None:
        expected_rows = _positive_int(expected_rows, "HuggingFace.ExpectedRows")
        if count != expected_rows:
            raise ValueError(
                f"{adapter} split has {count} rows; expected {expected_rows}"
            )
    print(f"[dataset] materialized {count} {adapter} rows -> {output}")


# ---------------------------------------------------------------------------
# RULER generator
# ---------------------------------------------------------------------------


def _run_ruler(command: List[str], cwd: Path, env: Mapping[str, str]) -> str:
    """Run unmodified, pinned upstream scripts with this interpreter."""
    try:
        completed = subprocess.run(
            command, cwd=cwd, env=dict(env), check=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            timeout=1800,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"official RULER script failed:\n{exc.stdout}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("official RULER script exceeded 30 minutes") from exc
    if completed.stdout:
        print(completed.stdout, end="", flush=True)
    return completed.stdout or ""


def _cached_ruler_download(url: str, path: Path, expected: str) -> bytes:
    body = path.read_bytes() if path.is_file() else _download(url)
    if _sha256(body) != expected:
        raise ValueError(f"RULER checksum mismatch: {url} (cache: {path})")
    if not path.is_file():
        _write_bytes(path, body)
    return body


def _ruler_source(config: Mapping[str, Any], work: Path, cache: Path) -> Path:
    revision = _text(config.get("Revision"))
    checksum = _text(config.get("ArchiveSha256"))
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Ruler.Revision must pin a full official Git commit")
    if not re.fullmatch(r"[0-9a-f]{64}", checksum):
        raise ValueError("Ruler.ArchiveSha256 must pin the source archive checksum")
    archive = _cached_ruler_download(
        f"https://codeload.github.com/NVIDIA/RULER/zip/{revision}",
        cache / "source.zip", checksum,
    )
    prefix = f"RULER-{revision}/scripts/"
    scripts = work / "scripts"
    with zipfile.ZipFile(io.BytesIO(archive)) as source:
        for member in source.infolist():
            if member.is_dir() or not member.filename.startswith(prefix):
                continue
            relative = _relative_path(member.filename[len(prefix):], "RULER source member")
            _write_bytes(scripts / relative, source.read(member))
    if not (scripts / "data/prepare.py").is_file():
        raise ValueError("official RULER archive contains no generation scripts")
    return scripts


def _ruler_assets(scripts: Path, cache: Path, revision: str, needs_essays: bool,
                  env: Mapping[str, str]) -> Dict[str, str]:
    assets = scripts / "data/synthetic/json"
    pointer = (assets / "english_words.json").read_text(encoding="utf-8")
    match = re.search(r"^oid sha256:([0-9a-f]{64})$", pointer, re.MULTILINE)
    if not match:
        raise ValueError("official RULER English word list has no Git LFS checksum")
    words = _cached_ruler_download(
        f"https://media.githubusercontent.com/media/NVIDIA/RULER/{revision}/"
        "scripts/data/synthetic/json/english_words.json",
        cache / "english_words.json", match.group(1),
    )
    _write_bytes(assets / "english_words.json", words)
    hashes = {"english_words.json": _sha256(words)}
    if needs_essays:
        from scripts.PrepareRulerCorpus import prepare_sources

        # Never reuse the incomplete corpus produced by the old best-effort fetch.
        sources = prepare_sources(assets / "PaulGrahamEssays_URLs.txt", cache)
        source_count = len(json.loads(sources.read_text())["sources"])
        source_hash = _file_sha256(sources)
        corpus_path, report_path = cache / "PaulGrahamEssays.json", cache / "corpus_sources.json"
        saved = json.loads(report_path.read_text()) if report_path.is_file() else {}
        valid = (
            saved.get("complete") is True
            and saved.get("raw_manifest_sha256") == source_hash
            and corpus_path.is_file()
            and saved.get("corpus_sha256") == _file_sha256(corpus_path)
        )
        if not valid:
            print("[dataset] rendering the complete official NIAH corpus from verified source files", flush=True)
            wrapper = ProjectRoot / "scripts/PrepareRulerCorpus.py"
            log = _run_ruler([
                sys.executable, str(wrapper), "--render", str(assets / "download_paulgraham_essay.py"),
                "--manifest", str(sources),
            ], assets, env)
            _write_bytes(cache / "essay_download.log", log.encode("utf-8"))
            counts = [int(value) for value in re.findall(r"Download (\d+) essays", log)]
            if "Fail download " in log or sum(counts) != source_count:
                raise RuntimeError("official RULER parser skipped source files; refusing a partial corpus")
            corpus = (assets / "PaulGrahamEssays.json").read_bytes()
            if not _text(json.loads(corpus).get("text")).strip():
                raise ValueError("official RULER parser produced an empty corpus")
            _write_bytes(corpus_path, corpus)
            _write_json(report_path, {
                "complete": True, "requested_urls": source_count,
                "downloaded_essays": source_count, "failed_downloads": 0,
                "raw_manifest_sha256": source_hash, "corpus_sha256": _sha256(corpus),
            })
        corpus = corpus_path.read_bytes()
        _write_bytes(assets / "PaulGrahamEssays.json", corpus)
        hashes["PaulGrahamEssays.json"] = _sha256(corpus)
        hashes["corpus_sources_manifest.json"] = source_hash
    return hashes


def _ruler_environment(cache: Path) -> Dict[str, str]:
    required = ("nltk", "wonderwords", "html2text", "bs4", "tenacity")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(
            f"official RULER generation needs {', '.join(missing)}; "
            "install requirement.txt first"
        )
    import nltk

    # Keep NLTK downloads inside the project, including in upstream subprocesses.
    nltk_dir = cache / "nltk_data"
    nltk_dir.mkdir(parents=True, exist_ok=True)
    for resource in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{resource}", paths=[str(nltk_dir)])
        except LookupError:
            if not nltk.download(resource, download_dir=str(nltk_dir), quiet=True, raise_on_error=True):
                raise RuntimeError(f"could not download NLTK {resource} for official RULER")
    env = dict(os.environ)
    env["NLTK_DATA"] = str(nltk_dir)
    return env


def _validate_ruler_output(path: Path, samples: int, length: int) -> None:
    if not path.is_file():
        raise RuntimeError(f"official RULER did not generate the requested test file: {path}")
    rows = _read_json_records(path)
    if len(rows) != samples:
        raise ValueError(f"official RULER generated {len(rows)} rows, expected {samples}: {path}")
    for index, row in enumerate(rows):
        outputs = row.get("outputs")
        if not _text(row.get("input")).strip() or not isinstance(outputs, list) or not outputs:
            raise ValueError(f"official RULER row {index} has no input or answers: {path}")
        if any(value is None or not str(value).strip() for value in outputs):
            raise ValueError(f"official RULER row {index} has an empty answer: {path}")
        if not isinstance(row.get("length"), int) or not 0 < row["length"] <= length:
            raise ValueError(f"official RULER row {index} exceeds length {length}: {path}")


def _prepare_ruler(config: Mapping[str, Any], stage: Path, full_config: Mapping[str, Any]) -> None:
    tasks = config.get("Tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Ruler.Tasks must be a non-empty list")
    if config.get("Split") != "test":
        raise ValueError("Ruler.Split must explicitly be test")
    tokenizer_name = config.get("Tokenizer")
    if tokenizer_name == "ModelPath":
        tokenizer_name = full_config.get("ModelPath")
    if not isinstance(tokenizer_name, str) or not tokenizer_name:
        raise ValueError("Ruler.Tokenizer must be a path or ModelPath")
    tokenizer_path = Path(tokenizer_name).expanduser()
    if tokenizer_path.exists():
        tokenizer_name = str(tokenizer_path.resolve())

    revision = _text(config.get("Revision"))
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Ruler.Revision must pin a full official Git commit")
    cache = ProjectRoot / ".cache/ruler" / revision
    cache.mkdir(parents=True, exist_ok=True)
    env = _ruler_environment(cache)
    provenance: Dict[str, Any] = {
        "repository": "https://github.com/NVIDIA/RULER", "revision": revision,
        "archive_sha256": config.get("ArchiveSha256"), "split": "test",
        "tokenizer": tokenizer_name, "model_template": "base", "files": [],
    }
    with tempfile.TemporaryDirectory(prefix="kvbench-ruler-") as temporary:
        work = Path(temporary)
        scripts = _ruler_source(config, work, cache)
        official = yaml.safe_load((scripts / "synthetic.yaml").read_text())
        constants = runpy.run_path(str(scripts / "data/synthetic/constants.py"))["TASKS"]
        generators = {"niah": "niah", "vt": "variable_tracking", "cwe": "common_words_extraction"}
        resolved = []
        for raw in tasks:
            task = _mapping(raw, "Ruler.Tasks[]")
            name, official_name = task.get("Task"), task.get("OfficialTask")
            if name not in generators or official_name not in official:
                raise ValueError("Ruler task must declare Task=niah/vt/cwe and an official OfficialTask")
            definition = official[official_name]
            if definition["task"] != generators[name]:
                raise ValueError(f"Ruler task {name} does not match official task {official_name}")
            lengths = task.get("Lengths")
            if not isinstance(lengths, list) or not lengths:
                raise ValueError("Ruler.Tasks[].Lengths must be a non-empty list")
            samples = _positive_int(task.get("Samples"), "Ruler.Tasks[].Samples")
            resolved.append((task, definition, constants[definition["task"]], samples))
        provenance["assets_sha256"] = _ruler_assets(
            scripts, cache, revision,
            any(definition["args"].get("type_haystack") == "essay" for _, definition, _, _ in resolved), env,
        )
        corpus_report = cache / "corpus_sources.json"
        if corpus_report.is_file() and "PaulGrahamEssays.json" in provenance["assets_sha256"]:
            provenance["corpus_download"] = json.loads(corpus_report.read_text())
            _write_bytes(stage / "ruler/_metadata/corpus_sources.json",
                         (cache / "corpus_sources_manifest.json").read_bytes())
            report = provenance["corpus_download"]
            print(
                f"[dataset] official NIAH corpus: {report['downloaded_essays']}/"
                f"{report['requested_urls']} essays, {report['failed_downloads']} failed URLs; "
                "recorded in provenance", flush=True,
            )
        for task, definition, defaults, samples in resolved:
            name, official_name = task["Task"], task["OfficialTask"]
            seed = int(task.get("Seed", 42))
            budget = _positive_int(task.get("TokensToGenerate", defaults["tokens_to_generate"]),
                                   "Ruler.Tasks[].TokensToGenerate")
            for raw_length in task["Lengths"]:
                length = _positive_int(raw_length, "Ruler.Tasks[].Lengths[]")
                output_dir = work / "generated" / str(length)
                command = [
                    sys.executable, str(scripts / f"data/synthetic/{definition['task']}.py"),
                    "--save_dir", str(output_dir), "--save_name", official_name,
                    "--subset", "test", "--tokenizer_path", tokenizer_name,
                    "--tokenizer_type", "hf", "--max_seq_length", str(length),
                    "--tokens_to_generate", str(budget), "--num_samples", str(samples),
                    "--random_seed", str(seed),
                    "--template", defaults["template"] + defaults.get("answer_prefix", ""),
                ]
                for key, value in definition["args"].items():
                    command.extend([f"--{key}", str(value)])
                print(f"[dataset] official RULER {official_name}/test length={length} samples={samples}", flush=True)
                _run_ruler(command, scripts, env)
                generated = output_dir / official_name / "test.jsonl"
                _validate_ruler_output(generated, samples, length)
                output = stage / "ruler" / f"{name}_len{length}.jsonl"
                body = generated.read_bytes()
                _write_bytes(output, body)
                provenance["files"].append({
                    "file": output.name, "official_task": official_name, "samples": samples,
                    "max_seq_length": length, "seed": seed, "tokens_to_generate": budget,
                    "args": definition["args"], "sha256": _sha256(body),
                })
        _write_json(stage / "ruler/provenance.json", provenance)


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
    for group in ("Direct", "HuggingFace"):
        for raw in preparation.get(group, []):
            entry = _mapping(raw, f"DatasetPreparation.{group}[]")
            if entry.get("Split") != "test" and not _text(entry.get("SplitReason")).strip():
                raise ValueError(
                    f"{entry.get('Output')}: use Split=test, or explain the labeled "
                    "evaluation exception in SplitReason"
                )
    ruler = _mapping(preparation.get("Ruler"), "DatasetPreparation.Ruler")
    if ruler.get("Split") != "test":
        raise ValueError("Ruler.Split must explicitly be test")
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


def _record_source(stage: Path, entry: Mapping[str, Any], kind: str) -> None:
    directory = _top_level_directory(_relative_path(entry.get("Output"), f"{kind}.Output"))
    urls = entry.get("Urls") or ([entry["Url"]] if entry.get("Url") else [])
    if entry.get("GoogleDriveId"):
        urls = [_google_drive_url(entry["GoogleDriveId"])]
    artifacts = []
    for url in urls:
        metadata = _source_cache(url) / "source.json"
        if metadata.is_file():
            artifacts.append(json.loads(metadata.read_text()))
    _write_json(stage / directory / "_metadata/source.json", {"kind": kind, **entry, "downloaded_artifacts": artifacts})


def _preparation_fingerprint(config: Mapping[str, Any]) -> str:
    ruler = config["DatasetPreparation"]["Ruler"]
    tokenizer = config.get("ModelPath") if ruler.get("Tokenizer") == "ModelPath" else ruler.get("Tokenizer")
    return _sha256(json.dumps({"preparation": config["DatasetPreparation"], "tokenizer": tokenizer},
                              sort_keys=True, ensure_ascii=False).encode())


def _generator_hashes() -> Dict[str, str]:
    return {f"scripts/{name}": _file_sha256(Path(__file__).with_name(name))
            for name in ("PrepareDataset.py", "PrepareRulerCorpus.py")}


def _write_preparation_manifest(stage: Path, config: Mapping[str, Any]) -> None:
    preparation = config["DatasetPreparation"]
    files = []
    for directory in preparation["DatasetDirectories"]:
        for path in sorted((stage / directory).rglob("*")):
            if path.is_file():
                files.append({"path": path.relative_to(stage).as_posix(),
                              "sha256": _file_sha256(path), "bytes": path.stat().st_size})
    sources = [{"kind": group, **entry} for group in ("Direct", "HuggingFace")
               for entry in preparation.get(group, [])]
    _write_json(stage / "preparation.json", {
        "schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "preparation_sha256": _preparation_fingerprint(config),
        "generators": _generator_hashes(), "sources": sources,
        "ruler": json.loads((stage / "ruler/provenance.json").read_text()),
        "directories": preparation["DatasetDirectories"], "files": files,
    })


def verify(config_path: Path) -> Dict[str, Any]:
    """Verify the installed bytes against the producing code and source plan."""
    config = _load_config(config_path)
    root = _dataset_root(config, config_path)
    manifest_path = root / "preparation.json"
    if not manifest_path.is_file():
        raise ValueError(f"no preparation manifest: {manifest_path}; regenerate the data first")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != 1:
        raise ValueError("unsupported preparation manifest schema")
    if manifest["preparation_sha256"] != _preparation_fingerprint(config) or manifest["generators"] != _generator_hashes():
        raise ValueError("dataset configuration or producing scripts changed; regenerate the data")
    expected = {entry["path"] for entry in manifest["files"]}
    actual = {path.relative_to(root).as_posix() for directory in manifest["directories"]
              for path in (root / directory).rglob("*") if path.is_file()}
    if actual != expected:
        raise ValueError(f"prepared data files changed: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
    for entry in manifest["files"]:
        path = root / _relative_path(entry["path"], "manifest file")
        if path.stat().st_size != entry["bytes"] or _file_sha256(path) != entry["sha256"]:
            raise ValueError(f"prepared data failed integrity verification: {path}")
    print(f"[dataset] verified {len(expected)} files against current configuration and producing scripts: {root}")
    return manifest


def _install_prepared(stage: Path, root: Path, directories: Sequence[str]) -> Path:
    """Keep the old datasets and roll back if any installation step fails."""
    backup_parent = root.parent / f".{root.name}.backups"
    backup_parent.mkdir(parents=True, exist_ok=True)
    backup = Path(tempfile.mkdtemp(prefix=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-"), dir=backup_parent))
    root.mkdir(parents=True, exist_ok=True)
    units = [*directories, "preparation.json"]
    installed, saved = [], []
    try:
        for name in units:
            destination = root / name
            if destination.exists():
                destination.rename(backup / name)
                saved.append(name)
            (stage / name).rename(destination)
            installed.append(name)
    except Exception:
        for name in reversed(installed):
            (root / name).rename(stage / name)
        for name in reversed(saved):
            (backup / name).rename(root / name)
        raise
    print(f"[dataset] previous data retained at {backup}")
    return backup


def prepare(config_path: Path, dry_run: bool = False) -> Path:
    config = _load_config(config_path)
    preparation = _validate_preparation_config(config)
    root = _dataset_root(config, config_path)
    directories = _planned_outputs(preparation)

    if dry_run:
        print(f"[dataset] DatasetPath: {root}")
        for entry in preparation.get("Direct", []):
            source = entry.get("Url", entry.get("Urls")) or f"GoogleDrive:{entry.get('GoogleDriveId')}"
            print(f"[dataset] direct: {source} split={entry.get('Split')} -> {entry.get('Output')}")
        for entry in preparation.get("HuggingFace", []):
            print(
                f"[dataset] hf: {entry.get('Dataset')}"
                f"[{entry.get('Config', '')}]/{entry.get('Split')} -> {entry.get('Output')}"
            )
        for directory in directories:
            print(f"[dataset] replaces: {directory}/")
        ruler = preparation["Ruler"]
        for task in ruler.get("Tasks", []):
            print(
                f"[dataset] official RULER: {task.get('OfficialTask')}/{ruler.get('Split')} "
                f"lengths={task.get('Lengths')} samples={task.get('Samples')} "
                f"revision={ruler.get('Revision')}"
            )
        return root

    root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{root.name}.prepare-", dir=root.parent))
    try:
        for raw in preparation.get("Direct", []):
            entry = _mapping(raw, "DatasetPreparation.Direct[]")
            _prepare_direct(entry, stage)
            _record_source(stage, entry, "Direct")
        for raw in preparation.get("HuggingFace", []):
            entry = _mapping(raw, "DatasetPreparation.HuggingFace[]")
            _prepare_huggingface(entry, stage)
            _record_source(stage, entry, "HuggingFace")
        _prepare_ruler(_mapping(preparation.get("Ruler"), "DatasetPreparation.Ruler"), stage, config)

        for directory in directories:
            staged = stage / directory
            if not staged.is_dir():
                raise RuntimeError(f"configured dataset directory was not generated: {directory}")
        _write_preparation_manifest(stage, config)
        _install_prepared(stage, root, directories)
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
    parser.add_argument("--verify", action="store_true", help="check installed file hashes and producing configuration/code without downloading")
    args = parser.parse_args()
    if args.verify:
        verify(args.config.resolve())
        return
    root = prepare(args.config.resolve(), dry_run=args.dry_run)
    print(f"[dataset] ready: {root}")


if __name__ == "__main__":
    main()
