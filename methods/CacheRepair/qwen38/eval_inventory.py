"""Reusable content inventory of local KVBench inputs, without chat boilerplate."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import random
import re
import sqlite3
import unicodedata

from .sources import write_json


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", str(text)).casefold().split())


def text_hash(text):
    return hashlib.sha256(normalize(text).encode()).hexdigest()


def json_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        if Path(path).suffix == ".json":
            yield from json.load(stream)
        else:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


def row_contents(workload, row):
    """Adapters for the inspected on-disk schemas, not guessed dataset names."""
    if workload in {"musique", "2wikimultihopqa", "samsum"}:
        for context in row["ctxs"]:
            text = context["text"]
            if context.get("title"):
                text = context["title"] + "\n\n" + text
            yield "document", text
        if workload != "samsum":
            yield "question", row["question"]
    elif workload in {"hotpotqa", "triviaqa", "multinews", "govreport"}:
        for passage in re.split(r"(?m)^Passage(?: \d+)?:[ \t]*\n", row["context"]):
            if passage.strip():
                yield "document", passage
        if row.get("input"):
            yield "question", row["input"]
    elif workload == "gsm8k":
        yield "question", row["question"]
    elif workload == "humaneval":
        yield "code", row["prompt"]
        # Also protect held-out solutions/tests, though they are not input prompts.
        yield "reference_code", row["canonical_solution"]
        yield "reference_code", row["test"]
    elif workload.startswith("ruler_"):
        body = row["input"].replace("[INST]", "").replace("[/INST]", "").strip()
        # The first line is the generic task instruction. Keep the haystack,
        # needles, variables and actual question; the shuffle variants use it too.
        yield "synthetic", body.split("\n", 1)[-1]
    else:
        raise ValueError(f"No evaluation-content adapter for {workload}")


def build_inventory(data_root, destination, *, skills_root=None, synthetic_cases=100):
    destination, root = Path(destination), Path(data_root)
    destination.mkdir(parents=True, exist_ok=False)
    connection = sqlite3.connect(destination / "contents.sqlite")
    connection.executescript("""
        CREATE TABLE contents(hash TEXT PRIMARY KEY, text TEXT NOT NULL);
        CREATE TABLE refs(hash TEXT, workload TEXT, record_id TEXT, kind TEXT);
        CREATE INDEX refs_hash ON refs(hash);
    """)
    coverage, files = {}, []
    content_fingerprint = hashlib.sha256()

    def add(workload, record_id, kind, text):
        if not isinstance(text, str):
            raise ValueError(f"Non-text evaluation content at {workload}/{record_id}/{kind}")
        if not text.strip():
            return
        digest = text_hash(text)
        content_fingerprint.update(json.dumps([workload, str(record_id), kind, text], ensure_ascii=False).encode())
        connection.execute("INSERT OR IGNORE INTO contents VALUES (?,?)", (digest, text))
        connection.execute("INSERT INTO refs VALUES (?,?,?,?)", (digest, workload, str(record_id), kind))
        coverage[workload]["content_units"] += 1

    patterns = {
        "musique": "musique/*.jsonl", "2wikimultihopqa": "2wikimultihopqa/*.json",
        "samsum": "samsum/*.jsonl", "hotpotqa": "hotpotqa/*.jsonl",
        "triviaqa": "triviaqa/*.jsonl", "multinews": "multinews/*.jsonl",
        "govreport": "govreport/*.jsonl", "gsm8k": "gsm8k/*.jsonl",
        "humaneval": "humaneval/*.jsonl", "mmlu": "mmlu/test/*.csv",
        **{f"ruler_{task}": f"ruler/{task}_len*.jsonl" for task in ("niah", "cwe", "vt")},
    }
    for workload, pattern in patterns.items():
        selected = sorted(root.glob(pattern))
        coverage[workload] = {"status": "available" if selected else "missing", "records": 0, "content_units": 0}
        for path in selected:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                while block := stream.read(8 * 1024 * 1024):
                    digest.update(block)
            files.append({"path": str(path.resolve()), "sha256": digest.hexdigest(), "bytes": path.stat().st_size})
            if workload == "mmlu":
                with path.open(newline="", encoding="utf-8") as stream:
                    for index, row in enumerate(csv.reader(stream)):
                        if len(row) < 6:
                            raise ValueError(f"Invalid MMLU CSV row in {path}:{index}")
                        coverage[workload]["records"] += 1
                        add(workload, f"{path.name}:{index}", "question", row[0])
                        add(workload, f"{path.name}:{index}", "question_with_choices", "\n".join(row[:5]))
                continue
            for index, row in enumerate(json_rows(path)):
                record_id = f"{path.name}:{row.get('_id', row.get('task_id', row.get('index', index)))}"
                coverage[workload]["records"] += 1
                for kind, text in row_contents(workload, row):
                    add(workload, record_id, kind, text)
            connection.commit()
        print(json.dumps({"inventory_workload": workload, **coverage[workload]}), flush=True)

    for workload in ("copy", "fresh_gap"):
        coverage[workload] = {"status": "available", "records": synthetic_cases, "content_units": 0,
                              "scope": "repository defaults, seed=42, cases [0,N); custom settings need re-inventory"}
    rng = random.Random(42)
    for index in range(synthetic_cases):
        add("copy", index, "synthetic", " ".join(rng.choices(["Δ", "Ω"], k=1000)))
        add("fresh_gap", index, "synthetic", f"The transient code is {731946 + index}.")
    from tasks.FreshGap import FreshGapTask
    fresh = object.__new__(FreshGapTask)
    fresh.linesPerChunk = 192
    for label in ("A", "C"):
        add("fresh_gap", label, "synthetic", fresh._Filler(label))

    # Static task/skill text is inspectable; tool results and generated agent
    # transcripts are dynamic and must never be represented as zero overlap.
    selected = [] if skills_root is None else sorted(Path(skills_root).glob("tasks/**/task.md"))
    if skills_root:
        selected += sorted(Path(skills_root).glob("tasks/**/SKILL.md"))
    coverage["agent_benchflow"] = {"status": "partial" if selected else "missing", "records": len(selected),
                                    "content_units": 0, "unavailable": "runtime tool outputs, task assets and generated conversations"}
    for path in selected:
        add("agent_benchflow", str(path.relative_to(skills_root)), "task_or_skill", path.read_text())
    connection.commit()
    unique = connection.execute("SELECT count(*) FROM contents").fetchone()[0]
    connection.close()
    report = {"format": "kvbench-evaluation-content-v1", "coverage": coverage, "files": files,
              "content_sha256": content_fingerprint.hexdigest(),
              "unique_content_units": unique, "normalization": "NFKC, casefold, whitespace collapse",
              "boilerplate": "no chat wrappers/Task role instructions; remove Passage labels and RULER generic opening line",
              "scope": "all rows of local registered static datasets; RULER shuffle variants share these source bodies"}
    write_json(destination / "manifest.json", report)
    return report


def inventory_contents(directory):
    connection = sqlite3.connect(Path(directory) / "contents.sqlite")
    try:
        for digest, text in connection.execute("SELECT hash,text FROM contents ORDER BY rowid"):
            refs = connection.execute("SELECT workload,record_id,kind FROM refs WHERE hash=?", (digest,)).fetchall()
            yield digest, text, refs
    finally:
        connection.close()
