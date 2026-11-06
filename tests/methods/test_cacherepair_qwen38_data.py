"""Schema, overlap screening and source-level split safety for repair data."""

import hashlib
import json
import pickle
import sqlite3

import numpy as np
import pytest

from methods.CacheRepair.qwen38.decontam import MinHashLSH, screen, window_hashes
from methods.CacheRepair.qwen38.eval_inventory import build_inventory, normalize, row_contents, text_hash
from methods.CacheRepair.qwen38.prepare import compile_examples
from methods.CacheRepair.qwen38.sources import local_checkpoint_spec
from methods.CacheRepair.qwen38 import sources


class Words:
    def encode(self, text, *, add_special_tokens=False):
        return [int.from_bytes(hashlib.blake2b(word.encode(), digest_size=4).digest(), "little")
                for word in text.split()]


def make_inventory(path, values):
    path.mkdir()
    with sqlite3.connect(path / "contents.sqlite") as connection:
        connection.executescript("CREATE TABLE contents(hash TEXT PRIMARY KEY,text TEXT);"
                                 "CREATE TABLE refs(hash TEXT,workload TEXT,record_id TEXT,kind TEXT);")
        for index, text in enumerate(values):
            digest = text_hash(text)
            connection.execute("INSERT INTO contents VALUES (?,?)", (digest, text))
            connection.execute("INSERT INTO refs VALUES (?,?,?,?)", (digest, "test", str(index), "document"))


def test_unicode_normalization_and_schema_adapters():
    assert normalize("ＡBC\u00a0  Def\n") == "abc def"
    assert text_hash("ＡBC  Def") == text_hash("abc\ndef")
    assert list(row_contents("musique", {"ctxs": [{"title": "Title", "text": "Body"}], "question": "Q"})) == [
        ("document", "Title\n\nBody"), ("question", "Q")]
    assert list(row_contents("hotpotqa", {"context": "Passage 1:\nAlpha\nPassage 2:\nBeta", "input": "Q"})) == [
        ("document", "Alpha\n"), ("document", "Beta"), ("question", "Q")]
    with pytest.raises(ValueError, match="adapter"):
        list(row_contents("unknown", {}))


def test_local_snapshot_revision_provenance(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors.index.json").write_text("{}")
    (tmp_path / ".msc").write_bytes(pickle.dumps([
        {"Path": "model-01.safetensors", "Revision": "pinned-modelscope-commit"},
        {"Path": "tokenizer.json", "Revision": "pinned-tokenizer-commit"},
    ]))
    spec = local_checkpoint_spec(tmp_path)
    assert spec["weight_revisions"] == ["pinned-modelscope-commit"]
    assert spec["file_revisions"]["tokenizer.json"] == "pinned-tokenizer-commit"


def test_isolated_corpus_worker_preserves_ids_and_provenance(tmp_path, monkeypatch):
    records = {2: {"id": "102", "contents": "Passage", "title": "Title", "wikipedia_id": "article"}}

    class Source:
        def __init__(self, repo, revision, cache):
            assert (repo, revision) == (sources.CORPUS_REPO, sources.CORPUS_REVISION)
            self.cache = tmp_path

        def rows(self, name, indices):
            assert name == "shard.parquet" and indices == [2]
            return records

    monkeypatch.setattr(sources, "ParquetSource", Source)
    result = sources._fetch_corpus_rows((str(tmp_path), "shard.parquet", [2], 100))
    assert result["102"]["source_row"] == 2
    assert result["102"]["source_file"] == "shard.parquet"
    assert json.loads((tmp_path / "passage-102.json").read_text()) == result["102"]
    records[2]["id"] = "wrong"
    with pytest.raises(ValueError, match="positional-ID"):
        sources._fetch_corpus_rows((str(tmp_path), "shard.parquet", [2], 100))


def test_inventory_fingerprint_covers_changed_skill_content(tmp_path):
    root, skills = tmp_path / "data", tmp_path / "skills"
    root.mkdir()
    task = skills / "tasks" / "task" / "task.md"
    task.parent.mkdir(parents=True)
    task.write_text("Original held out task text")
    first = build_inventory(root, tmp_path / "first", skills_root=skills, synthetic_cases=1)
    task.write_text("Changed held out task text")
    second = build_inventory(root, tmp_path / "second", skills_root=skills, synthetic_cases=1)
    assert first["coverage"] == second["coverage"]
    assert first["content_sha256"] != second["content_sha256"]
    assert first["coverage"]["hotpotqa"]["status"] == "missing"
    assert first["coverage"]["agent_benchflow"]["status"] == "partial"


def test_rolling_hash_is_position_independent_and_includes_all_windows():
    ids = list(range(130))
    hashes = window_hashes(ids)
    assert len(hashes) == 81
    for offset in (0, 1, 37, 80):
        assert hashes[offset] == window_hashes(ids[offset:offset + 50])[0]
    assert len(window_hashes(ids[:49])) == 0
    assert not np.array_equal(hashes, window_hashes(list(reversed(ids))))


def test_exact_spans_and_questions_are_audited(tmp_path):
    exact = "An entire held out passage"
    span = " ".join(f"token{i}" for i in range(70))
    acquired = {"passages": {"exact": {"contents": exact}, "span": {"contents": "intro " + span + " outro"},
                             "clean": {"contents": "Completely independent source document."}},
                "records": [{"query": "A protected question?", "query_id": "q", "config": "nq"}]}
    inventory = tmp_path / "inventory"
    make_inventory(inventory, [exact.upper(), "extra " + " ".join(f"token{i}" for i in range(10, 65)), "A protected question?"])
    result = screen(acquired, inventory, Words())
    assert set(result["removed_passage_ids"]) == {"exact", "span"}
    assert any(item["reason"] == "verified_50_token_span" and item["passage_id"] == "span" for item in result["removals"])
    assert result["question_matches"][0]["query_id"] == "q"
    assert result["evaluation_content_units_scanned"] == 3


def test_minhash_reports_scored_candidates():
    index = MinHashLSH()
    text = " ".join(f"word{i}" for i in range(40))
    index.add("source", text)
    assert list(index.query(text)) == [("source", 1.0)]
    assert not list(index.query("Entirely unrelated brief passage"))


def test_deterministic_source_document_split_and_limits():
    records, passages = [], {}
    for index in range(14):
        ids = [f"{index}-{part}" for part in range(3)]
        for identifier in ids:
            passages[identifier] = {
                "id": identifier, "title": f"Title {identifier}",
                "contents": " ".join(["unique" + identifier] * 90),
                # Pairs of queries share one source article; they must stay together.
                "wikipedia_id": f"article-{index // 2}", "source_file": "source", "source_row": index,
            }
        records.append({"source": "corag/kilt", "revision": "pinned", "config": "nq", "split": "train",
                        "source_file": "queries", "source_row": index, "query_id": str(index),
                        "query": f"question {index}", "answers": [],
                        "passage_ids": ids, "original_passage_ids": ids})
    first = compile_examples(records, passages, Words(), seed=7, train_size=8, validation_size=2,
                             max_tokens=180, max_chunk_tokens=128, max_chunks=3)
    second = compile_examples(records, passages, Words(), seed=7, train_size=8, validation_size=2,
                              max_tokens=180, max_chunk_tokens=128, max_chunks=3)
    assert first == second
    rows, manifest = first
    assert len(rows["train"]) == 8 and len(rows["validation"]) == 2
    train_docs = {p["wikipedia_id"] for row in manifest if row["split"] == "train" for p in row["passages"]}
    validation_docs = {p["wikipedia_id"] for row in manifest if row["split"] == "validation" for p in row["passages"]}
    assert not train_docs & validation_docs
    for examples in rows.values():
        for row in examples:
            assert 2 <= len(row["chunks"]) <= 3
            assert sum(map(len, row["chunks"])) <= 180
            assert all(0 < len(chunk) <= 128 for chunk in row["chunks"])
