"""Deterministic full-scale CoRAG approximation, with resumable acquisition.

This is a new manifest, not the unavailable published 50,000-example manifest.
Source ratios and length bands follow Appendix J; tokenization/KB formatting
follow the actual Qwen3.8 KVBench deployment. Query order is seeded random.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import sqlite3
import tempfile

import numpy as np

from .decontam import MinHashLSH, screen, window_hashes
from .eval_inventory import inventory_contents, text_hash
from .resume import atomic_json, fingerprint
from .sources import (CORPUS_REPO, CORPUS_REVISION, QUERY_REPO, QUERY_REVISION,
                      SOURCE_CONFIGS, inventory, local_tokenizer_spec, write_json)

SHARES = dict(zip(SOURCE_CONFIGS, (2, 2, 2, 2, 1, 1)))
BANDS = (("short", 1, 1024, 1), ("medium", 1025, 2048, 1), ("long", 2049, 4096, 8))


def acquire_full(cache, *, seed=7, candidates=150000, workers=4):
    """Download pinned Parquet shards; cache selected rows in one SQLite DB."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    metadata_path = cache / "source-inventory.json"
    if not metadata_path.exists():
        atomic_json(metadata_path, inventory())
    metadata = json.loads(metadata_path.read_text())
    spec = {"seed": seed, "candidates": candidates, "inventory": metadata}
    marker = cache / "acquisition.json"
    if marker.exists():
        if json.loads(marker.read_text())["identity"] != fingerprint(spec):
            raise ValueError("Acquisition cache has different selection settings")
        return

    requests = [(source, item) for source in metadata["sources"] for item in source["files"]]

    def download(request):
        source, item = request
        path = hf_hub_download(source["id"], item["path"], repo_type="dataset", revision=source["revision"],
                               cache_dir=str(cache / "hub"))
        if Path(path).stat().st_size != item["bytes"]:
            raise ValueError(f"Downloaded size differs: {item['path']}")
        print(json.dumps({"downloaded": source["id"] + "/" + item["path"]}), flush=True)
        return source["id"], item["path"], path

    with ThreadPoolExecutor(max_workers=workers) as pool:
        paths = {(repo, name): path for repo, name, path in pool.map(download, requests)}
    records = []
    query_source = next(s for s in metadata["sources"] if s["id"] == QUERY_REPO)
    for config in SOURCE_CONFIGS:
        files = sorted(f["path"] for f in query_source["files"] if f["path"].startswith(config + "/"))
        tables = [(name, pq.ParquetFile(paths[QUERY_REPO, name])) for name in files]
        count = sum(p.metadata.num_rows for _, p in tables)
        wanted = candidates * SHARES[config] // 10
        if wanted > count:
            raise ValueError(f"Not enough {config} records: {wanted}>{count}")
        rng = random.Random(f"{seed}:{config}")
        indices = rng.sample(range(count), wanted)
        ranks = {index: rank for rank, index in enumerate(indices)}
        cursor = 0
        for name, parquet in tables:
            for group in range(parquet.num_row_groups):
                length = parquet.metadata.row_group(group).num_rows
                selected = sorted(i - cursor for i in indices if cursor <= i < cursor + length)
                if selected:
                    rows = parquet.read_row_group(group).take(selected).to_pylist()
                    for offset, row in zip(selected, rows):
                        if not all(key in row for key in ("query", "query_id", "context_doc_ids", "answers")):
                            raise ValueError("Pinned query schema differs")
                        records.append({"source": QUERY_REPO, "revision": QUERY_REVISION, "config": config,
                                        "split": "train", "source_file": name,
                                        "source_row": cursor + offset - sum(p.metadata.num_rows for n, p in tables if n < name),
                                        "selection_rank": ranks[cursor + offset], "query_id": str(row["query_id"]),
                                        "query": row["query"], "answers": row["answers"],
                                        "passage_ids": list(dict.fromkeys(map(str, row["context_doc_ids"][:20])))})
                cursor += length
        print(json.dumps({"query_source": config, "selected": wanted, "available": count}), flush=True)
    records.sort(key=lambda row: (SOURCE_CONFIGS.index(row["config"]), row["selection_rank"]))
    atomic_json(cache / "queries.json", records)
    required = {int(identifier) for row in records for identifier in row["passage_ids"]}
    connection = sqlite3.connect(cache / "passages.sqlite")
    connection.execute("CREATE TABLE IF NOT EXISTS passages(id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
    connection.execute("CREATE TABLE IF NOT EXISTS completed_shards(name TEXT PRIMARY KEY)")
    done = {r[0] for r in connection.execute("SELECT name FROM completed_shards")}
    cursor = 0
    for name in sorted(n for repo, n in paths if repo == CORPUS_REPO):
        parquet = pq.ParquetFile(paths[CORPUS_REPO, name])
        length = parquet.metadata.num_rows
        if name not in done:
            selected = sorted(i - cursor for i in required if cursor <= i < cursor + length)
            row_cursor, picked = 0, 0
            selected_array = np.asarray(selected, dtype=np.int64)
            for group in range(parquet.num_row_groups):
                group_size = parquet.metadata.row_group(group).num_rows
                lo, hi = np.searchsorted(selected_array, [row_cursor, row_cursor + group_size])
                offsets = selected_array[lo:hi] - row_cursor
                if len(offsets):
                    rows = parquet.read_row_group(group).take(offsets).to_pylist()
                    for offset, row in zip(offsets, rows):
                        original_row = row_cursor + int(offset)
                        if str(row["id"]) != str(cursor + original_row):
                            raise ValueError("Corpus positional IDs differ")
                        value = {**row, "source": CORPUS_REPO, "revision": CORPUS_REVISION,
                                 "split": "train", "source_file": name, "source_row": original_row}
                        connection.execute("INSERT OR REPLACE INTO passages VALUES (?,?)",
                                           (str(row["id"]), json.dumps(value)))
                    picked += len(rows)
                row_cursor += group_size
            connection.execute("INSERT INTO completed_shards VALUES (?)", (name,))
            connection.commit()
            print(json.dumps({"acquired_shard": name, "passages": picked}), flush=True)
        cursor += length
    actual = connection.execute("SELECT COUNT(*) FROM passages").fetchone()[0]
    connection.close()
    if actual != len(required):
        raise ValueError("Missing or unexpected corpus passages")
    atomic_json(marker, {"identity": fingerprint(spec), "records": len(records), "passages": actual,
                         "policy": "seeded random sampling without replacement within six source train splits; first 20 retrievals"})


def compile_full(records, passages, tokenizer, *, seed=7, train_size=50000, validation_size=500,
                 max_chunk_tokens=512):
    """Reserve validation content first, then remove its source content from train.

    Unlike whole-corpus connected components, this remains useful when retrieval
    queries form a giant component. Related training passages are removed before
    example construction; no held-out source document is allowed back in train.
    """
    from tasks.Musique import MusiqueTask
    if train_size % 100 or validation_size % 100:
        raise ValueError("Exact source/length proportions require split sizes divisible by 100")
    prefix = tokenizer.encode(MusiqueTask.prefixPrompt, add_special_tokens=False)
    ending = tokenizer.encode("\n\n", add_special_tokens=False)
    encoded = {}

    def tokens(identifier, first=False):
        key = (identifier, first)
        if key not in encoded:
            p = passages[identifier]
            title, body = p["title"], p["contents"]
            if body.startswith(title + "\n"):
                body = body[len(title) + 1:]
            text = (MusiqueTask.prefixPrompt if first else "") + title + "\n\n" + body + "\n\n"
            ids = tokenizer.encode(text, add_special_tokens=False)
            quota = max_chunk_tokens + (len(prefix) if first else 0)
            encoded[key] = ids[:quota - len(ending)] + ending if len(ids) > quota else ids
        return encoded[key]

    rng = random.Random(seed)
    choices = {"train": [], "validation": []}
    held_docs, held_texts, held_windows = set(), set(), set()
    near = MinHashLSH()
    held_queries, selected_queries = set(), set()
    reserved_passages = set()

    def overlaps(identifier):
        p = passages[identifier]
        if p["wikipedia_id"] in held_docs or text_hash(p["contents"]) in held_texts:
            return True
        ids = tokenizer.encode(p["contents"], add_special_tokens=False)
        if any(int(digest) in held_windows for digest in window_hashes(ids)):
            return True  # Hash collisions conservatively exclude a candidate.
        return bool(list(near.query(p["contents"])))

    # Bands are filled from longest to shortest so short groups cannot consume
    # all long-eligible queries. Quotas are exact and all retrieval order retained.
    for split, size in (("validation", validation_size), ("train", train_size)):
        for config in SOURCE_CONFIGS:
            candidates = [r for r in records if r["config"] == config]
            for band, low, high, share in reversed(BANDS):
                wanted = size * SHARES[config] * share // 100
                picked = 0
                for row in candidates:
                    query_key = text_hash(row["query"])
                    identity = (config, row["query_id"])
                    if identity in selected_queries or query_key in held_queries and split == "train":
                        continue
                    identifiers = row["passage_ids"]
                    if split == "train":
                        identifiers = [i for i in identifiers if not overlaps_cached(i)]
                    lengths, running = [], 0
                    for position, identifier in enumerate(identifiers[:20]):
                        running += len(tokens(identifier, first=position == 0))
                        lengths.append(running)
                    possible = [k for k in range(2, len(lengths) + 1) if low <= lengths[k - 1] <= high]
                    if not possible:
                        continue
                    count = rng.choice(possible)
                    kept = identifiers[:count]
                    choices[split].append((row, kept, band))
                    selected_queries.add(identity)
                    if split == "validation":
                        held_queries.add(query_key)
                        reserved_passages.update(kept)
                    picked += 1
                    if picked == wanted:
                        break
                if picked != wanted:
                    raise ValueError(f"Insufficient {split}/{config}/{band}: {picked}/{wanted}; increase candidates")
                print(json.dumps({"compiled_split": split, "source": config, "band": band, "examples": picked}), flush=True)
        if split == "validation":
            for identifier in sorted(reserved_passages):
                p = passages[identifier]
                held_docs.add(p["wikipedia_id"])
                held_texts.add(text_hash(p["contents"]))
                held_windows.update(map(int, window_hashes(tokenizer.encode(p["contents"], add_special_tokens=False))))
                near.add(identifier, p["contents"])
            overlap_cache = {}

            def overlaps_cached(identifier):
                if identifier not in overlap_cache:
                    overlap_cache[identifier] = overlaps(identifier)
                return overlap_cache[identifier]

    result, manifest = {}, []
    for split, selected in choices.items():
        rng.shuffle(selected)
        result[split] = []
        for row, ids, band in selected:
            chunks = [tokens(i, first=position == 0)[:] for position, i in enumerate(ids)]
            # Prefix is an instruction, and can add a few tokens over the plain
            # passage cap. Keep that distinction explicit in the manifest.
            result[split].append({"chunks": chunks})
            manifest.append({**{k: v for k, v in row.items() if k != "passage_ids"},
                             "example_id": fingerprint(chunks), "split": split, "seed": seed,
                             "band": band, "chunk_count": len(chunks), "token_count": sum(map(len, chunks)),
                             "corpus_revision": CORPUS_REVISION,
                             "passages": [{"id": i, "wikipedia_id": passages[i]["wikipedia_id"],
                                           "content_sha256": text_hash(passages[i]["contents"]),
                                           "source_file": passages[i]["source_file"], "source_row": passages[i]["source_row"],
                                           "retrieval_rank": row["original_passage_ids"].index(i)} for i in ids]})
    train_docs = {p["wikipedia_id"] for r in manifest if r["split"] == "train" for p in r["passages"]}
    if train_docs & held_docs:
        raise AssertionError("Source documents leaked across splits")
    return result, manifest


def build_full(args):
    from transformers import AutoTokenizer
    cache, output = Path(args.cache), Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    acquire_full(cache, seed=args.seed, candidates=args.candidates, workers=args.workers)
    records = json.loads((cache / "queries.json").read_text())
    with sqlite3.connect(cache / "passages.sqlite") as db:
        passages = {i: json.loads(payload) for i, payload in db.execute("SELECT id,payload FROM passages ORDER BY CAST(id AS INTEGER)")}
    coverage = json.loads((Path(args.evaluation_inventory) / "manifest.json").read_text())
    if any(v["status"] != "available" for k, v in coverage["coverage"].items() if k != "agent_benchflow"):
        raise ValueError("Required evaluation coverage is missing")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer_spec = local_tokenizer_spec(args.model)
    audit_key = fingerprint({"acquisition": json.loads((cache / "acquisition.json").read_text()),
                             "evaluation": coverage, "tokenizer": tokenizer_spec, "screen_format": "packed-v1"})
    audit_path = cache / f"screen-{audit_key}.json"
    if not audit_path.exists():
        audit = screen({"records": records, "passages": passages}, args.evaluation_inventory, tokenizer)
        atomic_json(audit_path, audit)
    audit = json.loads(audit_path.read_text())
    rejected = set(audit["removed_passage_ids"]) | set(audit["pending_review_passage_ids"])
    questions = {(m["config"], m["query_id"]) for m in audit["question_matches"]}
    clean = []
    for row in records:
        ids, seen = [], set()
        for i in row["passage_ids"]:
            digest = text_hash(passages[i]["contents"])
            if i not in rejected and digest not in seen and passages[i]["contents"].strip():
                ids.append(i)
                seen.add(digest)
        if len(ids) >= 2 and (row["config"], row["query_id"]) not in questions:
            clean.append({**row, "original_passage_ids": row["passage_ids"], "passage_ids": ids})
    examples, manifest = compile_full(clean, passages, tokenizer, seed=args.seed,
                                      train_size=args.train_size, validation_size=args.validation_size)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="." + output.name + "-writing-", dir=output.parent))
    for split, rows in examples.items():
        with (temporary / f"{split}.jsonl").open("x") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
    with (temporary / "examples.manifest.jsonl").open("x") as stream:
        for row in manifest:
            stream.write(json.dumps({**row, "tokenizer": tokenizer_spec}, ensure_ascii=False) + "\n")
    audit.update({"evaluation_inventory": str(Path(args.evaluation_inventory).resolve()), "coverage": coverage["coverage"],
                  "acquisition": json.loads((cache / "acquisition.json").read_text()),
                  "train_validation_source_documents_disjoint": True,
                  "split_policy": "validation reserved first; training passages excluded by Wikipedia/content/50-token/LSH overlap and query text",
                  "band_policy": "10/10/80 percent formatted Qwen3.8 lengths at <=1024/2048/4096; differs from paper document-only Qwen2.5 lengths",
                  "source_policy": "20/20/20/20/10/10 percent in each length band",
                  "length_statistics": {split: {"examples": len(rows),
                    "total_tokens": sum(sum(map(len, r['chunks'])) for r in rows),
                    "max_tokens": max(sum(map(len, r['chunks'])) for r in rows),
                    "bands": dict(Counter(r["band"] for r in manifest if r["split"] == split))} for split, rows in examples.items()}})
    write_json(temporary / "deduplication.json", audit)
    write_json(temporary / "dataset.json", {"format": "kvbench-qwen38-full-v1", "tokenizer": tokenizer_spec,
        "files_sha256": {f"{s}.jsonl": hashlib.sha256((temporary / f"{s}.jsonl").read_bytes()).hexdigest() for s in examples},
        "sizes": {s: len(r) for s, r in examples.items()}, "max_tokens": 4096, "seed": args.seed,
        "policy": "Reproducible CoRAG approximation, not exact published training manifest"})
    temporary.rename(output)
    print(json.dumps({"dataset_completed": str(output), "sizes": {s: len(r) for s, r in examples.items()}}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("acquire", "build"))
    parser.add_argument("--cache", required=True)
    parser.add_argument("--model")
    parser.add_argument("--evaluation-inventory")
    parser.add_argument("--output")
    parser.add_argument("--candidates", type=int, default=150000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-size", type=int, default=50000)
    parser.add_argument("--validation-size", type=int, default=500)
    args = parser.parse_args()
    if args.command == "acquire":
        acquire_full(args.cache, seed=args.seed, candidates=args.candidates, workers=args.workers)
    else:
        if not all((args.model, args.evaluation_inventory, args.output)):
            parser.error("build requires --model, --evaluation-inventory, --output")
        build_full(args)


if __name__ == "__main__":
    main()
