"""Pinned CoRAG/KILT acquisition using Hub metadata and Parquet range reads."""

from __future__ import annotations

from bisect import bisect_right
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import hashlib
import io
import json
import multiprocessing
from pathlib import Path
import pickle


SOURCE_CONFIGS = ("eli5", "fever", "nq", "wow", "trex", "structured_zeroshot")
QUERY_REPO = "corag/kilt"
QUERY_REVISION = "d3f3f906826b1524e7db4301723cecad770be0a7"
CORPUS_REPO = "corag/kilt-corpus"
CORPUS_REVISION = "bf95d9b4b19c68420395efad7498d0ef5052ab4b"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def local_tokenizer_spec(path):
    """A content revision for local tokenizer files, without asserting Hub ancestry."""
    path = Path(path).resolve()
    files = {}
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "vocab.json", "merges.txt"):
        if (path / name).is_file():
            files[name] = hashlib.sha256((path / name).read_bytes()).hexdigest()
    if "tokenizer.json" not in files:
        raise ValueError("A local Qwen tokenizer.json is required")
    revision = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"id": "Qwen/Qwen3.8-27B", "path": str(path),
            "revision": "local-sha256:" + revision, "files": files}


def local_checkpoint_spec(path):
    """Record local snapshot ancestry without confusing cache metadata with hashes."""
    path = Path(path).resolve()
    result = {"path": str(path), "files_sha256": {
        name: hashlib.sha256((path / name).read_bytes()).hexdigest()
        for name in ("config.json", "model.safetensors.index.json")
    }}
    metadata = path / ".msc"
    if metadata.is_file():
        class DataOnlyUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                raise pickle.UnpicklingError("ModelScope cache metadata must contain primitive data only")

        data = DataOnlyUnpickler(io.BytesIO(metadata.read_bytes())).load()
        revisions = {row["Path"]: row["Revision"] for row in data}
        result["modelscope_cache_metadata_sha256"] = hashlib.sha256(metadata.read_bytes()).hexdigest()
        result["file_revisions"] = revisions
        result["weight_revisions"] = sorted({revision for name, revision in revisions.items() if name.endswith(".safetensors")})
        result["revision_verification"] = "declared by local ModelScope download metadata; weights not independently rehashed"
    else:
        result["weight_revisions"] = []
        result["revision_verification"] = "upstream weight revision unavailable; config/index hashes are not full weight hashes"
    return result


def inventory():
    """Metadata only: no query records or passage data downloaded."""
    from huggingface_hub import HfApi

    entries = []
    for repo, revision in ((QUERY_REPO, QUERY_REVISION), (CORPUS_REPO, CORPUS_REVISION)):
        info = HfApi().dataset_info(repo, revision=revision, files_metadata=True)
        files = [{"path": item.rfilename, "bytes": item.size} for item in info.siblings
                 if item.rfilename.endswith(".parquet") and (
                     repo == CORPUS_REPO or any(item.rfilename.startswith(c + "/train-") for c in SOURCE_CONFIGS))]
        entries.append({"id": repo, "revision": info.sha, "split": "train", "gated": info.gated,
                        "license": (info.card_data or {}).get("license"), "files": files,
                        "full_download_bytes": sum(item["bytes"] for item in files)})
    return {"sources": entries, "selection": "six source train splits; ordered first-k retrieval IDs",
            "acquisition": "range reads of selected Parquet row groups, with decoded-record cache",
            "range_cost_note": "a row group has 1000 rows; actual transfer exceeds selected passage text"}


class ParquetSource:
    def __init__(self, repo, revision, cache):
        self.repo, self.revision = repo, revision
        self.cache = Path(cache) / repo.replace("/", "--") / revision
        self.cache.mkdir(parents=True, exist_ok=True)

    def _open(self, name):
        from huggingface_hub import HfFileSystem
        import pyarrow.parquet as pq

        stream = HfFileSystem().open(f"datasets/{self.repo}@{self.revision}/{name}",
                                   "rb", block_size=65536)
        return stream, pq.ParquetFile(stream)

    def metadata(self, name):
        path = self.cache / (name.replace("/", "--") + ".metadata.json")
        if path.is_file():
            return json.loads(path.read_text())
        stream, parquet = self._open(name)
        try:
            result = {"path": name, "rows": parquet.metadata.num_rows,
                      "row_groups": [parquet.metadata.row_group(i).num_rows for i in range(parquet.num_row_groups)],
                      "columns": parquet.schema_arrow.names}
        finally:
            stream.close()
        write_json(path, result)
        return result

    def rows(self, name, indices):
        metadata = self.metadata(name)
        starts, cursor = [], 0
        for count in metadata["row_groups"]:
            starts.append(cursor)
            cursor += count
        wanted, result = {}, {}
        for index in indices:
            if not 0 <= index < cursor:
                raise ValueError(f"Row {index} is outside {self.repo}/{name}")
            group = bisect_right(starts, index) - 1
            wanted.setdefault(group, []).append(index)
        stream, parquet = self._open(name)
        try:
            for group, group_indices in wanted.items():
                table = parquet.read_row_group(group)
                for index in group_indices:
                    result[index] = table.slice(index - starts[group], 1).to_pylist()[0]
        finally:
            stream.close()
        return result


def _fetch_corpus_rows(request):
    """Isolate Hub's shared HTTP client to one process per shard worker.

    Hub resets its process-global client on connection errors. Sharing that
    client between concurrent range-read threads can close their live sockets.
    """
    cache, name, indices, start = request
    corpus = ParquetSource(CORPUS_REPO, CORPUS_REVISION, cache)
    found = corpus.rows(name, indices)
    result = {}
    for row_index, row in found.items():
        expected = str(start + row_index)
        if row["id"] != expected:
            raise ValueError("Corpus positional-ID assumption does not hold")
        value = {**row, "source": CORPUS_REPO, "revision": CORPUS_REVISION,
                 "split": "train", "source_file": name, "source_row": row_index}
        write_json(corpus.cache / f"passage-{expected}.json", value)
        result[expected] = value
    print(json.dumps({"acquired_shard": name, "new_passages": len(result)}), flush=True)
    return result


def acquire(cache, *, records_per_source=16, passages_per_query=6, workers=6):
    """Deterministic prefix sample, preserving published retrieval order.

    Passage IDs are checked against their actual row IDs. If the corpus ceases
    to use positional IDs, fail rather than retrieving an unrelated passage.
    """
    if records_per_source < 1 or passages_per_query < 2:
        raise ValueError("Need positive records_per_source and at least two passages")
    cache = Path(cache)
    key = f"queries-{records_per_source}-passages-{passages_per_query}.json"
    query_source = ParquetSource(QUERY_REPO, QUERY_REVISION, cache)
    corpus = ParquetSource(CORPUS_REPO, CORPUS_REVISION, cache)
    acquired = cache / (QUERY_REVISION + "-" + CORPUS_REVISION + "-" + key)
    if acquired.is_file():
        return json.loads(acquired.read_text())

    def queries(config):
        name = config + "/train-00000-of-0000" + ("2" if config == "eli5" else "1") + ".parquet"
        rows = query_source.rows(name, range(records_per_source))
        output = []
        for index, row in sorted(rows.items()):
            if not all(k in row for k in ("query", "query_id", "context_doc_ids", "answers")):
                raise ValueError(f"Unexpected CoRAG query schema in {name}")
            output.append({"source": QUERY_REPO, "revision": QUERY_REVISION, "config": config,
                           "split": "train", "source_file": name, "source_row": index,
                           "query_id": str(row["query_id"]), "query": row["query"], "answers": row["answers"],
                           "passage_ids": row["context_doc_ids"][:passages_per_query]})
        return output

    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = [row for group in pool.map(queries, SOURCE_CONFIGS) for row in group]
    names = [f"data/train-{i:05d}-of-00037.parquet" for i in range(37)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        metadata = list(pool.map(corpus.metadata, names))
    starts, cursor = [], 0
    for item in metadata:
        if item["columns"] != ["id", "contents", "title", "wikipedia_id"]:
            raise ValueError("Unexpected CoRAG passage schema")
        starts.append(cursor)
        cursor += item["rows"]
    requested = sorted({str(identifier) for row in records for identifier in row["passage_ids"]}, key=int)
    passages, groups = {}, {}
    for identifier in requested:
        cached = corpus.cache / f"passage-{identifier}.json"
        if cached.is_file():
            passages[identifier] = json.loads(cached.read_text())
            continue
        index = int(identifier)
        if not 0 <= index < cursor:
            raise ValueError(f"Passage ID {identifier} is outside the pinned corpus")
        shard = bisect_right(starts, index) - 1
        groups.setdefault(shard, []).append(index - starts[shard])

    requests = [(str(cache), names[shard], indices, starts[shard]) for shard, indices in groups.items()]
    # spawn avoids inheriting active native-library threads and HTTP sockets.
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for group in pool.map(_fetch_corpus_rows, requests):
            passages.update(group)
    result = {"records": records, "passages": passages,
              "policy": "first-N source train records; first-k published retrieval IDs; no shuffle within records"}
    write_json(acquired, result)
    return result
