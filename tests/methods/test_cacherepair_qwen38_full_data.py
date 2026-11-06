"""Full-scale policy checks on small source-balanced fixtures."""
from collections import Counter
import json
import sqlite3

import pytest

from methods.CacheRepair.qwen38.full_data import SHARES, compile_full
from tests.methods.test_cacherepair_qwen38_data import Words


def test_full_manifest_ratios_bands_and_source_isolation():
    records, passages = [], {}
    for config, share in SHARES.items():
        for index in range(share * 30):
            ids = []
            for chunk in range(20):
                identifier = f"{config}-{index}-{chunk}"
                ids.append(identifier)
                passages[identifier] = {"title": identifier,
                    "contents": " ".join(f"{identifier}-word{i}" for i in range(120)),
                    "wikipedia_id": f"{config}-{index}", "source_file": "corpus", "source_row": chunk}
            records.append({"source": "corag/kilt", "revision": "pinned", "config": config, "split": "train",
                            "source_file": "queries", "source_row": index, "query_id": f"{config}-{index}",
                            "query": f"query {config} {index}", "answers": [],
                            "passage_ids": ids, "original_passage_ids": ids})
    examples, manifest = compile_full(records, passages, Words(), train_size=100, validation_size=100)
    for split in ("train", "validation"):
        selected = [r for r in manifest if r["split"] == split]
        assert len(examples[split]) == len(selected) == 100
        assert Counter(r["band"] for r in selected) == {"short": 10, "medium": 10, "long": 80}
        assert Counter(r["config"] for r in selected) == {c: s * 10 for c, s in SHARES.items()}
        for row in selected:
            low, high = {"short": (1, 1024), "medium": (1025, 2048), "long": (2049, 4096)}[row["band"]]
            assert low <= row["token_count"] <= high
            assert 2 <= row["chunk_count"] <= 20
    docs = {s: {p["wikipedia_id"] for r in manifest if r["split"] == s for p in r["passages"]}
            for s in examples}
    assert not docs["train"] & docs["validation"]


def test_full_acquisition_tracks_original_shard_rows_and_reuses_cache(tmp_path, monkeypatch):
    import huggingface_hub
    import pyarrow as pa
    import pyarrow.parquet as pq
    from methods.CacheRepair.qwen38 import full_data as module
    raw, cache = tmp_path / "raw", tmp_path / "cache"
    raw.mkdir()
    resources = []
    for repo, revision in ((module.QUERY_REPO, module.QUERY_REVISION), (module.CORPUS_REPO, module.CORPUS_REVISION)):
        files = []
        if repo == module.QUERY_REPO:
            for config in SHARES:
                for shard in range(2 if config == "eli5" else 1):
                    name = f"{config}/train-{shard:05d}.parquet"
                    path = raw / repo.replace("/", "--") / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    table = pa.Table.from_pylist([{"query": f"{config}:{shard}:{i}", "query_id": f"{shard}:{i}",
                        "context_doc_ids": list(map(str, range(20))), "answers": []} for i in range(6)])
                    pq.write_table(table, path, row_group_size=2)
                    files.append({"path": name, "bytes": path.stat().st_size})
        else:
            for shard in range(2):
                name = f"data/train-{shard:05d}.parquet"
                path = raw / repo.replace("/", "--") / name
                path.parent.mkdir(parents=True, exist_ok=True)
                table = pa.Table.from_pylist([{"id": str(shard * 10 + i), "contents": f"Passage {i}",
                                             "title": str(i), "wikipedia_id": str(shard * 10 + i)} for i in range(10)])
                pq.write_table(table, path, row_group_size=3)
                files.append({"path": name, "bytes": path.stat().st_size})
        resources.append({"id": repo, "revision": revision, "files": files})
    monkeypatch.setattr(module, "inventory", lambda: {"sources": resources})
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo, name, **kw: str(raw / repo.replace("/", "--") / name))
    module.acquire_full(cache, candidates=10, workers=2)
    rows = json.loads((cache / "queries.json").read_text())
    assert len(rows) == 10
    for row in rows:
        path = raw / module.QUERY_REPO.replace("/", "--") / row["source_file"]
        original = pq.read_table(path).slice(row["source_row"], 1).to_pylist()[0]
        assert row["query"] == original["query"]
    with sqlite3.connect(cache / "passages.sqlite") as db:
        assert db.execute("SELECT COUNT(*) FROM passages").fetchone()[0] == 20
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **kw: pytest.fail("Cache was not reused"))
    module.acquire_full(cache, candidates=10, workers=2)
    with pytest.raises(ValueError, match="different selection"):
        module.acquire_full(cache, candidates=20, workers=2)
