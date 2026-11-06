"""Dataset provenance and official RULER process-boundary regressions."""

import hashlib
import io
import json
import os
from pathlib import Path
import zipfile

import pytest
import yaml

from scripts import PrepareDataset as preparation


@pytest.fixture
def official_ruler(tmp_path, monkeypatch):
    """Small upstream stand-in; run it as a subprocess, without network/packages."""
    revision = "a" * 40
    # This executable deliberately emits different bytes for validation/test.
    generator = '''import argparse, json
from pathlib import Path
p = argparse.ArgumentParser()
for name in ("save_dir", "save_name", "subset", "tokenizer_path", "tokenizer_type",
             "max_seq_length", "tokens_to_generate", "num_samples", "random_seed", "template"):
    p.add_argument("--" + name, required=True)
args = p.parse_args()
output = Path(args.save_dir) / args.save_name / (args.subset + ".jsonl")
output.parent.mkdir(parents=True, exist_ok=True)
row = {"input": args.subset + ":" + args.tokenizer_path, "outputs": ["gold"],
       "answer_prefix": " Answer:", "length": int(args.max_seq_length), "upstream_field": 17}
output.write_text((json.dumps(row, separators=(",", ":")) + "\\n") * int(args.num_samples))
'''
    definitions = {
        "niah_single_1": {"task": "niah", "args": {}},
        "vt": {"task": "variable_tracking", "args": {}},
        "cwe": {"task": "common_words_extraction", "args": {}},
    }
    constants = {entry["task"]: {"template": "{context}", "tokens_to_generate": 2}
                 for entry in definitions.values()}
    words = b'{"0": "alpha", "1": "beta"}'
    word_hash = hashlib.sha256(words).hexdigest()
    members = {
        "data/prepare.py": "# official entry point\n",
        "synthetic.yaml": yaml.safe_dump(definitions),
        "data/synthetic/constants.py": "TASKS = " + repr(constants),
        "data/synthetic/json/english_words.json":
            f"version https://git-lfs.github.com/spec/v1\noid sha256:{word_hash}\nsize {len(words)}\n",
        **{f"data/synthetic/{entry['task']}.py": generator for entry in definitions.values()},
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, body in members.items():
            archive.writestr(f"RULER-{revision}/scripts/{name}", body)
    source = buffer.getvalue()
    monkeypatch.setattr(preparation, "ProjectRoot", tmp_path / "project")
    monkeypatch.setattr(preparation, "_download",
                        lambda url: source if "codeload.github.com" in url else words)
    monkeypatch.setattr(preparation, "_ruler_environment", lambda cache: dict(os.environ))
    return {
        "Revision": revision, "ArchiveSha256": hashlib.sha256(source).hexdigest(),
        "Split": "test", "Tokenizer": "model name with spaces",
        "Tasks": [{"Task": task, "OfficialTask": official, "Lengths": [32, 64],
                   "Samples": 2, "Seed": 42}
                  for task, official in (("niah", "niah_single_1"), ("vt", "vt"), ("cwe", "cwe"))],
    }


def test_official_ruler_runs_test_generators_and_preserves_bytes(official_ruler, tmp_path):
    stage = tmp_path / "stage"
    preparation._prepare_ruler(official_ruler, stage, {})
    provenance = json.loads((stage / "ruler/provenance.json").read_text())
    assert provenance["split"] == "test"
    assert len(provenance["files"]) == 6
    for entry in provenance["files"]:
        body = (stage / "ruler" / entry["file"]).read_bytes()
        rows = [json.loads(line) for line in body.splitlines()]
        assert len(rows) == 2
        assert all(row["input"] == "test:model name with spaces" for row in rows)
        assert rows[0]["upstream_field"] == 17
        assert b'": "' not in body  # no reserialization of upstream records
        assert hashlib.sha256(body).hexdigest() == entry["sha256"]


def test_ruler_failure_keeps_installed_data(official_ruler, tmp_path, monkeypatch):
    root = tmp_path / "data/ruler"
    root.mkdir(parents=True)
    old = root / "niah_len32.jsonl"
    old.write_bytes(b"installed-data\n")
    config = {"DatasetPath": str(root.parent), "DatasetPreparation": {
        "DatasetDirectories": ["ruler"], "Direct": [], "HuggingFace": [], "Ruler": official_ruler,
    }}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))

    def wrong_split(command, cwd, env):
        output = Path(command[command.index("--save_dir") + 1])
        name = command[command.index("--save_name") + 1]
        (output / name).mkdir(parents=True, exist_ok=True)
        (output / name / "validation.jsonl").write_text("{}\n")

    monkeypatch.setattr(preparation, "_run_ruler", wrong_split)
    with pytest.raises(RuntimeError, match="requested test file"):
        preparation.prepare(path)
    assert old.read_bytes() == b"installed-data\n"
    assert list(tmp_path.glob(".data.prepare-*")) == []


@pytest.mark.parametrize("change", [
    {"outputs": []}, {"outputs": [None]}, {"length": 65}, {"input": ""},
])
def test_reject_invalid_upstream_ruler_records(tmp_path, change):
    row = {"input": "prompt", "outputs": ["gold"], "length": 32, **change}
    path = tmp_path / "test.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError):
        preparation._validate_ruler_output(path, 1, 64)


def test_corrupt_ruler_source_is_not_executed(official_ruler, tmp_path):
    official_ruler["ArchiveSha256"] = "0" * 64
    with pytest.raises(ValueError, match="checksum mismatch"):
        preparation._prepare_ruler(official_ruler, tmp_path / "stage", {})


def test_official_essay_download_failures_are_recorded(official_ruler, tmp_path, monkeypatch):
    from scripts import PrepareRulerCorpus

    cache = tmp_path / "cache"
    scripts = preparation._ruler_source(official_ruler, tmp_path / "source", cache)
    assets = scripts / "data/synthetic/json"
    (assets / "PaulGrahamEssays_URLs.txt").write_text("http://example.org/one.html\nhttps://example.org/missing.txt\n")
    manifest = cache / "corpus_sources_manifest.json"
    manifest.write_text(json.dumps({"sources": [{}, {}]}))
    monkeypatch.setattr(PrepareRulerCorpus, "prepare_sources", lambda urls, cache: manifest)

    def download(command, cwd, env):
        (cwd / "PaulGrahamEssays.json").write_text('{"text": "An official essay."}')
        return "Fail download missing.txt, (network error)\nDownload 0 essays from repo\nDownload 1 essays from html\n"

    monkeypatch.setattr(preparation, "_run_ruler", download)
    with pytest.raises(RuntimeError, match="refusing a partial corpus"):
        preparation._ruler_assets(scripts, cache, official_ruler["Revision"], True, {})
    assert "network error" in (cache / "essay_download.log").read_text()
    assert not (cache / "PaulGrahamEssays.json").exists()


def test_real_sources_require_test_or_documented_exception():
    config = preparation._load_config(preparation.DefaultConfigPath)
    preparation._validate_preparation_config(config)
    sources = config["DatasetPreparation"]["HuggingFace"]
    next(entry for entry in sources if entry["Adapter"] == "multinews")["Split"] = "train"
    with pytest.raises(ValueError, match="SplitReason"):
        preparation._validate_preparation_config(config)


def test_hidden_labels_cannot_be_scored_as_literal_placeholders():
    with pytest.raises(ValueError, match="private labels"):
        preparation._adapt_musique_ans({"question": "Test question", "paragraphs": []})
    with pytest.raises(ValueError, match="no public answer"):
        preparation._answers_from_trivia({"answer": {"value": "<unk>", "aliases": []}})


def test_musique_archive_member_must_match_split(tmp_path):
    with pytest.raises(ValueError, match="must match"):
        preparation._prepare_direct({
            "Output": "musique/test.jsonl", "Format": "musique_ans_google_drive_zip",
            "GoogleDriveId": "source", "Split": "test", "ArchiveFile": "data/musique_ans_v1.0_dev.jsonl",
        }, tmp_path)


def test_2wiki_dev_adapter_cannot_be_relabelled_as_test(tmp_path, monkeypatch):
    monkeypatch.setattr(preparation, "_download", lambda url: b"not-needed")
    with pytest.raises(ValueError, match="requires Split=dev"):
        preparation._prepare_direct({
            "Output": "2wikimultihopqa/test.json", "Format": "2wikimultihopqa_dev_zip",
            "Url": "https://example.org/data.zip", "Split": "test",
            "ArchiveFile": "data_ids/dev.json",
        }, tmp_path)


def test_source_metadata_does_not_become_a_qa_input(tmp_path):
    entry = {"Output": "samsum/samsum_test.jsonl", "Split": "test"}
    preparation._record_source(tmp_path, entry, "Direct")
    assert list((tmp_path / "samsum").glob("*.json")) == []
    assert json.loads((tmp_path / "samsum/_metadata/source.json").read_text())["Split"] == "test"


def test_installed_data_is_verified_and_tampering_is_detected(official_ruler, tmp_path):
    config = {"DatasetPath": str(tmp_path / "data"), "DatasetPreparation": {
        "DatasetDirectories": ["ruler"], "Direct": [], "HuggingFace": [], "Ruler": official_ruler,
    }}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    root = preparation.prepare(path)
    manifest = preparation.verify(path)
    assert len(manifest["files"]) == 7
    target = root / "ruler/niah_len32.jsonl"
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="integrity verification"):
        preparation.verify(path)


def test_installation_failure_restores_all_previous_datasets(tmp_path, monkeypatch):
    stage, root = tmp_path / "stage", tmp_path / "data"
    for base, text in ((stage, "new"), (root, "old")):
        for name in ("alpha", "beta"):
            (base / name).mkdir(parents=True)
            (base / name / "data.txt").write_text(text)
        (base / "preparation.json").write_text(text)
    original = Path.rename

    def fail_second(self, target):
        if self == stage / "beta":
            raise OSError("simulated installation failure")
        return original(self, target)

    monkeypatch.setattr(Path, "rename", fail_second)
    with pytest.raises(OSError, match="simulated"):
        preparation._install_prepared(stage, root, ["alpha", "beta"])
    assert (root / "alpha/data.txt").read_text() == "old"
    assert (root / "beta/data.txt").read_text() == "old"
    assert (root / "preparation.json").read_text() == "old"


def test_modified_cached_upstream_source_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(preparation, "ProjectRoot", tmp_path)
    url = "https://example.org/source.zip"
    cache = preparation._source_cache(url)
    cache.mkdir(parents=True)
    payload = cache / "payload"
    payload.write_bytes(b"original")
    (cache / "source.json").write_text(json.dumps({"url": url, "sha256": hashlib.sha256(b"original").hexdigest()}))
    assert preparation._download_file(url) == payload
    payload.write_bytes(b"modified")
    with pytest.raises(ValueError, match="cached source was modified"):
        preparation._download_file(url)


def test_cached_corpus_render_rejects_tampered_source(tmp_path):
    from scripts.PrepareRulerCorpus import render

    raw = tmp_path / "essay.raw"
    raw.write_bytes(b"modified")
    manifest = tmp_path / "sources.json"
    manifest.write_text(json.dumps({"sources": [{"url": "https://example.org/essay", "file": str(raw),
                                               "sha256": hashlib.sha256(b"original").hexdigest()}]}))
    with pytest.raises(ValueError, match="cache was modified"):
        render(tmp_path / "not_executed.py", manifest)
