"""CLI readiness must work with a not-yet-created nested output directory."""
import hashlib
import json

from tests.backends.test_cacherepair_qwen38_native import native


def test_pool_readiness_creates_nested_output_parent(native, tmp_path, monkeypatch):
    from methods.CacheRepair.qwen38 import train, pair_pool
    target, _ = native
    data = tmp_path / "data"
    data.mkdir()
    row = {"chunks": [[1, 2], [3]]}
    files = {}
    for split in ("train", "validation"):
        path = data / f"{split}.jsonl"
        path.write_text(json.dumps(row) + "\n")
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (data / "dataset.json").write_text(json.dumps({"tokenizer": {}, "files_sha256": files}))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"target_model_id": "tiny-qwen", "expected_kv_shape": [2, 2, 2, 16],
        "repair_config": {"d_b": 16, "n_heads": 2, "n_blocks": 1, "d_seg": 2},
        "training": {"seed": 7, "initialization_seed": 1, "epochs": 1, "schedule_epochs": 1}}))
    output = tmp_path / "new" / "nested" / "training"
    monkeypatch.setattr(train.sys, "argv", ["train", "--model", "tiny", "--dataset", str(data),
        "--config", str(config), "--output", str(output), "--gpus", "0,1", "--pair-groups", "0,1"])
    monkeypatch.setattr(train, "local_tokenizer_spec", lambda _: {})
    monkeypatch.setattr(train, "local_checkpoint_spec", lambda _: {})
    monkeypatch.setattr(train, "load_native_target", lambda *a, **kw: target)
    monkeypatch.setattr(train.torch.cuda, "set_device", lambda *_: None)

    class Pool:
        def __init__(self, *a, **kw):
            pass
        def close(self):
            pass

    def run(*args, **kwargs):
        assert (output.parent / "resources-ready.json").is_file()
        return {"interrupted": True, "latest_restart": None}

    monkeypatch.setattr(pair_pool, "PairPool", Pool)
    monkeypatch.setattr(train, "run_upstream", run)
    train.main()
