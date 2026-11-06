"""RMS prefix reuse requires unchanged semantics and matching equivalence proof."""
import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from methods.CacheRepair.qwen38.resume import Restart, fingerprint, validated_statistics_prefix


def test_import_requires_matching_data_target_code_and_real_target_evidence(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    geometry = {"full_attention_layer_indices": [1], "n_kv_heads": 2, "head_dim": 4}
    previous = {"target_config": geometry, "train_sha256": "same-rows", "implementation_sha256": {"old": "code"},
                "backend_spec": {"attention_backend": "sdpa", "pack_chunks": True}}
    identity = copy.deepcopy(previous)
    identity["implementation_sha256"] = {"new": "code"}
    identity["backend_spec"]["pipeline_depth"] = 3
    rows = [{"chunks": [[1, 2], [3]]}, {"chunks": [[4], [5, 6]]}]
    (source / "run-identity.json").write_text(json.dumps(previous))
    sums = {key: torch.ones(1, 2, 2, 4, dtype=torch.float64) for key in ("sigma_stale", "sigma_delta")}
    torch.save({"identity": fingerprint({"run": previous, "rows": rows}), "index": 1, "count": 3, "sums": sums},
               source / "statistics-progress.pt")
    import methods.CacheRepair.qwen38.resume as module
    proof = {"status": "passed", "kv_bitwise_equal": True, "rms_sums_bitwise_equal": True,
        "target_geometry": geometry, "source_run_identity_sha256": fingerprint(previous),
        "source_progress_sha256": hashlib.sha256((source / "statistics-progress.pt").read_bytes()).hexdigest(),
        "pipeline_source_sha256": hashlib.sha256((Path(module.__file__).parent / "experimental/pipeline.py").read_bytes()).hexdigest()}
    report = tmp_path / "proof.json"
    report.write_text(json.dumps(proof))
    actual = validated_statistics_prefix(source, identity, rows, report)
    assert actual["index"] == 1 and actual["count"] == 3
    for key in sums:
        assert torch.equal(actual["sums"][key], sums[key])
    different = copy.deepcopy(identity)
    different["train_sha256"] = "different-data"
    with pytest.raises(ValueError, match="differs"):
        validated_statistics_prefix(source, different, rows, report)
    proof["kv_bitwise_equal"] = False
    report.write_text(json.dumps(proof))
    with pytest.raises(ValueError, match="evidence"):
        validated_statistics_prefix(source, identity, rows, report)


def test_checkpoint_interval_is_relative_to_imported_cursor(tmp_path):
    rows = [{"chunks": [[1], [2]]} for _ in range(200)]
    torch.save({"identity": fingerprint({"run": {}, "rows": rows}), "index": 80, "count": 160, "sums": {}},
               tmp_path / "statistics-progress.pt")
    restart = Restart(tmp_path, {}, None, "tiny")
    assert restart.restore_statistics(rows)[2] == 80
    restart.after_statistics(180, {}, 360)
    assert torch.load(tmp_path / "statistics-progress.pt", weights_only=True)["index"] == 180
