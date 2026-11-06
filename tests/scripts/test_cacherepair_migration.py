"""An audited source update must preserve numerical state and fail closed."""
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from scripts.MigrateCacheRepairTraining import audited_identity, digest, migrate, same_tree


@pytest.fixture
def transition(tmp_path):
    root, audit = tmp_path / "repo", tmp_path / "audit"
    source, destination = tmp_path / "old/training", tmp_path / "new/training"
    name = "methods/CacheRepair/qwen38/train.py"
    shared = "methods/CacheRepair/backend/cacherepair/model.py"
    for base in (root, audit / "candidate"):
        for relative, contents in ((name, "reviewed update"), (shared, "unchanged repair blocks")):
            path = base / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents)
    before = tmp_path / "before.py"
    before.write_text("old control flow")
    (audit / "candidate-sha256.json").write_text(json.dumps({name: {"before": digest(before), "candidate": digest(root / name)}}))
    (audit / "full-candidate-tests.log").write_text("73 passed\n")
    identity = {"train_size": 8, "schedule_epochs": 2, "implementation_sha256": {name: digest(before)}}
    checkpoint = source / "resume-checkpoints/step-00000001"
    checkpoint.mkdir(parents=True)
    stats = {key: torch.ones(1, 1, 2, 4) for key in ("sigma_stale", "sigma_delta")}
    save_file({**stats, "weight": torch.arange(4.)}, str(checkpoint / "model.safetensors"))
    (checkpoint / "config.json").write_text("{}")
    state = {"format": "kvbench-qwen38-trainer-v1", "identity": identity, "step": 1, "epoch": 0,
        "offset": 4, "horizon": 4, "optimizer": {"state": {1: {"exp_avg": torch.tensor([.2, .5])}}},
        "torch_rng": torch.get_rng_state(), "cuda_rng": [], "python_rng": (1, 2, 3),
        "model_sha256": digest(checkpoint / "model.safetensors")}
    torch.save(state, checkpoint / "trainer.pt")
    (source / "run-identity.json").write_text(json.dumps(identity))
    (source / "status.json").write_text('{"status":"interrupted"}')
    (source / "latest.json").write_text(json.dumps({"checkpoint": str(checkpoint), "step": 1}))
    (source / "loss.jsonl").write_text('{"step":1,"mse":0.5}\n')
    save_file(stats, str(source / "statistics.safetensors"))
    return root, audit, source, destination, state


def test_migration_preserves_optimizer_rng_model_rms_and_data_cursor(transition):
    root, audit, source, destination, before = transition
    receipt = migrate(source, destination, audit, root=root)
    after = torch.load(destination / "resume-checkpoints/step-00000001/trainer.pt", weights_only=True)
    assert same_tree({k: v for k, v in before.items() if k != "identity"}, {k: v for k, v in after.items() if k != "identity"})
    assert after["identity"] != before["identity"]
    assert receipt["optimizer_rng_cursor_bitwise_preserved"] and not receipt["rms_recomputed"]
    for name in ("loss.jsonl", "statistics.safetensors"):
        assert (source / name).read_bytes() == (destination / name).read_bytes()
    assert json.loads((destination / "latest.json").read_text())["checkpoint"].startswith(str(destination))
    with pytest.raises(FileExistsError):
        migrate(source, destination, audit, root=root)


def test_migration_rejects_running_source(transition):
    root, audit, source, destination, _ = transition
    (source / "status.json").write_text('{"status":"training"}')
    with pytest.raises(ValueError, match="stopped"):
        migrate(source, destination, audit, root=root)
    assert not destination.exists()


def test_migration_rejects_unreviewed_source_change(transition):
    root, audit, source, destination, _ = transition
    (root / "methods/CacheRepair/qwen38/train.py").write_text("unexpected change")
    with pytest.raises(ValueError, match="reviewed"):
        migrate(source, destination, audit, root=root)
    assert not destination.exists()


def test_migration_rejects_damaged_model(transition):
    root, audit, source, destination, _ = transition
    with (source / "resume-checkpoints/step-00000001/model.safetensors").open("ab") as stream:
        stream.write(b"damaged")
    with pytest.raises(ValueError, match="checksum"):
        migrate(source, destination, audit, root=root)
    assert not destination.exists()


def test_execution_change_requires_matching_numerical_evidence(transition):
    import copy
    root, audit, _, _, state = transition
    identity = copy.deepcopy(state["identity"])
    identity["backend_spec"] = {"pipeline_depth": 3, "root_training_pipeline_depth": 2,
        "attention_backend": "sdpa"}
    proof = {"status": "passed", "optimizer_update_bitwise_equal": True,
        "root_depth_before": 2, "root_depth_after": 3,
        "implementation_sha256": {"methods/CacheRepair/qwen38/train.py": digest(root / "methods/CacheRepair/qwen38/train.py")}}
    (audit / "execution-update.json").write_text(json.dumps(proof))
    updated, _ = audited_identity(identity, audit, root)
    assert updated["backend_spec"] == {**identity["backend_spec"], "root_training_pipeline_depth": 3}
    proof["optimizer_update_bitwise_equal"] = False
    (audit / "execution-update.json").write_text(json.dumps(proof))
    with pytest.raises(ValueError, match="numerical evidence"):
        audited_identity(identity, audit, root)
