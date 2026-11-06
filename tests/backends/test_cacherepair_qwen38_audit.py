"""Fault injection at the RMS/optimizer/epoch-checkpoint commit boundaries."""
import json
import shutil

import pytest
import torch
from safetensors.torch import load_file

from methods.CacheRepair.qwen38.model import RepairConfig
from methods.CacheRepair.qwen38.train import run_upstream
from tests.backends.test_cacherepair_qwen38_native import native


def inputs():
    rows = [{"chunks": [[i + 1, 2], [3, i + 5]]} for i in range(8)]
    options = dict(model_id="tiny-qwen", device="cpu", epochs=2, schedule_epochs=2,
        repair_config=RepairConfig(d_b=16, n_heads=2, n_blocks=1, d_seg=2),
        stats_examples=2, validate_train=False, checkpoint_steps=1, keep_restarts=4)
    return rows, options


def assert_same_result(reference, recovered, step=4):
    assert (reference / "loss.jsonl").read_bytes() == (recovered / "loss.jsonl").read_bytes()
    expected = load_file(str(reference / f"checkpoint-{step}/model.safetensors"))
    actual = load_file(str(recovered / f"checkpoint-{step}/model.safetensors"))
    assert all(torch.equal(expected[key], actual[key]) for key in expected)


@pytest.mark.parametrize("previous_pointer", [False, True])
def test_resume_after_committed_directory_before_latest_pointer(native, tmp_path, previous_pointer):
    target, _ = native
    rows, options = inputs()
    full, split = tmp_path / "full", tmp_path / "split"
    run_upstream(target, rows, rows[:2], full, **options)
    stop = 2 if previous_pointer else 1
    run_upstream(target, rows, rows[:2], split, stop_after_steps=stop, **options)
    pointer = split / "latest.json"
    if previous_pointer:
        pointer.write_text(json.dumps({"checkpoint": str(split / "resume-checkpoints/step-00000001"), "step": 1}))
    else:
        pointer.unlink()
    # The renamed, fsynced checkpoint exists, but publishing latest was lost.
    run_upstream(target, rows, rows[:2], split, resume_from="auto", **options)
    assert_same_result(full, split)


def test_resume_recovers_torn_rms_final_file_before_first_committed_update(native, tmp_path):
    target, _ = native
    rows, options = inputs()
    full, split = tmp_path / "full", tmp_path / "split"
    run_upstream(target, rows, rows[:2], full, **options)
    run_upstream(target, rows, rows[:2], split, stop_after_steps=1, **options)
    (split / "latest.json").unlink()
    shutil.rmtree(split / "resume-checkpoints")
    (split / "statistics.safetensors").write_bytes(b"interrupted safetensors write")
    run_upstream(target, rows, rows[:2], split, resume_from="auto", **options)
    assert_same_result(full, split)


@pytest.mark.parametrize("partial", ["empty", "weights_without_validation"])
def test_resume_finishes_partial_epoch_checkpoint(native, tmp_path, partial):
    target, _ = native
    rows, options = inputs()
    output = tmp_path / "training"
    run_upstream(target, rows, rows[:2], output, stop_after_steps=2, **options)
    checkpoint = output / "checkpoint-2"
    checkpoint.mkdir()
    if partial == "weights_without_validation":
        restart = output / "resume-checkpoints/step-00000002"
        for name in ("config.json", "model.safetensors"):
            shutil.copyfile(restart / name, checkpoint / name)
    report = run_upstream(target, rows, rows[:2], output, resume_from="auto", **options)
    assert (checkpoint / "config.json").is_file()
    assert (checkpoint / "model.safetensors").is_file()
    assert any(row["path"] == str(checkpoint) and row["reload_exact"] for row in report["checkpoints"])
    assert json.loads((checkpoint / "validation.json").read_text())["reload_exact"]


@pytest.mark.parametrize("damage", ["identity", "weights"])
def test_orphan_recovery_rejects_damaged_committed_state(native, tmp_path, damage):
    target, _ = native
    rows, options = inputs()
    output = tmp_path / "training"
    run_upstream(target, rows, rows[:2], output, stop_after_steps=1, **options)
    (output / "latest.json").unlink()
    checkpoint = output / "resume-checkpoints/step-00000001"
    if damage == "identity":
        path = checkpoint / "trainer.pt"
        state = torch.load(path, weights_only=True)
        state["identity"]["train_size"] += 1
        torch.save(state, path)
    else:
        with (checkpoint / "model.safetensors").open("ab") as stream:
            stream.write(b"damaged")
    with pytest.raises(ValueError, match="differs"):
        run_upstream(target, rows, rows[:2], output, resume_from="auto", **options)
    assert not (output / "latest.json").exists()


def test_interrupted_rms_commit_preserves_previous_final_file(tmp_path, monkeypatch):
    from methods.CacheRepair.qwen38 import resume
    path = tmp_path / "statistics.safetensors"
    values = {key: torch.ones(1, 1, 2, 4) for key in ("sigma_stale", "sigma_delta")}
    restart = resume.Restart(tmp_path, {}, None, "unused")
    restart.save_statistics(values, path)
    original = path.read_bytes()

    def interrupted_save(values, temporary):
        from pathlib import Path
        Path(temporary).write_bytes(b"incomplete write")
        raise RuntimeError("injected write failure")

    monkeypatch.setattr(resume, "save_file", interrupted_save)
    with pytest.raises(RuntimeError, match="injected"):
        restart.save_statistics(values, path)
    assert path.read_bytes() == original


def test_nonfinite_gradient_is_rejected_before_optimizer_commit(native, tmp_path, monkeypatch):
    from methods.CacheRepair.qwen38 import train
    target, _ = native
    rows, options = inputs()
    original = train.QwenCacheRepair.__init__

    def poison(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.v_head.weight.register_hook(lambda gradient: torch.full_like(gradient, float("nan")))

    monkeypatch.setattr(train.QwenCacheRepair, "__init__", poison)
    output = tmp_path / "training"
    with pytest.raises(FloatingPointError, match="nonfinite"):
        run_upstream(target, rows, rows[:2], output, **options)
    assert not (output / "latest.json").exists()
    assert (output / "loss.jsonl").read_text() == ""


def test_shared_official_repair_blocks_are_source_pinned(native, tmp_path):
    import hashlib
    import inspect
    from pathlib import Path
    from methods.CacheRepair.qwen38.train import OfficialRepairer
    from scripts import RunCacheRepairTrainingFast as runner
    target, _ = native
    rows, options = inputs()
    output = tmp_path / "training"
    run_upstream(target, rows, rows[:2], output, stop_after_steps=1, **options)
    source = Path(inspect.getfile(OfficialRepairer))
    identity = json.loads((output / "run-identity.json").read_text())
    assert identity["implementation_sha256"]["methods/CacheRepair/backend/cacherepair/model.py"] == hashlib.sha256(source.read_bytes()).hexdigest()
    root = Path(runner.__file__).parent.parent
    assert root / "methods/CacheRepair/backend/cacherepair/model.py" in runner.protected_source_paths(root, 3)
