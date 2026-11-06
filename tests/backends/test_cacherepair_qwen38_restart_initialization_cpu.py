"""Tiny CPU regression for init-from recovery before the first update."""
import json
import shutil
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from methods.CacheRepair.qwen38.target import TargetGeometry
from methods.CacheRepair.qwen38.model import RepairConfig
from methods.CacheRepair.qwen38.train import run_upstream


class _TinyTarget(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(16, 8)
        self.config = SimpleNamespace(model_type="tiny")

    def get_input_embeddings(self):
        return self.embed_tokens


class _TinyPairPool:
    asynchronous = False

    def __init__(self):
        self.last_event = {"compute_seconds": 0.001}

    def set_sequence(self, rows):
        self.rows = rows

    def get(self, chunks):
        tokens = torch.tensor([token for chunk in chunks for token in chunk], dtype=torch.long).view(1, -1)
        ids = torch.cat([torch.full((len(chunk),), i, dtype=torch.long) for i, chunk in enumerate(chunks)]).view(1, -1)
        length = tokens.shape[1]
        stale = torch.arange(length * 4, dtype=torch.float32).view(1, length, 1, 1, 2, 2) / 20
        joint = stale + 0.125
        self.last_event = {"compute_seconds": 0.001}
        return {"stale": stale, "joint": joint, "tokens": tokens, "chunk_ids": ids}


def _install_tiny_target_geometry(monkeypatch):
    geometry = TargetGeometry(target_model_id="tiny-qwen", config_sha256="tiny-config",
        full_attention_layer_indices=(0,), num_hidden_layers=1, n_kv_heads=1, head_dim=2,
        hidden_size=8, n_attention_heads=2, vocab_size=16, max_position_embeddings=32,
        target_rope_spec={"rotary_dim": 2, "attention_scaling": 1.0})
    monkeypatch.setattr(TargetGeometry, "from_target", classmethod(lambda cls, target, target_model_id=None: geometry))
    return geometry


def _equal_tree(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(_equal_tree(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return isinstance(right, type(left)) and len(left) == len(right) and all(_equal_tree(a, b) for a, b in zip(left, right))
    return left == right


def test_parent_init_survives_baseline_failure_and_auto_resume(monkeypatch, tmp_path):
    from methods.CacheRepair.qwen38.scheduled_validation import ScheduledValidation
    _install_tiny_target_geometry(monkeypatch)
    target = _TinyTarget().eval()
    pool = _TinyPairPool()
    rows = [{"chunks": [[1 + i, 2], [3, 4 + i % 2]]} for i in range(4)]
    options = dict(model_id="tiny-qwen", repair_config=RepairConfig(d_b=16, n_heads=2, n_blocks=1, d_seg=2),
        stats_examples=2, device="cpu", validate_train=False, checkpoint_steps=100, epochs=1, schedule_epochs=1,
        pair_pool=pool)
    parent_dir = tmp_path / "parent"
    run_upstream(target, rows, rows[:2], parent_dir, **options)
    parent = parent_dir / "checkpoint-1"
    parent_bytes = (parent / "model.safetensors").read_bytes()
    plan = {"enabled": True, "hourly_validation_examples": 2, "hourly_interval_seconds": 3600}

    reference = tmp_path / "reference"
    run_upstream(target, rows, rows[:2], reference, init_from=parent, evaluation_plan=plan, **options)

    interrupted = tmp_path / "interrupted"
    original_run = ScheduledValidation.run
    def fail_baseline(self, *args, **kwargs):
        if kwargs.get("kind") == "baseline":
            raise RuntimeError("simulated baseline failure")
        return original_run(self, *args, **kwargs)
    monkeypatch.setattr(ScheduledValidation, "run", fail_baseline)
    with pytest.raises(RuntimeError, match="simulated baseline failure"):
        run_upstream(target, rows, rows[:2], interrupted, init_from=parent, evaluation_plan=plan, **options)
    monkeypatch.setattr(ScheduledValidation, "run", original_run)

    step0 = interrupted / "resume-checkpoints/step-00000000"
    pointer_path = interrupted / "latest.json"
    pointer = json.loads(pointer_path.read_text())
    assert pointer["step"] == 0 and pointer["checkpoint"] == str(step0.resolve())
    # Simulate directory commit succeeding before latest.json publication.
    pointer_path.unlink()
    from methods.CacheRepair.qwen38.resume import Restart
    saved_identity = json.loads((interrupted / "run-identity.json").read_text())
    recovered = Restart(interrupted, saved_identity, target, "tiny-qwen", resume="auto")
    assert recovered.resume == step0.resolve()
    assert json.loads(pointer_path.read_text()) == {"checkpoint": str(step0.resolve()), "step": 0}
    assert json.loads((interrupted / "recovered-latest.json").read_text())["step"] == 0
    from safetensors.torch import load_file
    parent_tensors = load_file(str(parent / "model.safetensors"))
    step0_tensors = load_file(str(step0 / "model.safetensors"))
    assert all(torch.equal(parent_tensors[name], step0_tensors[name]) for name in parent_tensors)

    # The initialization source is immutable even though a step-zero restart exists.
    (parent / "model.safetensors").write_bytes(parent_bytes + b"changed")
    with pytest.raises(ValueError, match="Initial checkpoint model changed"):
        run_upstream(target, rows, rows[:2], interrupted, resume_from="auto", evaluation_plan=plan, **options)
    (parent / "model.safetensors").write_bytes(parent_bytes)

    # Explicit path resume must inherit the same provenance from run-identity.
    run_upstream(target, rows, rows[:2], interrupted, resume_from=step0, evaluation_plan=plan, **options)
    assert json.loads((interrupted / "run-identity.json").read_text())["initialization"]["model_sha256"]

    # A new output directory can recover provenance from the embedded trainer state.
    explicit_output = tmp_path / "explicit-new-output"
    explicit_step0 = explicit_output / "resume-checkpoints/step-00000000"
    explicit_step0.parent.mkdir(parents=True)
    shutil.copytree(step0, explicit_step0)
    run_upstream(target, rows, rows[:2], explicit_output, resume_from=explicit_step0,
                 evaluation_plan=plan, **options)
    assert json.loads((explicit_output / "run-identity.json").read_text())["initialization"]["model_sha256"]

    assert (reference / "loss.jsonl").read_bytes() == (interrupted / "loss.jsonl").read_bytes()
    assert (reference / "loss.jsonl").read_bytes() == (explicit_output / "loss.jsonl").read_bytes()
    ref_model = load_file(str(reference / "checkpoint-1/model.safetensors"))
    resumed_model = load_file(str(interrupted / "checkpoint-1/model.safetensors"))
    explicit_model = load_file(str(explicit_output / "checkpoint-1/model.safetensors"))
    assert ref_model.keys() == resumed_model.keys() == explicit_model.keys()
    assert all(torch.equal(ref_model[name], resumed_model[name]) for name in ref_model)
    assert all(torch.equal(ref_model[name], explicit_model[name]) for name in ref_model)
    ref_state = torch.load(reference / "resume-checkpoints/step-00000001/trainer.pt", map_location="cpu", weights_only=True)
    resumed_state = torch.load(interrupted / "resume-checkpoints/step-00000001/trainer.pt", map_location="cpu", weights_only=True)
    explicit_state = torch.load(explicit_output / "resume-checkpoints/step-00000001/trainer.pt", map_location="cpu", weights_only=True)
    assert (ref_state["step"], ref_state["epoch"], ref_state["offset"], ref_state["horizon"]) == (
        resumed_state["step"], resumed_state["epoch"], resumed_state["offset"], resumed_state["horizon"])
    assert _equal_tree(ref_state["optimizer"], resumed_state["optimizer"])
    assert _equal_tree(ref_state["optimizer"], explicit_state["optimizer"])
    assert ref_state["model_sha256"] == resumed_state["model_sha256"] == explicit_state["model_sha256"]


def test_auto_resume_unknown_origin_without_snapshot_fails_closed(tmp_path):
    from methods.CacheRepair.qwen38.resume import Restart, atomic_json
    identity = {"train_size": 4}  # Legacy/unknown initialization source.
    atomic_json(tmp_path / "run-identity.json", identity)
    restart = Restart(tmp_path, identity, None, "unused", resume="auto")
    with pytest.raises(ValueError, match="no durable restart or recorded initialization"):
        restart.restore(None, None, 100)


def test_auto_resume_before_first_stats_file_uses_parent_rms_directly(monkeypatch, tmp_path):
    """An init metadata commit alone must recover parent RMS, not recompute all rows."""
    import methods.CacheRepair.qwen38.train as train_module
    from safetensors.torch import load_file
    _install_tiny_target_geometry(monkeypatch)
    target = _TinyTarget().eval()
    pool = _TinyPairPool()
    rows = [{"chunks": [[1 + i, 2], [3, 4 + i % 2]]} for i in range(4)]
    options = dict(model_id="tiny-qwen", repair_config=RepairConfig(d_b=16, n_heads=2, n_blocks=1, d_seg=2),
        stats_examples=2, device="cpu", validate_train=False, checkpoint_steps=100, epochs=1, schedule_epochs=1,
        pair_pool=pool)
    parent_dir = tmp_path / "parent"
    run_upstream(target, rows, rows[:2], parent_dir, **options)
    parent = parent_dir / "checkpoint-1"
    reference = tmp_path / "reference"
    run_upstream(target, rows, rows[:2], reference, init_from=parent, **options)

    interrupted = tmp_path / "before-stats"
    original_resumed_main = train_module.resumed_main
    monkeypatch.setattr(train_module, "resumed_main", lambda *args, **kwargs:
        (lambda: (_ for _ in ()).throw(RuntimeError("simulated before RMS file"))))
    with pytest.raises(RuntimeError, match="simulated before RMS file"):
        run_upstream(target, rows, rows[:2], interrupted, init_from=parent, **options)
    monkeypatch.setattr(train_module, "resumed_main", original_resumed_main)
    assert (interrupted / "run-identity.json").is_file()
    assert not (interrupted / "statistics.safetensors").exists()
    assert not (interrupted / "latest.json").exists()

    original_resumed_stats = train_module.resumed_stats
    def guarded_resumed_stats(function, bindings):
        actual = original_resumed_stats(function, bindings)
        def forbidden_full_rms(*args, **kwargs):
            raise AssertionError("auto recovery recomputed RMS instead of using the recorded parent")
        return forbidden_full_rms
    monkeypatch.setattr(train_module, "resumed_stats", guarded_resumed_stats)
    # Old source reaches this guard; the two-line fix passes parent RMS via --stats.
    run_upstream(target, rows, rows[:2], interrupted, resume_from="auto", **options)

    parent_tensors = load_file(str(parent / "model.safetensors"))
    imported_stats = load_file(str(interrupted / "statistics.safetensors"))
    final_tensors = load_file(str(interrupted / "checkpoint-1/model.safetensors"))
    for name in ("sigma_stale", "sigma_delta"):
        assert torch.equal(imported_stats[name], parent_tensors[name])
        assert torch.equal(final_tensors[name], parent_tensors[name])

    reference_tensors = load_file(str(reference / "checkpoint-1/model.safetensors"))
    assert reference_tensors.keys() == final_tensors.keys()
    assert all(torch.equal(reference_tensors[name], final_tensors[name]) for name in reference_tensors)
