"""Native supervision and execution of the actual upstream loop on a tiny model."""

import json
from pathlib import Path

import pytest
import torch

from methods.CacheRepair.backend.cacherepair import train as official
from methods.CacheRepair.qwen38 import QwenCacheRepair, RepairConfig, TargetGeometry
from methods.CacheRepair.qwen38.supervision import make_pair
from methods.CacheRepair.qwen38.train import _bind, run_upstream, validation_diagnostics
from tests.backends.test_cacherepair_qwen38_native import native


def test_native_supervision_pair_order_and_target_freeze(native):
    target, _ = native
    with pytest.raises(ValueError, match="frozen"):
        make_pair(target, [[1, 2], [3, 4]])
    target.requires_grad_(False).eval()
    geometry = TargetGeometry.from_target(target, target_model_id="tiny-qwen")
    pair = make_pair(target, [[1, 2, 3], [4, 5]], geometry=geometry)
    assert pair["stale"].shape == pair["joint"].shape == (1, 5, 2, 2, 2, 16)
    assert pair["stale"].dtype == pair["joint"].dtype == torch.float32
    assert torch.equal(pair["tokens"], torch.tensor([[1, 2, 3, 4, 5]]))
    assert torch.equal(pair["chunk_ids"], torch.tensor([[0, 0, 0, 1, 1]]))
    assert torch.isfinite(pair["stale"]).all() and torch.isfinite(pair["joint"]).all()
    assert not pair["stale"].requires_grad and not pair["joint"].requires_grad


def test_batched_independent_chunks_preserve_causal_supervision(native):
    target, _ = native
    target.requires_grad_(False).eval()
    geometry = TargetGeometry.from_target(target, target_model_id="tiny-qwen")
    chunks = [[1], [2, 3, 4, 5], [6, 7], [8, 9, 10]]
    expected = make_pair(target, chunks, geometry=geometry)
    for batch in (2, 4):
        actual = make_pair(target, chunks, geometry=geometry, chunk_batch_size=batch)
        for key in ("tokens", "chunk_ids"):
            assert torch.equal(actual[key], expected[key])
        for key in ("stale", "joint"):
            torch.testing.assert_close(actual[key], expected[key], rtol=1e-5, atol=1e-6)


def test_official_token_weighted_rms_is_reused():
    def pairs(target, chunks):
        length, value = chunks
        stale = torch.full((1, length, 1, 1, 2, 2), value)
        return {"stale": stale, "joint": stale * 3, "tokens": torch.ones(1, length, dtype=torch.long)}
    bindings = {**official.__dict__, "make_pair": pairs}
    stats = _bind(official.compute_stats, bindings)(None, [{"chunks": (1, 1.)}, {"chunks": (3, 3.)}])
    assert stats["sigma_stale"].shape == (1, 1, 2, 2)
    torch.testing.assert_close(stats["sigma_stale"], torch.full((1, 1, 2, 2), 7. ** 0.5))
    torch.testing.assert_close(stats["sigma_delta"], stats["sigma_stale"] * 2)


def test_validation_zero_head_and_token_weighted_layer_baseline(native):
    target, _ = native
    model = QwenCacheRepair.from_target(
        target, target_model_id="tiny-qwen",
        repair_config=RepairConfig(d_b=16, n_heads=2, n_blocks=1, d_seg=2),
    )
    rows = [{"chunks": [[1], [2, 3]]}, {"chunks": [[4, 5, 6], [7, 8]]}]
    pairs = lambda target, chunks: make_pair(target, chunks, geometry=model.target_config)
    report = validation_diagnostics(model, target, rows, pairs)
    assert report["normalized_residual_mse"] == report["zero_residual_baseline_mse"]
    assert report["relative_residual_l2_error"] == 1
    assert report["predicted_to_true_residual_l2"] == 0
    error = torch.tensor(report["per_attention_layer_k_v_normalized_mse"])
    torch.testing.assert_close(error, torch.tensor(report["per_attention_layer_k_v_zero_residual_mse"]))
    numerator = sum((pair["joint"] - pair["stale"]).square().sum(dim=(0, 1, 3, 5))
                    for pair in (pairs(target, row["chunks"]) for row in rows))
    torch.testing.assert_close(error, numerator / (8 * 2 * 16))


@pytest.mark.parametrize("cache_pairs", [False, True])
def test_original_training_loop_accumulation_checkpoint_and_reload(native, tmp_path, cache_pairs):
    target, _ = native
    original = official.make_pair
    rows = [{"chunks": [[1, 2, 3][:1 + i % 3], [4, 5]]} for i in range(5)]
    output = tmp_path / "training"
    report = run_upstream(
        target, rows, rows[:2], output, model_id="tiny-qwen",
        repair_config=RepairConfig(d_b=16, n_heads=2, n_blocks=1, d_seg=2),
        stats_examples=2, device="cpu", validate_train=False,
        cache_pairs=cache_pairs,
    )
    assert official.make_pair is original
    assert len(report["gradient_steps"]) == 2  # Four examples, then the actual final one.
    assert report["target_frozen"]
    assert report["target_parameter_sample_before"] == report["target_parameter_sample_after"]
    assert report["checkpoints"][0]["reload_exact"]
    assert report["checkpoints"][0]["validation"]["zero_residual_baseline_mse"] > 0
    assert Path(report["checkpoints"][0]["path"], "model.safetensors").is_file()
    assert len((output / "loss.jsonl").read_text().splitlines()) == 2
    assert json.loads((output / "diagnostics.json").read_text())["statistics_examples"] == 2
    if cache_pairs:
        assert report["pair_cache_hits"] > 0
        assert list((output / "supervision-pairs").glob("*.safetensors"))


@pytest.mark.parametrize("stop_at", [2, 3, 6])
def test_mid_epoch_restart_matches_uninterrupted_training(native, tmp_path, stop_at):
    from safetensors.torch import load_file
    target, _ = native
    rows = [{"chunks": [[i + 1, 2, 3], [4, i + 5]]} for i in range(9)]
    options = dict(model_id="tiny-qwen", repair_config=RepairConfig(d_b=16, n_heads=2, n_blocks=1, d_seg=2),
                   stats_examples=2, device="cpu", validate_train=False, epochs=2, schedule_epochs=2,
                   checkpoint_steps=2)
    full, split = tmp_path / "full", tmp_path / "split"
    run_upstream(target, rows, rows[:2], full, **options)
    expected_rng = torch.get_rng_state().clone()
    stopped = run_upstream(target, rows, rows[:2], split, stop_after_steps=stop_at, **options)
    assert stopped["interrupted"]
    # Simulate a crash after logging a later update but before a durable save.
    with (split / "loss.jsonl").open("a") as log:
        log.write(json.dumps({"step": stop_at + 1, "mse": 99}) + "\n")
    run_upstream(target, rows, rows[:2], split, resume_from="auto", **options)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert (split / "loss.jsonl").read_bytes() == (full / "loss.jsonl").read_bytes()
    expected = load_file(str(full / "checkpoint-6/model.safetensors"))
    actual = load_file(str(split / "checkpoint-6/model.safetensors"))
    assert expected.keys() == actual.keys()
    assert all(torch.equal(expected[name], actual[name]) for name in expected)
    assert (split / "replayed-loss.jsonl").exists()
    with pytest.raises(ValueError, match="identity"):
        run_upstream(target, rows[::-1], rows[:2], split, resume_from="auto", **options)


def test_restart_before_first_durable_update_archives_old_logs(tmp_path):
    from methods.CacheRepair.qwen38.resume import Restart, atomic_json
    identity = {"train_size": 4, "initialization": None}
    atomic_json(tmp_path / "run-identity.json", identity)
    (tmp_path / "loss.jsonl").write_text(json.dumps({"step": 1, "mse": .5}) + "\n")
    restart = Restart(tmp_path, identity, None, "unused", resume="auto")
    assert restart.restore(None, None, 100) == (0, 0, 0)
    assert (tmp_path / "loss.jsonl").read_text() == ""
    assert json.loads((tmp_path / "replayed-loss.jsonl").read_text())["step"] == 1


def test_statistics_resume_preserves_fp64_sums(native, tmp_path):
    from methods.CacheRepair.qwen38.resume import Restart, TrainingInterrupted, resumed_stats
    from safetensors.torch import load_file
    target, _ = native
    target.requires_grad_(False).eval()
    rows = [{"chunks": [[1, 2, 3], [4, 5]]}, {"chunks": [[8, 9], [10]]}]
    restart = Restart(tmp_path, {"train_size": 2}, target, "tiny-qwen")
    pair_fn = lambda target, chunks: make_pair(target, chunks, geometry=TargetGeometry.from_target(target, target_model_id="tiny-qwen"))
    bindings = {**official.__dict__, "make_pair": pair_fn,
                "restore_statistics": restart.restore_statistics, "after_statistics": restart.after_statistics}
    stats = resumed_stats(official.compute_stats, bindings)
    restart.requested = "SIGTERM"
    with pytest.raises(TrainingInterrupted):
        stats(target, rows)
    restart.requested = None
    actual = stats(target, rows)
    expected = _bind(official.compute_stats, {**official.__dict__, "make_pair": pair_fn})(target, rows)
    assert all(torch.equal(actual[name], expected[name]) for name in actual)


def test_initialize_from_weights_uses_new_optimizer(native, tmp_path):
    target, _ = native
    rows = [{"chunks": [[1, 2], [3, 4]]} for _ in range(4)]
    options = dict(model_id="tiny-qwen", repair_config=RepairConfig(d_b=16, n_heads=2, n_blocks=1, d_seg=2),
                   stats_examples=1, device="cpu", validate_train=False)
    run_upstream(target, rows, rows[:1], tmp_path / "old", **options)
    report = run_upstream(target, rows, rows[:1], tmp_path / "new",
                          init_from=tmp_path / "old/checkpoint-1", **options)
    assert "fresh optimizer" in report["initialization"]
    assert len(report["gradient_steps"]) == 1


def test_memory_bounded_finetune_retains_rms_and_reloads_exactly(native,tmp_path):
    from safetensors.torch import load_file
    target,_=native;rows=[{'chunks':[[1,2],[3,4]]} for _ in range(4)]
    options=dict(model_id='tiny-qwen',repair_config=RepairConfig(d_b=16,n_heads=2,n_blocks=1,d_seg=2),
        stats_examples=1,device='cpu',validate_train=False)
    run_upstream(target,rows,rows[:1],tmp_path/'old',**options)
    report=run_upstream(target,rows,rows[:1],tmp_path/'new',**options,
        init_from=tmp_path/'old/checkpoint-1',learning_rate_scale=1/3,
        backend_spec={'memory_efficient_repairer':True,'repairer_tile':2})
    assert report['checkpoints'][0]['reload_exact']
    before=load_file(str(tmp_path/'old/checkpoint-1/model.safetensors'))
    after=load_file(str(tmp_path/'new/checkpoint-1/model.safetensors'))
    for name in ('sigma_stale','sigma_delta'):assert torch.equal(before[name],after[name])
    assert len(report['gradient_steps'])==1 and report['learning_rate_scale']==1/3


def test_memory_bounded_finetune_exact_resume(native,tmp_path):
    from safetensors.torch import load_file
    target,_=native;rows=[{'chunks':[[1,2],[3,4+i%2]]} for i in range(5)]
    base=dict(model_id='tiny-qwen',repair_config=RepairConfig(d_b=16,n_heads=2,n_blocks=1,d_seg=2),
        stats_examples=1,device='cpu',validate_train=False,checkpoint_steps=1)
    run_upstream(target,rows,rows[:1],tmp_path/'parent',**base)
    options={**base,'epochs':2,'schedule_epochs':2,'learning_rate_scale':1/3,
        'backend_spec':{'memory_efficient_repairer':True,'repairer_tile':2}}
    parent=tmp_path/'parent/checkpoint-2'
    run_upstream(target,rows,rows[:1],tmp_path/'full',init_from=parent,**options)
    stopped=run_upstream(target,rows,rows[:1],tmp_path/'split',init_from=parent,
        stop_after_steps=1,**options)
    assert stopped['interrupted']
    run_upstream(target,rows,rows[:1],tmp_path/'split',resume_from='auto',**options)
    full=load_file(str(tmp_path/'full/checkpoint-4/model.safetensors'))
    resumed=load_file(str(tmp_path/'split/checkpoint-4/model.safetensors'))
    assert full.keys()==resumed.keys()
    for name in full:assert torch.equal(full[name],resumed[name]),name
    assert (tmp_path/'full/loss.jsonl').read_bytes()==(tmp_path/'split/loss.jsonl').read_bytes()


def test_hourly_validation_and_epoch_quality_resume_preserve_training(native,tmp_path,monkeypatch):
    from safetensors.torch import load_file
    from methods.CacheRepair.qwen38.scheduled_validation import ScheduledValidation
    target,_=native
    rows=[{'chunks':[[1+i,2],[3,4]]} for i in range(9)]
    options=dict(model_id='tiny-qwen',repair_config=RepairConfig(d_b=16,n_heads=2,n_blocks=1,d_seg=2),
        stats_examples=2,device='cpu',validate_train=False,checkpoint_steps=1,epochs=2,schedule_epochs=2)
    full=tmp_path/'reference'
    run_upstream(target,rows,rows[:2],full,**options)
    expected_rng=torch.get_rng_state()
    # Every non-final update becomes due, independent of machine speed.
    monkeypatch.setattr(ScheduledValidation,'due',lambda self:True)
    plan=dict(enabled=True,hourly_validation_examples=2,hourly_interval_seconds=3600,
        epoch_quality=dict(enabled=True))
    output=tmp_path/'scheduled/training'
    report=run_upstream(target,rows,rows[:2],output,evaluation_plan=plan,**options)
    assert report['interrupted'] and report['reason']=='epoch_quality_evaluation'
    for step in [3,6]:
        request=json.loads((output/'epoch-quality-request.json').read_text())
        assert request['step']==step
        assert json.loads((output/'latest.json').read_text())['step']==step
        assert (output/f'checkpoint-{step}/validation.json').is_file()
        done=output.parent/'quality-evaluations'/f'step-{step:08d}'/'completed.json'
        done.parent.mkdir(parents=True)
        done.write_text(json.dumps(dict(status='completed',checkpoint=request['checkpoint'])))
        report=run_upstream(target,rows,rows[:2],output,resume_from='auto',evaluation_plan=plan,**options)
    assert not report.get('interrupted') and torch.equal(expected_rng,torch.get_rng_state())
    assert (full/'loss.jsonl').read_bytes()==(output/'loss.jsonl').read_bytes()
    a,b=[load_file(str(p/'checkpoint-6/model.safetensors')) for p in [full,output]]
    assert all(torch.equal(v,b[k]) for k,v in a.items())
    validations=list((output/'scheduled-validation').glob('*.json'))
    assert len(validations)==7 # baseline, four hourly, two epoch validations
    assert all(json.loads(p.read_text())['validation']['examples']==2 for p in validations)


def test_hourly_evaluation_failure_keeps_latest_optimizer_update(native,tmp_path,monkeypatch):
    from methods.CacheRepair.qwen38.scheduled_validation import ScheduledValidation
    target,_=native
    rows=[{'chunks':[[1+i,2],[3,4]]} for i in range(5)]
    options=dict(model_id='tiny-qwen',repair_config=RepairConfig(d_b=16,n_heads=2,n_blocks=1,d_seg=2),
        stats_examples=2,device='cpu',validate_train=False,checkpoint_steps=100)
    plan=dict(enabled=True,hourly_validation_examples=2)
    original=ScheduledValidation.run
    def evaluate(self,*args,**kwargs):
        if kwargs['kind']=='hourly':raise RuntimeError('simulated evaluation failure')
        return original(self,*args,**kwargs)
    monkeypatch.setattr(ScheduledValidation,'due',lambda self:True)
    monkeypatch.setattr(ScheduledValidation,'run',evaluate)
    output=tmp_path/'training'
    with pytest.raises(RuntimeError,match='evaluation failure'):
        run_upstream(target,rows,rows[:2],output,evaluation_plan=plan,**options)
    latest=json.loads((output/'latest.json').read_text())
    assert latest['step']==1
    assert json.loads((output/'loss.jsonl').read_text().splitlines()[-1])['step']==1
    assert (Path(latest['checkpoint'])/'trainer.pt').is_file()
    monkeypatch.setattr(ScheduledValidation,'run',original)
    monkeypatch.setattr(ScheduledValidation,'due',lambda self:False)
    report=run_upstream(target,rows,rows[:2],output,resume_from='auto',evaluation_plan=plan,**options)
    assert not report.get('interrupted')


def test_auto_resume_without_snapshot_or_initialization_fails_closed(tmp_path):
    from methods.CacheRepair.qwen38.resume import Restart, atomic_json
    identity = {"train_size": 4}  # Pre-metadata/unknown origin must not silently become random init.
    atomic_json(tmp_path / "run-identity.json", identity)
    restart = Restart(tmp_path, identity, None, "unused", resume="auto")
    with pytest.raises(ValueError, match="no durable restart or recorded initialization"):
        restart.restore(None, None, 100)


def test_init_from_recovers_after_baseline_failure_before_first_update(native, tmp_path, monkeypatch):
    from safetensors.torch import load_file
    from methods.CacheRepair.qwen38.scheduled_validation import ScheduledValidation
    target, _ = native
    rows = [{"chunks": [[1 + i, 2], [3, 4 + (i % 2)]]} for i in range(4)]
    options = dict(model_id="tiny-qwen", repair_config=RepairConfig(d_b=16, n_heads=2, n_blocks=1, d_seg=2),
        stats_examples=2, device="cpu", validate_train=False, checkpoint_steps=100, epochs=1, schedule_epochs=1)
    parent_dir = tmp_path / "parent"
    run_upstream(target, rows, rows[:2], parent_dir, **options)
    parent = parent_dir / "checkpoint-1"
    parent_bytes = (parent / "model.safetensors").read_bytes()
    plan = dict(enabled=True, hourly_validation_examples=2, hourly_interval_seconds=3600)

    reference = tmp_path / "reference"
    run_upstream(target, rows, rows[:2], reference, init_from=parent, evaluation_plan=plan, **options)

    interrupted = tmp_path / "interrupted"
    original_run = ScheduledValidation.run
    def fail_first_baseline(self, *args, **kwargs):
        if kwargs.get("kind") == "baseline":
            raise RuntimeError("simulated baseline interruption")
        return original_run(self, *args, **kwargs)
    monkeypatch.setattr(ScheduledValidation, "run", fail_first_baseline)
    with pytest.raises(RuntimeError, match="simulated baseline interruption"):
        run_upstream(target, rows, rows[:2], interrupted, init_from=parent, evaluation_plan=plan, **options)
    monkeypatch.setattr(ScheduledValidation, "run", original_run)

    step0 = interrupted / "resume-checkpoints/step-00000000"
    pointer = json.loads((interrupted / "latest.json").read_text())
    assert pointer["step"] == 0 and Path(pointer["checkpoint"]) == step0.resolve()
    parent_tensors = load_file(str(parent / "model.safetensors"))
    step0_tensors = load_file(str(step0 / "model.safetensors"))
    assert all(torch.equal(parent_tensors[name], step0_tensors[name]) for name in parent_tensors)

    # Parent mutation is detected from the immutable run identity before any
    # attempt to load a restart or initialize weights.
    (parent / "model.safetensors").write_bytes(parent_bytes + b"changed")
    with pytest.raises(ValueError, match="Initial checkpoint model changed"):
        run_upstream(target, rows, rows[:2], interrupted, resume_from="auto", evaluation_plan=plan, **options)
    (parent / "model.safetensors").write_bytes(parent_bytes)

    run_upstream(target, rows, rows[:2], interrupted, resume_from="auto", evaluation_plan=plan, **options)
    assert (reference / "loss.jsonl").read_bytes() == (interrupted / "loss.jsonl").read_bytes()
    reference_model = load_file(str(reference / "checkpoint-1/model.safetensors"))
    resumed_model = load_file(str(interrupted / "checkpoint-1/model.safetensors"))
    assert reference_model.keys() == resumed_model.keys()
    assert all(torch.equal(reference_model[name], resumed_model[name]) for name in reference_model)

    ref_state = torch.load(reference / "resume-checkpoints/step-00000001/trainer.pt", map_location="cpu", weights_only=True)
    resumed_state = torch.load(interrupted / "resume-checkpoints/step-00000001/trainer.pt", map_location="cpu", weights_only=True)
    assert (ref_state["step"], ref_state["epoch"], ref_state["offset"], ref_state["horizon"]) == (
        resumed_state["step"], resumed_state["epoch"], resumed_state["offset"], resumed_state["horizon"])
    def equal_tree(left, right):
        if isinstance(left, torch.Tensor):
            return isinstance(right, torch.Tensor) and torch.equal(left, right)
        if isinstance(left, dict):
            return isinstance(right, dict) and left.keys() == right.keys() and all(equal_tree(left[k], right[k]) for k in left)
        if isinstance(left, (list, tuple)):
            return isinstance(right, type(left)) and len(left) == len(right) and all(equal_tree(a, b) for a, b in zip(left, right))
        return left == right
    assert equal_tree(ref_state["optimizer"], resumed_state["optimizer"])
    assert ref_state["model_sha256"] == resumed_state["model_sha256"]
