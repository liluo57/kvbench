import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.SmokeCacheRepairVllmTraining import (
    acquire_gpu_locks,
    expected_training_order,
    parse_physical_gpus,
    place_long_row_in_first_batch,
    row_key,
    restart_cli_arguments,
    select_smoke_rows,
    validate_gate_checks,
    validate_prompt_events,
    validate_restart_state,
    _phase_groups,
)
from scripts import SmokeCacheRepairVllmTraining as smoke


def _row(seed, lengths):
    next_id = seed
    chunks = []
    for length in lengths:
        chunks.append(list(range(next_id, next_id + length)))
        next_id += length + 1
    return {"chunks": chunks}


def test_physical_gpu_argument_requires_exactly_six_distinct_ids():
    assert parse_physical_gpus("7,2,9,4,1,8") == (7, 2, 9, 4, 1, 8)
    for value in ("0,1", "0,1,2,3,4,4", "0,1,2,3,4,-1", "x,1,2,3,4,5"):
        with pytest.raises(ValueError):
            parse_physical_gpus(value)


def test_smoke_selection_keeps_real_long_row_and_distinct_short_rows(tmp_path):
    path = tmp_path / "train.jsonl"
    rows = [_row(1000 + i * 100, [3 + i, 4 + i]) for i in range(15)]
    long_row = _row(100_000, [8000, 8000])
    # A duplicate prompt occurrence must not make the restart-order key ambiguous.
    with path.open("w") as stream:
        for row in [*rows[:3], long_row, rows[3], rows[3], *rows[4:]]:
            stream.write(json.dumps(row) + "\n")
    selected, evidence = select_smoke_rows(path, short_count=5)
    assert selected[0] == long_row
    assert sum(map(len, selected[0]["chunks"])) == 16_000
    assert len(selected) == 6
    assert len({row_key(row["chunks"]) for row in selected}) == len(selected)
    assert evidence["tokens"][0] == 16_000
    assert evidence["source_indices"][0] == 3


def test_long_prompt_is_in_first_six_gpu_optimizer_batch_for_any_seed():
    rows = [_row(3000 + i * 100, [4 + i, 5 + i]) for i in range(11)]
    rows.insert(0, _row(200_000, [8000, 8000]))
    long_key = row_key(rows[0]["chunks"])
    selection = {"source_indices": list(range(12)),
                 "tokens": [sum(map(len, row["chunks"])) for row in rows],
                 "row_keys": [row_key(row["chunks"]) for row in rows]}
    rows, selection = place_long_row_in_first_batch(rows, selection, seed=8)
    first_batch = expected_training_order(rows, seed=8, epochs=1)[:4]
    assert first_batch[0] == long_key
    assert selection["tokens"][selection["row_keys"].index(first_batch[0])] == 16_000


def test_expected_order_matches_production_shuffle_and_preserves_occurrences():
    rows = [_row(i * 1000, [3, 4]) for i in range(12)]
    order = expected_training_order(rows, seed=7, epochs=2)
    assert len(order) == 24
    assert order[:12] == expected_training_order(rows, seed=7, epochs=1)
    assert set(order[:12]) == {row_key(row["chunks"]) for row in rows}


def test_prompt_event_lengths_follow_keys_after_shuffled_order():
    rows = [_row(i * 10_000, [3 + i, 9 + 2 * i]) for i in range(12)]
    seed = 8
    original_keys = [row_key(row["chunks"]) for row in rows]
    shuffled_keys = expected_training_order(rows, seed=seed, epochs=1)
    assert shuffled_keys != original_keys
    tokens_by_key = {row_key(row["chunks"]): sum(map(len, row["chunks"])) for row in rows}
    events = [{"pair_key": key, "tokens": tokens_by_key[key]} for key in shuffled_keys]
    assert validate_prompt_events(events, rows, seed=seed, epochs=1)

    # Catch the old bug: shuffled index 0 is not necessarily rows[0], and these
    # examples intentionally have different lengths.
    wrong_lengths = [dict(event) for event in events]
    wrong_lengths[0]["tokens"] = sum(map(len, rows[0]["chunks"]))
    with pytest.raises(ValueError, match="token lengths"):
        validate_prompt_events(wrong_lengths, rows, seed=seed, epochs=1)
    with pytest.raises(ValueError, match="request order"):
        validate_prompt_events(list(reversed(events)), rows, seed=seed, epochs=1)


def test_resume_uses_production_auto_path_and_requires_expected_latest(tmp_path):
    output = tmp_path / "training"
    checkpoint = output / "resume-checkpoints/step-00000001"
    checkpoint.mkdir(parents=True)
    (output / "latest.json").write_text(json.dumps({"checkpoint": str(checkpoint), "step": 1}))
    assert restart_cli_arguments(output, checkpoint, tmp_path / "epoch3") == ["--resume", "auto"]
    assert restart_cli_arguments(output, None, tmp_path / "epoch3") == [
        "--init-from", str(tmp_path / "epoch3")]
    with pytest.raises(ValueError, match="latest checkpoint"):
        restart_cli_arguments(output, output / "resume-checkpoints/step-00000000",
                              tmp_path / "epoch3")


def test_restart_cursor_and_state_require_a_real_adamw_commit():
    state = {
        "format": "kvbench-qwen38-trainer-v1", "step": 2, "epoch": 0,
        "offset": 8, "horizon": 18, "optimizer": {"state": {0: {"step": 2}}},
        "python_rng": (3, (1, 2, 3), None), "torch_rng": bytearray(b"rng"),
        "cuda_rng": [b"first", b"second"], "identity": {"run": "test"},
    }
    assert validate_restart_state(state, expected_step=2, expected_epoch=0,
                                  expected_offset=8, train_size=12, cuda_rng_count=2)
    for bad in (
        {**state, "offset": 6},
        {**state, "optimizer": {"state": {}}},
        {**state, "cuda_rng": [b"one"]},
        {**state, "epoch": 1},
    ):
        with pytest.raises(ValueError):
            validate_restart_state(bad, expected_step=2, expected_epoch=0,
                                  expected_offset=8, train_size=12, cuda_rng_count=2)


def test_gate_rejects_missing_or_failed_required_checks():
    checks = {
        "six_gpu_update": True, "four_gpu_exact_resume_update": True,
        "two_gpu_exact_resume_update": True, "optimizer_state_restored": True,
        "rng_state_present": True, "progress_cursor_restored": True,
        "model_updated": True, "rms_unchanged": True,
        "init_checkpoint_unchanged": True, "finite_losses": True,
        "prompt_order_verified": True, "source_unchanged": True,
    }
    assert validate_gate_checks(checks)
    with pytest.raises(ValueError):
        validate_gate_checks({key: value for key, value in checks.items() if key != "rms_unchanged"})
    with pytest.raises(ValueError):
        validate_gate_checks({**checks, "rms_unchanged": False})


def test_gpu_lock_acquisition_refuses_busy_and_foreign_process_snapshot():
    snapshot = [{"index": index, "uuid": f"GPU-{index}", "total_mib": 46068,
                 "used_mib": 0, "utilization": 0, "compute_pids": []}
                for index in range(6)]
    snapshot[2]["used_mib"] = 4096
    with pytest.raises(RuntimeError, match="GPU use"):
        acquire_gpu_locks(tuple(range(6)), snapshot=snapshot)
    snapshot[2]["used_mib"] = 0
    snapshot[4]["compute_pids"] = [99999999]
    # An exited PID is still a busy compute-process entry in the captured snapshot;
    # the utility refuses it rather than claiming a possibly-racing allocation.
    with pytest.raises(RuntimeError, match="GPU use"):
        acquire_gpu_locks(tuple(range(6)), snapshot=snapshot)



def test_group_layout_supports_six_four_and_two_gpu_allocations():
    assert _phase_groups(6, dedicated=False) == "0,1;2,3;4,5"
    assert _phase_groups(4, dedicated=False) == "0,1;2,3"
    assert _phase_groups(2, dedicated=False) == "0,1"
    assert _phase_groups(6, dedicated=True) == "2,3;4,5"
    assert _phase_groups(4, dedicated=True) == "2,3"
    assert _phase_groups(2, dedicated=True) == "0,1"


def test_gpu_locks_allow_small_driver_residue_and_refuse_compute_processes(monkeypatch):
    snapshot = [{"index": index, "uuid": f"GPU-{index}", "total_mib": 46068,
                 "used_mib": 14, "utilization": 0, "compute_pids": []}
                for index in range(6)]
    monkeypatch.setattr(smoke, "gpu_snapshot", lambda: snapshot)
    monkeypatch.setattr(smoke.Path, "open", lambda self, *args, **kwargs: io.StringIO())
    monkeypatch.setattr(smoke.fcntl, "flock", lambda *args: None)
    handles = acquire_gpu_locks(tuple(range(6)), snapshot=snapshot)
    assert len(handles) == 6
    for handle in handles:
        handle.close()

    snapshot[3]["compute_pids"] = [999999]
    with pytest.raises(RuntimeError, match="GPU use"):
        acquire_gpu_locks(tuple(range(6)), snapshot=snapshot)


def test_gpu_lock_error_identifies_foreign_compute_owner(monkeypatch):
    from types import SimpleNamespace
    snapshot = [{"index": index, "uuid": f"GPU-{index}", "total_mib": 46068,
                 "used_mib": 14, "utilization": 0, "compute_pids": []}
                for index in range(6)]
    snapshot[0]["compute_pids"] = [12345]
    original_stat = smoke.Path.stat
    monkeypatch.setattr(smoke.Path, "stat", lambda self, *args, **kwargs:
        SimpleNamespace(st_uid=987654) if str(self) == "/proc/12345" else original_stat(self, *args, **kwargs))
    original_getpwuid = smoke.pwd.getpwuid
    monkeypatch.setattr(smoke.pwd, "getpwuid", lambda uid:
        SimpleNamespace(pw_name="foreign-user") if uid == 987654 else original_getpwuid(uid))
    with pytest.raises(RuntimeError, match="non-owned compute use"):
        acquire_gpu_locks(tuple(range(6)), snapshot=snapshot)

def test_cli_help_is_cpu_only_and_does_not_import_the_training_stack():
    script = Path(__file__).resolve().parents[2] / "scripts/SmokeCacheRepairVllmTraining.py"
    result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True,
                            text=True, check=True)
    assert "--init-from" in result.stdout
    assert "--physical-gpus" in result.stdout
    assert "six-gpu-smoke.json" in result.stdout and "full success" in result.stdout
