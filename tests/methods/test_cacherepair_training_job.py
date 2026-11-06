import json
import sqlite3
import pytest

from scripts.RunCacheRepairTraining import available_gpus, merge_inventory
from tests.methods.test_cacherepair_qwen38_data import make_inventory


def test_gpu_selection_excludes_existing_processes_and_small_devices():
    base = dict(total_mib=46068, used_mib=0, utilization=0, has_compute_process=False)
    snapshot = [{**base, "index": 0}, {**base, "index": 1, "has_compute_process": True},
                {**base, "index": 2, "used_mib": 40000}, {**base, "index": 3, "utilization": 20},
                {**base, "index": 4, "total_mib": 24000}, {**base, "index": 5}]
    assert available_gpus(snapshot) == [0, 5]
    assert available_gpus(snapshot, {5}) == [5]


def test_merge_inventories_protects_both_static_and_actual_inputs(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    make_inventory(first, ["Shared passage", "Static evaluation"])
    make_inventory(second, ["Shared passage", "Actual frozen input"])
    (first / "manifest.json").write_text(json.dumps({"coverage": {"test": {"status": "available"}}}))
    output = tmp_path / "merged"
    merge_inventory([first, second], output)
    with sqlite3.connect(output / "contents.sqlite") as connection:
        assert connection.execute("SELECT COUNT(*) FROM contents").fetchone()[0] == 3
        assert connection.execute("SELECT COUNT(*) FROM refs").fetchone()[0] == 4
    assert json.loads((output / "manifest.json").read_text())["content_sha256"]


def test_five_hour_policy_keeps_original_deadline_on_restart():
    from scripts.RunCacheRepairTrainingFast import due_for_pause
    policy = {"six_gpu_started_unix": 100., "six_gpu_deadline_unix": 18100.}
    assert not due_for_pause(policy, 18099.)
    assert due_for_pause(policy, 18100.)
    # Reloading the policy cannot reset the budget for another five hours.
    resumed = json.loads(json.dumps(policy))
    assert due_for_pause(resumed, 20000.)
    assert not due_for_pause({}, 20000.)


def test_allocation_pauses_and_supports_four_or_two_gpu_resume():
    from scripts.RunCacheRepairTrainingFast import allocation_policy, due_for_pause, gpu_groups
    assert gpu_groups([0, 1, 2, 3, 4, 5]) == "0,1;2,3;4,5"
    assert gpu_groups([2, 3, 4, 5]) == "0,1;2,3"
    assert gpu_groups([4, 5]) == "0,1"
    policy = allocation_policy({"six_gpu_started_unix": 100., "six_gpu_deadline_unix": 18100.}, 5., [0, 1])
    assert policy["after_window"] == "pause_for_review"
    assert policy["deadline_unix"] == 18100.
    assert due_for_pause(policy, 18100.)
    reviewed = allocation_policy(policy, 3., [0, 1, 2, 3], new_window=True)
    assert reviewed["started_unix"] is None and reviewed["deadline_unix"] is None
    assert reviewed["physical_ids"] == [0, 1, 2, 3]


def test_training_identity_allows_allocation_changes_only():
    from scripts.RunCacheRepairTrainingFast import immutable_plan
    base = dict(model="target", dataset="data", config="config", python="python", chunk_batch_size=8,
                config_sha256="abc", dataset_manifest_sha256="def", initialization="fresh")
    assert immutable_plan({**base, "physical_gpus": "0,1", "six_hours": 3}) == immutable_plan(
        {**base, "physical_gpus": "0,1,2,3", "six_hours": 5})
    assert immutable_plan({**base, "config_sha256": "changed"}) != immutable_plan(base)
    assert immutable_plan({**base,'init_from':'epoch3','init_from_sha256':'abc'})!=immutable_plan(base)


def test_finetune_starts_new_optimizer_but_exact_resume_keeps_cursor(tmp_path):
    from scripts.RunCacheRepairTrainingFast import initialization_arguments
    training=tmp_path/'training'
    assert initialization_arguments(training,'epoch3')==['--init-from','epoch3']
    training.mkdir()
    assert initialization_arguments(training,'epoch3')==['--resume','auto']


def test_default_policy_cancels_previous_deadline_without_resetting_training():
    from scripts.RunCacheRepairTrainingFast import allocation_policy, due_for_pause
    policy = allocation_policy({"started_unix": 100., "deadline_unix": 18100.}, None, [0, 1, 2, 3, 4, 5])
    assert policy["started_unix"] == 100.
    assert policy["deadline_unix"] is None
    assert policy["after_window"] == "continue_until_requested"
    assert not due_for_pause(policy, 20000.)


def test_attached_child_never_signals_a_reused_pid(tmp_path, monkeypatch):
    from scripts import RunCacheRepairTrainingFast as runner
    signals = []
    monkeypatch.setattr(runner, "process_identity", lambda pid: ("S", "different-birth"))
    monkeypatch.setattr(runner.os, "kill", lambda *args: signals.append(args))
    child = runner.AttachedChild(123, "original-birth", tmp_path)
    child.send_signal(runner.signal.SIGTERM)
    assert child.returncode == 1 and not signals


def test_attach_owned_process_preserves_pid_and_allows_explicit_stop(tmp_path):
    import os
    import subprocess
    import sys
    from scripts import RunCacheRepairTrainingFast as runner
    training = tmp_path / "training"
    training.mkdir()
    process = subprocess.Popen([sys.executable, "-c", "import time; print('ready', flush=True); time.sleep(30)",
        "methods.CacheRepair.qwen38.train", "--output", str(training)],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "0,1"}, start_new_session=True, stdout=subprocess.PIPE)
    try:
        assert process.stdout.readline() == b"ready\n"
        (tmp_path / "resources-ready.json").write_text(json.dumps({"pid": process.pid}))
        birth = runner.validate_attachment(process.pid, tmp_path, [0, 1])
        child = runner.AttachedChild(process.pid, birth, training)
        assert child.pid == process.pid and child.poll() is None
        (training / "status.json").write_text('{"status":"interrupted"}')
        child.send_signal(runner.signal.SIGTERM)
        process.wait(timeout=5)
        assert child.wait(timeout=5) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


@pytest.mark.parametrize("physical", ["0,1", "2,3,4,5", "0,1,2,3,4,5"])
def test_deadline_saves_and_exits_without_starting_another_allocation(tmp_path, monkeypatch, physical):
    from scripts import RunCacheRepairTrainingFast as runner
    job, dataset = tmp_path / "job", tmp_path / "data"
    job.mkdir()
    dataset.mkdir()
    (dataset / "dataset.json").write_text("{}")
    config = tmp_path / "config.json"
    config.write_text("{}")
    (job / "fast-validation-native-norm.json").write_text('{"status":"passed"}')
    (job / "six-gpu-smoke.json").write_text('{"passed":true}')
    started, signalled = [], []

    class Child:
        returncode = None

        def __init__(self, command, **kwargs):
            started.append(command)
            (job / "resources-ready.json").write_text(json.dumps({"time_unix": runner.time.time() - 1}))
            (job / "training").mkdir()
            (job / "training/status.json").write_text('{"status":"statistics","examples_done":100}')

        def poll(self):
            return self.returncode

        def send_signal(self, number):
            signalled.append(number)
            self.returncode = 0

    ids = list(map(int, physical.split(",")))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner.sys, "argv", ["supervisor", "--job", str(job), "--dataset", str(dataset),
        "--config", str(config), "--model", "target", "--physical-gpus", physical, "--run-hours", "0.000001"])
    monkeypatch.setattr(runner, "gpu_snapshot", lambda: [])
    monkeypatch.setattr(runner, "available_gpus", lambda *args: ids)
    # Simulated children must not wait on the real production GPU leases.
    monkeypatch.setattr(runner.fcntl, "flock", lambda *args: None)
    monkeypatch.setattr(runner.subprocess, "Popen", Child)
    monkeypatch.setattr(runner.signal, "signal", lambda *args: None)
    runner.main()
    assert len(started) == 1
    command = started[0]
    assert command[command.index("--pair-groups") + 1] == runner.gpu_groups(ids)
    assert signalled == [runner.signal.SIGTERM]
    assert json.loads((job / "job-status.json").read_text())["status"] == "paused_for_review"
    assert json.loads((job / "five-hour-review.json").read_text())["automatic_restart"] is False


def test_dedicated_vllm_can_resume_with_smaller_gpu_allocation():
    from scripts.RunCacheRepairTrainingFast import supervision_groups
    assert supervision_groups([0,1,2,3,4,5],dedicated=True,backend='vllm')=='2,3;4,5'
    assert supervision_groups([2,3,4,5],dedicated=True,backend='vllm')=='2,3'
    assert supervision_groups([4,5],dedicated=True,backend='vllm')=='0,1'
    assert supervision_groups([0,1,2,3],backend='vllm')=='0,1;2,3'
    import pytest
    with pytest.raises(ValueError):supervision_groups([0,1],dedicated=True,backend='native')
