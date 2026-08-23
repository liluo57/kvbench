import json
import inspect
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from core.Engine import (
    BenchmarkInitializationError,
    BenchmarkResourceReleaseError,
    Engine,
)
from helpers.Gpu import GpuInfo
from core.Method import Method
from core.Result import NumOutputTokensKey, Result, TotalTimeKey, TtftKey
from core.Task import Case, Task
from methods import (
    CacheblendLmcache,
    CacheblendRepo,
    FullPrefillTransformer,
    FullPrefillVllm,
    NaiveTransformer,
)
from workload.RAGWorkload import RAGInput, RAGWorkload


class FakeTask(Task):
    def __init__(self, name, prompt):
        self.name = name
        self.prompt = prompt

    def Cases(self):
        data = RAGInput(prepare_input=[], run_input=self.prompt)
        yield Case(data, RAGWorkload(0, data), {"expected": self.prompt})

    def Evaluate(self, result, metadata):
        return {"accuracy": float(result.output == metadata["expected"])}


class FakeMethod(Method):
    name = "fake"

    def __init__(self, *, failMode="", **kwargs):
        super().__init__(**kwargs)
        self.failMode = failMode
        self.seen = set()

    def Initialize(self, gpuIds):
        super().Initialize(gpuIds)
        print(f"initialized on {self.gpuIds}", flush=True)
        if self.failMode == "initialize":
            raise RuntimeError("fake initialization failure")
        if self.failMode == "initialize_sleep":
            time.sleep(2)

    def Prepare(self, data):
        pass

    def Run(self, data, retainOutput=None):
        prompt = data[0]
        print(f"running {prompt}", flush=True)
        if self.failMode == "always" or (
            self.failMode == "once" and prompt not in self.seen
        ):
            self.seen.add(prompt)
            raise RuntimeError(f"fake run failure: {prompt}")
        if self.failMode == "sleep":
            time.sleep(2)
        if self.failMode == "crash":
            os._exit(7)
        return [
            Result(
                output=prompt,
                performance={TtftKey: 0.1, TotalTimeKey: 0.2,
                             NumOutputTokensKey: 2},
            )
        ]

    def Reset(self):
        pass


def _gpu(index):
    return GpuInfo(index, "fake", 100, 0, 0.0, 0)


def _run(tmp_path, tasks, methods, gpu_count=2, **kwargs):
    snapshot = [_gpu(index) for index in range(gpu_count)]
    engine_kwargs = {
        "initializeTimeout": 5,
        "taskTimeout": 5,
        "shutdownGracePeriod": 1,
        "gpuReleaseStableSeconds": 0,
        **kwargs,
    }
    with patch(
        "core.Engine.ResolveGpuIds",
        return_value=(list(range(gpu_count)), snapshot),
    ), patch("core.Engine.QueryGpus", return_value=snapshot):
        engine = Engine(
            availableGpuIds="auto",
            outputRoot=tmp_path,
            tui=False,
            verbose=False,
            **engine_kwargs,
        )
        return engine, engine.Evaluate(tasks, methods, [])


def test_pair_retries_in_same_worker_and_persists_logs(tmp_path):
    engine, report = _run(
        tmp_path,
        [FakeTask("flaky", "flaky")],
        [FakeMethod(failMode="once")],
        gpu_count=1,
    )
    assert report["status"] == "completed"
    assert len(report["runs"]) == 1
    assert not report["failures"]
    events = [
        json.loads(line)
        for line in (engine.outputDir / "events.jsonl").read_text().splitlines()
    ]
    starts = [event for event in events if event["type"] == "task_started"]
    assert [event["attempt"] for event in starts] == [1, 2]
    assert len({event["worker_id"] for event in starts}) == 1
    assert Path(starts[0]["log_path"]).read_text().strip() == "running flaky"
    assert (engine.outputDir / "results" / "full.json").exists()


def test_second_pair_failure_isolated_and_worker_continues(tmp_path):
    engine, report = _run(
        tmp_path,
        [FakeTask("bad", "bad"), FakeTask("good", "good")],
        [FakeMethod(failMode="always"), FakeMethod()],
        gpu_count=2,
    )
    assert report["status"] == "completed"
    assert len(report["runs"]) == 2
    assert len(report["failures"]) == 2
    assert all(item["attempts"] == 2 for item in report["failures"])


def test_initialization_failure_aborts_benchmark(tmp_path):
    snapshot = [_gpu(0)]
    with patch("core.Engine.ResolveGpuIds", return_value=([0], snapshot)), patch(
        "core.Engine.QueryGpus", return_value=snapshot
    ):
        engine = Engine(
            outputRoot=tmp_path,
            tui=False,
            verbose=False,
            initializeTimeout=5,
            taskTimeout=5,
            shutdownGracePeriod=0.5,
        )
        with pytest.raises(BenchmarkInitializationError):
            engine.Evaluate(
                [FakeTask("task", "text")],
                [FakeMethod(failMode="initialize")],
                [],
            )
    manifest = json.loads((engine.outputDir / "manifest.json").read_text())
    assert manifest["status"] == "initialization_failed"


def test_initialization_timeout_aborts_benchmark(tmp_path):
    snapshot = [_gpu(0)]
    with patch("core.Engine.ResolveGpuIds", return_value=([0], snapshot)), patch(
        "core.Engine.QueryGpus", return_value=snapshot
    ):
        engine = Engine(
            outputRoot=tmp_path,
            tui=False,
            verbose=False,
            initializeTimeout=0.2,
            taskTimeout=5,
            shutdownGracePeriod=0.2,
        )
        with pytest.raises(BenchmarkInitializationError, match="timed out"):
            engine.Evaluate(
                [FakeTask("task", "text")],
                [FakeMethod(failMode="initialize_sleep")],
                [],
            )


def test_unschedulable_method_does_not_abort(tmp_path):
    _, report = _run(
        tmp_path,
        [FakeTask("task", "text")],
        [FakeMethod(gpuNums=2), FakeMethod()],
        gpu_count=1,
    )
    assert report["status"] == "completed"
    assert len(report["runs"]) == 1
    assert report["failures"][0]["kind"] == "unschedulable"


def test_task_timeout_retries_then_fails(tmp_path):
    engine, report = _run(
        tmp_path,
        [FakeTask("slow", "slow")],
        [FakeMethod(failMode="sleep")],
        gpu_count=1,
        taskTimeout=0.2,
        shutdownGracePeriod=0.2,
    )
    assert report["status"] == "completed"
    assert not report["runs"]
    assert report["failures"][0]["kind"] in ("task_timeout", "worker_exit")
    assert report["failures"][0]["attempts"] == 2


def test_worker_crash_retries_in_a_replacement_process(tmp_path):
    engine, report = _run(
        tmp_path,
        [FakeTask("crash", "crash")],
        [FakeMethod(failMode="crash")],
        gpu_count=1,
    )
    assert report["status"] == "completed"
    assert report["failures"][0]["kind"] == "worker_exit"
    assert report["failures"][0]["attempts"] == 2
    events = [
        json.loads(line)
        for line in (engine.outputDir / "events.jsonl").read_text().splitlines()
    ]
    dispatches = [event for event in events if event["type"] == "task_dispatched"]
    assert len({event["worker_id"] for event in dispatches}) == 2


def test_method_resource_validation():
    with pytest.raises(ValueError):
        FakeMethod(gpuNums=0)
    with pytest.raises(ValueError):
        FakeMethod(perfWeight=0)
    for methodClass in (CacheblendRepo, FullPrefillTransformer, NaiveTransformer):
        with pytest.raises(ValueError):
            methodClass(gpuNums=2)
    for methodClass in (CacheblendLmcache, FullPrefillVllm):
        parameters = inspect.signature(methodClass).parameters
        assert "gpuNums" in parameters
        assert "perfWeight" in parameters
        assert "tensorParallelSize" not in parameters


def test_perf_weight_drives_initial_instance_allocation(tmp_path):
    tasks = [FakeTask(f"task-{index}", str(index)) for index in range(6)]
    engine, report = _run(
        tmp_path,
        tasks,
        [FakeMethod(perfWeight=3), FakeMethod(perfWeight=1)],
        gpu_count=3,
    )
    assert len(report["runs"]) == 12
    events = [
        json.loads(line)
        for line in (engine.outputDir / "events.jsonl").read_text().splitlines()
    ]
    initial = [event for event in events if event["type"] == "worker_spawned"][:3]
    assert [event["method_index"] for event in initial].count(0) == 2
    assert [event["method_index"] for event in initial].count(1) == 1


def test_tui_gpu_snapshot_is_refreshed_from_nvml(tmp_path):
    baseline = [_gpu(0)]
    live = [GpuInfo(0, "fake", 100, 60, 0.6, 87)]
    queried = iter([baseline, live])

    class RecordingTui:
        enabled = True
        cancelRequested = False

        def __init__(self):
            self.snapshots = []

        def Start(self):
            pass

        def Update(self, snapshot):
            self.snapshots.append(snapshot)

        def FinishAndWait(self):
            pass

        def Stop(self):
            pass

    dashboard = RecordingTui()

    def query():
        return next(queried, baseline)

    with patch("core.Engine.ResolveGpuIds", return_value=([0], baseline)), patch(
        "core.Engine.QueryGpus", side_effect=query
    ), patch("core.Engine.BenchmarkTui", return_value=dashboard), patch(
        "core.Engine._GPU_SNAPSHOT_INTERVAL", 0
    ):
        engine = Engine(
            outputRoot=tmp_path,
            tui=True,
            verbose=False,
            initializeTimeout=5,
            taskTimeout=5,
            shutdownGracePeriod=1,
            gpuReleaseStableSeconds=0,
        )
        engine.Evaluate([FakeTask("task", "text")], [FakeMethod()], [])

    assert any(
        snapshot["gpu_snapshot"][0]["utilization"] == 87
        and snapshot["gpu_snapshot"][0]["memoryRatio"] == 0.6
        for snapshot in dashboard.snapshots
    )


def test_gpu_is_not_rescheduled_until_nvml_reports_release(tmp_path):
    baseline = [_gpu(0)]
    busy = [GpuInfo(0, "fake", 100, 50, 0.5, 0)]
    polls = 0

    snapshots = iter([baseline, busy, busy])

    def query():
        nonlocal polls
        polls += 1
        return next(snapshots, baseline)

    with patch("core.Engine.ResolveGpuIds", return_value=([0], baseline)), patch(
        "core.Engine.QueryGpus", side_effect=query
    ):
        engine = Engine(
            outputRoot=tmp_path,
            tui=False,
            verbose=False,
            initializeTimeout=5,
            taskTimeout=5,
            shutdownGracePeriod=1,
            gpuReleaseTimeout=2,
            gpuReleaseStableSeconds=0,
            gpuReleaseMemoryToleranceMiB=0,
        )
        report = engine.Evaluate(
            [FakeTask("task", "text")],
            [FakeMethod(tag="first"), FakeMethod(tag="second")],
            [],
        )
    assert report["status"] == "completed"
    events = [
        json.loads(line)
        for line in (engine.outputDir / "events.jsonl").read_text().splitlines()
    ]
    released = next(
        index for index, event in enumerate(events)
        if event["type"] == "gpu_released"
    )
    secondSpawn = next(
        index for index, event in enumerate(events)
        if event["type"] == "worker_spawned" and event["method_index"] == 1
    )
    assert released < secondSpawn


def test_gpu_release_timeout_is_a_distinct_fatal_error(tmp_path):
    baseline = [_gpu(0)]
    busy = [GpuInfo(0, "fake", 100, 50, 0.5, 0)]
    with patch("core.Engine.ResolveGpuIds", return_value=([0], baseline)), patch(
        "core.Engine.QueryGpus", return_value=busy
    ):
        engine = Engine(
            outputRoot=tmp_path,
            tui=False,
            verbose=False,
            initializeTimeout=5,
            taskTimeout=5,
            shutdownGracePeriod=0.2,
            gpuReleaseTimeout=0.3,
            gpuReleaseStableSeconds=0,
            gpuReleaseMemoryToleranceMiB=0,
        )
        with pytest.raises(BenchmarkResourceReleaseError):
            engine.Evaluate(
                [FakeTask("task", "text")],
                [FakeMethod()],
                [],
            )
    manifest = json.loads((engine.outputDir / "manifest.json").read_text())
    assert manifest["status"] == "resource_release_failed"
    assert manifest["unreleased_gpus"]


def test_new_compute_pid_blocks_gpu_release_even_at_baseline_memory(tmp_path):
    baseline = [_gpu(0)]
    contended = [GpuInfo(0, "fake", 100, 0, 0.0, 0, (98765,))]
    snapshots = iter([baseline])

    def query():
        return next(snapshots, contended)

    with patch("core.Engine.ResolveGpuIds", return_value=([0], baseline)), patch(
        "core.Engine.QueryGpus", side_effect=query
    ):
        engine = Engine(
            outputRoot=tmp_path,
            tui=False,
            verbose=False,
            initializeTimeout=5,
            taskTimeout=5,
            shutdownGracePeriod=0.2,
            gpuReleaseTimeout=0.3,
            gpuReleaseStableSeconds=0,
            gpuReleaseMemoryToleranceMiB=0,
        )
        with pytest.raises(BenchmarkResourceReleaseError, match="98765"):
            engine.Evaluate([FakeTask("task", "text")], [FakeMethod()], [])

    manifest = json.loads((engine.outputDir / "manifest.json").read_text())
    unreleased = manifest["unreleased_gpus"]["0"]
    assert unreleased["external_compute_pids"] == [98765]


def test_gpu_must_remain_clean_for_stability_window(tmp_path):
    baseline = [_gpu(0)]
    with patch("core.Engine.ResolveGpuIds", return_value=([0], baseline)), patch(
        "core.Engine.QueryGpus", return_value=baseline
    ):
        engine = Engine(
            outputRoot=tmp_path,
            tui=False,
            verbose=False,
            initializeTimeout=5,
            taskTimeout=5,
            shutdownGracePeriod=1,
            gpuReleaseTimeout=2,
            gpuReleaseStableSeconds=0.25,
            gpuReleaseMemoryToleranceMiB=0,
        )
        report = engine.Evaluate(
            [FakeTask("task", "text")],
            [FakeMethod(tag="first"), FakeMethod(tag="second")],
            [],
        )

    assert report["status"] == "completed"
    events = [
        json.loads(line)
        for line in (engine.outputDir / "events.jsonl").read_text().splitlines()
    ]
    releases = [event for event in events if event["type"] == "gpu_released"]
    assert releases
    assert all(event["stable_duration"] >= 0.25 for event in releases)
