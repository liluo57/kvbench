import pytest

from core.Method import Method
from core.Result import NumOutputTokensKey, Result, TotalTimeKey, TtftKey
from core.Task import Case, Task
from core.Workload import Action, ActionKind, Workload
from core.Worker import (
    EvaluatePair,
    _Initialize,
    _RunCommandLoop,
    _RunOneAttempt,
    _Shutdown,
)
from metrics import TTFTMetric, ThroughputMetric


class _TwoRunWorkload(Workload):
    def __init__(self, caseId):
        self.case_id = caseId
        self.step = 0
        self.observed = []

    def next(self):
        if self.step == 0:
            action = Action(ActionKind.PREPARE, self.case_id, [f"ctx-{self.case_id}"])
        elif self.step == 1:
            action = Action(
                ActionKind.RUN,
                self.case_id,
                f"case-{self.case_id}:first",
                "first",
                True,
            )
        elif self.step == 2:
            action = Action(
                ActionKind.RUN,
                self.case_id,
                f"case-{self.case_id}:final",
                "final",
                False,
            )
        else:
            return None
        self.step += 1
        return [action]

    def observe(self, results):
        self.observed.extend(results)

    @property
    def finished(self):
        return self.step >= 3

    @property
    def final_result(self):
        return self.observed[-1].result if self.observed else None


class _BatchTask(Task):
    name = "batch"

    def __init__(self, count=3):
        self.count = count
        self.workloads = []
        self.evaluated = []

    def Cases(self):
        for caseId in range(self.count):
            workload = _TwoRunWorkload(caseId)
            self.workloads.append(workload)
            yield Case(
                input=caseId,
                workload=workload,
                metadata={"expected": f"case-{caseId}:final"},
            )

    def Evaluate(self, result, metadata):
        self.evaluated.append(result.output)
        return {"accuracy": float(result.output == metadata["expected"])}


class _RecordingMethod(Method):
    name = "recording"
    method_metrics = ("reuse_ratio",)

    def __init__(self):
        super().__init__()
        self.prepareCalls = []
        self.runCalls = []
        self.resetCalls = 0

    def Prepare(self, data):
        self.prepareCalls.append(data)

    def Run(self, data, retainOutput=None):
        self.runCalls.append((list(data), list(retainOutput or [])))
        results = []
        for prompt in data:
            final = prompt.endswith(":final")
            results.append(
                Result(
                    output=prompt,
                    performance={
                        TtftKey: 2.0 if final else 1.0,
                        NumOutputTokensKey: 6 if final else 2,
                        TotalTimeKey: 2.0 if final else 1.0,
                    },
                    metadata={
                        "reuse_ratio": 0.75 if final else 0.25,
                        "n_input": 3 if final else 1,
                    },
                )
            )
        return results

    def Reset(self):
        self.resetCalls += 1


def test_evaluate_pair_batches_actions_and_aggregates_every_run():
    task = _BatchTask(count=3)
    method = _RecordingMethod()
    ttft = TTFTMetric()
    throughput = ThroughputMetric()
    ttft.Update(Result(performance={TtftKey: 999}))

    report = EvaluatePair(task, method, [ttft, throughput], batchSize=2)

    assert report["method"] == "recording"
    assert report["task"] == "batch"
    assert report["cases"] == 3
    assert report["task_metrics"] == {"accuracy": {"mean": 1.0}}
    assert report["system_metrics"]["ttft"]["ttft_count"] == 6
    assert report["system_metrics"]["ttft"]["ttft_mean"] == 1.5
    assert report["system_metrics"]["throughput"]["throughput_mean"] == 2.5
    assert report["system_metrics"]["throughput"][
        "throughput_total_tokens_per_sec"
    ] == pytest.approx(8 / 3)

    reuse = report["method_metrics"]["reuse_ratio"]
    assert reuse["reuse_ratio_count"] == 6
    assert reuse["reuse_ratio_mean"] == pytest.approx(0.625)
    assert reuse["reuse_ratio_weight_total"] == 12

    assert method.prepareCalls == [
        [["ctx-0"], ["ctx-1"]],
        [["ctx-2"]],
    ]
    assert method.runCalls == [
        (["case-0:first", "case-1:first"], [True, True]),
        (["case-0:final", "case-1:final"], [False, False]),
        (["case-2:first"], [True]),
        (["case-2:final"], [False]),
    ]
    assert method.resetCalls == 2
    assert task.evaluated == ["case-0:final", "case-1:final", "case-2:final"]
    assert all(len(workload.observed) == 3 for workload in task.workloads)


class _OneActionWorkload(Workload):
    def __init__(self, caseId, kind):
        self.case_id = caseId
        self.kind = kind
        self.started = False

    def next(self):
        self.started = True
        data = ["context"] if self.kind == ActionKind.PREPARE else "prompt"
        return [Action(self.kind, self.case_id, data)]

    def observe(self, results):
        pass

    @property
    def finished(self):
        return self.started


class _MixedTask(Task):
    name = "mixed"

    def Cases(self):
        yield Case(workload=_OneActionWorkload(0, ActionKind.PREPARE))
        yield Case(workload=_OneActionWorkload(1, ActionKind.RUN))

    def Evaluate(self, result, metadata):
        return {}


def test_evaluate_pair_rejects_mixed_action_kinds_in_one_step():
    with pytest.raises(RuntimeError, match="Mixed action kinds"):
        EvaluatePair(_MixedTask(), _RecordingMethod(), [], batchSize=2)


class _ShortMethod(_RecordingMethod):
    def Run(self, data, retainOutput=None):
        return []


class _RunTask(Task):
    name = "wrong-result-count"

    def Cases(self):
        yield Case(workload=_OneActionWorkload(0, ActionKind.RUN))

    def Evaluate(self, result, metadata):
        return {}


def test_evaluate_pair_validates_method_result_count():
    with pytest.raises(RuntimeError, match=r"returned 0 result\(s\) for 1 action"):
        EvaluatePair(_RunTask(), _ShortMethod(), [], batchSize=1)


class _StalledWorkload(Workload):
    case_id = 0

    def next(self):
        return None

    def observe(self, results):
        pass

    @property
    def finished(self):
        return False


class _StalledTask(Task):
    name = "stalled"

    def Cases(self):
        yield Case(workload=_StalledWorkload())

    def Evaluate(self, result, metadata):
        return {"accuracy": 1.0}


def test_evaluate_pair_rejects_a_stalled_unfinished_workload():
    with pytest.raises(RuntimeError, match="unfinished"):
        EvaluatePair(_StalledTask(), _RecordingMethod(), [], batchSize=1)


class _PrepareOnlyTask(Task):
    name = "prepare-only"

    def Cases(self):
        yield Case(workload=_OneActionWorkload(0, ActionKind.PREPARE))

    def Evaluate(self, result, metadata):
        return {"accuracy": 1.0}


def test_evaluate_pair_requires_a_final_run_result():
    with pytest.raises(RuntimeError, match="without a RUN result"):
        EvaluatePair(_PrepareOnlyTask(), _RecordingMethod(), [], batchSize=1)


class _EmptyActionWorkload(_StalledWorkload):
    def next(self):
        return []


class _EmptyActionTask(_StalledTask):
    name = "empty-actions"

    def Cases(self):
        yield Case(workload=_EmptyActionWorkload())


def test_evaluate_pair_rejects_an_empty_action_step():
    with pytest.raises(RuntimeError, match="empty Action list"):
        EvaluatePair(_EmptyActionTask(), _RecordingMethod(), [], batchSize=1)


# ---------------------------------------------------------------------------
# WorkerMain lifecycle helpers — each one event-emitter pure so it can be
# tested without spawning a real worker process.
# ---------------------------------------------------------------------------


class _CapturingQueue:
    """Minimal queue stand-in that records every event for assertions."""

    def __init__(self):
        self.events = []

    def put(self, event):
        self.events.append(event)


class _RecordingWorkerMethod(_RecordingMethod):
    """A Method whose lifecycle methods record their arguments."""

    def __init__(self):
        super().__init__()
        self.initializeCalls = []
        self.closeCalls = 0

    def Initialize(self, gpuIds):
        self.initializeCalls.append(list(gpuIds))


    def Close(self):
        self.closeCalls += 1


class _FailingInitializeMethod(_RecordingMethod):
    def Initialize(self, gpuIds):
        raise RuntimeError("backend down")


class _FailingRunMethod(_RecordingMethod):
    def Run(self, data, retainOutput=None):
        raise RuntimeError("decode crashed")


class _IdempotentCloseMethod(_RecordingWorkerMethod):
    def Close(self):
        self.closeCalls += 1


def test_initialize_emits_started_and_done_on_success(tmp_path):
    method = _RecordingWorkerMethod()
    queue = _CapturingQueue()

    ok = _Initialize(
        method, [0, 1], queue, "w0", methodIndex=2, batchSize=4,
        instanceLog=str(tmp_path / "log.txt"),
    )

    assert ok is True
    kinds = [event["type"] for event in queue.events]
    assert kinds == ["initialize_started", "initialize_done"]
    assert method.initializeCalls == [[0, 1]]
    started = queue.events[0]
    assert started["method_index"] == 2
    assert started["gpu_ids"] == [0, 1]
    assert started["batch_size"] == 4
    assert "duration" in queue.events[1]


def test_initialize_emits_failed_on_exception(tmp_path):
    method = _FailingInitializeMethod()
    queue = _CapturingQueue()

    ok = _Initialize(
        method, [0], queue, "w0", methodIndex=0, batchSize=1,
        instanceLog=str(tmp_path / "log.txt"),
    )

    assert ok is False
    kinds = [event["type"] for event in queue.events]
    assert kinds == ["initialize_started", "initialize_failed"]
    failed = queue.events[1]
    assert "RuntimeError" in failed["error"]
    assert "backend down" in failed["error"]
    assert "Traceback" in failed["traceback"]


def test_shutdown_closes_method_and_emits_worker_closed(tmp_path):
    method = _IdempotentCloseMethod()
    queue = _CapturingQueue()

    class _Connection:
        def close(self_inner):
            self_inner.closed = True

    connection = _Connection()
    connection.closed = False

    _Shutdown(method, connection, queue, "w0")

    assert method.closeCalls == 1
    assert connection.closed is True
    assert queue.events[-1]["type"] == "worker_closed"
    assert queue.events[-1]["error"] is None


def test_shutdown_swallows_method_close_errors(tmp_path):
    method = _RecordingWorkerMethod()

    def _bad_close():
        raise OSError("cuda gone")
    method.Close = _bad_close  # type: ignore[method-assign]

    queue = _CapturingQueue()

    class _Connection:
        def close(self_inner):
            pass

    _Shutdown(method, _Connection(), queue, "w0")

    closed = queue.events[-1]
    assert closed["type"] == "worker_closed"
    assert "OSError" in closed["error"]
    assert "cuda gone" in closed["error"]


def test_run_one_attempt_emits_done_when_evaluate_pair_succeeds(tmp_path):
    task = _BatchTask(count=1)
    method = _RecordingWorkerMethod()
    queue = _CapturingQueue()
    logPath = str(tmp_path / "attempt.log")

    completed = _RunOneAttempt(
        method, task, 0, 0, 1, 2, logPath, [], 1,
        queue, "w0",
    )

    assert completed is True
    kinds = [event["type"] for event in queue.events]
    assert kinds == ["task_started", "task_done"]
    done = queue.events[-1]
    assert "report" in done
    assert done["attempt"] == 1
    assert done["task"] == "batch"


def test_run_one_attempt_emits_task_failed_when_attempts_exhausted(tmp_path):
    task = _BatchTask(count=1)
    method = _FailingRunMethod()
    queue = _CapturingQueue()
    logPath = str(tmp_path / "attempt.log")

    completed = _RunOneAttempt(
        method, task, 0, 0, 1, 1, logPath, [], 1,
        queue, "w0",
    )

    assert completed is False
    kinds = [event["type"] for event in queue.events]
    assert "task_attempt_failed" in kinds
    assert kinds[-1] == "task_failed"
    failed = queue.events[-1]
    assert "RuntimeError" in failed["error"]


def test_run_command_loop_processes_task_then_shutdown(tmp_path):
    method = _RecordingWorkerMethod()
    queue = _CapturingQueue()
    logPath = str(tmp_path / "attempt.log")
    task = _BatchTask(count=1)

    sentCommands = [
        {
            "op": "task",
            "task": task,
            "task_index": 0,
            "start_attempt": 1,
            "max_attempts": 1,
            "log_paths": {1: logPath},
        },
        {"op": "shutdown"},
    ]

    class _PipeEnd:
        def __init__(self, commands):
            self._commands = list(commands)

        def recv(self_inner):
            return self_inner._commands.pop(0)

    _RunCommandLoop(
        method, [], 1, _PipeEnd(sentCommands), queue, "w0", 0,
        str(tmp_path / "instance.log"),
    )

    kinds = [event["type"] for event in queue.events]
    assert "task_started" in kinds
    assert "task_done" in kinds
    assert kinds[-1] == "worker_idle"


def test_run_command_loop_breaks_on_eof(tmp_path):
    method = _RecordingWorkerMethod()
    queue = _CapturingQueue()

    class _ClosedPipe:
        def recv(self_inner):
            raise EOFError

    _RunCommandLoop(
        method, [], 1, _ClosedPipe(), queue, "w0", 0,
        str(tmp_path / "instance.log"),
    )

    # The only emitted event is the spurious one if the loop were miscoded;
    # an immediate EOFError should leave the queue empty.
    assert queue.events == []
