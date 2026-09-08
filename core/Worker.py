"""Spawn-safe method worker and the single-pair evaluation loop."""

import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .Metrics import AggregateStats, Metric
from .Method import Method
from .Result import AggregateScores, NormalizeScores, Result
from .Task import Case, Task
from .Workflow import Action, ActionKind, ActionResult, Workflow


def EvaluatePair(
    task: Task,
    method: Method,
    metrics: List[Metric],
    batchSize: int,
    externalRunCallback: Optional[Callable[[int, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Evaluate one pair entirely inside its method worker."""
    for metric in metrics:
        metric.Reset()

    taskScores: Dict[str, List[float]] = {}
    methodScores: Dict[str, List[float]] = {
        name: [] for name in method.method_metrics
    }
    methodWeights: Dict[str, List[float]] = {
        name: [] for name in method.method_metrics
    }
    nCases = 0
    # AgentBenchFlow cases are independent external rollouts.  Keep them
    # one-at-a-time so a Method.Run failure can be attributed to exactly one
    # case and the following rollout can still use the same worker.
    effectiveBatchSize = (
        1 if getattr(task, "continueOnCaseFailure", False) else batchSize
    )
    batch: List[Case] = []
    for case in task.Cases():
        batch.append(case)
        if len(batch) >= effectiveBatchSize:
            nCases += _ProcessCaseBatch(
                task, method, metrics, batch, taskScores, methodScores,
                methodWeights, externalRunCallback=externalRunCallback,
            )
            batch = []
    if batch:
        nCases += _ProcessCaseBatch(
            task, method, metrics, batch, taskScores, methodScores, methodWeights,
            externalRunCallback=externalRunCallback,
        )

    report: Dict[str, Any] = {
        "method": method.Label,
        "task": task.Label,
        "cases": nCases,
        "task_metrics": AggregateScores(taskScores),
        "system_metrics": {metric.name: metric.Summary() for metric in metrics},
    }
    if method.method_metrics:
        report["method_metrics"] = {}
        for name, values in methodScores.items():
            stats = AggregateStats(values, name=name)
            weights = methodWeights[name]
            if values and weights and sum(weights) > 0:
                stats[f"{name}_mean"] = sum(
                    value * weight
                    for value, weight in zip(values, weights)
                ) / sum(weights)
                stats[f"{name}_weight_total"] = sum(weights)
            report["method_metrics"][name] = stats
    return report


def _ProcessCaseBatch(
    task: Task,
    method: Method,
    metrics: List[Metric],
    batch: List[Case],
    taskScores: Dict[str, List[float]],
    methodScores: Dict[str, List[float]],
    methodWeights: Dict[str, List[float]],
    externalRunCallback: Optional[Callable[[int, Dict[str, Any]], None]] = None,
) -> int:
    """Process a batch, isolating failures for tasks that opt in.

    The normal engine contract intentionally keeps pair failures fatal after
    retries.  AgentBenchFlow opts into the narrower case-level contract: its
    external rollout is scored as zero when it fails, then the worker moves on
    to the next case.
    """
    try:
        return _ProcessBatch(
            task,
            method,
            metrics,
            batch,
            taskScores,
            methodScores,
            methodWeights,
            externalRunCallback=externalRunCallback,
        )
    except BaseException as exc:
        if not getattr(task, "continueOnCaseFailure", False) or len(batch) != 1:
            raise

        case = batch[0]
        fail = getattr(case.workflow, "fail", None)
        if callable(fail):
            try:
                fail(exc)
            except BaseException:
                traceback.print_exc()

        # A BenchFlow case may be scored as zero only after it has completed
        # at least one model RUN.  Before that point, a zero Result is a
        # startup/control-plane failure and must reach the pair-level retry
        # and failures.json path instead of masquerading as a valid score.
        if _HasUnsuccessfulExternalRun(batch):
            raise

        failureScorer = getattr(task, "CaseFailureScores", None)
        if not callable(failureScorer):
            raise
        scores = NormalizeScores(failureScorer(case.metadata, exc))
        for name, value in scores.items():
            taskScores.setdefault(name, []).append(float(value))
        try:
            method.Reset()
        except BaseException:
            traceback.print_exc()
        return 1


def _ProcessBatch(
    task: Task,
    method: Method,
    metrics: List[Metric],
    batch: List[Case],
    taskScores: Dict[str, List[float]],
    methodScores: Dict[str, List[float]],
    methodWeights: Dict[str, List[float]],
    externalRunCallback: Optional[Callable[[int, Dict[str, Any]], None]] = None,
) -> int:
    workflows = [case.workflow for case in batch]
    finalResults: Dict[int, Result] = {}
    reportedExternalRuns: Dict[int, Dict[str, Any]] = {}

    def reportExternalRun(caseId: int, descriptor: Any) -> None:
        if externalRunCallback is None or not isinstance(descriptor, dict):
            return
        if reportedExternalRuns.get(caseId) == descriptor:
            return
        reportedExternalRuns[caseId] = descriptor
        externalRunCallback(caseId, descriptor)

    # Register a runner as soon as it starts, including the interval while
    # it is waiting for its first provider request.  The post-next() probe is
    # retained for lightweight/custom workflows that do not implement the
    # setter but expose a descriptor directly.
    for workflow in workflows:
        setter = getattr(workflow, "SetExternalCleanupCallback", None)
        if callable(setter):
            setter(
                lambda descriptor, caseId=workflow.case_id: reportExternalRun(
                    caseId, descriptor
                )
            )

    while True:
        stepActions: List[Action] = []
        workflowSlices: List[Tuple[Workflow, int, int]] = []
        for workflow in workflows:
            if workflow.finished:
                continue
            actions = workflow.next()
            descriptorGetter = getattr(
                workflow, "ExternalCleanupDescriptor", None
            )
            if callable(descriptorGetter):
                reportExternalRun(workflow.case_id, descriptorGetter())
            if actions is None:
                if not workflow.finished:
                    raise RuntimeError(
                        f"Workflow case_id={workflow.case_id} returned no "
                        "Actions while unfinished"
                    )
                continue
            if not actions:
                raise RuntimeError(
                    f"Workflow case_id={workflow.case_id} returned an empty "
                    "Action list"
                )
            start = len(stepActions)
            stepActions.extend(actions)
            workflowSlices.append((workflow, start, start + len(actions)))

        if not stepActions:
            unfinished = [
                workflow.case_id for workflow in workflows if not workflow.finished
            ]
            if unfinished:
                raise RuntimeError(
                    f"Workflows produced no Actions while unfinished: {unfinished}"
                )
            break
        kinds = {action.kind for action in stepActions}
        if len(kinds) != 1:
            raise RuntimeError(
                f"Mixed action kinds in one step: {kinds}. All actions in one "
                "step must be either PREPARE or RUN."
            )
        kind = kinds.pop()
        if kind == ActionKind.PREPARE:
            method.Prepare([action.data for action in stepActions])
            stepResults = [
                ActionResult(action.case_id, Result(), action.tag)
                for action in stepActions
            ]
        else:
            results = method.Run(
                [action.data for action in stepActions],
                [action.retainOutput for action in stepActions],
            )
            if len(results) != len(stepActions):
                raise RuntimeError(
                    f"{method.Label}.Run returned {len(results)} result(s) for "
                    f"{len(stepActions)} action(s)"
                )
            stepResults = [
                ActionResult(action.case_id, result, action.tag)
                for action, result in zip(stepActions, results)
            ]
            for stepResult in stepResults:
                finalResults[stepResult.case_id] = stepResult.result
                for metric in metrics:
                    metric.Update(stepResult.result)
                for name in method.method_metrics:
                    value = stepResult.result.metadata.get(name)
                    if value is None:
                        continue
                    methodScores[name].append(float(value))
                    weight = (
                        stepResult.result.metadata.get("n_input", 1.0)
                        if name == "reuse_ratio"
                        else 1.0
                    )
                    methodWeights[name].append(float(weight or 0.0))

        for workflow, start, end in workflowSlices:
            workflow.observe(stepResults[start:end])

    missingResults = [
        case.workflow.case_id
        for case in batch
        if (
            case.workflow.case_id not in finalResults
            and getattr(case.workflow, "final_result", None) is None
        )
    ]
    if missingResults:
        raise RuntimeError(
            f"Workflows finished without a RUN result: {missingResults}"
        )

    for case in batch:
        # Most workflows expose the last inference Result via ``finalResults``;
        # workflows whose scoring signal is *not* an inference output (e.g.
        # AgentBenchFlowWorkflow, which exposes the rollout's ``result.json``)
        # may override ``final_result`` to surface a different Result.
        result = (
            getattr(case.workflow, "final_result", None)
            or finalResults[case.workflow.case_id]
        )
        scores = NormalizeScores(task.Evaluate(result, case.metadata))
        for name, value in scores.items():
            taskScores.setdefault(name, []).append(float(value))
    method.Reset()
    return len(batch)


def _HasUnsuccessfulExternalRun(batch: List[Case]) -> bool:
    """Return whether a case explicitly failed before its first model RUN."""
    for case in batch:
        checker = getattr(case.workflow, "HasSuccessfulRun", None)
        if callable(checker) and not checker():
            return True
    return False


def _Redirect(path: str) -> None:
    """Redirect Python and native stdout/stderr at the file-descriptor level."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.dup2(fd, 1)
        os.dup2(fd, 2)
    finally:
        os.close(fd)
    sys.stdout = os.fdopen(1, "w", buffering=1, closefd=False)
    sys.stderr = os.fdopen(2, "w", buffering=1, closefd=False)


def _Emit(queue, workerId: str, kind: str, **fields: Any) -> None:
    queue.put(
        {
            "type": kind,
            "worker_id": workerId,
            "time": time.time(),
            **fields,
        }
    )


def _Initialize(
    method: Method,
    gpuIds: List[int],
    eventQueue,
    workerId: str,
    methodIndex: int,
    batchSize: int,
    instanceLog: str,
) -> bool:
    """Initialize the method's backend; emit the matching lifecycle event.

    Returns ``True`` iff :meth:`Method.Initialize` returned normally. A
    failure here is globally fatal for the run — the coordinator converts
    the ``initialize_failed`` event into a ``BenchmarkInitializationError``
    — so any exception path emits ``initialize_failed`` with the formatted
    traceback rather than swallowing it.
    """
    initStart = time.perf_counter()
    _Emit(
        eventQueue,
        workerId,
        "initialize_started",
        method_index=methodIndex,
        method=method.Label,
        gpu_ids=gpuIds,
        batch_size=batchSize,
        pid=os.getpid(),
        process_group=os.getpgrp(),
        log_path=instanceLog,
    )
    try:
        method.Initialize(gpuIds)
    except BaseException as exc:  # initialization failure is globally fatal
        _Emit(
            eventQueue,
            workerId,
            "initialize_failed",
            duration=time.perf_counter() - initStart,
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
        return False
    _Emit(
        eventQueue,
        workerId,
        "initialize_done",
        duration=time.perf_counter() - initStart,
    )
    return True


def _RunOneAttempt(
    method: Method,
    task: Task,
    taskIndex: int,
    methodIndex: int,
    attempt: int,
    maxAttempts: int,
    logPath: str,
    metrics: List[Metric],
    batchSize: int,
    eventQueue,
    workerId: str,
    controlConnection=None,
) -> bool:
    """Run :func:`EvaluatePair` once, emitting one of the matching events.

    Returns ``True`` if the pair completed successfully (the coordinator
    should advance to the next pair). Returns ``False`` if the attempt
    failed; the caller decides whether to retry or surface
    ``task_failed``. A failure to reset the method is diagnostic-only and
    does not affect the return value — the worker keeps running.
    """
    methodLabel = method.Label
    attemptStart = time.perf_counter()
    _Emit(
        eventQueue,
        workerId,
        "task_started",
        method_index=methodIndex,
        task_index=taskIndex,
        method=methodLabel,
        task=task.Label,
        attempt=attempt,
        log_path=logPath,
    )
    try:
        def reportExternalRun(caseId: int, descriptor: Dict[str, Any]) -> None:
            fields = {
                "type": "external_run_started",
                "worker_id": workerId,
                "time": time.time(),
                "method_index": methodIndex,
                "task_index": taskIndex,
                "method": methodLabel,
                "task": task.Label,
                "attempt": attempt,
                "case_id": caseId,
                "cleanup": descriptor,
            }
            if controlConnection is None:
                _Emit(eventQueue, workerId, fields.pop("type"), **fields)
                return
            try:
                # Pipe.send reaches the coordinator without a feeder thread,
                # so a subsequent SIGKILL cannot discard this registration.
                controlConnection.send(fields)
            except (BrokenPipeError, EOFError, OSError):
                _Emit(eventQueue, workerId, fields.pop("type"), **fields)

        report = EvaluatePair(
            task,
            method,
            metrics,
            batchSize,
            externalRunCallback=reportExternalRun,
        )
    except BaseException as exc:  # keep this worker alive by design
        duration = time.perf_counter() - attemptStart
        error = f"{type(exc).__name__}: {exc}"
        tb = traceback.format_exc()
        _Emit(
            eventQueue,
            workerId,
            "task_attempt_failed",
            method_index=methodIndex,
            task_index=taskIndex,
            method=methodLabel,
            task=task.Label,
            attempt=attempt,
            duration=duration,
            error=error,
            traceback=tb,
            log_path=logPath,
        )
        try:
            method.Reset()
        except BaseException:  # reset failure is diagnostic only
            traceback.print_exc()
        if attempt == maxAttempts:
            _Emit(
                eventQueue,
                workerId,
                "task_failed",
                method_index=methodIndex,
                task_index=taskIndex,
                method=methodLabel,
                task=task.Label,
                attempts=attempt,
                duration=duration,
                error=error,
                traceback=tb,
                log_path=logPath,
            )
        return False

    duration = time.perf_counter() - attemptStart
    _Emit(
        eventQueue,
        workerId,
        "task_done",
        method_index=methodIndex,
        task_index=taskIndex,
        method=methodLabel,
        task=task.Label,
        attempt=attempt,
        duration=duration,
        report=report,
        log_path=logPath,
    )
    return True


def _RunCommandLoop(
    method: Method,
    metrics: List[Metric],
    batchSize: int,
    connection,
    eventQueue,
    workerId: str,
    methodIndex: int,
    instanceLog: str,
) -> None:
    """Drain coordinator-issued commands until ``shutdown`` or EOF.

    Each ``task`` command goes through one or more ``_RunOneAttempt`` calls
    (up to ``max_attempts``); the loop emits ``worker_idle`` after every
    task so the coordinator can dispatch the next pair. A ``shutdown``
    command (or an EOFError — the coordinator's end closed) terminates
    the loop and lets :func:`_Shutdown` run.
    """
    while True:
        try:
            command = connection.recv()
        except EOFError:
            break
        if command.get("op") == "shutdown":
            break
        if command.get("op") != "task":
            continue

        task: Task = command["task"]
        taskIndex = int(command["task_index"])
        startAttempt = int(command.get("start_attempt", 1))
        maxAttempts = int(command.get("max_attempts", 2))
        completed = False

        for attempt in range(startAttempt, maxAttempts + 1):
            logPath = command["log_paths"][attempt]
            _Redirect(logPath)
            if _RunOneAttempt(
                method, task, taskIndex, methodIndex,
                attempt, maxAttempts, logPath, metrics, batchSize,
                eventQueue, workerId, connection,
            ):
                completed = True
                break

        _Redirect(instanceLog)
        _Emit(
            eventQueue,
            workerId,
            "worker_idle",
            method_index=methodIndex,
            task_index=taskIndex,
            completed=completed,
        )


def _Shutdown(
    method: Method,
    connection,
    eventQueue,
    workerId: str,
) -> None:
    """Tear down the method's backend; emit ``worker_closed``; close the pipe.

    Closes the method even after a partial initialization. A backend may
    have spawned its own process before raising, so skipping ``Close`` here
    can orphan it. Errors during ``Close`` are diagnostic-only — process
    exit still releases CUDA, so we capture the error string but do not
    re-raise.
    """
    closeStart = time.perf_counter()
    closeError = None
    try:
        method.Close()
    except BaseException as exc:  # process exit still releases CUDA
        closeError = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    _Emit(
        eventQueue,
        workerId,
        "worker_closed",
        duration=time.perf_counter() - closeStart,
        error=closeError,
    )
    try:
        connection.close()
    except Exception:
        pass


def WorkerMain(
    workerId: str,
    methodIndex: int,
    method: Method,
    metrics: List[Metric],
    gpuIds: List[int],
    batchSize: int,
    connection,
    eventQueue,
    instanceLog: str,
) -> None:
    """Own one initialized method and execute coordinator-issued tasks.

    The three lifecycle phases — ``_Initialize`` → ``_RunCommandLoop`` →
    ``_Shutdown`` — are independent helpers, each one event-emitter pure
    so they can be unit-tested without spawning a real worker process.
    :func:`WorkerMain` itself only wires them together: it puts the
    worker in its own process group, redirects stdout / stderr to the
    instance log, and guarantees ``_Shutdown`` runs even when
    ``_Initialize`` fails.
    """
    try:
        os.setsid()
    except OSError:
        pass
    _Redirect(instanceLog)
    initialized = _Initialize(
        method, gpuIds, eventQueue, workerId, methodIndex, batchSize, instanceLog,
    )
    try:
        if initialized:
            _RunCommandLoop(
                method, metrics, batchSize, connection,
                eventQueue, workerId, methodIndex, instanceLog,
            )
    finally:
        _Shutdown(method, connection, eventQueue, workerId)
