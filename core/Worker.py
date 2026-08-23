"""Spawn-safe method worker and the single-pair evaluation loop."""

import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .Metrics import AggregateStats, Metric
from .Method import Method
from .Result import Result
from .Task import Case, Task
from .Workload import Action, ActionKind, ActionResult, Workload


def _NormalizeScores(scores: Any) -> Dict[str, float]:
    if scores is None:
        return {}
    if isinstance(scores, (int, float)):
        return {"score": float(scores)}
    return dict(scores)


def _AggregateScores(perCase: Dict[str, List[float]]) -> Dict[str, Any]:
    return {
        name: {"mean": (sum(values) / len(values)) if values else None}
        for name, values in perCase.items()
    }


def EvaluatePair(
    task: Task,
    method: Method,
    metrics: List[Metric],
    batchSize: int,
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
    batch: List[Case] = []
    for case in task.Cases():
        batch.append(case)
        if len(batch) >= batchSize:
            nCases += _ProcessBatch(
                task,
                method,
                metrics,
                batch,
                taskScores,
                methodScores,
                methodWeights,
            )
            batch = []
    if batch:
        nCases += _ProcessBatch(
            task,
            method,
            metrics,
            batch,
            taskScores,
            methodScores,
            methodWeights,
        )

    report: Dict[str, Any] = {
        "method": method.Label,
        "task": task.name,
        "cases": nCases,
        "task_metrics": _AggregateScores(taskScores),
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


def _ProcessBatch(
    task: Task,
    method: Method,
    metrics: List[Metric],
    batch: List[Case],
    taskScores: Dict[str, List[float]],
    methodScores: Dict[str, List[float]],
    methodWeights: Dict[str, List[float]],
) -> int:
    workloads = [case.workload for case in batch]
    finalResults: Dict[int, Result] = {}

    while True:
        stepActions: List[Action] = []
        workloadSlices: List[Tuple[Workload, int, int]] = []
        for workload in workloads:
            if workload.finished:
                continue
            actions = workload.next()
            if actions is None:
                if not workload.finished:
                    raise RuntimeError(
                        f"Workload case_id={workload.case_id} returned no "
                        "Actions while unfinished"
                    )
                continue
            if not actions:
                raise RuntimeError(
                    f"Workload case_id={workload.case_id} returned an empty "
                    "Action list"
                )
            start = len(stepActions)
            stepActions.extend(actions)
            workloadSlices.append((workload, start, start + len(actions)))

        if not stepActions:
            unfinished = [
                workload.case_id for workload in workloads if not workload.finished
            ]
            if unfinished:
                raise RuntimeError(
                    f"Workloads produced no Actions while unfinished: {unfinished}"
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

        for workload, start, end in workloadSlices:
            workload.observe(stepResults[start:end])

    missingResults = [
        case.workload.case_id
        for case in batch
        if case.workload.case_id not in finalResults
    ]
    if missingResults:
        raise RuntimeError(
            f"Workloads finished without a RUN result: {missingResults}"
        )

    for case in batch:
        result = finalResults[case.workload.case_id]
        scores = _NormalizeScores(task.Evaluate(result, case.metadata))
        for name, value in scores.items():
            taskScores.setdefault(name, []).append(float(value))
    method.Reset()
    return len(batch)


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
    """Own one initialized method and execute coordinator-issued tasks."""
    try:
        os.setsid()
    except OSError:
        pass
    _Redirect(instanceLog)
    initialized = False
    initStart = time.perf_counter()
    _Emit(
        eventQueue,
        workerId,
        "initialize_started",
        method_index=methodIndex,
        method=method.Label,
        gpu_ids=gpuIds,
        pid=os.getpid(),
        process_group=os.getpgrp(),
        log_path=instanceLog,
    )
    try:
        method.Initialize(gpuIds)
        initialized = True
        _Emit(
            eventQueue,
            workerId,
            "initialize_done",
            duration=time.perf_counter() - initStart,
        )
    except BaseException as exc:  # initialization failure is globally fatal
        _Emit(
            eventQueue,
            workerId,
            "initialize_failed",
            duration=time.perf_counter() - initStart,
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )

    try:
        while initialized:
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
            methodLabel = method.Label
            completed = False

            for attempt in range(startAttempt, maxAttempts + 1):
                logPath = command["log_paths"][attempt]
                _Redirect(logPath)
                attemptStart = time.perf_counter()
                _Emit(
                    eventQueue,
                    workerId,
                    "task_started",
                    method_index=methodIndex,
                    task_index=taskIndex,
                    method=methodLabel,
                    task=task.name,
                    attempt=attempt,
                    log_path=logPath,
                )
                try:
                    report = EvaluatePair(task, method, metrics, batchSize)
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
                        task=task.name,
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
                            task=task.name,
                            attempts=attempt,
                            duration=duration,
                            error=error,
                            traceback=tb,
                            log_path=logPath,
                        )
                    continue

                duration = time.perf_counter() - attemptStart
                _Emit(
                    eventQueue,
                    workerId,
                    "task_done",
                    method_index=methodIndex,
                    task_index=taskIndex,
                    method=methodLabel,
                    task=task.name,
                    attempt=attempt,
                    duration=duration,
                    report=report,
                    log_path=logPath,
                )
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
    finally:
        closeStart = time.perf_counter()
        closeError = None
        # Close even after partial initialization. A backend may have spawned
        # its own process before raising, so skipping Close here can orphan it.
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
