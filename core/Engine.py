"""The KVBench evaluation engine."""

from typing import Any, Dict, Iterable, List

from .Metrics import AggregateStats, Metric
from .Method import Method
from .Task import Case, Task


def _NormalizeScores(scores: Any) -> Dict[str, float]:
    """Accept a dict of task-metric scores, a bare float, or None."""
    if scores is None:
        return {}
    if isinstance(scores, (int, float)):
        return {"score": float(scores)}
    return dict(scores)


def _AggregateScores(perCase: Dict[str, List[float]]) -> Dict[str, Any]:
    """Summarize per-case task scores into {metric: {mean, per_case}}."""
    return {
        name: {
            "mean": (sum(values) / len(values)) if values else None,
            #"per_case": values,
        }
        for name, values in perCase.items()
    }


def _MeanValue(stats: Dict[str, Any]) -> Any:
    """Extract a metric's mean from its summary dict.

    Task metrics are ``{"mean": value}``; system/method metrics are
    ``{"<name>_mean": value, ...}`` (or ``{name: None}`` when empty).
    """
    if not stats:
        return None
    if "mean" in stats:
        return stats["mean"]
    for key, value in stats.items():
        if key.endswith("_mean"):
            return value
    return next(iter(stats.values()), None)


def _CoreReport(run: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten one run into ``{method, task, <metric>: mean, ...}``."""
    core: Dict[str, Any] = {"method": run["method"], "task": run["task"]}
    for group in ("task_metrics", "system_metrics", "method_metrics"):
        for name, stats in run.get(group, {}).items():
            core[name] = _MeanValue(stats)
    return core


class Engine:
    """Controls the evaluation process.

    Evaluates the cartesian product ``methods x tasks``. For each pair the
    engine groups cases into batches of :attr:`batchSize`, runs each batch
    through ``Prepare -> Run``, then feeds every result through
    ``Evaluate`` and all system :class:`Metric` objects, and aggregates the
    task's own metrics and the system metrics into a report.
    """

    def __init__(self, verbose: bool = True, batchSize: int = 1):
        self.verbose = verbose
        self.batchSize = max(1, int(batchSize))

    def Evaluate(
        self,
        tasks: Iterable[Task],
        methods: Iterable[Method],
        metrics: Iterable[Metric],
    ) -> Dict[str, Any]:
        """Run the benchmark and return the full report.

        Args:
            tasks: one or more :class:`Task` instances.
            methods: one or more :class:`Method` instances.
            metrics: one or more system :class:`Metric` instances.

        Returns:
            A report dict with one entry per (method, task) pair.
        """
        tasks = list(tasks)
        methods = list(methods)
        metrics = list(metrics)

        if self.verbose:
            print(
                f"[engine] evaluating {len(tasks)} task(s) x {len(methods)} method(s) "
                f"with {len(metrics)} metric(s)"
            )

        runs = [
            self._evaluatePair(task, method, metrics)
            for method in methods
            for task in tasks
        ]
        return {
            "runs": runs,
            "cores": [_CoreReport(run) for run in runs],
        }

    def _evaluatePair(
        self,
        task: Task,
        method: Method,
        metrics: List[Metric],
    ) -> Dict[str, Any]:
        if self.verbose:
            print(f"[engine]   method={method.Label!r} task={task.name!r}")

        for metric in metrics:
            metric.Reset()

        taskScores: Dict[str, List[float]] = {}
        methodScores: Dict[str, List[float]] = {
            name: [] for name in method.method_metrics
        }
        nCases = 0

        batch: List[Case] = []
        for case in task.Cases():
            batch.append(case)
            if len(batch) >= self.batchSize:
                nCases += self._processBatch(
                    task, method, metrics, batch, taskScores, methodScores
                )
                batch = []
        if batch:
            nCases += self._processBatch(
                task, method, metrics, batch, taskScores, methodScores
            )

        report = {
            "method": method.Label,
            "task": task.name,
            "cases": nCases,
            "task_metrics": _AggregateScores(taskScores),
            "system_metrics": {m.name: m.Summary() for m in metrics},
        }
        if method.method_metrics:
            report["method_metrics"] = {
                name: AggregateStats(values, name=name)
                for name, values in methodScores.items()
            }
        return report

    def _processBatch(
        self,
        task: Task,
        method: Method,
        metrics: List[Metric],
        batch: List[Case],
        taskScores: Dict[str, List[float]],
        methodScores: Dict[str, List[float]],
    ) -> int:
        """Run one batch of cases through ``Prepare -> Run``.

        ``batch`` is a list of :class:`Case` objects (at most ``batchSize``).
        The method is prepared on all cases at once, run on all prompts at once
        (results in the same order), then each result is scored per-sample and
        fed to every system :class:`Metric`; finally :meth:`Method.Reset` clears
        the batch's state. Returns the number of cases in the batch.
        """
        method.Prepare([c.prepare_input for c in batch])
        results = method.Run([c.run_input for c in batch])

        for case, result in zip(batch, results):
            for name, value in _NormalizeScores(
                task.Evaluate(result, case.metadata)
            ).items():
                taskScores.setdefault(name, []).append(value)
            for metric in metrics:
                metric.Update(result)
            for name in method.method_metrics:
                value = result.metadata.get(name)
                if value is not None:
                    methodScores[name].append(float(value))

        method.Reset()
        return len(batch)
