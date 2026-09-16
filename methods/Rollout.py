"""Repeated-evaluation wrapper for existing KVBench Methods.

The wrapper repeats a Method.Run call for each input action, not each Task
Case.  The input batch shape is therefore unchanged: a batch of ``N``
original samples still returns ``N`` aggregate Results.  This is important for
the Engine's existing sample and system-metric accounting.

Task metrics are intentionally not calculated here.  A Method only sees raw
inference Results, while a Task owns the scorer (and may score textual model
outputs). The Worker asks this wrapper for its temporary raw Results and
averages those task scores once per original Case.
"""

from __future__ import annotations

import inspect
import math
import time
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Sequence

from core.Method import Method, ResolveMaxNewTokens
from core.Result import NormalizeScores, Result, TotalTimeKey


def _IsScalar(value: Any) -> bool:
    """Whether ``value`` is a numeric scalar suitable for statistics."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _Stats(values: Sequence[Any]) -> Optional[Dict[str, float]]:
    """Return population statistics for a non-empty numeric sequence."""
    if not values or not all(_IsScalar(value) for value in values):
        return None
    numbers = [float(value) for value in values]
    mean = sum(numbers) / len(numbers)
    variance = sum((value - mean) ** 2 for value in numbers) / len(numbers)
    return {
        "mean": mean,
        "variance": variance,
        "std": math.sqrt(variance),
    }


def _AggregateMapping(
    values: Sequence[Mapping[str, Any]],
    *,
    includeSamples: bool = False,
) -> tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Aggregate only scalar mapping fields; retain metadata otherwise.

    Numeric fields are averaged independently. Nested structures and
    identifiers are not silently averaged: the first value is retained when
    present in every rollout, which mirrors the existing diagnostic metadata
    behavior while keeping the numeric path generic.
    """
    if not values:
        return {}, {}

    keys = []
    seen = set()
    for value in values:
        for key in value:
            if key not in seen:
                seen.add(key)
                keys.append(key)

    aggregate: Dict[str, Any] = {}
    metrics: Dict[str, Dict[str, Any]] = {}
    for key in keys:
        present = [value[key] for value in values if key in value]
        if len(present) != len(values):
            # A value that is absent in one rollout is not a well-defined
            # rollout statistic. Preserve it only if it is available at all.
            if present:
                aggregate[key] = present[0]
            continue
        stats = _Stats(present)
        if stats is not None:
            aggregate[key] = stats["mean"]
            metricStats: Dict[str, Any] = dict(stats)
            if includeSamples:
                # Keep the original per-rollout metric values even when the
                # much larger raw Result objects are omitted from the report.
                metricStats["samples"] = list(present)
            metrics[str(key)] = metricStats
        else:
            aggregate[key] = present[0]
    return aggregate, metrics


def _CallRun(
    method: Method,
    data: List[str],
    retainOutput: List[bool],
    maxNewTokens: int,
) -> List[Result]:
    """Call a wrapped Method while preserving older two-argument Methods."""
    try:
        parameters = inspect.signature(method.Run).parameters.values()
    except (TypeError, ValueError):
        parameters = ()
    acceptsBudget = any(
        parameter.name == "maxNewTokens"
        or parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )
    if acceptsBudget:
        return method.Run(data, retainOutput, maxNewTokens=maxNewTokens)
    return method.Run(data, retainOutput)


class RolloutMethod(Method):
    """Run an existing Method repeatedly and aggregate within each sample.

    ``Run`` makes ``num_rollouts`` calls to ``base_method.Run`` using the same
    input batch. It returns one Result per input, so the Engine continues to
    observe one sample per original Case rather than a flattened sample ×
    rollout workload.

    Population variance (``ddof=0``) is used because the requested rollouts
    are treated as the observed distribution for each sample. If a wrapped
    Method raises, the exception is propagated unchanged; no failed rollout is
    converted into a score of zero.
    """

    name = "rollout"
    # The Worker only activates the three generic Case-result hooks for an
    # explicit opt-in. Ordinary Methods retain the historical report shape.
    _supportsCaseResultHooks = True
    _includeTaskMetricStatistics = True
    _rolloutMetadataKey = "rollout"

    def __init__(
        self,
        base_method: Method,
        num_rollouts: int = 10,
        keep_individual_results: bool = True,
        *,
        keep_optional_metadata: bool = True,
        tag: Optional[str] = None,
    ):
        if not isinstance(base_method, Method):
            raise TypeError("base_method must be a Method")
        if isinstance(num_rollouts, bool) or not isinstance(num_rollouts, int):
            raise TypeError("num_rollouts must be an integer")
        if num_rollouts < 1:
            raise ValueError("num_rollouts must be at least 1")
        if not isinstance(keep_individual_results, bool):
            raise TypeError("keep_individual_results must be a bool")
        if not isinstance(keep_optional_metadata, bool):
            raise TypeError("keep_optional_metadata must be a bool")

        # Match the wrapped lifecycle contract. In particular, stateful
        # methods that require one Case per batch must keep that restriction.
        self.maxCaseBatchSize = base_method.maxCaseBatchSize
        super().__init__(
            gpuNums=base_method.gpuNums,
            perfWeight=base_method.perfWeight,
            tag=tag,
        )
        self.base_method = base_method
        self.num_rollouts = num_rollouts
        self.keep_individual_results = keep_individual_results
        self.keep_optional_metadata = keep_optional_metadata
        self.method_metrics = tuple(base_method.method_metrics)
        self._pendingRolloutResults: Dict[int, List[Result]] = {}

    # -------------------------------------------------------------- lifecycle
    def Initialize(self, gpuIds: Sequence[int]) -> None:
        super().Initialize(gpuIds)
        self.base_method.Initialize(gpuIds)

    def Prepare(self, data: List[List[str]]) -> None:
        self.base_method.Prepare(data)

    def Reset(self) -> None:
        self.base_method.Reset()
        self._pendingRolloutResults.clear()

    def Close(self) -> None:
        self.base_method.Close()

    # ------------------------------------------------------------------ Run
    def Run(
        self,
        data: List[str],
        retainOutput: Optional[List[bool]] = None,
        maxNewTokens: Optional[int] = None,
    ) -> List[Result]:
        maxNewTokens = ResolveMaxNewTokens(maxNewTokens)
        retain = list(retainOutput or [False] * len(data))
        if len(retain) < len(data):
            retain.extend([False] * (len(data) - len(retain)))

        rolloutResults: List[List[Result]] = []
        elapsedByRollout: List[float] = []
        for _ in range(self.num_rollouts):
            started = time.perf_counter()
            results = _CallRun(self.base_method, data, retain, maxNewTokens)
            elapsedByRollout.append(time.perf_counter() - started)
            if len(results) != len(data):
                raise RuntimeError(
                    f"{self.base_method.Label}.Run returned {len(results)} "
                    f"result(s) for {len(data)} action(s)"
                )
            rolloutResults.append(results)

        aggregated: List[Result] = []
        for sampleIndex in range(len(data)):
            individual = [results[sampleIndex] for results in rolloutResults]
            result = self._AggregateSample(
                individual,
                elapsedByRollout=elapsedByRollout,
            )
            if not self.keep_individual_results:
                # Task.Evaluate still needs the raw Results briefly, but the
                # heavy objects are kept outside Result.metadata and released
                # after the Worker scores this Case.
                self._pendingRolloutResults[id(result)] = individual
            aggregated.append(result)
        return aggregated

    def RolloutResults(self, result: Result) -> Optional[List[Result]]:
        """Return raw Results for Worker-side Task.Evaluate, when available."""
        pending = self._pendingRolloutResults.get(id(result))
        if pending is not None:
            return pending
        rollout = result.metadata.get(self._rolloutMetadataKey)
        if isinstance(rollout, Mapping):
            values = rollout.get("results")
            if isinstance(values, list):
                return values
        return None

    def ReleaseRolloutResults(self, result: Result) -> None:
        """Release the private raw-result bridge for one aggregate Result."""
        self._pendingRolloutResults.pop(id(result), None)

    def EvaluateCase(
        self,
        task: Any,
        result: Result,
        metadata: Dict[str, Any],
    ) -> Dict[str, float]:
        """Score each raw rollout through the existing Task scorer.

        This optional Method hook is invoked only by the Worker when a Method
        provides it. Keeping the hook here means the Worker does not need to
        know what a rollout is or how its statistics are represented.
        """
        individual = self.RolloutResults(result)
        if not individual:
            return NormalizeScores(task.Evaluate(result, metadata))

        perRollout = [
            NormalizeScores(task.Evaluate(item, metadata))
            for item in individual
        ]
        names: List[str] = []
        seen = set()
        for scores in perRollout:
            for name in scores:
                if name not in seen:
                    seen.add(name)
                    names.append(name)

        aggregate: Dict[str, float] = {}
        rolloutStats: Dict[str, Dict[str, Any]] = {}
        for name in names:
            values = [scores[name] for scores in perRollout if name in scores]
            if len(values) != len(perRollout):
                # A missing metric is not silently treated as zero.
                continue
            stats = _Stats(values)
            if stats is None:
                continue
            aggregate[name] = stats["mean"]
            metricStats: Dict[str, Any] = dict(stats)
            # Task scores are metrics, not metadata. Keep every raw score so
            # callers can recompute distributions without retaining Results.
            metricStats["samples"] = list(values)
            rolloutStats[name] = metricStats

        rollout = result.metadata.get(self._rolloutMetadataKey)
        if isinstance(rollout, Mapping):
            rollout["task_metrics"] = rolloutStats
            self._SetPrimaryTaskStats(result, rolloutStats)
        return aggregate

    @staticmethod
    def _SetPrimaryTaskStats(
        result: Result, taskMetrics: Mapping[str, Mapping[str, Any]]
    ) -> None:
        """Expose a stable primary task metric on the legacy rollout aliases.

        ``rollout.mean`` predates task-aware scoring and originally referred to
        numeric ``Result.output``. Language-generation outputs are strings, so
        that value was ``null`` even though accuracy had already been scored.
        Prefer accuracy-like names, then use the first returned task metric.
        Per-metric values remain authoritative under ``task_metrics``.
        """
        if not taskMetrics:
            return
        name = next(
            (
                candidate
                for candidate in ("accuracy", "acc", "score", "reward")
                if candidate in taskMetrics
            ),
            next(iter(taskMetrics)),
        )
        stats = taskMetrics[name]
        if not isinstance(stats, Mapping):
            return
        rollout = result.metadata.get(RolloutMethod._rolloutMetadataKey)
        if not isinstance(rollout, Mapping):
            return
        rollout["primary_metric"] = name
        for key in ("mean", "variance", "std"):
            rollout[key] = stats.get(key)
        result.metadata.update(
            {
                "rollout_mean": stats.get("mean"),
                "rollout_variance": stats.get("variance"),
                "rollout_std": stats.get("std"),
            }
        )

    def SampleResult(
        self,
        case: Any,
        result: Result,
        scores: Dict[str, float],
    ) -> Dict[str, Any]:
        """Return one serializable record for one original Case."""
        record: Dict[str, Any] = {
            "sample_id": case.workflow.case_id,
            "task_metrics": dict(scores),
        }
        # Flat aliases make the common one-metric analysis table convenient;
        # task_metrics remains the unambiguous namespace for multiple metrics.
        record.update(scores)
        rollout = result.metadata.get(self._rolloutMetadataKey)
        if isinstance(rollout, Mapping):
            taskMetrics = rollout.get("task_metrics", {})
            record["rollout"] = _SerializeValue(
                rollout,
                keepOptionalMetadata=self.keep_optional_metadata,
            )
            record["rollout_metrics"] = _SerializeValue(
                taskMetrics,
                keepOptionalMetadata=self.keep_optional_metadata,
            )
            if isinstance(taskMetrics, Mapping) and taskMetrics:
                # These aliases intentionally mirror the primary metric chosen
                # by _SetPrimaryTaskStats (accuracy/acc/score/reward first).
                record.update(
                    {
                        "rollout_mean": rollout.get("mean"),
                        "rollout_variance": rollout.get("variance"),
                        "rollout_std": rollout.get("std"),
                    }
                )
        return record

    def _AggregateSample(
        self,
        individual: List[Result],
        *,
        elapsedByRollout: Sequence[float],
    ) -> Result:
        outputs = [result.output for result in individual]
        outputStats = _Stats(outputs)
        output = outputStats["mean"] if outputStats is not None else outputs[0]

        performance, performanceMetrics = _AggregateMapping(
            [result.performance for result in individual], includeSamples=True
        )
        metadata, metadataMetrics = _AggregateMapping(
            [result.metadata for result in individual], includeSamples=True
        )

        # Numeric output is the most natural scalar quality signal for direct
        # Method users. For normal language-generation Methods, Task.Evaluate
        # supplies the quality statistics in the Worker instead.
        primaryStats = outputStats
        if primaryStats is None:
            qualityValues = [
                result.metadata.get("quality") for result in individual
            ]
            primaryStats = _Stats(qualityValues)
        primaryMetric = None
        if outputStats is not None:
            primaryMetric = "output"
        elif primaryStats is not None:
            primaryMetric = "quality"

        metricSummaries: Dict[str, Any] = {
            # Result.performance is the mandatory raw metric namespace.
            "performance": performanceMetrics,
        }
        if self.keep_optional_metadata:
            metricSummaries["metadata"] = metadataMetrics

        throughputValues = []
        for raw in individual:
            tokens = raw.performance.get("num_output_tokens")
            total = raw.performance.get(TotalTimeKey)
            if _IsScalar(tokens) and _IsScalar(total) and float(total) > 0:
                throughputValues.append(float(tokens) / float(total))
        throughputStats = _Stats(throughputValues)
        if throughputStats is not None:
            throughputMetricStats: Dict[str, Any] = dict(throughputStats)
            throughputMetricStats["samples"] = list(throughputValues)
            # Throughput is derived from mandatory performance fields, so it
            # remains available even when all metadata is disabled.
            metricSummaries["throughput"] = throughputMetricStats

        if outputStats is not None:
            metricSummaries["output"] = dict(outputStats)

        rollout: Dict[str, Any] = {
            "num_rollouts": len(individual),
            "primary_metric": primaryMetric,
            "mean": primaryStats["mean"] if primaryStats else None,
            "variance": primaryStats["variance"] if primaryStats else None,
            "std": primaryStats["std"] if primaryStats else None,
            "metrics": metricSummaries,
            # These are batch-level timings: one Method.Run on a batch can
            # serve multiple original samples. Per-sample performance remains
            # in Result.performance and its rollout metric summaries.
            "execution": {
                "rollout_batch_elapsed_seconds": list(elapsedByRollout),
                "wrapper_total_elapsed_seconds": sum(elapsedByRollout),
            },
        }
        if self.keep_individual_results:
            rollout["results"] = individual

        metadata.update(
            {
                "num_rollouts": len(individual),
                "rollout_mean": rollout["mean"],
                "rollout_variance": rollout["variance"],
                "rollout_std": rollout["std"],
                self._rolloutMetadataKey: rollout,
            }
        )
        if TotalTimeKey in performance:
            rollout["total_rollout_time"] = sum(
                float(result.performance[TotalTimeKey])
                for result in individual
                if _IsScalar(result.performance.get(TotalTimeKey))
            )

        return Result(
            output=output,
            performance=performance,
            metadata=metadata,
        )


def _SerializeValue(
    value: Any,
    *,
    keepOptionalMetadata: bool = True,
) -> Any:
    """Serialize nested rollout Results without changing core Result.

    Performance is always retained. When optional metadata is disabled, raw
    Result metadata is omitted entirely. Metrics have their own stable
    locations: performance metrics stay under ``rollout.metrics.performance``
    and task metrics stay under ``rollout.task_metrics``. This prevents a
    metadata field from accidentally becoming part of the metric contract
    while still retaining all metric samples.
    """
    if isinstance(value, Result):
        rawMetadata = value.metadata
        if not keepOptionalMetadata:
            rawMetadata = {}
        return {
            "output": _SerializeValue(
                value.output,
                keepOptionalMetadata=keepOptionalMetadata,
            ),
            "performance": _SerializeValue(
                value.performance,
                keepOptionalMetadata=keepOptionalMetadata,
            ),
            "metadata": _SerializeValue(
                rawMetadata,
                keepOptionalMetadata=keepOptionalMetadata,
            ),
        }
    if isinstance(value, Mapping):
        return {
            str(key): _SerializeValue(
                item,
                keepOptionalMetadata=keepOptionalMetadata,
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _SerializeValue(
                item,
                keepOptionalMetadata=keepOptionalMetadata,
            )
            for item in value
        ]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


__all__ = ["RolloutMethod"]
