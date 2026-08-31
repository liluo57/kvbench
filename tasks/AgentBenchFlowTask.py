"""SkillsBench task selection and scoring for the real BenchFlow runtime."""

from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Union

from core.Config import AgentBenchFlowDefaults
from core.Result import Result
from core.Task import Case, Task
from workload.AgentBenchFlowWorkload import AgentBenchFlowInput, AgentBenchFlowWorkload


_REWARD_KEYS = ("reward", "score", "rewards", "scores")


class AgentBenchFlowTask(Task):
    """Expose one real BenchFlow rollout as one KVBench Case.

    KVBench uses a local SkillsBench checkout only to enumerate development
    cases. BenchFlow remains the source of task parsing, Docker setup, skills,
    agent execution, verification, and the official result artifact. For
    reproducible runs, ``source_mode="dataset"`` selects a pinned registry
    dataset such as ``skillsbench@1.1``.
    """

    name = "agent_benchflow"

    # Each task id owns an independent BenchFlow rollout.  A broken rollout
    # must not prevent the remaining task ids from being evaluated.
    continueOnCaseFailure = True

    def __init__(
        self,
        *,
        skillsbench_dir: Optional[Union[str, Path]] = None,
        task_ids: Optional[Sequence[str]] = None,
        exclude_task_ids: Optional[Sequence[str]] = None,
        max_samples: Optional[int] = None,
        source_mode: Optional[str] = None,
        dataset: Optional[str] = None,
        agent: Optional[str] = None,
        sandbox: Optional[str] = None,
        skill_mode: Optional[str] = None,
        provider_host: Optional[str] = None,
        endpoint_host: Optional[str] = None,
        port: Optional[int] = None,
        model_id: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
        result_json_timeout: Optional[float] = None,
        thinking: Optional[bool] = None,
        provider_api_key: Optional[str] = None,
        provider_api_key_env: Optional[str] = None,
        bench_command: Optional[str] = None,
        bench_extra_args: Optional[Sequence[str]] = None,
    ):
        cfg = AgentBenchFlowDefaults()
        self.sourceMode = source_mode or cfg.get("SourceMode", "dataset")
        if self.sourceMode not in {"dataset", "local"}:
            raise ValueError("source_mode must be 'dataset' or 'local'")
        self.dataset = dataset if dataset is not None else cfg.get("Dataset", "skillsbench@1.1")
        configuredRepo = cfg.get("SkillsBenchRepo")
        repoValue = skillsbench_dir if skillsbench_dir is not None else configuredRepo
        self.skillsbenchDir = Path(repoValue) if repoValue else None
        if self.sourceMode == "local" and self.skillsbenchDir is None:
            raise FileNotFoundError(
                "local AgentBenchFlow source requires SkillsBenchRepo or skillsbench_dir"
            )
        self.excludeTaskIds = set(exclude_task_ids or ())
        self.maxSamples = max_samples
        self.agent = agent or cfg.get("Agent", "pi-acp")
        self.sandbox = sandbox or cfg.get("Sandbox", "docker")
        self.skillMode = skill_mode or cfg.get("SkillMode", "with-skill")
        if self.skillMode not in {"with-skill", "no-skill"}:
            raise ValueError("skill_mode must be 'with-skill' or 'no-skill'")
        self.providerHost = provider_host or cfg.get("ProviderHost", "127.0.0.1")
        self.endpointHost = endpoint_host or cfg.get("EndpointHost", "0.0.0.0")
        self.port = 0 if port is None else int(port)
        self.modelId = model_id if model_id is not None else cfg.get("ModelId")
        configuredOutput = output_dir if output_dir is not None else cfg.get("OutputDir")
        self.outputDir = Path(configuredOutput) if configuredOutput else None
        self.resultJsonTimeout = (
            float(result_json_timeout)
            if result_json_timeout is not None
            else float(cfg.get("ResultJsonTimeoutSec", 3600))
        )
        self.thinking = thinking if thinking is not None else cfg.get("Thinking")
        self.providerApiKey = provider_api_key
        self.providerApiKeyEnv = provider_api_key_env or cfg.get(
            "ProviderApiKeyEnv", "KVBENCH_PROVIDER_API_KEY"
        )
        self.benchCommand = bench_command or cfg.get("BenchCommand", "bench")
        self.benchExtraArgs = list(
            bench_extra_args if bench_extra_args is not None else cfg.get("BenchExtraArgs") or []
        )
        self._resolvedTaskIds = self._ResolveTaskIds(task_ids)

    def _ResolveTaskIds(self, task_ids: Optional[Sequence[str]]) -> List[str]:
        if task_ids is None:
            if self.skillsbenchDir is None:
                raise ValueError(
                    "task_ids is required for dataset mode when no local SkillsBench "
                    "checkout is configured"
                )
            tasksRoot = self.skillsbenchDir / "tasks"
            if not tasksRoot.is_dir():
                raise FileNotFoundError(
                    f"no tasks/ directory under {self.skillsbenchDir} "
                    "(used only for local case enumeration)"
                )
            discovered = sorted(
                entry.name
                for entry in tasksRoot.iterdir()
                if (entry / "task.md").is_file()
            )
        else:
            if not task_ids:
                raise ValueError(
                    "task_ids=[] is almost certainly a typo; pass None to discover "
                    "from the local checkout or a non-empty list"
                )
            discovered = list(task_ids)

        filtered = [taskId for taskId in discovered if taskId not in self.excludeTaskIds]
        if self.maxSamples is not None:
            filtered = filtered[: int(self.maxSamples)]
        if not filtered:
            raise ValueError(
                "no BenchFlow tasks to run after filtering "
                f"(exclude={sorted(self.excludeTaskIds)})"
            )
        return filtered

    def _CaseOutputDir(self, taskId: str) -> Optional[Path]:
        if self.outputDir is None:
            return None
        return self.outputDir / taskId

    def Cases(self) -> Iterator[Case]:
        for index, taskId in enumerate(self._resolvedTaskIds):
            caseOutputDir = self._CaseOutputDir(taskId)
            data = AgentBenchFlowInput(
                task_id=taskId,
                source_mode=self.sourceMode,
                dataset=self.dataset,
                skillsbench_dir=(
                    str(self.skillsbenchDir) if self.skillsbenchDir is not None else None
                ),
                agent=self.agent,
                sandbox=self.sandbox,
                skill_mode=self.skillMode,
                provider_host=self.providerHost,
                endpoint_host=self.endpointHost,
                port=self.port,
                model_id=self.modelId,
                output_dir=str(caseOutputDir) if caseOutputDir is not None else None,
                result_json_timeout=self.resultJsonTimeout,
                thinking=self.thinking,
                provider_api_key=self.providerApiKey,
                provider_api_key_env=self.providerApiKeyEnv,
                bench_command=self.benchCommand,
                bench_extra_args=self.benchExtraArgs,
            )
            yield Case(
                input=data,
                workload=AgentBenchFlowWorkload(case_id=index, data=data),
                metadata={
                    "case_id": index,
                    "task_id": taskId,
                    "source_mode": self.sourceMode,
                    "dataset": self.dataset,
                    "skill_mode": self.skillMode,
                    "agent": self.agent,
                },
            )

    def Evaluate(self, result: Result, metadata: Dict[str, Any]) -> Dict[str, float]:
        reward = self._ExtractReward(result.output)
        return {"reward": float(reward), "accuracy": float(reward)}

    def CaseFailureScores(
        self, metadata: Dict[str, Any], error: BaseException
    ) -> Dict[str, float]:
        """Return the score for a rollout that could not be completed.

        This is deliberately separate from :meth:`Evaluate`: a BenchFlow
        startup/bridge failure is a failed case, not a failed KVBench task.
        The worker uses this hook to record zero and continue with the next
        task id.
        """
        return {"reward": 0.0, "accuracy": 0.0}

    @staticmethod
    def _ExtractReward(payload: Any) -> float:
        """Extract the scalar correctness reward from current BenchFlow output."""
        if not isinstance(payload, Mapping):
            return float(payload) if isinstance(payload, (int, float)) else 0.0
        for key in _REWARD_KEYS:
            if key not in payload:
                continue
            value = _CoerceReward(payload[key])
            if value is not None:
                return value
        # Some result producers put canonical metrics below final_metrics.
        finalMetrics = payload.get("final_metrics")
        if isinstance(finalMetrics, Mapping):
            for key in ("reward", "score", "accuracy", "pass_rate"):
                value = _CoerceReward(finalMetrics.get(key))
                if value is not None:
                    return value
        return 0.0


def _CoerceReward(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    if isinstance(value, Mapping):
        # Canonical rewards commonly use {"reward": value} or a map of
        # criterion values. Prefer the named scalar before averaging criteria.
        for key in ("reward", "score", "value", "mean"):
            if key in value:
                direct = _CoerceReward(value[key])
                if direct is not None:
                    return direct
        values = [item for item in (_CoerceReward(v) for v in value.values()) if item is not None]
        return sum(values) / len(values) if values else None
    if isinstance(value, (list, tuple)):
        values = [item for item in (_CoerceReward(v) for v in value) if item is not None]
        return sum(values) / len(values) if values else None
    return None


__all__ = ["AgentBenchFlowTask"]
