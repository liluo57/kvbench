"""SkillsBench task selection and scoring for the real BenchFlow runtime."""

from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from core.Config import Get
from core.Result import Result
from core.Task import Case, Task
from workload.AgentBenchFlowWorkload import AgentBenchFlowInput, AgentBenchFlowWorkload


_REWARD_KEYS = ("reward", "score", "rewards", "scores")


class AgentBenchFlowTask(Task):
    """Expose one real BenchFlow rollout as one KVBench Case.

    KVBench selects the task id; a local SkillsBench checkout supplies the
    selected task files when local source mode is used. BenchFlow remains the
    source of task parsing, Docker setup, skills,
    agent execution, verification, and the official result artifact. For
    reproducible runs, ``source_mode="dataset"`` selects a pinned registry
    dataset such as ``skillsbench@1.1``.
    """

    name = "agent_benchflow"

    # Each task id owns an independent BenchFlow rollout.  A broken rollout
    # must not prevent the remaining task ids from being evaluated.
    continueOnCaseFailure = True

    # Cache of (skillsbench tasks root, task id) pairs whose Docker image has
    # been confirmed available locally. Constructing many AgentBenchFlowTask
    # instances over the same source checkout must not re-run `docker image
    # inspect` per instance.
    _validatedTaskKeys: set = set()

    def __init__(self, task_id: str):
        """Create one task from the shared AgentBenchFlow config.

        Task selection remains in ``Main.py``; runtime behavior is configured
        only through ``config.yaml``.
        """
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("task_id must be a non-empty string")
        super().__init__(tag=task_id)
        abf = Get("AgentBenchFlow", {}) or {}
        self.sourceMode = abf.get("SourceMode", "dataset")
        if self.sourceMode not in {"dataset", "local"}:
            raise ValueError("source_mode must be 'dataset' or 'local'")
        self.dataset = abf.get("Dataset", "skillsbench@1.1")
        configuredRepo = abf.get("SkillsBenchRepo")
        repoValue = configuredRepo
        self.skillsbenchDir = Path(repoValue) if repoValue else None
        if self.sourceMode == "local" and self.skillsbenchDir is None:
            raise FileNotFoundError(
                "local AgentBenchFlow source requires AgentBenchFlow.SkillsBenchRepo"
            )
        self.agent = abf.get("Agent", "pi-acp")
        self.sandbox = abf.get("Sandbox", "docker")
        self.skillMode = abf.get("SkillMode", "with-skill")
        if self.skillMode not in {"with-skill", "no-skill"}:
            raise ValueError("skill_mode must be 'with-skill' or 'no-skill'")
        self.providerHost = abf.get("ProviderHost", "127.0.0.1")
        self.endpointHost = abf.get("EndpointHost", "0.0.0.0")
        self.port = int(abf.get("Port", 0))
        configuredPortRange = abf.get("EndpointPortRange")
        if configuredPortRange is None:
            self.endpointPortRange = None
        else:
            values = tuple(int(value) for value in configuredPortRange)
            if len(values) != 2:
                raise ValueError(
                    "endpoint_port_range must contain [first_port, last_port]"
                )
            firstPort, lastPort = values
            if not (1 <= firstPort <= lastPort <= 65535):
                raise ValueError(
                    "endpoint_port_range must be an inclusive range within ports 1-65535"
                )
            self.endpointPortRange = (firstPort, lastPort)
        self.modelId = abf.get("ModelId")
        configuredOutput = abf.get("OutputDir")
        self.outputDir = Path(configuredOutput) if configuredOutput else None
        self.resultJsonTimeout = float(abf.get("ResultJsonTimeoutSec", 3600))
        self.thinking = abf.get("Thinking")
        self.providerApiKey = abf.get("ProviderApiKey")
        self.providerApiKeyEnv = abf.get(
            "ProviderApiKeyEnv", "KVBENCH_PROVIDER_API_KEY"
        )
        self.benchCommand = abf.get("BenchCommand", "bench")
        self.benchExtraArgs = list(abf.get("BenchExtraArgs") or [])
        remote = abf.get("RemoteDocker", {}) or {}
        if not isinstance(remote, Mapping):
            raise ValueError("AgentBenchFlow.RemoteDocker must be a mapping")
        self.remoteEndpoint = remote.get("Endpoint")
        self.remoteAdvertiseHost = remote.get("KVBenchAdvertiseHost")
        self.remoteAuthTokenEnv = remote.get(
            "AuthTokenEnv", "KVBENCH_REMOTE_TOKEN"
        )
        self.remoteConnectTimeout = float(remote.get("ConnectTimeoutSec", 10))
        self.remotePollInterval = float(remote.get("PollIntervalSec", 1))
        self.remoteArtifactDownloadRetries = int(
            remote.get("ArtifactDownloadRetries", 3)
        )
        if self.sandbox == "remote-docker" and not self.remoteEndpoint:
            raise ValueError(
                "AgentBenchFlow.RemoteDocker.Endpoint is required when "
                "Sandbox=remote-docker"
            )
        self._resolvedTaskIds = [task_id]
        if (
            self.sourceMode == "local"
            and self.skillsbenchDir is not None
            and self.sandbox != "remote-docker"
        ):
            AgentBenchFlowTask._EnsureLocalImages(
                self.skillsbenchDir / "tasks", self._resolvedTaskIds
            )

    @classmethod
    def _EnsureLocalImages(
        cls,
        skillsbenchTasksRoot: Path,
        taskIds: Sequence[str],
    ) -> None:
        """Verify BenchFlow Docker images are prebuilt; raise before Engine.

        Mirrors what used to live as :func:`ValidateSkillsbenchImages` in
        ``Main.py``: it refuses to silently trigger a BenchFlow Dockerfile
        build at run-time. The check is memoised on
        ``(skillsbenchTasksRoot, taskId)`` so callers that construct one
        ``AgentBenchFlowTask`` per task id do not re-run ``docker image
        inspect`` for every instance.
        """
        # Imports kept local so the cold path does not pay the benchflow
        # import cost in dataset-only runs.
        import shutil
        import subprocess

        try:
            from benchflow.task.document import TaskDocument
        except ImportError as exc:
            raise RuntimeError(
                "BenchFlow is not importable; install the same environment "
                "that provides the `bench` command"
            ) from exc

        if not shutil.which("docker"):
            raise RuntimeError(
                "docker is not on PATH; run scripts/PrepareSkillsbench.py first "
                "after installing Docker"
            )

        missingConfiguration: List[str] = []
        missingImages: List[str] = []
        newlyVerified = 0
        for taskId in taskIds:
            key: Tuple[str, str] = (str(skillsbenchTasksRoot), taskId)
            if key in cls._validatedTaskKeys:
                continue
            taskFile = skillsbenchTasksRoot / taskId / "task.md"
            try:
                image = TaskDocument.from_path(taskFile).config.sandbox.docker_image
            except Exception as exc:  # noqa: BLE001 - report the task clearly
                raise RuntimeError(
                    f"could not read BenchFlow config for {taskId}: {exc}"
                ) from exc
            if not image:
                missingConfiguration.append(taskId)
                continue
            inspected = subprocess.run(
                ["docker", "image", "inspect", image],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if inspected.returncode != 0:
                missingImages.append(f"{taskId} ({image})")
            else:
                cls._validatedTaskKeys.add(key)
                newlyVerified += 1

        if missingConfiguration or missingImages:
            details: List[str] = []
            if missingConfiguration:
                details.append("no image in task.md: " + ", ".join(missingConfiguration))
            if missingImages:
                details.append("local image missing: " + ", ".join(missingImages))
            raise RuntimeError(
                "SkillsBench is not initialized; AgentBenchFlowTask refuses to "
                "trigger task Dockerfile builds. "
                + "; ".join(details)
                + ". Run: python scripts/PrepareSkillsbench.py --proxy <proxy>"
            )
        if newlyVerified:
            print(f"[main] verified {newlyVerified} prebuilt SkillsBench images")

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
                endpoint_port_range=self.endpointPortRange,
                model_id=self.modelId,
                output_dir=str(caseOutputDir) if caseOutputDir is not None else None,
                result_json_timeout=self.resultJsonTimeout,
                thinking=self.thinking,
                provider_api_key=self.providerApiKey,
                provider_api_key_env=self.providerApiKeyEnv,
                bench_command=self.benchCommand,
                bench_extra_args=self.benchExtraArgs,
                remote_endpoint=(
                    str(self.remoteEndpoint) if self.remoteEndpoint is not None else None
                ),
                remote_advertise_host=(
                    str(self.remoteAdvertiseHost)
                    if self.remoteAdvertiseHost is not None
                    else None
                ),
                remote_auth_token_env=str(self.remoteAuthTokenEnv),
                remote_connect_timeout=self.remoteConnectTimeout,
                remote_poll_interval=self.remotePollInterval,
                remote_artifact_download_retries=(
                    self.remoteArtifactDownloadRetries
                ),
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
                    "sandbox": self.sandbox,
                },
            )

    def Evaluate(self, result: Result, metadata: Dict[str, Any]) -> Dict[str, float]:
        reward = self._ExtractReward(result.output)
        scores: Dict[str, float] = {
            "reward": float(reward),
            "accuracy": float(reward),
        }
        # The workload records one TTFT / reuse_ratio / prompt length reading
        # per case, taken from that case's first inference result (the
        # Skill-inlined turn). A case that failed before its first RUN
        # completes simply omits these keys, so the per-case mean in the
        # report excludes it.
        firstTtft = result.metadata.get("first_run_ttft")
        if firstTtft is not None:
            scores["first_run_ttft"] = float(firstTtft)
        firstReuse = result.metadata.get("first_run_reuse_ratio")
        if firstReuse is not None:
            scores["first_run_reuse_ratio"] = float(firstReuse)
        firstPromptLength = result.metadata.get("first_run_prompt_length")
        if firstPromptLength is not None:
            scores["first_run_prompt_length"] = float(firstPromptLength)
        return scores

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
