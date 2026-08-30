"""SkillsBench, treated as a kvbench Task.

SkillsBench's expert-curated tasks become Cases: each rollout is one Case,
driven by :class:`workload.AgentBenchFlowWorkload.AgentBenchFlowWorkload`.
The Task is responsible only for *case generation* and *scoring* — it
doesn't know about HTTP, subprocesses, or how the rollout is executed.

Configuration
-------------
The Task reads defaults from ``config.yaml`` (``AgentBenchFlow.*``). All
constructor parameters may be omitted to pick up the config defaults; the
constructor signature also lets callers override anything explicitly.

Case discovery
--------------
By default the Task scans ``<skillsbench_dir>/tasks/*/task.md`` and yields one
Case per directory. Pass ``task_ids`` explicitly to run a subset; pass
``exclude_task_ids`` to drop known-broken entries without editing the list.

Scoring
-------
:meth:`Evaluate` reads the rollout's ``result.json`` (passed via
``workload.final_result.output``) and extracts the reward. The synthetic
result.json we write in direct mode uses the ``reward`` key directly; in
bench mode the same key is used by ``bench eval run``.
"""

from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Union

from core.Config import AgentBenchFlowDefaults, AgentBenchFlowSkillsBenchRepo
from core.Result import Result
from core.Task import Case, Task

from workload.AgentBenchFlowWorkload import AgentBenchFlowInput, AgentBenchFlowWorkload


_REWARD_KEYS = ("reward", "rewards", "score", "scores")


class AgentBenchFlowTask(Task):
    """One SkillsBench rollout per Case.

    Args:
        skillsbench_dir: Root of the cloned SkillsBench repo. Falls back to
            ``config.yaml`` ``AgentBenchFlow.SkillsBenchRepo``.
        task_ids: Subset of task ids to run; ``None`` (default) scans the
            ``tasks/`` directory.
        exclude_task_ids: Task ids to skip from the discovered/listed set.
        max_samples: Optional cap on the number of cases yielded.
        run_mode: ``"direct"`` (default) spawns the agent CLI directly;
            ``"bench"`` invokes ``bench eval run``.
        agent: BenchFlow agent name (only used when ``run_mode="bench"``).
        agent_command: CLI binary to spawn (only used when ``run_mode="direct"``).
        model: Model id passed through to the agent; empty string lets the
            agent pick its own default.
        sandbox: Sandbox backend forwarded to ``bench eval run``.
        endpoint_env_key: Env var the agent's LLM client reads for its base URL.
        endpoint_env_overrides: Extra env vars injected into the agent subprocess.
        output_dir: Optional shared directory for all cases' output files.
        agent_extra_args / bench_extra_args: Extra CLI flags appended to the
            agent / bench invocation.
        result_json_timeout: Seconds to wait for the agent + verifier.
    """

    name = "agent_benchflow"

    def __init__(
        self,
        *,
        skillsbench_dir: Optional[Union[str, Path]] = None,
        task_ids: Optional[Sequence[str]] = None,
        exclude_task_ids: Optional[Sequence[str]] = None,
        max_samples: Optional[int] = None,
        sandbox_type: Optional[str] = None,
        image_override: Optional[str] = None,
        image_overrides: Optional[Mapping[str, str]] = None,
        agent_command: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
        agent_extra_args: Optional[Sequence[str]] = None,
        result_json_timeout: Optional[float] = None,
        thinking: Optional[bool] = True,
    ):
        cfg = AgentBenchFlowDefaults()
        self.skillsbenchDir = Path(
            skillsbench_dir if skillsbench_dir is not None else AgentBenchFlowSkillsBenchRepo()
        )
        self.excludeTaskIds = set(exclude_task_ids or ())
        self.maxSamples = max_samples
        self.sandboxType = sandbox_type or cfg.get("SandboxType", "docker")
        # image_override = single URI used for every task (e.g. "docker://python:3.11-slim").
        # image_overrides = per-task dict, overrides the single one if the task id is present.
        # Falls back to per-task dict in config, then to image_override, then to None
        # (which makes the sandbox parse the FROM line of the task's Dockerfile).
        cfgOverrides = cfg.get("ImageOverrides") or {}
        self.imageOverridesDict: Dict[str, str] = dict(image_overrides or cfgOverrides)
        self.imageOverrideDefault: Optional[str] = (
            image_override if image_override is not None else cfg.get("ImageOverride")
        )
        self.agentCommand = agent_command or cfg.get("AgentCommand", "mini-swe-agent")
        self.outputDir = Path(output_dir) if output_dir else (
            Path(cfg["OutputDir"]) if cfg.get("OutputDir") else None
        )
        self.agentExtraArgs = list(agent_extra_args or cfg.get("AgentExtraArgs") or [])
        self.resultJsonTimeout = (
            float(result_json_timeout)
            if result_json_timeout is not None
            else float(cfg.get("ResultJsonTimeoutSec", 3600))
        )
        # CoT toggle for the agent pipeline. ``True`` (default) lets the model
        # reason; ``False`` suppresses CoT. The per-arch kwarg translation
        # lives in ``helpers.ModelAdapter.render_chat``. ``None`` collapses to
        # ``True``.
        self.thinking = thinking if thinking is not None else True
        self._resolvedTaskIds = self._ResolveTaskIds(task_ids)

    def _ResolveImageOverride(self, task_id: str) -> Optional[str]:
        return self.imageOverridesDict.get(task_id, self.imageOverrideDefault)

    # ------------------------------------------------------------- case source
    def _ResolveTaskIds(self, task_ids: Optional[Sequence[str]]) -> List[str]:
        if task_ids is None:
            tasksRoot = self.skillsbenchDir / "tasks"
            if not tasksRoot.is_dir():
                raise FileNotFoundError(
                    f"no tasks/ directory under {self.skillsbenchDir} "
                    f"(expected SkillsBench repo layout)"
                )
            discovered = sorted(
                entry.name
                for entry in tasksRoot.iterdir()
                if (entry / "task.md").is_file()
            )
        else:
            if not task_ids:
                raise ValueError(
                    "task_ids=[] is almost certainly a typo; pass None to "
                    "discover from <skillsbench_dir>/tasks/ or a non-empty list"
                )
            discovered = list(task_ids)

        filtered = [tid for tid in discovered if tid not in self.excludeTaskIds]
        if self.maxSamples is not None:
            filtered = filtered[: int(self.maxSamples)]
        if not filtered:
            raise ValueError(
                f"no SkillsBench tasks to run after filtering "
                f"(skillsbench_dir={self.skillsbenchDir}, "
                f"exclude={sorted(self.excludeTaskIds)})"
            )
        return filtered

    def _CaseOutputDir(self, task_id: str) -> Optional[Path]:
        if self.outputDir is None:
            return None
        return self.outputDir / task_id

    # -------------------------------------------------------------- core.Task
    def Cases(self) -> Iterator[Case]:
        for index, taskId in enumerate(self._resolvedTaskIds):
            caseOutputDir = self._CaseOutputDir(taskId)
            data = AgentBenchFlowInput(
                skillsbench_dir=str(self.skillsbenchDir),
                task_id=taskId,
                sandbox_type=self.sandboxType,
                image_override=self._ResolveImageOverride(taskId),
                agent_command=self.agentCommand,
                agent_extra_args=self.agentExtraArgs,
                output_dir=str(caseOutputDir) if caseOutputDir is not None else None,
                result_json_timeout=self.resultJsonTimeout,
                thinking=self.thinking,
            )
            yield Case(
                input=data,
                workload=AgentBenchFlowWorkload(case_id=index, data=data),
                metadata={
                    "case_id": index,
                    "task_id": taskId,
                    "skillsbench_dir": str(self.skillsbenchDir),
                },
            )

    def Evaluate(self, result: Result, metadata: Dict[str, Any]) -> Dict[str, float]:
        payload = result.output
        reward = self._ExtractReward(payload)
        return {
            "reward": float(reward),
            "accuracy": float(reward),
        }

    # ---------------------------------------------------------------- scoring
    @staticmethod
    def _ExtractReward(payload: Any) -> float:
        if payload is None:
            return 0.0
        if isinstance(payload, dict):
            for key in _REWARD_KEYS:
                if key not in payload:
                    continue
                value = payload[key]
                coerced = _CoerceReward(value)
                if coerced is not None:
                    return coerced
        if isinstance(payload, (int, float)):
            return float(payload)
        return 0.0


def _CoerceReward(value: Any) -> Optional[float]:
    """Best-effort cast of a reward-like field to a single float."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    if isinstance(value, dict):
        numbers: List[float] = []
        for item in value.values():
            coerced = _CoerceReward(item)
            if coerced is not None:
                numbers.append(coerced)
        if numbers:
            return sum(numbers) / len(numbers)
        return None
    if isinstance(value, (list, tuple)):
        numbers = []
        for item in value:
            coerced = _CoerceReward(item)
            if coerced is not None:
                numbers.append(coerced)
        if numbers:
            return sum(numbers) / len(numbers)
        return None
    return None


__all__ = ["AgentBenchFlowTask"]
