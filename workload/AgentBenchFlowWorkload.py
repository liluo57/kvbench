"""Small Workload bridge between a real BenchFlow rollout and KVBench."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from core.Config import ModelPath
from core.Result import Result
from core.Workload import Action, ActionKind, ActionResult, Workload

from helpers.benchflow import BenchflowRunner


@dataclass
class AgentBenchFlowInput:
    """Per-case configuration passed to :class:`AgentBenchFlowWorkload`."""

    task_id: str
    source_mode: str = "dataset"
    dataset: Optional[str] = "skillsbench@1.1"
    skillsbench_dir: Optional[str] = None
    agent: str = "pi-acp"
    sandbox: str = "docker"
    skill_mode: str = "with-skill"
    provider_host: str = "127.0.0.1"
    endpoint_host: str = "0.0.0.0"
    port: int = 0
    model_id: Optional[str] = None
    output_dir: Optional[str] = None
    result_json_timeout: float = 3600.0
    thinking: Optional[bool] = None
    provider_api_key: Optional[str] = None
    provider_api_key_env: str = "KVBENCH_PROVIDER_API_KEY"
    bench_command: str = "bench"
    bench_extra_args: Sequence[str] = field(default_factory=tuple)
    #: Filled after the runner starts; useful to callers and diagnostics.
    endpoint_url: str = field(default="", init=False)


class AgentBenchFlowWorkload(Workload):
    """Convert each external agent request into an ordinary RUN Action."""

    def __init__(
        self,
        case_id: int,
        data: AgentBenchFlowInput,
        *,
        runner: Optional[BenchflowRunner] = None,
    ):
        self.case_id = case_id
        self._data = data
        self._runner = runner
        self._pending = None
        self._lastResult: Optional[Result] = None
        self._finalResult: Optional[Result] = None
        self._finished = False

    def next(self) -> Optional[List[Action]]:
        if self._finished:
            return None
        try:
            if self._runner is None:
                self._runner = BenchflowRunner(
                    taskId=self._data.task_id,
                    modelPath=ModelPath(),
                    sourceMode=self._data.source_mode,
                    dataset=self._data.dataset,
                    skillsbenchDir=self._data.skillsbench_dir,
                    agent=self._data.agent,
                    sandbox=self._data.sandbox,
                    skillMode=self._data.skill_mode,
                    providerHost=self._data.provider_host,
                    endpointHost=self._data.endpoint_host,
                    port=self._data.port,
                    modelId=self._data.model_id,
                    jobsDir=self._data.output_dir,
                    resultJsonTimeout=self._data.result_json_timeout,
                    thinking=self._data.thinking,
                    providerApiKey=self._data.provider_api_key,
                    providerApiKeyEnv=self._data.provider_api_key_env,
                    benchCommand=self._data.bench_command,
                    extraArgs=self._data.bench_extra_args,
                )
            self._runner.start()
            self._data.endpoint_url = getattr(self._runner, "endpointUrl", "")
            request = self._runner.wait_for_request()
        except BaseException as exc:
            self.fail(exc)
            return None
        if request is None:
            self._finalResult = self._BuildFinalResult()
            self._finished = True
            return None

        if self._pending is not None:
            raise RuntimeError("BenchFlow produced a request before the prior one was observed")
        self._pending = request
        return [
            Action(
                kind=ActionKind.RUN,
                case_id=self.case_id,
                data=request.prompt,
                tag="agent_turn",
                retainOutput=True,
            )
        ]

    def observe(self, results: List[ActionResult]) -> None:
        if len(results) != 1:
            raise ValueError(
                "AgentBenchFlowWorkload expects exactly one ActionResult per "
                f"step, got {len(results)}"
            )
        if self._pending is None:
            raise RuntimeError("AgentBenchFlowWorkload observed a result without a pending request")
        result = results[0].result
        self._lastResult = result
        request = self._pending
        self._pending = None
        output = "" if result.output is None else str(result.output)
        if self._runner is None:
            raise RuntimeError("AgentBenchFlowWorkload has no BenchFlow runner")
        try:
            self._runner.respond(request, output)
        except BaseException as exc:
            self.fail(exc)

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def final_result(self) -> Optional[Result]:
        if self._finalResult is not None:
            return self._finalResult
        return self._lastResult

    def fail(self, error: BaseException | str) -> None:
        """Finish this rollout as a zero-score case and release its runner."""
        if self._finished:
            return
        message = (
            f"{type(error).__name__}: {error}"
            if isinstance(error, BaseException)
            else str(error)
        )
        metadata: Dict[str, Any] = {}
        if self._runner is not None:
            try:
                metadata.update(self._runner.Diagnostics())
            except BaseException:
                pass
            metadata["benchflow_error"] = message
            try:
                self._runner.stop()
            except BaseException:
                pass
        self._pending = None
        self._finalResult = Result(
            output={"reward": 0.0, "error": message},
            performance={},
            metadata=metadata,
        )
        self._finished = True

    def _BuildFinalResult(self) -> Result:
        payload = self._runner.officialResult if self._runner is not None else None
        metadata: Dict[str, Any] = (
            self._runner.Diagnostics() if self._runner is not None else {}
        )
        if self._runner is not None:
            # The monitor has already collected BenchFlow's result. Close the
            # listening endpoint before the case is handed to Task.Evaluate.
            self._runner.stop()
        return Result(output=payload, performance={}, metadata=metadata)

    def close(self) -> None:
        if self._runner is not None:
            self._runner.stop()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001 - best effort only
            pass


__all__ = ["AgentBenchFlowInput", "AgentBenchFlowWorkload"]
