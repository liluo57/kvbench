"""Small Workload bridge between a real BenchFlow rollout and KVBench."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from core.Config import ModelPath
from core.Result import Result
from core.Workload import Action, ActionKind, ActionResult, Workload

from helpers.benchflow import BenchflowRunner, RemoteBenchflowRunner
from helpers.endpoint import OpenAIRequest


def _ToolCallArguments(toolCall: Any) -> Optional[Dict[str, Any]]:
    """Return decoded arguments for an OpenAI tool call, when available."""
    if not isinstance(toolCall, dict):
        return None
    function = toolCall.get("function")
    if not isinstance(function, dict):
        return None
    arguments = function.get("arguments")
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str):
        return None
    try:
        decoded = json.loads(arguments)
    except (TypeError, ValueError):
        return None
    return decoded if isinstance(decoded, dict) else None


def _MessageText(content: Any) -> Optional[str]:
    """Extract text from a string or an OpenAI content-part list."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    parts: List[str] = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") != "text":
            continue
        text = part.get("text")
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts) if parts else None


def _ExtractSkillDocuments(messages: Sequence[Dict[str, Any]]) -> List[str]:
    """Extract only Skill document bodies from a provider message history.

    BenchFlow exposes a Skill in two separate messages: an assistant tool call
    reading ``.../SKILL.md`` followed by the matching ``tool`` response.  The
    system message contains only the available-Skill index and the user
    message contains the task; neither is a preparation target.
    """
    skillToolCallIds = set()
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        toolCalls = message.get("tool_calls")
        if not isinstance(toolCalls, list):
            continue
        for toolCall in toolCalls:
            arguments = _ToolCallArguments(toolCall)
            path = arguments.get("path") if arguments is not None else None
            if not isinstance(path, str) or Path(path).name.casefold() != "skill.md":
                continue
            callId = toolCall.get("id") if isinstance(toolCall, dict) else None
            if callId is not None:
                skillToolCallIds.add(str(callId))

    documents: List[str] = []
    seen = set()
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "tool":
            continue
        callId = message.get("tool_call_id")
        if callId is None or str(callId) not in skillToolCallIds:
            continue
        content = _MessageText(message.get("content"))
        if content and content not in seen:
            seen.add(content)
            documents.append(content)
    return documents


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
    remote_endpoint: Optional[str] = None
    remote_advertise_host: Optional[str] = None
    remote_auth_token_env: str = "KVBENCH_REMOTE_TOKEN"
    remote_connect_timeout: float = 10.0
    remote_poll_interval: float = 1.0
    remote_artifact_download_retries: int = 3
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
        self._pendingKind: Optional[ActionKind] = None
        self._pendingActionSent = False
        self._lastResult: Optional[Result] = None
        self._finalResult: Optional[Result] = None
        self._finished = False
        self._skillDocuments: List[str] = []
        self._skillDocumentSet = set()
        self._skillsPrepared = False
        self._RememberSkillDocuments(self._LoadBundledSkillDocuments())

    def _LoadBundledSkillDocuments(self) -> List[str]:
        """Load local task skills so they can be prepared before turn one.

        Local runs have the exact SkillsBench source tree that BenchFlow
        uploads into the sandbox.  Preparing these files up front avoids one
        prepare/reset cycle per Skill as the agent progressively reads them.
        Dataset-mode runs do not expose that tree here and use the request
        history fallback in :meth:`next` instead.
        """
        if self._data.skill_mode != "with-skill" or not self._data.skillsbench_dir:
            return []
        skillsRoot = (
            Path(self._data.skillsbench_dir)
            / "tasks"
            / self._data.task_id
            / "environment"
            / "skills"
        )
        if not skillsRoot.is_dir():
            return []
        documents: List[str] = []
        for skillPath in sorted(skillsRoot.glob("*/SKILL.md")):
            try:
                content = skillPath.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
            if content:
                documents.append(content)
        return documents

    def _RememberSkillDocuments(self, documents: Sequence[str]) -> List[str]:
        """Add unseen Skill bodies and return the newly observed ones."""
        newDocuments: List[str] = []
        for document in documents:
            if not document or document in self._skillDocumentSet:
                continue
            self._skillDocumentSet.add(document)
            self._skillDocuments.append(document)
            newDocuments.append(document)
        return newDocuments

    def _RunAction(self, request: OpenAIRequest) -> Action:
        return Action(
            kind=ActionKind.RUN,
            case_id=self.case_id,
            data=request.prompt,
            tag="agent_turn",
            retainOutput=True,
        )

    def next(self) -> Optional[List[Action]]:
        if self._finished:
            return None
        if self._pending is not None:
            if self._pendingKind != ActionKind.RUN or self._pendingActionSent:
                raise RuntimeError(
                    "AgentBenchFlowWorkload.next() called before the prior "
                    "action was observed"
                )
            self._pendingActionSent = True
            return [self._RunAction(self._pending)]
        try:
            if self._runner is None:
                runnerArgs = dict(
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
                if self._data.sandbox == "remote-docker":
                    if not self._data.remote_endpoint:
                        raise ValueError(
                            "AgentBenchFlow.RemoteDocker.Endpoint is required "
                            "when Sandbox=remote-docker"
                        )
                    self._runner = RemoteBenchflowRunner(
                        remoteEndpoint=self._data.remote_endpoint,
                        advertiseHost=self._data.remote_advertise_host,
                        remoteAuthTokenEnv=self._data.remote_auth_token_env,
                        remoteConnectTimeout=self._data.remote_connect_timeout,
                        remotePollInterval=self._data.remote_poll_interval,
                        artifactDownloadRetries=(
                            self._data.remote_artifact_download_retries
                        ),
                        **runnerArgs,
                    )
                else:
                    self._runner = BenchflowRunner(**runnerArgs)
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

        self._pending = request
        newDocuments = self._RememberSkillDocuments(
            _ExtractSkillDocuments(request.messages)
        )
        # A local task can be prepared before the first RUN.  In dataset mode,
        # the first Skill body becomes visible only after the agent has read it;
        # in either case the data sent to Method.Prepare is *only* Skill text.
        shouldPrepare = bool(newDocuments) or (
            bool(self._skillDocuments) and not self._skillsPrepared
        )
        if shouldPrepare:
            self._pendingKind = ActionKind.PREPARE
            self._pendingActionSent = True
            return [
                Action(
                    kind=ActionKind.PREPARE,
                    case_id=self.case_id,
                    data=list(self._skillDocuments),
                    tag="skill_documents",
                )
            ]

        self._pendingKind = ActionKind.RUN
        self._pendingActionSent = True
        return [self._RunAction(request)]

    def observe(self, results: List[ActionResult]) -> None:
        if len(results) != 1:
            raise ValueError(
                "AgentBenchFlowWorkload expects exactly one ActionResult per "
                f"step, got {len(results)}"
            )
        if self._pending is None:
            raise RuntimeError("AgentBenchFlowWorkload observed a result without a pending request")
        if self._pendingKind == ActionKind.PREPARE:
            # PREPARE has no model result and must not release the provider's
            # request.  The next step runs the same complete prompt.
            self._skillsPrepared = True
            self._pendingKind = ActionKind.RUN
            self._pendingActionSent = False
            return
        if self._pendingKind != ActionKind.RUN:
            raise RuntimeError("AgentBenchFlowWorkload has an invalid pending action")
        result = results[0].result
        self._lastResult = result
        request = self._pending
        self._pending = None
        self._pendingKind = None
        self._pendingActionSent = False
        output = "" if result.output is None else str(result.output)
        if self._runner is None:
            raise RuntimeError("AgentBenchFlowWorkload has no BenchFlow runner")
        try:
            # Keep compatibility with lightweight/custom runners that expose
            # the original respond(request, output) contract. The native
            # stop fields are present only on the vLLM Result metadata.
            stopMetadata = {}
            if "finish_reason" in result.metadata:
                stopMetadata["finishReason"] = result.metadata["finish_reason"]
            if "stop_reason" in result.metadata:
                stopMetadata["stopReason"] = result.metadata["stop_reason"]
            self._runner.respond(request, output, **stopMetadata)
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
        self._pendingKind = None
        self._pendingActionSent = False
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
