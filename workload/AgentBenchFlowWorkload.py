"""One SkillsBench rollout, driven by mini-swe-agent (or another CLI agent).

The Workload is the kvbench-facing side of the bridge; the HTTP server, the
agent subprocess, and the verifier live in :class:`helpers.BenchflowHelper`.
The Workload's contract is the usual :class:`core.Workload.Workload`:

- :meth:`next` blocks until the agent's next LLM call arrives, then emits a
  single :class:`~core.Workload.Action` carrying the rendered prompt. The
  Engine turns that into ``Method.Run``, which is the only place our KV cache
  optimization enters the picture — Method itself has no idea that the
  prompt came from an agent over HTTP.
- :meth:`observe` hands the assistant text back to the Helper, which unblocks
  the waiting HTTP request and lets the agent continue its turn.
- :meth:`final_result` returns the parsed ``result.json`` so
  :class:`tasks.AgentBenchFlowTask.AgentBenchFlowTask.Evaluate` can read the
  rollout's reward.

The rollout ends when the Helper's watchdog detects the agent subprocess
exit + (in direct mode) the verifier has finished scoring; :meth:`next` then
returns ``None``.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from core.Result import Result
from core.Workload import Action, ActionKind, ActionResult, Workload

from helpers.BenchflowHelper import BenchflowHelper


@dataclass
class AgentBenchFlowInput:
    """Per-case data for :class:`AgentBenchFlowWorkload`.

    The fields are a thin pass-through to :class:`BenchflowHelper`; defaults
    match what the Helper would pick if left to its own devices.
    """

    skillsbench_dir: str
    task_id: str
    # ── Orchestration ──────────────────────────────────────────────────────
    sandbox_type: str = "docker"  # "docker" | "apptainer" | "local"
    # ── Sandbox image override (apptainer only) ───────────────────────────
    # When non-None, the sandbox pulls this OCI/Docker URI instead of the
    # FROM line of environment/Dockerfile. Used to skip Dockerfile RUN steps
    # (which need newuidmap/fakeroot, unavailable in rootless installs) by
    # pointing at a pre-built image that already has the required tools.
    image_override: Optional[str] = None
    # ── Agent settings ─────────────────────────────────────────────────────
    agent_command: str = "mini-swe-agent"
    agent_extra_args: Sequence[str] = field(default_factory=tuple)
    # ── CoT toggle (Qwen3 + Muse Glimmer) ──────────────────────────────────
    # ``True`` lets the model reason (Qwen3 omits the empty pre-closed
    # ``<think></think>`` block; Muse Glimmer sets
    # ``Reasoning strength: high.``). ``False`` suppresses CoT. ``None``
    # collapses to ``True``.
    thinking: Optional[bool] = True
    # ── Network ────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: Optional[int] = None
    # ── Output / lifecycle ─────────────────────────────────────────────────
    output_dir: Optional[str] = None
    result_json_timeout: float = 3600.0
    #: Populated by the Helper on first ``next`` so callers can inspect where
    #: the rollout landed without recomputing paths.
    endpoint_url: str = field(default="", init=False)


class AgentBenchFlowWorkload(Workload):
    """Bridge a SkillsBench rollout through a CLI agent and ``Method.Run``."""

    def __init__(self, case_id: int, data: AgentBenchFlowInput):
        self.case_id = case_id
        self._data = data
        self._helper: Optional[BenchflowHelper] = None
        self._pending: Optional[Any] = None
        self._lastResult: Optional[Result] = None
        self._finalResult: Optional[Result] = None

    # ------------------------------------------------------------------ Workload
    def next(self) -> Optional[List[Action]]:
        if self._helper is None:
            self._helper = BenchflowHelper(
                skillsbench_dir=self._data.skillsbench_dir,
                task_id=self._data.task_id,
                sandbox_type=self._data.sandbox_type,
                image_override=self._data.image_override,
                agent_command=self._data.agent_command,
                agent_extra_args=self._data.agent_extra_args,
                host=self._data.host,
                port=self._data.port,
                output_dir=self._data.output_dir,
                result_json_timeout=self._data.result_json_timeout,
                thinking=self._data.thinking,
            )
            self._data.endpoint_url = self._helper.endpointUrl

        item = self._helper.wait_for_request()
        if item is None:
            self._finalResult = self._SynthesiseFinalResult()
            return None

        prompt, _future = item
        self._pending = item
        return [Action(
            kind=ActionKind.RUN,
            case_id=self.case_id,
            data=prompt,
            tag="agent_turn",
            retainOutput=False,
        )]

    def observe(self, results: List[ActionResult]) -> None:
        if len(results) != 1:
            raise ValueError(
                f"AgentBenchFlowWorkload expects exactly one ActionResult per "
                f"step, got {len(results)}"
            )
        result = results[0].result
        self._lastResult = result
        if self._pending is None:
            return
        _prompt, future = self._pending
        self._pending = None
        output = "" if result.output is None else str(result.output)
        self._helper.respond(future, output)

    @property
    def finished(self) -> bool:
        if self._helper is None:
            return False
        return self._helper.is_done

    @property
    def final_result(self) -> Optional[Result]:
        if self._finalResult is not None:
            return self._finalResult
        return self._lastResult

    # --------------------------------------------------------------- helpers
    def _SynthesiseFinalResult(self) -> Result:
        payload = self._helper.final_result() if self._helper is not None else None
        output: Any = payload if payload is not None else self._lastResult
        metadata: Dict[str, Any] = {}
        if self._helper is not None:
            metadata["endpoint_url"] = self._helper.endpointUrl
            metadata["output_dir"] = str(self._helper.outputDir)
            metadata["agent_log"] = str(self._helper.agentLogPath)
            metadata["verifier_log"] = str(self._helper.verifierLogPath)
            metadata["run_mode"] = self._helper.runMode
        if isinstance(payload, dict):
            for key in ("reward", "rewards", "score", "scores", "agent_exit_code",
                        "verifier_ran", "verifier_error"):
                if key in payload:
                    metadata[f"benchflow_{key}"] = payload[key]
        return Result(output=output, performance={}, metadata=metadata)

    def __del__(self) -> None:
        helper = getattr(self, "_helper", None)
        if helper is not None:
            try:
                helper.stop()
            except Exception:  # noqa: BLE001 - destructor must not raise
                pass


__all__ = ["AgentBenchFlowInput", "AgentBenchFlowWorkload"]
