"""The :class:`BenchflowHelper` orchestrator.

Owns the four pieces of a SkillsBench rollout — HTTP endpoint, sandbox,
agent, and verifier — and exposes only the queue/sync the Workload needs.
Method.Run is unchanged; the Helper's only contact with kvbench's core is
the request queue.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import queue
import sys
import tempfile
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from helpers.SkillInjector import BuildSkillsBlock

from .http_handler import HelperHTTPHandler
from .sandbox import ApptainerSandbox, DockerSandbox, LocalSandbox, Sandbox
from .util import FindFreePort


class BenchflowHelper:
    """Own the HTTP endpoint, the sandbox, the agent, and the verifier for
    one SkillsBench rollout.
    """

    _DONE_SENTINEL: Optional[Tuple[str, "concurrent.futures.Future[str]"]] = None

    def __init__(
        self,
        *,
        skillsbench_dir: str | Path,
        task_id: str,
        model_path: str,
        # ── Sandbox selection ───────────────────────────────────────────────
        sandbox_type: str = "docker",  # "docker" | "apptainer" | "local"
        image_override: Optional[str] = None,  # ApptainerSandbox only
        # ── Agent settings ──────────────────────────────────────────────────
        agent_command: str = "mini-swe-agent",
        agent_extra_args: Sequence[str] = (),
        # ── Network ────────────────────────────────────────────────────────
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        # ── Output / lifecycle ──────────────────────────────────────────────
        output_dir: Optional[str | Path] = None,
        result_json_timeout: float = 3600.0,
        # ── CoT toggle (Qwen3 + Muse Glimmer) ───────────────────────────────
        # ``True`` (default) lets the model reason: Qwen3 omits the empty
        # pre-closed ``<think></think>`` block; Muse Glimmer sets the system
        # prompt's ``Reasoning strength: high.``. ``False`` suppresses CoT
        # (inject the Qwen block / set ``low.``). ``None`` collapses to
        # ``True`` (the helper does not auto-detect CoT).
        thinking: Optional[bool] = True,
    ):
        self.skillsbenchDir = Path(skillsbench_dir)
        self.taskId = task_id
        self.modelPath = model_path
        self.sandboxType = sandbox_type
        # Compatibility field used by workload diagnostics. This adapter
        # executes a direct CLI agent rather than a named BenchFlow agent.
        self.runMode = "direct"
        self.imageOverride = image_override
        self.agentCommand = agent_command
        self.agentExtraArgs = list(agent_extra_args)
        self.host = host
        self.resultJsonTimeout = float(result_json_timeout)
        self.thinking = thinking if thinking is not None else True

        self.outputDir = Path(output_dir) if output_dir else Path(
            tempfile.mkdtemp(prefix=f"kvbench-benchflow-{task_id}-")
        )
        self.outputDir.mkdir(parents=True, exist_ok=True)
        self.agentWorkDir = self.outputDir / "agent_workspace"
        self.agentWorkDir.mkdir(parents=True, exist_ok=True)

        self.port = int(port) if port else FindFreePort(host)
        # The server may listen on all interfaces, but the in-sandbox client
        # must use loopback so inherited HTTP proxy variables cannot intercept
        # the local OpenAI-compatible endpoint.
        clientHost = "127.0.0.1" if host in ("", "0.0.0.0") else host
        self.endpointUrl = f"http://{clientHost}:{self.port}"

        # Internal state.
        self._requestQueue: "queue.Queue[Optional[Tuple[str, concurrent.futures.Future[str]]]]" = queue.Queue()
        self._doneEvent = threading.Event()
        self._stopped = False
        self._stopLock = threading.Lock()
        self._finalResult: Optional[Dict[str, Any]] = None
        self._server: Optional[ThreadingHTTPServer] = None
        self._serverThread: Optional[threading.Thread] = None
        self._sandbox: Optional[Sandbox] = None
        self._watchdogThread: Optional[threading.Thread] = None

        self.agentLogPath = self.outputDir / "agent.log"
        self.verifierLogPath = self.outputDir / "verifier.log"

        # Pre-render the task's SkillsBench skills once so every LLM turn
        # starts from the same prefix (KV-cache friendly across rollouts of
        # the same task) and so the agent doesn't have to discover skills
        # itself — mini-swe-agent has no skill loader.
        self._skillsBlock = self._BuildSkillsBlock()

        # Muse Glimmer's CoT toggle now flows through ModelAdapter.render_chat's
        # ``chat_template_kwargs={"reasoning_strength": ...}``, which the
        # ATEM template's ``render_reasoning`` macro reads directly. No
        # string-prepending needed here — the adapter owns the translation.

        self._StartServer()
        self._StartSandboxAndAgent()
        self._StartWatchdog()

    # ------------------------------------------------------------- public API
    @property
    def is_done(self) -> bool:
        return self._doneEvent.is_set()

    def wait_for_request(self, timeout: Optional[float] = None) -> Optional[Tuple[str, concurrent.futures.Future[str]]]:
        try:
            item = self._requestQueue.get(timeout=timeout)
        except queue.Empty:
            return None
        if item is self._DONE_SENTINEL:
            return None
        return item

    def respond(self, future: concurrent.futures.Future[str], output_text: str) -> None:
        if not future.done():
            future.set_result(output_text)

    def final_result(self) -> Optional[Dict[str, Any]]:
        return self._finalResult

    def stop(self) -> None:
        with self._stopLock:
            if self._stopped:
                return
            self._stopped = True
            self._doneEvent.set()
            self._requestQueue.put(self._DONE_SENTINEL)
            if self._sandbox is not None:
                try:
                    self._sandbox.cleanup()
                except Exception:  # noqa: BLE001
                    pass
            if self._server is not None:
                try:
                    self._server.shutdown()
                    self._server.server_close()
                except Exception:  # noqa: BLE001
                    pass

    # --------------------------------------------------------- internals: HTTP
    def _BuildSkillsBlock(self) -> str:
        """Render the task's SkillsBench skills into a system-prompt prefix.

        Errors are logged and swallowed: a missing or malformed skills
        directory should not abort a rollout, only degrade it back to the
        discovery-by-failure mode we used to have.
        """
        try:
            return BuildSkillsBlock(self.skillsbenchDir, self.taskId)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[benchflow] failed to build skills block for {self.taskId}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return ""

    def _StartServer(self) -> None:
        server = ThreadingHTTPServer((self.host, self.port), HelperHTTPHandler)
        server.helper = self  # type: ignore[attr-defined]
        self._server = server
        self._serverThread = threading.Thread(
            target=server.serve_forever,
            name=f"benchflow-http-{self.taskId}",
            daemon=True,
        )
        self._serverThread.start()

    # ----------------------------------------------------- internals: sandbox
    def _StartSandboxAndAgent(self) -> None:
        if self.sandboxType == "docker":
            self._sandbox = DockerSandbox()
        elif self.sandboxType == "apptainer":
            # The SIF is a reusable immutable image. Keeping it in the
            # configured case output directory makes retries independent of
            # Docker Hub availability after the first successful pull.
            self._sandbox = ApptainerSandbox(
                image_override=self.imageOverride,
                cleanup_image=False,
            )
        elif self.sandboxType == "local":
            self._sandbox = LocalSandbox(self.agentWorkDir)
        else:
            raise ValueError(
                f"unknown sandbox_type={self.sandboxType!r}; "
                f"expected 'docker', 'apptainer', or 'local'"
            )
        # Docker / Apptainer sandboxes need a back-reference to resolve the
        # SkillsBench path without re-passing it through the constructor.
        if isinstance(self._sandbox, (DockerSandbox, ApptainerSandbox)):
            self._sandbox._helper = self  # type: ignore[attr-defined]

        taskDir = self.skillsbenchDir / "tasks" / self.taskId
        self._sandbox.prepare(self.taskId, taskDir)

        self._agentEnv = self._BuildAgentEnv()
        self._agentCmd = self._BuildAgentCmd()

    def _BuildAgentEnv(self) -> Dict[str, str]:
        """Env vars for the agent subprocess, pointing its LLM client at us.

        Hardcoded because mini-swe-agent uses litellm with the OpenAI SDK
        convention: ``OPENAI_API_BASE`` is the base URL, the SDK appends
        ``/chat/completions``, so we put ``/v1`` in the env var to match
        our ``/v1/chat/completions`` route. Auth is not enforced — the dummy
        key just keeps SDKs that require a non-empty value happy.
        """
        env = dict(os.environ)
        env["OPENAI_API_BASE"] = f"{self.endpointUrl}/v1"
        env.setdefault("OPENAI_API_KEY", "dummy")
        noProxyValues = {
            item.strip()
            for key in ("NO_PROXY", "no_proxy")
            for item in env.get(key, "").split(",")
            if item.strip()
        }
        noProxyValues.update({"127.0.0.1", "localhost", "0.0.0.0"})
        noProxy = ",".join(sorted(noProxyValues))
        env["NO_PROXY"] = noProxy
        env["no_proxy"] = noProxy
        # Skip mini-swe-agent's first-run interactive setup.
        env.setdefault("MSWEA_CONFIGURED", "true")
        env.setdefault("MSWEA_GLOBAL_CONFIG_FILE", str(self.outputDir / "mswea.env"))
        # Our endpoint has no cost table; without this, mini-swe-agent raises.
        env.setdefault("MSWEA_COST_TRACKING", "ignore_errors")
        # mini-swe-agent refuses to start without a model name. From kvbench's
        # perspective the value is irrelevant (our endpoint ignores the
        # ``model`` field); the litellm provider prefix is what matters.
        env.setdefault("MSWEA_MODEL_NAME", "openai/any-model")
        return env

    def _BuildAgentCmd(self) -> List[str]:
        """Build the agent CLI command for the sandbox to execute."""
        taskText = self._BuildAgentTaskText()
        if self.sandboxType == "local":
            cmd: List[str] = [
                self.agentCommand,
                "--task", taskText,
                "--yolo",
                "--exit-immediately",
                "--cost-limit", "0",
                "--output", str(self.outputDir / "trajectory.json"),
            ]
            cmd.extend(self.agentExtraArgs)
            return cmd

        # The task text is passed explicitly.  The file is also mounted for
        # agents/scripts that refer to /root/task.md, but relying on that file
        # as the CLI argument made the old adapter lose the skill context.
        if self.sandboxType == "apptainer":
            # ApptainerSandbox binds host_tools -> /host_tools and prepends
            # /host_tools to PATH, so ``<agentCommand>`` resolves to the
            # host install (python-launcher script). task.md still lives at
            # /root/task.md.
            return [
                self.agentCommand,
                "--task", taskText,
                "--yolo",
                "--exit-immediately",
                "--cost-limit", "0",
                "--output", "/root/.mini-swe-agent/trajectory.json",
                *self.agentExtraArgs,
            ]

        return [
            self.agentCommand,
            "--task", taskText,
            "--yolo",
            "--exit-immediately",
            "--cost-limit", "0",
            "--output", "/root/.mini-swe-agent/trajectory.json",
            *self.agentExtraArgs,
        ]

    def _BuildAgentTaskText(self) -> str:
        """Add the adapter context needed by a generic terminal agent.

        BenchFlow normally injects skills through an agent-specific skill
        loader. mini-SWE-agent has no such loader, so the available skill
        directory and the expected workflow must be made explicit in its
        task prompt. This does not reveal verifier/oracle data or a solution.
        """
        taskMd = self.skillsbenchDir / "tasks" / self.taskId / "task.md"
        if not taskMd.is_file():
            raise FileNotFoundError(
                f"SkillsBench task {self.taskId!r} has no task.md at {taskMd}"
            )
        taskText = taskMd.read_text(encoding="utf-8")
        # The YAML front matter is useful to BenchFlow, but it is noise for a
        # terminal agent and causes mini-SWE's generic "solve an issue" prompt
        # to compete with the actual user request.  Keep the human task body.
        if taskText.startswith("---"):
            _, separator, body = taskText.partition("\n---")
            if separator:
                taskText = body.lstrip("\n")
        skillsDir = taskMd.parent / "environment" / "skills"
        if not skillsDir.is_dir():
            return taskText
        return "\n\n## Task\n" + taskText.rstrip()

    # --------------------------------------------------- internals: watchdog
    def _StartWatchdog(self) -> None:
        self._watchdogThread = threading.Thread(
            target=self._WatchdogLoop,
            name=f"benchflow-watchdog-{self.taskId}",
            daemon=True,
        )
        self._watchdogThread.start()

    def _WatchdogLoop(self) -> None:
        """Run the agent + verifier inside the sandbox, score, and report."""
        try:
            agentExitCode = self._sandbox.run_agent(  # type: ignore[union-attr]
                task_id=self.taskId,
                cmd=self._agentCmd,
                env=self._agentEnv,
                log_path=self.agentLogPath,
            )
        except Exception as exc:  # noqa: BLE001 - surface as a failed rollout
            self._finalResult = {
                "reward": 0.0,
                "error": f"agent failed to start: {exc}",
                "sandbox_type": self.sandboxType,
                "endpoint_url": self.endpointUrl,
            }
            self._WriteResultJson()
            self._doneEvent.set()
            self._requestQueue.put(self._DONE_SENTINEL)
            return

        # Run the verifier (in-sandbox) regardless of agent exit code.
        taskDir = self.skillsbenchDir / "tasks" / self.taskId
        verifierScript = taskDir / "verifier" / "test.sh"

        verifierRan = verifierScript.is_file()
        verifierExitCode: Optional[int] = None
        verifierError: Optional[str] = None
        rewardPath: Optional[Path] = None

        if verifierRan:
            try:
                verifierExitCode, rewardPath = self._sandbox.run_verifier(  # type: ignore[union-attr]
                    task_id=self.taskId,
                    verifier_test_sh=verifierScript,
                    log_path=self.verifierLogPath,
                )
            except Exception as exc:  # noqa: BLE001
                verifierError = f"{type(exc).__name__}: {exc}"

        reward = self._ReadReward(verifierExitCode, rewardPath)

        self._finalResult = {
            "reward": reward,
            "agent_exit_code": agentExitCode,
            "verifier_ran": verifierRan,
            "verifier_exit_code": verifierExitCode,
            "verifier_error": verifierError,
            "reward_file": str(rewardPath) if rewardPath else None,
            "sandbox_type": self.sandboxType,
            "endpoint_url": self.endpointUrl,
            "output_dir": str(self.outputDir),
            "agent_log": str(self.agentLogPath),
            "verifier_log": str(self.verifierLogPath),
        }
        self._WriteResultJson()
        self._doneEvent.set()
        self._requestQueue.put(self._DONE_SENTINEL)

    @staticmethod
    def _ReadReward(
        verifierExitCode: Optional[int],
        rewardPath: Optional[Path],
    ) -> float:
        """Extract the rollout reward from verifier outputs.

        Priority order:
        1. The SkillsBench verifier's ``reward.txt`` (``1`` or ``0``).
        2. The verifier's own exit code (0 = pass → 1.0).
        3. ``0.0`` (something failed).
        """
        if rewardPath is not None and rewardPath.is_file():
            try:
                text = rewardPath.read_text(encoding="utf-8").strip()
                if text:
                    return float(text)
            except (OSError, ValueError):
                pass
        if verifierExitCode is not None:
            return 1.0 if verifierExitCode == 0 else 0.0
        return 0.0

    def _WriteResultJson(self) -> None:
        if self._finalResult is None:
            return
        resultJsonPath = self.outputDir / "result.json"
        try:
            resultJsonPath.write_text(json.dumps(self._finalResult, indent=2))
        except OSError:
            pass