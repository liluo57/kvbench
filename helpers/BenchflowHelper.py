"""Drive a SkillsBench rollout end-to-end: HTTP endpoint + sandbox + agent + verifier.

The Helper owns four things and exposes only the queue/sync the Workload
needs (mirroring the original split — endpoint plumbing is hidden from the
Workload, and Method is untouched):

- **HTTP server**: an OpenAI-compatible ``POST /v1/chat/completions`` endpoint
  backed by ``http.server.ThreadingHTTPServer``. Each agent request becomes a
  ``(prompt, future)`` pair in a queue; the handler blocks on the future
  until the Workload responds.
- **Sandbox**: the SkillsBench task's environment. The agent and the verifier
  both run *inside* the sandbox (DockerSandbox in production; LocalSandbox
  for dev / smoke tests). The sandbox is what guarantees the agent sees the
  task's installed dependencies and the verifier's expected paths
  (``/root/answer.json``, ``/logs/verifier/reward.txt``).
- **Watchdog**: detects when the agent exits, runs the verifier inside the
  sandbox, extracts the reward, and writes a synthetic ``result.json``.

The Helper's ``sandbox_type`` selects the isolation strategy:

- ``docker`` (default): builds the task's ``environment/Dockerfile``, runs the
  agent + verifier inside containers with the SkillsBench-expected mounts,
  extracts the reward from ``/logs/verifier/reward.txt``.
- ``apptainer``: pulls the ``FROM``-line base image of the task's
  ``environment/Dockerfile`` as a rootless SIF via ``apptainer pull`` and
  executes the agent + verifier with bind mounts. RUN steps in the upstream
  Dockerfile are NOT applied (non-root apptainer lacks fuse-overlayfs /
  newuidmap to mutate the container filesystem), so any in-container
  dependencies the task relies on (``apt-get install``, ``pip install``,
  ``COPY``) must already be present in the base image. Build/run failures
  surface as a non-zero exit code and a ``reward.txt`` of ``0`` -- the
  helper never silently falls back to a different sandbox.
- ``local``: runs everything on the host. Faster (no image build), no
  isolation — only for dev / smoke tests where the task's expected
  filesystem layout isn't required.

Cleanup happens in :meth:`stop`, which is idempotent.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import queue
import re
import shutil
import socket
import subprocess
import sys
import shlex
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from core.Config import ModelPath as _ModelPath

#: Qwen3 chat-template tokens. Mirrors :mod:`tasks.TemplateHelper`.
_ImStart = "<|im_start|>"
_ImEnd = "<|im_end|>"
_NonThinkingBlock = "<think>\n\n</think>\n\n"
_MaxToolResponseChars = 5000

#: Muse Glimmer chat-template tokens (Meta-derived ``<|start|>...<|message|>...<|eot|>``
#: style; see :mod:`tasks.TemplateHelper._MuseSystemPrefix` for the verified KB-path
#: usage).
_MGStart = "<|start|>"
_MGMessage = "<|message|>"
_MGEot = "<|eot|>"

#: Architectures whose chat default uses ``<|im_start|>...`` (Qwen3-Instruct).
_Qwen3Archs = ("Qwen3ForCausalLM",)

#: Architectures that use the Meta-style ``<|start|>/<|message|>/<|eot|>`` chat
#: format. vLLM maps ``MuseGlimmerForConditionalGeneration`` to the
#: ``MuseGlimmerForCausalLM`` class (text-only), so both names appear.
#: Duplicated from :mod:`tasks.TemplateHelper` to keep this module
#: self-contained — the SkillsBench / agent pipeline is otherwise unaware of
#: the RULER / KB-template code path.
_MuseGlimmerArchs = (
    "MuseGlimmerForCausalLM",
    "MuseGlimmerForConditionalGeneration",
)

#: Test seam: when non-None, :func:`_DetectArch` returns this verbatim
#: instead of reading ``config.json``. Production code never sets this.
_ArchOverride: Optional[str] = None

# Local import kept here (rather than at module top) to avoid a circular
# import: SkillInjector imports nothing from this module, but keeping the
# import lazy makes the helper's standalone usage in tests easier to set up.
from helpers.SkillInjector import BuildSkillsBlock  # noqa: E402

# mini-SWE-agent sends this schema in every request.  vLLM's offline LLM API
# only receives a rendered prompt, so the schema has to be rendered into the
# Qwen prompt as well; merely returning it in the HTTP response is too late.
_DefaultBashTool = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                }
            },
            "required": ["command"],
        },
    },
}


def _MessageContent(message: Mapping[str, Any]) -> str:
    """Convert OpenAI text or simple multimodal content to plain text."""
    content = message.get("content", "")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: List[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, Mapping):
                text = item.get("text")
                if text is not None:
                    chunks.append(str(text))
        return "\n".join(chunks)
    return str(content)


def _CompactToolResponse(content: str) -> str:
    """Keep long shell transcripts from consuming the whole agent context."""
    if len(content) <= _MaxToolResponseChars:
        return content
    head = _MaxToolResponseChars // 2
    tail = _MaxToolResponseChars - head
    return (
        content[:head]
        + "\n... [middle of tool output elided by kvbench] ...\n"
        + content[-tail:]
    )


def _RenderToolInstructions(tools: Sequence[Mapping[str, Any]]) -> str:
    """Render the tool section used by Qwen3's native chat template."""
    serialized = "\n".join(
        json.dumps(tool, ensure_ascii=False, separators=(",", ": "))
        for tool in tools
    )
    return (
        "# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        f"<tools>\n{serialized}\n</tools>\n\n"
        "Every assistant turn must contain exactly one bash function call. "
        "When the task is complete, call bash with `echo "
        "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` as the only command.\n\n"
        "For each function call, return a json object with function name and "
        "arguments within <tool_call></tool_call> XML tags:\n"
        "<tool_call>\n"
        '{"name": <function-name>, "arguments": <args-json-object>}\n'
        "</tool_call>"
    )


def _ToolCallFunction(tool_call: Mapping[str, Any]) -> Mapping[str, Any]:
    function = tool_call.get("function")
    return function if isinstance(function, Mapping) else tool_call


# -------------------------------------------------------------- arch detection
@lru_cache(maxsize=1)
def _DetectArch() -> str:
    """Return the chat-format arch for the configured model.

    Reads ``<ModelPath>/config.json`` and inspects the ``architectures`` list:

    - ``"muse_glimmer"`` if any architecture is in :data:`_MuseGlimmerArchs`
    - ``"qwen3"`` if any is in :data:`_Qwen3Archs`
    - ``"other"`` otherwise (plain Qwen-style header, no thinking handling)

    Any I/O failure (missing model dir / malformed config) falls back to
    ``"other"``. Honors :data:`_ArchOverride` for tests; the cache is cleared
    by :func:`_SetArchForTesting` whenever the override changes.
    """
    if _ArchOverride is not None:
        return _ArchOverride
    modelPath = _ModelPath()
    if not modelPath:
        return "other"
    try:
        with open(Path(modelPath) / "config.json", encoding="utf-8") as f:
            archs = json.load(f).get("architectures", [])
    except (OSError, ValueError):
        return "other"
    if any(a in _MuseGlimmerArchs for a in archs):
        return "muse_glimmer"
    if any(a in _Qwen3Archs for a in archs):
        return "qwen3"
    return "other"


def _SetArchForTesting(arch: Optional[str]) -> None:
    """Force :func:`_DetectArch` to return ``arch`` until cleared.

    Pass ``None`` to restore production behavior. Test-only — production code
    never touches :data:`_ArchOverride`.
    """
    global _ArchOverride
    _ArchOverride = arch
    _DetectArch.cache_clear()


# ----------------------------------------------------------- Glimmer renderer
def _RenderGlimmerToolInstructions(tools: Sequence[Mapping[str, Any]]) -> str:
    """Render the Meta-style tool instructions for Muse Glimmer.

    Differs from :func:`_RenderToolInstructions` (Qwen3) in two ways:

    - Tool schemas are emitted as plain JSON lines, not wrapped in a
      ``<tools>...</tools>`` XML envelope (Meta convention).
    - Wording follows the Llama-3 tool-use phrasing, and the tool-call tag
      uses the plain ``<tool_call>`` form (no zero-width space) so the
      Meta-tuned tokenizer treats it as a single special token sequence.
    """
    serialized = "\n".join(
        json.dumps(tool, ensure_ascii=False, separators=(",", ": "))
        for tool in tools
    )
    return (
        "You have access to a set of tools you can use to answer the user's question.\n\n"
        "# Tools\n\n"
        f"{serialized}\n\n"
        "If you intend to call a tool, output the tool call in the following format:\n"
        "<tool_call>\n"
        '{"name": <function-name>, "arguments": <args-json-object>}\n'
        "</tool_call>\n\n"
        "Every assistant turn must contain exactly one bash function call. "
        "When the task is complete, call bash with `echo "
        "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` as the only command.\n"
    )


def _RenderGlimmerChatPrompt(
    messages: List[Dict[str, Any]],
    tools: Optional[Sequence[Mapping[str, Any]]] = None,
    system_prefix: str = "",
    thinking: Optional[bool] = True,
) -> str:
    """Render an OpenAI message list using the Muse Glimmer chat format.

    All turns use the Meta envelope ``<|start|>{role}<|message|>{content}<|eot|>``.
    The assistant header ends with ``<|message|>`` (not ``<|start|>assistant``)
    so the model does not fall into the ``to=self`` reasoning mode.

    CoT is controlled by the system prompt's ``Reasoning strength`` line, which
    the BenchflowHelper appends to ``system_prefix`` before this function is
    called. ``thinking`` itself does not affect the renderer here — it is
    accepted for symmetry with the Qwen3 path so callers don't need arch-
    specific branches.
    """
    del thinking  # CoT for Glimmer is set upstream via system_prefix's
                  # "Reasoning strength" line. Accept the parameter for symmetry.

    normalizedTools = list(tools or (_DefaultBashTool,))
    parts: List[str] = []
    first = True
    for message in messages:
        role = str(message.get("role", "user"))
        content = _MessageContent(message)
        if role == "system" and first:
            if system_prefix:
                systemContent = f"{system_prefix.rstrip()}\n\n{content}"
            else:
                systemContent = content
            parts.append(
                f"{_MGStart}system{_MGMessage}{systemContent}\n\n"
                f"{_RenderGlimmerToolInstructions(normalizedTools)}{_MGEot}"
            )
        elif role in ("system", "user"):
            parts.append(f"{_MGStart}{role}{_MGMessage}{content}{_MGEot}")
        elif role == "assistant":
            text = content
            if text:
                parts.append(f"{_MGStart}assistant{_MGMessage}{text}")
            else:
                parts.append(f"{_MGStart}assistant{_MGMessage}")
            for tool_call in message.get("tool_calls") or []:
                if not isinstance(tool_call, Mapping):
                    continue
                function = _ToolCallFunction(tool_call)
                name = str(function.get("name", "bash"))
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {"command": arguments}
                if text or tool_call is not (message.get("tool_calls") or [])[0]:
                    parts.append("\n")
                parts.append(
                    "<tool_call>\n"
                    + json.dumps(
                        {"name": name, "arguments": arguments},
                        ensure_ascii=False,
                    )
                    + "\n</tool_call>"
                )
            parts.append(_MGEot)
        elif role == "tool":
            toolContent = _CompactToolResponse(content)
            parts.append(
                f"{_MGStart}user{_MGMessage}<tool_response>\n{toolContent}\n"
                f"</tool_response>{_MGEot}"
            )
        else:
            parts.append(f"{_MGStart}user{_MGMessage}{content}{_MGEot}")
        first = False
    # Final assistant header. ``<|message|>`` suffix avoids the model's
    # ``to=self`` reasoning-mode fork.
    parts.append(f"{_MGStart}assistant{_MGMessage}")
    return "".join(parts)


def _RenderChatPrompt(
    messages: List[Dict[str, Any]],
    tools: Optional[Sequence[Mapping[str, Any]]] = None,
    system_prefix: str = "",
    *,
    thinking: Optional[bool] = True,
) -> str:
    """Render OpenAI messages for the configured model.

    Dispatches on :func:`_DetectArch`:

    - ``"muse_glimmer"`` → :func:`_RenderGlimmerChatPrompt` (Meta-style
      envelope). CoT for Glimmer is controlled upstream by appending a
      ``Reasoning strength: high|low`` line to ``system_prefix``; the
      ``thinking`` arg is accepted for symmetry and ignored.
    - otherwise → Qwen3 path (this function body). CoT is gated by the
      assistant-header suffix: ``thinking=False`` injects
      :data:`_NonThinkingBlock` to suppress Qwen3-Instruct's default
      thinking mode; ``None`` and ``True`` keep the header bare so the
      model emits a ``<think>…</think>`` trace.

    ``FullPrefillVllm`` accepts a text prompt rather than an OpenAI chat
    request, and vLLM's offline LLM API only receives the rendered prompt
    string.  The old renderer dropped both the tool schema and prior tool
    calls, which made an agent conversation indistinguishable from ordinary
    chat.  This is intentionally kept in sync with Qwen3's tokenizer
    template, including ``<tool_response>`` blocks.

    ``system_prefix`` is prepended to the first ``system`` message's content
    (separated by ``\\n\\n``). It is the injection point used by
    :class:`helpers.SkillInjector.BuildSkillsBlock` to deliver the task's
    curated SkillsBench skills directly to the LLM, skipping the discovery
    step that mini-swe-agent has no native support for.
    """
    if _DetectArch() == "muse_glimmer":
        return _RenderGlimmerChatPrompt(messages, tools, system_prefix, thinking)
    normalizedTools = list(tools or (_DefaultBashTool,))
    parts: List[str] = []
    first = True
    for message in messages:
        role = str(message.get("role", "user"))
        content = _MessageContent(message)
        if role == "system" and first:
            if system_prefix:
                systemContent = f"{system_prefix.rstrip()}\n\n{content}"
            else:
                systemContent = content
            parts.append(
                f"{_ImStart}system\n{systemContent}\n\n"
                f"{_RenderToolInstructions(normalizedTools)}{_ImEnd}\n"
            )
        elif role in ("system", "user"):
            parts.append(f"{_ImStart}{role}\n{content}{_ImEnd}\n")
        elif role == "assistant":
            text = content
            if text:
                parts.append(f"{_ImStart}assistant\n{text}")
            else:
                parts.append(f"{_ImStart}assistant\n")
            for tool_call in message.get("tool_calls") or []:
                if not isinstance(tool_call, Mapping):
                    continue
                function = _ToolCallFunction(tool_call)
                name = str(function.get("name", "bash"))
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        # Preserve malformed history without crashing the
                        # bridge; the model can repair it on the next turn.
                        arguments = {"command": arguments}
                if text or tool_call is not (message.get("tool_calls") or [])[0]:
                    parts.append("\n")
                parts.append(
                    "<tool_call>\n"
                    + json.dumps(
                        {"name": name, "arguments": arguments},
                        ensure_ascii=False,
                    )
                    + "\n</tool_call>"
                )
            parts.append(f"{_ImEnd}\n")
        elif role == "tool":
            toolContent = _CompactToolResponse(content)
            parts.append(
                f"{_ImStart}user\n<tool_response>\n{toolContent}\n"
                f"</tool_response>{_ImEnd}\n"
            )
        else:
            parts.append(f"{_ImStart}user\n{content}{_ImEnd}\n")
        first = False
    # ``thinking is False`` suppresses Qwen3-Instruct's default CoT by injecting
    # the empty pre-closed think block. ``None`` and ``True`` leave the header
    # bare so the model emits its own ``<think>…</think>`` trace.
    suffix = "" if thinking is not False else _NonThinkingBlock
    parts.append(f"{_ImStart}assistant\n{suffix}")
    return "".join(parts)


#: Matches the ``<\xe2\x80\x8btool_call>...</\xe2\x80\x8btool_call>`` blocks emitted by both
#: :func:`_RenderChatPrompt` (Qwen3 path) and :func:`_RenderGlimmerChatPrompt`
#: (Meta path). Both renderers emit the plain form (no zero-width space);
#: the regex matches plain ``<\xe2\x80\x8btool_call>`` only.
_ToolCallBlock = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_CodeBlock = re.compile(
    r"```(?:mswea_bash_command|bash|sh|shell)?\s*\n?(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


def _ParseToolCalls(output: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Extract Qwen XML tool calls, with a compatibility fallback for code."""
    calls: List[Dict[str, Any]] = []
    spans: List[Tuple[int, int]] = []
    for match in _ToolCallBlock.finditer(output):
        try:
            payload = json.loads(match.group(1).strip())
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, Mapping):
            continue
        name = str(payload.get("name", "bash"))
        arguments = payload.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"command": arguments}
        if not isinstance(arguments, Mapping) or "command" not in arguments:
            continue
        calls.append({
            "id": f"call_{time.time_ns()}_{len(calls)}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(dict(arguments), ensure_ascii=False),
            },
        })
        spans.append(match.span())

    if not calls:
        # Some compatible local models follow mini-SWE's legacy fenced-command
        # wording even when the native Qwen tool instructions are present.
        match = _CodeBlock.search(output)
        if match:
            command = match.group(1).strip()
            if command:
                calls.append({
                    "id": f"call_{time.time_ns()}_0",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": json.dumps(
                            {"command": command}, ensure_ascii=False
                        ),
                    },
                })
                spans.append(match.span())

    content = output
    for start, end in reversed(spans):
        content = content[:start] + content[end:]
    return content.strip(), calls


def _FindFreePort(host: str) -> int:
    """Bind ``host:0`` and return the OS-assigned port number."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _ResolveHostSitePackages() -> Optional[Path]:
    """Locate the host's site-packages dir (so we can mount it into the SIF).

    mini-swe-agent is installed on the host (e.g.
    ``/data/lyh/miniconda3/lib/python3.13/site-packages``). Inside the SIF
    the only Python is whatever the base image ships (often empty), so we
    stage the entire host site-packages tree at ``/host_tools/_lib`` and
    point ``PYTHONPATH`` at it.
    """
    # Strategy 1: ask the active interpreter where its ``purelib`` is.
    try:
        import sysconfig
        purelib = sysconfig.get_path("purelib")
        if purelib and Path(purelib).is_dir():
            return Path(purelib)
    except Exception:  # noqa: BLE001
        pass
    # Strategy 2: find an installed ``minisweagent`` package and walk up
    # to its parent (the site-packages root).
    try:
        import minisweagent  # type: ignore[import-not-found]
        pkgFile = Path(minisweagent.__file__).resolve()
        # minisweagent/__init__.py -> minisweagent/ -> site-packages/
        return pkgFile.parent.parent
    except Exception:  # noqa: BLE001
        pass
    return None


# --------------------------------------------------------------------- handler


class _HelperHTTPHandler(BaseHTTPRequestHandler):
    """HTTP handler that bridges one ``/v1/chat/completions`` request to the
    helper's request queue and blocks until the Workload responds.
    """

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def _WriteJSON(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 - stdlib name
        helper: "BenchflowHelper" = self.server.helper  # type: ignore[attr-defined]
        if self.path != "/v1/chat/completions":
            self._WriteJSON(404, {"error": f"not found: {self.path}"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._WriteJSON(400, {"error": "invalid Content-Length"})
            return
        if length <= 0:
            self._WriteJSON(400, {"error": "empty request body"})
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._WriteJSON(400, {"error": f"invalid JSON: {exc}"})
            return
        messages = payload.get("messages") or []
        if not isinstance(messages, list):
            self._WriteJSON(400, {"error": "messages must be a list"})
            return
        tools = payload.get("tools")
        if not isinstance(tools, list) or not tools:
            tools = [_DefaultBashTool]
        prompt = _RenderChatPrompt(
            messages,
            tools,
            system_prefix=helper._skillsBlock,
            thinking=helper.thinking,
        )

        if helper._doneEvent.is_set():
            self._WriteJSON(503, {"error": "endpoint is shutting down"})
            return
        future: concurrent.futures.Future[str] = concurrent.futures.Future()
        helper._requestQueue.put((prompt, future))
        try:
            output = future.result()
        except Exception as exc:  # noqa: BLE001 - convert any failure to HTTP
            self._WriteJSON(500, {"error": f"generation failed: {exc}"})
            return

        content, toolCalls = _ParseToolCalls(output)
        message: Dict[str, Any] = {
            "role": "assistant",
            "content": content,
        }
        # mini-SWE-agent parses OpenAI ``tool_calls``; returning the generated
        # XML only as content makes it report a format error forever.  Keep
        # the raw textual reasoning in content and expose executable calls in
        # the protocol field the client actually consumes.
        if toolCalls:
            message["tool_calls"] = toolCalls

        response = {
            "id": f"chatcmpl-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": str(payload.get("model", "")),
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if toolCalls else "stop",
                }
            ],
        }
        self._WriteJSON(200, response)


# --------------------------------------------------------------------- sandbox


class Sandbox(ABC):
    """Where the agent and verifier run. SkillsBench semantics require the
    task's Dockerfile container; LocalSandbox is the non-isolated fallback.
    """

    @abstractmethod
    def prepare(self, task_id: str, task_dir: Path) -> None:
        """One-time setup (image build, dir create). May be a no-op."""

    @abstractmethod
    def run_agent(
        self,
        *,
        task_id: str,
        cmd: Sequence[str],
        env: Mapping[str, str],
        log_path: Path,
    ) -> int:
        """Run the agent. Returns the exit code when finished (blocking)."""

    @abstractmethod
    def run_verifier(
        self,
        *,
        task_id: str,
        verifier_test_sh: Path,
        log_path: Path,
    ) -> Tuple[int, Optional[Path]]:
        """Run ``bash verifier/test.sh`` and report the reward-file path.

        Returns ``(exit_code, reward_file_path)``; the reward file lives at
        ``/logs/verifier/reward.txt`` inside the sandbox.
        """

    @abstractmethod
    def cleanup(self) -> None:
        """Tear down any container / tempdirs. Idempotent."""


class LocalSandbox(Sandbox):
    """No isolation — runs everything on the host.

    The SkillsBench verifier expects ``/logs/verifier/reward.txt`` (root-only)
    and other in-container paths; LocalSandbox will fail those checks. Use
    this only for smoke-testing the HTTP plumbing.
    """

    def __init__(self, work_dir: Path):
        self.workDir = Path(work_dir)
        self.workDir.mkdir(parents=True, exist_ok=True)

    def prepare(self, task_id: str, task_dir: Path) -> None:
        return

    def run_agent(
        self,
        *,
        task_id: str,
        cmd: Sequence[str],
        env: Mapping[str, str],
        log_path: Path,
    ) -> int:
        logFile = open(log_path, "w", encoding="utf-8")
        try:
            proc = subprocess.Popen(
                list(cmd),
                stdout=logFile,
                stderr=subprocess.STDOUT,
                env=dict(env),
                cwd=str(self.workDir),
            )
            return proc.wait()
        finally:
            logFile.close()

    def run_verifier(
        self,
        *,
        task_id: str,
        verifier_test_sh: Path,
        log_path: Path,
    ) -> Tuple[int, Optional[Path]]:
        logFile = open(log_path, "w", encoding="utf-8")
        try:
            proc = subprocess.run(
                ["bash", str(verifier_test_sh)],
                stdout=logFile,
                stderr=subprocess.STDOUT,
                cwd=str(verifier_test_sh.parent),
                check=False,
            )
        finally:
            logFile.close()
        # The SkillsBench verifier writes its reward at /logs/verifier/reward.txt.
        # LocalSandbox usually can't create that path; treat absence as 0.0.
        rewardPath = Path("/logs/verifier/reward.txt")
        return proc.returncode, (rewardPath if rewardPath.is_file() else None)

    def cleanup(self) -> None:
        return


class DockerSandbox(Sandbox):
    """Build the task's Dockerfile and run agent + verifier inside containers.

    Wiring:

    - The image is built once per task id, tagged ``kvbench-skillsbench:<id>``.
      We copy the task's ``environment/Dockerfile`` into a temp build context
      and append a final ``RUN pip install mini-swe-agent`` layer so the
      upstream SkillsBench Dockerfile stays untouched.
    - **Agent run**: mount ``tasks/<id>/task.md -> /root/task.md`` and the
      output-dir ``agent_workspace`` -> ``/root`` so the agent's writes
      (notably ``/root/answer.json``) persist for the verifier. Run with
      ``--network host`` so the agent's LLM client can reach our kvbench
      endpoint on ``127.0.0.1``.
    - **Verifier run**: mount ``tasks/<id>/verifier -> /verifier`` and a
      fresh ``verifier_workspace`` -> ``/logs`` so the verifier can write
      ``/logs/verifier/reward.txt`` and we can read it back from the host.
    """

    _IMAGE_TAG_PREFIX = "kvbench-skillsbench"

    def __init__(
        self,
        *,
        network: str = "host",
        install_mini_swe: bool = True,
        extra_docker_run_args: Sequence[str] = (),
    ):
        if not shutil.which("docker"):
            raise RuntimeError(
                "DockerSandbox requires the `docker` CLI in PATH "
                "(install Docker Desktop / docker-ce first)"
            )
        self.network = network
        self.installMiniSwe = install_mini_swe
        self.extraDockerRunArgs = list(extra_docker_run_args)
        self._imageTag: Optional[str] = None
        self._agentWorkspace: Optional[Path] = None
        self._verifierWorkspace: Optional[Path] = None
        self._activeContainers: List[str] = []

    # ----------------------------------------------------------- public API
    def prepare(self, task_id: str, task_dir: Path) -> None:
        """Build the task image (idempotent: ``docker build`` is cached)."""
        dockerfileDir = task_dir / "environment"
        dockerfile = dockerfileDir / "Dockerfile"
        if not dockerfile.is_file():
            raise FileNotFoundError(
                f"SkillsBench task {task_id!r} has no environment/Dockerfile at "
                f"{dockerfile}"
            )
        self._imageTag = f"{self._IMAGE_TAG_PREFIX}:{task_id}"

        with tempfile.TemporaryDirectory(prefix=f"kvbench-sb-{task_id}-") as tmp:
            tmpPath = Path(tmp)
            # Build context: symlink every file in environment/ into the
            # tempdir so COPY directives like ``COPY test.bib /root/test.bib``
            # resolve correctly.
            for entry in dockerfileDir.iterdir():
                target = tmpPath / entry.name
                if entry.is_file():
                    target.symlink_to(entry.resolve())

            # Append our extra layer to the end of the Dockerfile (without
            # modifying the upstream file).
            with open(tmpPath / "Dockerfile", "a", encoding="utf-8") as f:
                f.write("\n# === appended by kvbench AgentBenchFlowTask ===\n")
                if self.installMiniSwe:
                    f.write(
                        "RUN pip install --break-system-packages --quiet "
                        "mini-swe-agent\n"
                    )

            subprocess.run(
                ["docker", "build", "-t", self._imageTag, str(tmpPath)],
                check=True,
            )

    def run_agent(
        self,
        *,
        task_id: str,
        cmd: Sequence[str],
        env: Mapping[str, str],
        log_path: Path,
    ) -> int:
        """Run the agent in a fresh container; return its exit code."""
        assert self._imageTag is not None

        taskDir = self._TaskDir(task_id)
        taskMdPath = taskDir / "task.md"
        if not taskMdPath.is_file():
            raise FileNotFoundError(f"missing {taskMdPath}")

        # The agent's ``/root`` lives in this dir so its writes (notably
        # ``/root/answer.json``) survive across the agent->verifier handoff.
        self._agentWorkspace = self._agentWorkspace or self._MakeWorkspace("agent")
        containerName = f"kvbench-sb-{task_id}-{int(time.time())}"
        envArgs: List[str] = []
        for key, value in env.items():
            envArgs.extend(["-e", f"{key}={value}"])

        with open(log_path, "w", encoding="utf-8") as logFile:
            dockerArgs = [
                "docker", "run", "--rm",
                "--name", containerName,
                "--network", self.network,
                "-v", f"{self._agentWorkspace}:/root",
                # Mount the task after /root: the later file mount must not
                # be hidden by the writable agent workspace mount.
                "-v", f"{taskMdPath.resolve()}:/root/task.md:ro",
                *self._SkillDockerMounts(taskDir),
                *self.extraDockerRunArgs,
                *envArgs,
                self._imageTag,
                *cmd,
            ]
            self._activeContainers.append(containerName)
            try:
                proc = subprocess.run(
                    dockerArgs,
                    stdout=logFile,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                return proc.returncode
            finally:
                self._CleanupContainer(containerName)

    def run_verifier(
        self,
        *,
        task_id: str,
        verifier_test_sh: Path,
        log_path: Path,
    ) -> Tuple[int, Optional[Path]]:
        """Run ``verifier/test.sh`` and extract ``/logs/verifier/reward.txt``."""
        assert self._imageTag is not None

        self._agentWorkspace = self._agentWorkspace or self._MakeWorkspace("agent")
        self._verifierWorkspace = self._MakeWorkspace("verifier")
        containerName = f"kvbench-sb-verify-{task_id}-{int(time.time())}"

        # Mount the agent's ``/root`` (so the verifier sees ``/root/answer.json``)
        # and a fresh ``/logs`` we can read back from the host.
        with open(log_path, "w", encoding="utf-8") as logFile:
            dockerArgs = [
                "docker", "run", "--rm",
                "--name", containerName,
                "--network", self.network,
                "-v", f"{verifier_test_sh.parent.resolve()}:/verifier:ro",
                "-v", f"{self._agentWorkspace}:/root",
                "-v", f"{self._verifierWorkspace}:/logs",
                self._imageTag,
                "bash", f"/verifier/{verifier_test_sh.name}",
            ]
            self._activeContainers.append(containerName)
            try:
                proc = subprocess.run(
                    dockerArgs,
                    stdout=logFile,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            finally:
                self._CleanupContainer(containerName)

        rewardPath = self._verifierWorkspace / "verifier" / "reward.txt"
        if not rewardPath.is_file():
            rewardPath = self._verifierWorkspace / "reward.txt"
        return proc.returncode, (rewardPath if rewardPath.is_file() else None)

    def cleanup(self) -> None:
        for name in list(self._activeContainers):
            self._CleanupContainer(name)

    # ---------------------------------------------------------- internals
    def _TaskDir(self, task_id: str) -> Path:
        """Resolve the SkillsBench task directory from outside the sandbox.

        Mirrors ``BenchflowHelper.skillsbenchDir / tasks / task_id``.
        """
        helper = self._ResolveHelper()
        return helper.skillsbenchDir / "tasks" / task_id

    def _ResolveHelper(self) -> "BenchflowHelper":
        # Set in :meth:`BenchflowHelper._StartSandboxAndAgent` (lazy import
        # would create a cycle). We keep a back-reference so the sandbox can
        # read paths / config without duplicating them on the constructor.
        return self._helper  # type: ignore[attr-defined,return-value]

    def _MakeWorkspace(self, kind: str) -> Path:
        helper = self._ResolveHelper()
        workspace = helper.outputDir / f"{kind}_workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        return workspace

    def _SkillDockerMounts(self, task_dir: Path) -> List[str]:
        """Return mounts for the task's runtime skills, if it has any."""
        skillsDir = task_dir / "environment" / "skills"
        if not skillsDir.is_dir() or self._agentWorkspace is None:
            return []
        mounts: List[str] = []
        # These are the two conventional locations used by SkillsBench
        # agents.  Keeping both makes the portable skills usable by the
        # mini-SWE adapter even when a SKILL.md contains a home-relative path.
        for relative in (".agents/skills", ".claude/skills"):
            (self._agentWorkspace / relative).mkdir(parents=True, exist_ok=True)
            mounts.extend([
                "-v",
                f"{skillsDir.resolve()}:/root/{relative}:ro",
            ])
        return mounts

    def _CleanupContainer(self, name: str) -> None:
        if name in self._activeContainers:
            self._activeContainers.remove(name)
        subprocess.run(
            ["docker", "rm", "-f", name],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


class ApptainerSandbox(Sandbox):
    """Rootless container via Apptainer (no Docker / no root / no daemon).

    Why this design instead of "build the task's Dockerfile":

    - ``apptainer build`` from an arbitrary Dockerfile needs ``newuidmap`` /
      ``newgidmap`` / ``fuse-overlayfs``, none of which are installed in the
      conda env this helper targets. It is also unreliable for arbitrary
      RUN commands (apt-get, pip install, COPY) without root.
    - ``apptainer pull docker://<base>`` IS rootless; the OCI image is
      converted to a SIF transparently. We use that as the "image" for the
      agent and verifier.

    So ``prepare`` parses the first ``FROM`` line of
    ``environment/Dockerfile``, pulls that image, and stores the SIF path.
    The Dockerfile's RUN / COPY steps are intentionally not applied; any
    in-container dependency the task expects must already be in the base
    image. (For citation-check the FROM is ``ubuntu:24.04`` which has no
    Python, so the agent and verifier will both fail unless an
    ``image_override`` is supplied — this is the documented limitation.)

    Networking: ``--network host`` and ``--net`` both require root or a
    suid apptainer with /etc/subuid, neither of which is available here.
    With NO ``--network`` flag, the container shares the host network
    namespace by default (verified on the conda-forge 1.5.3 build) -- which
    is what we want, since the agent reaches the helper via
    ``127.0.0.1:<port>``.

    The helper also bind-mounts the host toolchain (``<output>/host_tools``)
    into the container at ``/host_tools`` and prepends it to PATH so
    ``mini-swe-agent`` resolves to the host install (the base image may
    not ship it).

    Cleanup deletes the pulled SIF on shutdown.
    """

    _SIF_PREFIX = "kvbench-skillsbench-sif"

    def __init__(
        self,
        *,
        apptainer_path: str = "apptainer",
        image_override: Optional[str] = None,
        cleanup_image: bool = True,
        # Rootless caveat: ``--network host`` and ``--net`` both require
        # root or a suid apptainer with /etc/subuid. The conda-forge build
        # supports neither. With NO --network flag, the container shares the
        # host network namespaces by default — which is what we want, since
        # the agent reaches the helper via 127.0.0.1:<port>. Pass an empty
        # string to skip the flag entirely; pass "none" to drop networking.
        network_mode: str = "",
        keep_image: bool = False,
        # ``--writable-tmpfs`` gives the in-container rootfs an in-memory
        # overlay (~64 MiB) without needing ``--fakeroot``. Without it, any
        # attempt by the agent or verifier to mutate the rootfs (apt install,
        # pip install, write to ``/root/.config``) fails silently with
        # ``Read-only file system``.
        writable_tmpfs: bool = True,
    ):
        apBin = shutil.which(apptainer_path)
        if not apBin:
            raise RuntimeError(
                "ApptainerSandbox requires the `apptainer` CLI in PATH "
                "(install apptainer / singularity first)"
            )
        self.apptainerPath = apBin
        self.imageOverride = image_override
        self.cleanupImage = bool(cleanup_image)
        self.networkMode = network_mode
        self.keepImage = bool(keep_image)
        self.writableTmpfs = bool(writable_tmpfs)

        self._sifPath: Optional[Path] = None
        self._resolvedImageURI: Optional[str] = None
        self._agentWorkspace: Optional[Path] = None
        self._verifierWorkspace: Optional[Path] = None
        self._hostTools: Optional[Path] = None
        self._hostSitePackages: Optional[Path] = None
        self._hostPythonRoot: Optional[Path] = None

    # ----------------------------------------------------------- public API
    def prepare(self, task_id: str, task_dir: Path) -> None:
        """Pull the base image as a SIF and stage host-side workspaces."""
        helper = self._ResolveHelper()
        # Shared asset dirs alongside the agent / verifier workspaces.
        self._agentWorkspace = self._MakeWorkspace("agent")
        self._verifierWorkspace = self._MakeWorkspace("verifier")
        self._hostTools = helper.outputDir / "host_tools"
        self._hostTools.mkdir(parents=True, exist_ok=True)
        # Resolve the host's site-packages so we can bind-mount it into
        # the SIF. Without this, mini-swe-agent (installed only on the
        # host) can't import its dependencies from inside the container.
        self._hostSitePackages = _ResolveHostSitePackages()
        self._hostPythonRoot = Path(sys.executable).resolve().parent.parent
        self._StageHostTools(self._hostTools)
        self._StageEnvironmentInputs(task_dir)

        # Resolve the image URI: override wins; else first FROM line.
        if self.imageOverride:
            imageURI = self.imageOverride
        else:
            imageURI = self._ResolveFromImage(task_dir)
        self._resolvedImageURI = imageURI

        # Pin SIF under the task workspace so cleanup can find it.
        taskWorkspace = helper.outputDir / "sif"
        taskWorkspace.mkdir(parents=True, exist_ok=True)
        self._sifPath = taskWorkspace / f"{self._SIF_PREFIX}-{task_id}.sif"

        # A completed SIF is immutable for the selected image and can be
        # reused by pair retries. Re-pulling with ``--disable-cache`` made a
        # transient registry failure turn a recoverable rollout into a second
        # failure.
        imageMarker = self._sifPath.with_suffix(".image")
        if self._sifPath.is_file():
            try:
                if imageMarker.read_text(encoding="utf-8").strip() == imageURI:
                    return
            except OSError:
                pass
            # The marker was absent or refers to an older override. Remove
            # only this resolved case image before rebuilding it.
            try:
                self._sifPath.unlink()
            except OSError:
                pass

        # Disable cache so a downstream rebuild doesn't accidentally reuse a
        # stale SIF. ``apptainer pull`` returns non-zero on any network /
        # manifest failure — we propagate that.
        subprocess.run(
            [
                self.apptainerPath, "pull",
                "--name", str(self._sifPath),
                "--disable-cache",
                imageURI,
            ],
            check=True,
        )
        try:
            imageMarker.write_text(imageURI + "\n", encoding="utf-8")
        except OSError:
            pass

    def run_agent(
        self,
        *,
        task_id: str,
        cmd: Sequence[str],
        env: Mapping[str, str],
        log_path: Path,
    ) -> int:
        """Run the agent inside the SIF with bind mounts; return exit code."""
        assert self._sifPath is not None and self._sifPath.exists(), (
            "ApptainerSandbox.prepare() must run before run_agent()"
        )
        assert self._agentWorkspace is not None
        helper = self._ResolveHelper()
        taskDir = helper.skillsbenchDir / "tasks" / task_id
        taskMdPath = taskDir / "task.md"
        if not taskMdPath.is_file():
            raise FileNotFoundError(f"missing {taskMdPath}")

        # Bind layout:
        #   * agent_workspace -> /root (writable; /root/answer.json lives here)
        #   * host_tools -> /host_tools (host's mini-swe-agent + python interpreter)
        binds = [
            f"{self._agentWorkspace.resolve()}:/root",
            f"{self._hostTools.resolve()}:/host_tools:ro",
        ]
        # Apptainer validates bind destinations before applying the /root
        # workspace mount. Stage the task file inside that workspace instead
        # of binding a not-yet-existing /root/task.md destination.
        try:
            shutil.copyfile(taskMdPath, self._agentWorkspace / "task.md")
        except OSError as exc:
            raise RuntimeError(f"could not stage {taskMdPath}: {exc}") from exc
        skillsDir = taskDir / "environment" / "skills"
        if skillsDir.is_dir():
            for relative in (".agents/skills", ".claude/skills"):
                (self._agentWorkspace / relative).mkdir(parents=True, exist_ok=True)
                binds.append(
                    f"{skillsDir.resolve()}:/root/{relative}:ro"
                )
        # Bind the host's site-packages into the SIF as ``/host_lib`` so the
        # Python inside the container can import minisweagent + its deps.
        # We do this via a real bind mount (not a symlink) because symlinks
        # pointing at host paths don't resolve inside the SIF.
        if self._hostSitePackages is not None:
            binds.append(f"{self._hostSitePackages.resolve()}:/host_lib:ro")
        if self._hostPythonRoot is not None and self._hostPythonRoot.is_dir():
            # Use the exact host interpreter that owns /host_lib. This avoids
            # loading CPython extension wheels (notably pydantic_core) built
            # for a different Python minor version in the base image.
            binds.append(f"{self._hostPythonRoot.resolve()}:/host_conda:ro")
        envArgs: List[str] = []
        for key, value in env.items():
            # Apptainer --env syntax: ``--env KEY=VAL``.
            envArgs.extend(["--env", f"{key}={value}"])

        # Always prepend /host_tools to PATH so ``mini-swe-agent`` resolves
        # to the host install (the container base may not ship it).
        pathENV = env.get("PATH", "")
        augmentedPath = f"/host_tools:{pathENV}" if pathENV else "/host_tools"
        envArgs.extend(["--env", f"PATH={augmentedPath}"])
        if self._hostSitePackages is not None:
            # Prepend so the agent's imports find minisweagent before any
            # container-shipped package.
            currentPP = env.get("PYTHONPATH", "")
            augmentedPP = f"/host_lib:{currentPP}" if currentPP else "/host_lib"
            envArgs.extend(["--env", f"PYTHONPATH={augmentedPP}"])

        apptainerArgs: List[str] = [
            self.apptainerPath, "exec",
        ]
        if self.networkMode:
            # empty string == keep the default (host-shared) network
            apptainerArgs.extend(["--network", self.networkMode])
        if self.writableTmpfs:
            apptainerArgs.append("--writable-tmpfs")
        apptainerArgs.extend([
            "--pwd", "/root",
            *sum((["--bind", b] for b in binds), []),
            *envArgs,
            str(self._sifPath),
            *cmd,
        ])
        # Fake root via ``unshare -U -r`` — gives us uid=0 inside the container
        # without requiring setuid ``newuidmap`` (which conda-forge apptainer
        # doesn't ship and /etc/subuid doesn't include ``lyh``). The kernel
        # maps only our host UID (3037) inside, so we can do everything
        # except chown to / chgrp to unmapped UIDs/GIDs — that's enough for
        # pip install, write to /root, etc. Apt-get fails on setgroups for
        # unmapped GIDs (e.g. nogroup); use image_override with a Python-
        # prebaked image to skip the apt layer.
        fullCmd = self._WrapWithUnshare(apptainerArgs)
        with open(log_path, "w", encoding="utf-8") as logFile:
            proc = subprocess.run(
                fullCmd,
                stdout=logFile,
                stderr=subprocess.STDOUT,
                check=False,
            )
            return proc.returncode

    def run_verifier(
        self,
        *,
        task_id: str,
        verifier_test_sh: Path,
        log_path: Path,
    ) -> Tuple[int, Optional[Path]]:
        """Run pytest directly and write ``/logs/verifier/reward.txt``.

        Why bypass ``verifier/test.sh``: SkillsBench's shell scripts hardcode
        ``apt-get install curl && curl ... | sh`` to bootstrap uv + pytest.
        ``apt-get`` fails in our rootless fakeroot namespace (setgroups is
        denied) and ``curl`` isn't installed in the base image. Rather than
        reimplement the curl-install-uv dance, we skip test.sh entirely and
        call ``python3 -m pytest`` directly — pytest lives in the host's
        ``site-packages`` which is bind-mounted at ``/host_lib`` so it's
        visible inside the SIF.

        Bind layout (mirror of DockerSandbox):

        - ``<verifier dir>:/verifier:ro``
        - ``<agent_workspace>:/root`` (so verifier sees /root/answer.json)
        - ``<verifier_workspace>:/logs`` (we write reward.txt here ourselves)
        """
        assert self._sifPath is not None and self._sifPath.exists()
        assert self._agentWorkspace is not None
        assert self._verifierWorkspace is not None

        binds = [
            f"{verifier_test_sh.parent.resolve()}:/verifier:ro",
            f"{self._agentWorkspace.resolve()}:/root",
            f"{self._verifierWorkspace.resolve()}:/logs",
        ]
        if self._hostSitePackages is not None:
            binds.append(f"{self._hostSitePackages.resolve()}:/host_lib:ro")
        if self._hostPythonRoot is not None and self._hostPythonRoot.is_dir():
            binds.append(f"{self._hostPythonRoot.resolve()}:/host_conda:ro")

        # Make sure the verifier can write its CTRF log + the reward file.
        # mkdir as fake root inside the writable-tmpfs overlay.
        logDir = self._verifierWorkspace / "verifier"
        logDir.mkdir(parents=True, exist_ok=True)

        apptainerArgs: List[str] = [
            self.apptainerPath, "exec",
        ]
        if self.networkMode:
            apptainerArgs.extend(["--network", self.networkMode])
        if self.writableTmpfs:
            apptainerArgs.append("--writable-tmpfs")
        apptainerArgs.extend([
            "--pwd", "/verifier",
            *sum((["--bind", b] for b in binds), []),
            "--env", "PYTHONPATH=/host_lib",
            "--env", "PATH=/host_lib:/host_tools:/usr/local/bin:/opt/conda/bin:/usr/bin:/bin",
            str(self._sifPath),
            # Skip test.sh entirely: invoke pytest directly. The path the
            # SkillsBench verifier writes to (``/logs/verifier/reward.txt``)
            # is derived from the CTRF JSON + exit code here.
            "/host_conda/bin/python", "-m", "pytest", "-rA", "-v",
            "--ctrf", "/logs/verifier/ctrf.json",
            f"/verifier/{verifier_test_sh.with_suffix('').name}_outputs.py",
        ])
        # Same fake-root trick as run_agent — see _WrapWithUnshare comment.
        fullCmd = self._WrapWithUnshare(apptainerArgs)
        with open(log_path, "w", encoding="utf-8") as logFile:
            proc = subprocess.run(
                fullCmd,
                stdout=logFile,
                stderr=subprocess.STDOUT,
                check=False,
            )

        # SkillsBench's test.sh writes ``echo 1 > /logs/verifier/reward.txt``
        # on pytest exit 0. We reproduce that contract here.
        rewardFile = self._verifierWorkspace / "verifier" / "reward.txt"
        try:
            rewardFile.write_text("1" if proc.returncode == 0 else "0")
        except OSError:
            pass

        return proc.returncode, (rewardFile if rewardFile.is_file() else None)

    def cleanup(self) -> None:
        if self.cleanupImage and self._sifPath is not None:
            try:
                self._sifPath.unlink()
            except OSError:
                pass

    def _StageEnvironmentInputs(self, task_dir: Path) -> None:
        """Stage Dockerfile input files into the mounted agent ``/root``.

        Apptainer mode intentionally uses a prebuilt base image and therefore
        cannot apply the task Dockerfile's ``COPY`` instructions.  Most
        SkillsBench inputs are copied to ``/root``; reproduce those copies
        locally so the agent sees the same task data as it would in Docker.
        """
        if self._agentWorkspace is None:
            return
        environmentDir = task_dir / "environment"
        dockerfile = environmentDir / "Dockerfile"
        staged: set[str] = set()
        copyPattern = re.compile(r"^\s*COPY\s+(.*?)\s+(\S+)\s*$", re.IGNORECASE)
        if dockerfile.is_file():
            for rawLine in dockerfile.read_text(encoding="utf-8").splitlines():
                match = copyPattern.match(rawLine)
                if not match:
                    continue
                try:
                    parts = shlex.split(match.group(1))
                except ValueError:
                    continue
                if len(parts) != 1:
                    # Multi-source COPY needs a directory destination; it is
                    # uncommon for simple agent inputs and is handled below
                    # by the root-level fallback.
                    continue
                sourceName, destination = parts[0], match.group(2)
                source = environmentDir / sourceName
                if not (
                    destination == "/root" or destination.startswith("/root/")
                ):
                    continue
                relativeDestination = destination.removeprefix("/root").lstrip("/")
                target = self._agentWorkspace / relativeDestination
                if source.is_dir():
                    # Match Docker's COPY semantics for directory sources:
                    # ``COPY <srcdir> /root/<dest>`` puts the source's
                    # contents into ``<dest>``. For ``COPY <srcdir> /root``
                    # (or trailing slash) the directory itself lands under
                    # ``/root/<srcdir>``. Without this branch a task whose
                    # Dockerfile uses ``COPY input /root/input`` ships an
                    # empty ``/root/input`` to the agent — which then
                    # invents a placeholder file and "solves" the wrong
                    # problem.
                    if (
                        not relativeDestination
                        or destination.endswith("/")
                    ):
                        target = target / source.name
                    if target.exists():
                        shutil.rmtree(target)
                    shutil.copytree(source, target)
                    staged.add(sourceName)
                    continue
                if not source.is_file():
                    continue
                # Match Docker's COPY semantics: a destination ending in `/`,
                # `/root` itself, or an existing directory receives the source
                # under its basename rather than being treated as a file.
                if (
                    not relativeDestination
                    or destination.endswith("/")
                    or target.is_dir()
                ):
                    target = target / source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                staged.add(sourceName)

        # Preserve the common Dockerfile convention even when a minimal or
        # custom task omits an explicit COPY line. Cover both single files
        # (the historical case) and directory trees (e.g. ``input/`` and
        # ``output_schema/`` from the typical SkillsBench Dockerfile).
        for source in environmentDir.iterdir():
            if source.name in staged or source.name in {"Dockerfile", "skills"}:
                continue
            target = self._agentWorkspace / source.name
            if target.exists():
                continue
            if source.is_file():
                shutil.copyfile(source, target)
            elif source.is_dir():
                shutil.copytree(source, target)

    # ---------------------------------------------------------- internals
    @staticmethod
    def _WrapWithUnshare(apptainerArgs: Sequence[str]) -> List[str]:
        """Wrap an apptainer command in ``unshare -U -r`` to give it fake root.

        Why this is necessary instead of apptainer's own ``--fakeroot``:

        - apptainer's ``--fakeroot`` requires a setuid-root ``newuidmap``
          binary on PATH plus a ``/etc/subuid`` entry for the invoking user.
          conda-forge apptainer doesn't ship ``newuidmap`` (no root to
          install it), and lyh isn't in ``/etc/subuid``.
        - Modern Linux (kernel >= 3.8) supports ``unshare -U -r``: creates a
          user namespace where our UID is mapped to root inside. No
          ``newuidmap`` needed, no ``/etc/subuid`` needed.
        - Apptainer inherits the namespace and runs as uid=0 inside the
          SIF. The container can write to ``/root`` (via ``--writable-tmpfs``),
          ``pip install``, etc.
        - The kernel still maps ONLY our host UID, so things needing
          arbitrary UIDs (``chown 1000``, apt-get's ``setgroups``) fail.
          That's the rootless tax — work around it by pulling a Python-
          prebaked base image via ``image_override``.
        """
        return ["unshare", "-U", "-r", "--map-root-user", *apptainerArgs]

    def _ResolveHelper(self) -> "BenchflowHelper":
        return self._helper  # type: ignore[attr-defined,return-value]

    def _MakeWorkspace(self, kind: str) -> Path:
        helper = self._ResolveHelper()
        workspace = helper.outputDir / f"{kind}_workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        return workspace

    @staticmethod
    def _ResolveFromImage(task_dir: Path) -> str:
        """Parse the first ``FROM <image>[:tag]`` from environment/Dockerfile.

        Returns a ``docker://<image>[:tag]`` URI for ``apptainer pull``.
        Raises if the Dockerfile is missing or has no FROM line.
        """
        dockerfile = task_dir / "environment" / "Dockerfile"
        if not dockerfile.is_file():
            raise FileNotFoundError(
                f"SkillsBench task has no environment/Dockerfile at {dockerfile}"
            )
        for rawLine in dockerfile.read_text(encoding="utf-8").splitlines():
            line = rawLine.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split()
            if not tokens or tokens[0].upper() != "FROM":
                continue
            if len(tokens) < 2:
                raise RuntimeError(
                    f"Dockerfile FROM line has no image: {rawLine!r}"
                )
            image = tokens[1]
            # Strip stage-AS alias (``FROM ubuntu AS base``).
            for sep in (" AS ", " as "):
                if sep in image:
                    image = image.split(sep)[0]
                    break
            return f"docker://{image}"
        raise RuntimeError(
            f"No FROM line found in {dockerfile}; cannot resolve base image"
        )

    @staticmethod
    def _StageHostTools(hostToolsDir: Path) -> None:
        """Stage host binaries the agent needs.

        The container's base image is expected to ship a ``python3`` (use
        ``image_override="docker://continuumio/miniconda3:latest"`` or any
        other image that already has Python 3.x). The launcher invokes
        ``python3`` (resolved via the container's PATH) on the staged
        script.

        In addition to the script, the host's ``site-packages`` is exposed
        at ``/host_tools/_lib`` and prepended to ``PYTHONPATH`` so the
        agent can import ``minisweagent`` (installed on the host, not in
        the container's base image).
        """
        msaScript = shutil.which("mini-swe-agent")
        if not msaScript:
            return
        baseName = os.path.basename(msaScript)

        # Stage the script under a non-conflicting name so the launcher
        # (which gets the canonical ``baseName``) can call it via
        # ``python3``.
        stagedScript = hostToolsDir / f"{baseName}.py"
        try:
            stagedScript.write_text(
                Path(msaScript).read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        except OSError:
            return
        try:
            stagedScript.chmod(0o755)
        except OSError:
            pass

        # Stage the host's site-packages dir at ``_lib`` so Python inside
        # the container can import minisweagent + its deps. Symlink, not
        # copy, so we don't bloat output dirs.
        hostSitePackages = _ResolveHostSitePackages()
        if hostSitePackages is not None:
            libDir = hostToolsDir / "_lib"
            try:
                if libDir.is_symlink() or libDir.exists():
                    libDir.unlink()
                libDir.symlink_to(hostSitePackages.resolve())
            except OSError:
                pass

        # Launcher at the canonical name: prepended to PATH inside the
        # container so ``mini-swe-agent`` resolves to the launcher, which
        # invokes ``python3`` on the staged script. PYTHONPATH points at
        # /host_lib (bound directly from the host's site-packages dir).
        launcher = hostToolsDir / baseName
        inContainerScript = f"/host_tools/{baseName}.py"
        launcher.write_text(
            f"#!/bin/sh\nexec env PYTHONPATH=/host_lib /host_conda/bin/python "
            f"{inContainerScript} \"$@\"\n",
            encoding="utf-8",
        )
        try:
            launcher.chmod(0o755)
        except OSError:
            pass


# --------------------------------------------------------------------- helper


# --------------------------------------------------------------------- helper


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

        self.port = int(port) if port else _FindFreePort(host)
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

        # For Muse Glimmer, CoT is gated by the system prompt's
        # ``Reasoning strength`` line (the model has no ``<think>`` tag idiom).
        # Append the right value here so every rollout of this task sees the
        # same prefix (KV-cache friendly). Skip when the agent already
        # supplied its own ``Reasoning strength:`` line so a user override
        # wins.
        if (
            _DetectArch() == "muse_glimmer"
            and self._skillsBlock
            and "Reasoning strength:" not in self._skillsBlock
        ):
            strength = "high" if self.thinking is not False else "low"
            self._skillsBlock = (
                f"{self._skillsBlock}\n\nReasoning strength: {strength}."
            )

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
        server = ThreadingHTTPServer((self.host, self.port), _HelperHTTPHandler)
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
        # adapterNotes = (
        #     "## Agent adapter notes\n"
        #     + "CRITICAL: this is a data/file-analysis task, not a software "
        #     + "repository issue. Ignore any generic codebase/reproduction "
        #     + "workflow in the agent instructions. Read these execution rules "
        #     + "before the task. You are operating as a terminal agent. Start "
        #     + "with a concise inspection command "
        #     + "(for example, a Python script that prints only each entry's "
        #     + "title, year, venue, and DOI); do not `cat` a large input file "
        #     + "or print every parsed dictionary. Before solving the task, "
        #     + "inspect the reusable guidance under `/root/.agents/skills` and "
        #     + "apply any relevant SKILL.md instructions. Use the available "
        #     + "terminal tool for every action, verify the requested output, "
        #     + "and finish only after the output file is complete. Do not use "
        #     + "oracle/ or verifier/ files to obtain the answer.\n\n"
        #     + "The evaluator reads the output file literally: it must be a "
        #     + "valid JSON object with the exact required key and an array of "
        #     + "clean values. Never redirect raw grep/awk output, entry keys, "
        #     + "or notes into the output file. A reliable workflow is to use a "
        #     + "short Python script to extract the relevant fields, do the "
        #     + "verification, then use json.dump to write the final object. "
        #     + "Before submitting, reopen the file with Python's json.load and "
        #     + "check its type, required key, value types, and sorting. Do not "
        #     + "issue the completion command until this validation succeeds. "
        #     + "For this kind of bibliography task, extracting all entry keys or "
        #     + "all titles is only an intermediate step: the final array must "
        #     + "contain only the entries you verified as fake, not every entry. "
        #     + "Make one compact pass over all entries and compare their title, "
        #     + "authors, venue, year, and DOI together; do not spend a separate "
        #     + "turn querying one API for every title. A single failed or empty "
        #     + "network lookup is not proof that a citation is fake, and a missing "
        #     + "DOI is not by itself proof either. Placeholder-looking DOI data, "
        #     + "generic metadata, and disagreement between the bibliographic "
        #     + "fields are useful signals that must be weighed together. "
        #     + "Use the exact shape `json.dump({'fake_citations': "
        #     + "sorted(fake_titles)}, output_file, indent=2)` after deciding "
        #     + "the list, then run `python3 -m json.tool /root/answer.json`."
        # )
        # return adapterNotes + "\n\n## Task\n" + taskText.rstrip()
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


__all__ = ["BenchflowHelper", "Sandbox", "LocalSandbox", "DockerSandbox", "ApptainerSandbox"]
