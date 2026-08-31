"""OpenAI-compatible HTTP handler for the BenchflowHelper.

The handler bridges one ``/v1/chat/completions`` request to the helper's
request queue and blocks until the Workload responds. The HTTP server itself
lives on :attr:`BenchflowHelper._server` — only the request/response plumbing
is here.
"""

from __future__ import annotations

import concurrent.futures
import json
import sys
import time
from http.server import BaseHTTPRequestHandler
from typing import TYPE_CHECKING, Any, Dict

from helpers.backends.ModelAdapter import parse_tool_calls, render_chat

if TYPE_CHECKING:
    from .helper import BenchflowHelper


class HelperHTTPHandler(BaseHTTPRequestHandler):
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
        prompt = render_chat(
            messages,
            modelPath=helper.modelPath,
            tools=tools,
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

        # Persist the raw vLLM output before parsing — this is what mini-
        # swe-agent actually receives, and is the only way to see why the
        # agent repeated-format-errored on a previous run. No-op if the
        # output dir is read-only / non-existent.
        try:
            debug_path = helper.outputDir / "debug_llm_io.jsonl"
            with open(debug_path, "a", encoding="utf-8") as f:
                rec = {
                    "phase": "response_raw",
                    "endpoint": "BenchflowHelper",
                    "output_text": output,
                    "output_len": len(output) if isinstance(output, str) else -1,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError:
            pass

        try:
            content, reasoning, toolCalls = parse_tool_calls(output, payload, modelPath=helper.modelPath)
        except Exception as exc:
            import traceback
            print(f"[benchflow] parse_tool_calls FAILED: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            self._WriteJSON(500, {"error": f"response parsing failed: {exc}"})
            return
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
        if reasoning:
            # Some clients (litellm-based agents, etc.) read this field
            # separately and will not double-count it as content.  Only emit
            # it when the parser actually extracted something.
            message["reasoning_content"] = reasoning

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
        print(f"[benchflow] HTTP response: tool_calls={len(toolCalls)} content_len={len(content)} reasoning_len={len(reasoning or '')}", file=sys.stderr, flush=True)
        if toolCalls:
            print(f"[benchflow]   first tool_call: {toolCalls[0]}", file=sys.stderr, flush=True)
        else:
            print(f"[benchflow]   content (head 200): {content[:200]!r}", file=sys.stderr, flush=True)
        self._WriteJSON(200, response)