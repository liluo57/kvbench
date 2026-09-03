"""A small OpenAI Chat Completions endpoint owned by KVBench.

The endpoint deliberately knows nothing about a benchmark, task packages,
skills, verifiers, or agents. It turns an incoming chat request into a
rendered prompt and queues it for the owner of the endpoint. The owner is
expected to call :meth:`respond` after the normal KVBench Engine/Worker loop
has produced a model result.
"""

from __future__ import annotations

import concurrent.futures
import copy
import hmac
import json
import queue
import threading
import time
import traceback
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from helpers.backends import ModelAdapter


class EndpointError(RuntimeError):
    """An endpoint lifecycle or protocol error."""


@dataclass
class EndpointResponse:
    """Parsed response fields plus the wire response payload."""

    payload: Dict[str, Any]
    stream: bool = False
    rawOutput: Optional[str] = None


@dataclass(eq=False)
class OpenAIRequest:
    """One queued request waiting for a synchronous KVBench generation."""

    requestId: str
    payload: Dict[str, Any]
    messages: List[Dict[str, Any]]
    tools: Optional[List[Mapping[str, Any]]]
    prompt: str
    stream: bool
    responseFuture: "concurrent.futures.Future[EndpointResponse]" = field(
        repr=False
    )
    receivedAt: float = field(default_factory=time.time)


_DONE = object()
_UNSUPPORTED_FIELDS = (
    "temperature",
    "top_p",
    "max_tokens",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "n",
    "response_format",
    "seed",
    "logprobs",
    "top_logprobs",
    "user",
)


class _EndpointServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class _OpenAIRequestHandler(BaseHTTPRequestHandler):
    """HTTP-only part of :class:`OpenAIEndpoint`."""

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    @property
    def endpoint(self) -> "OpenAIEndpoint":
        return self.server.endpoint  # type: ignore[attr-defined]

    def _WriteJSON(self, status: int, payload: Mapping[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _WriteError(
        self,
        status: int,
        message: str,
        errorType: str = "invalid_request_error",
    ) -> None:
        self._WriteJSON(
            status,
            {"error": {"message": message, "type": errorType}},
        )

    def _RequireAuthorization(self) -> bool:
        if self.endpoint._IsAuthorized(self.headers.get("Authorization")):
            return True
        self._WriteError(401, "invalid or missing bearer token", "authentication_error")
        return False

    def do_GET(self) -> None:  # noqa: N802 - stdlib name
        if not self._RequireAuthorization():
            return
        if self.path == "/health":
            self._WriteJSON(200, {"status": "ok"})
            return
        self._WriteError(404, f"not found: {self.path}")

    def do_POST(self) -> None:  # noqa: N802 - stdlib name
        if not self._RequireAuthorization():
            return
        if self.path != "/v1/chat/completions":
            self._WriteError(404, f"not found: {self.path}")
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._WriteError(400, "invalid Content-Length")
            return
        if length <= 0:
            self._WriteError(400, "empty request body")
            return

        try:
            decoded = self.rfile.read(length).decode("utf-8")
            payload = json.loads(decoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._WriteError(400, f"invalid JSON: {exc}")
            return
        if not isinstance(payload, dict):
            self._WriteError(400, "request body must be a JSON object")
            return

        try:
            request = self.endpoint._MakeRequest(payload)
        except (TypeError, ValueError, EndpointError) as exc:
            self._WriteError(400, str(exc))
            return

        self.endpoint._LogRequest(request)
        try:
            self.endpoint._Enqueue(request)
        except EndpointError as exc:
            self._WriteError(503, str(exc), "server_error")
            return
        try:
            response = request.responseFuture.result()
        except Exception as exc:  # noqa: BLE001 - convert generation errors to HTTP
            self.endpoint._LogError(request, exc)
            self._WriteError(502, f"generation failed: {exc}", "server_error")
            return

        try:
            self.endpoint._LogResponse(request, response)
            if response.stream:
                self._WriteSSE(response.payload)
            else:
                self._WriteJSON(200, response.payload)
        except (BrokenPipeError, ConnectionResetError):
            return

    def _WriteSSE(self, payload: Mapping[str, Any]) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        # This implementation emits a complete response in one/few chunks,
        # rather than holding an HTTP connection open for token streaming.
        # Closing after [DONE] lets ordinary OpenAI clients finish reading an
        # SSE body without requiring a token-count/content-length trailer.
        self.send_header("Connection", "close")
        self.end_headers()

        choice = payload["choices"][0]
        message = choice.get("message") or {}
        delta: Dict[str, Any] = {"role": "assistant"}
        if message.get("content"):
            delta["content"] = message["content"]
        if message.get("reasoning_content"):
            delta["reasoning_content"] = message["reasoning_content"]
        if message.get("tool_calls"):
            delta["tool_calls"] = message["tool_calls"]

        first = {
            "id": payload["id"],
            "object": "chat.completion.chunk",
            "created": payload["created"],
            "model": payload["model"],
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }
        self._SSEData(first)
        final = {
            "id": payload["id"],
            "object": "chat.completion.chunk",
            "created": payload["created"],
            "model": payload["model"],
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": choice.get("finish_reason", "stop"),
                }
            ],
        }
        self._SSEData(final)
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self.close_connection = True

    def _SSEData(self, payload: Mapping[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.wfile.write(b"data: " + body + b"\n\n")
        self.wfile.flush()


class OpenAIEndpoint:
    """Serve one benchmark-independent OpenAI-compatible model endpoint.

    ``render_chat`` and ``parse_tool_calls`` remain the only model-specific
    protocol hooks. The endpoint never executes tools and never changes the
    messages to add benchmark or skill instructions.
    """

    def __init__(
        self,
        *,
        modelPath: str,
        host: str = "0.0.0.0",
        port: int = 0,
        thinking: Optional[bool] = None,
        debugLogPath: Optional[str | Path] = None,
        apiKey: Optional[str] = None,
    ):
        self.modelPath = str(modelPath)
        self.host = host
        self.port = int(port)
        self.thinking = thinking
        self.debugLogPath = Path(debugLogPath) if debugLogPath else None
        self.apiKey = str(apiKey) if apiKey else None
        self._server: Optional[_EndpointServer] = None
        self._serverThread: Optional[threading.Thread] = None
        self._queue: "queue.Queue[object]" = queue.Queue()
        self._active: set[OpenAIRequest] = set()
        self._activeLock = threading.Lock()
        self._stateLock = threading.Lock()
        self._accepting = False
        self._stopped = False
        # ``finish`` is a graceful end-of-input signal.  A request remains
        # active after it has been handed to vLLM, so the endpoint must not
        # wake the workload until that request has produced a response.
        self._finishRequested = False
        self._doneQueued = False

    @property
    def is_running(self) -> bool:
        return self._server is not None and not self._stopped

    @property
    def endpointUrl(self) -> str:
        """Loopback URL for local health/debug clients, including ``/v1``."""
        return f"http://127.0.0.1:{self.port}/v1"

    @property
    def url(self) -> str:
        return self.endpointUrl

    def start(self) -> "OpenAIEndpoint":
        with self._stateLock:
            if self._server is not None:
                return self
            server = _EndpointServer((self.host, self.port), _OpenAIRequestHandler)
            server.endpoint = self  # type: ignore[attr-defined]
            self._server = server
            self.port = int(server.server_address[1])
            self._accepting = True
            self._stopped = False
            with self._activeLock:
                self._queue = queue.Queue()
                self._finishRequested = False
                self._doneQueued = False
            self._serverThread = threading.Thread(
                target=server.serve_forever,
                name=f"kvbench-endpoint-{self.port}",
                daemon=True,
            )
            self._serverThread.start()
        return self

    def wait_for_request(
        self, timeout: Optional[float] = None
    ) -> Optional[OpenAIRequest]:
        try:
            item = self._queue.get(timeout=timeout)
        except queue.Empty:
            return None
        if item is _DONE:
            return None
        return item  # type: ignore[return-value]

    def respond(self, request: OpenAIRequest, output: str) -> EndpointResponse:
        """Parse one raw Method output and release the waiting HTTP client."""
        try:
            content, reasoning, toolCalls = ModelAdapter.parse_tool_calls(
                str(output),
                request.payload,
                modelPath=self.modelPath,
            )
            response = self._BuildResponse(
                request,
                content=content or "",
                reasoning=reasoning,
                toolCalls=toolCalls,
                rawOutput=str(output),
            )
        except BaseException as exc:
            if not request.responseFuture.done():
                request.responseFuture.set_exception(exc)
            self._Forget(request)
            raise
        if not request.responseFuture.done():
            request.responseFuture.set_result(response)
        self._Forget(request)
        return response

    def fail(self, request: OpenAIRequest, error: BaseException | str) -> None:
        if not request.responseFuture.done():
            request.responseFuture.set_exception(
                error
                if isinstance(error, BaseException)
                else EndpointError(str(error))
            )
        self._Forget(request)

    def finish(self, error: Optional[str] = None) -> None:
        """Stop accepting requests and finish after in-flight work drains.

        Requests that are still waiting in the endpoint queue cannot be
        serviced after BenchFlow has exited, so they are failed immediately.
        A request already returned by :meth:`wait_for_request` is owned by
        the vLLM worker and must be allowed to complete; failing it here was
        the source of the endpoint/vLLM lifecycle race.
        """
        with self._stateLock:
            if self._stopped:
                return
            self._accepting = False
        failure = error or "endpoint finished before the request completed"
        with self._activeLock:
            self._finishRequested = True
            activeCount = len(self._active)
        self._Log(
            {
                "phase": "finish_requested",
                "error": failure,
                "active_requests": activeCount,
            }
        )
        self._DrainQueue(failure)
        self._QueueDoneIfDrained()

    def stop(self, error: str = "endpoint stopped") -> None:
        with self._stateLock:
            if self._stopped:
                return
            self._stopped = True
            self._accepting = False
        with self._activeLock:
            active = list(self._active)
        for request in active:
            self.fail(request, EndpointError(error))
        self._DrainQueue(error)
        self._QueueDone()
        server = self._server
        if server is not None:
            server.shutdown()
            server.server_close()
        thread = self._serverThread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self._server = None

    # -------------------------------------------------------------- HTTP bridge
    def _IsAuthorized(self, authorization: Optional[str]) -> bool:
        """Validate an optional bearer token without weakening local defaults."""
        if self.apiKey is None:
            return True
        prefix = "Bearer "
        if not authorization or not authorization.startswith(prefix):
            return False
        return hmac.compare_digest(authorization[len(prefix) :], self.apiKey)

    def _MakeRequest(self, payload: Mapping[str, Any]) -> OpenAIRequest:
        messages = payload.get("messages")
        if not isinstance(messages, list):
            raise ValueError("messages must be a list")
        if any(not isinstance(message, dict) for message in messages):
            raise ValueError("each messages item must be an object")
        toolsValue = payload.get("tools")
        if toolsValue is not None and not isinstance(toolsValue, list):
            raise ValueError("tools must be a list when supplied")
        stream = payload.get("stream", False)
        if not isinstance(stream, bool):
            raise ValueError("stream must be a boolean")
        prompt = ModelAdapter.render_chat(
            copy.deepcopy(messages),
            modelPath=self.modelPath,
            tools=copy.deepcopy(toolsValue) if toolsValue is not None else None,
            thinking=self.thinking,
        )
        if not isinstance(prompt, str):
            raise TypeError("render_chat must return a string")
        requestId = str(payload.get("id") or f"chatcmpl-{time.time_ns()}")
        return OpenAIRequest(
            requestId=requestId,
            payload=copy.deepcopy(dict(payload)),
            messages=copy.deepcopy(messages),
            tools=copy.deepcopy(toolsValue) if toolsValue is not None else None,
            prompt=prompt,
            stream=stream,
            responseFuture=concurrent.futures.Future(),
        )

    def _Enqueue(self, request: OpenAIRequest) -> None:
        with self._stateLock:
            if not self._accepting or self._stopped:
                raise EndpointError("endpoint is not accepting requests")
            # Keep the acceptance check, active-set insertion, and queue put
            # in one lifecycle critical section. This prevents finish() from
            # observing an empty active set between the check and insertion.
            with self._activeLock:
                self._active.add(request)
            self._queue.put(request)

    def _BuildResponse(
        self,
        request: OpenAIRequest,
        *,
        content: str,
        reasoning: Optional[str],
        toolCalls: Sequence[Mapping[str, Any]],
        rawOutput: str,
    ) -> EndpointResponse:
        message: Dict[str, Any] = {"role": "assistant", "content": content}
        if reasoning:
            message["reasoning_content"] = reasoning
        if toolCalls:
            message["tool_calls"] = [dict(call) for call in toolCalls]
        payload: Dict[str, Any] = {
            "id": request.requestId,
            "object": "chat.completion",
            "created": int(request.receivedAt),
            "model": str(request.payload.get("model", "")),
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if toolCalls else "stop",
                }
            ],
        }
        return EndpointResponse(
            payload=payload,
            stream=request.stream,
            rawOutput=rawOutput,
        )

    # --------------------------------------------------------------- diagnostics
    def _LogRequest(self, request: OpenAIRequest) -> None:
        unsupported = [key for key in _UNSUPPORTED_FIELDS if key in request.payload]
        self._Log(
            {
                "phase": "request",
                "request_id": request.requestId,
                "payload": request.payload,
                "prompt": request.prompt,
                "prompt_len": len(request.prompt),
                "unsupported_generation_fields": unsupported,
            }
        )

    def _LogResponse(self, request: OpenAIRequest, response: EndpointResponse) -> None:
        choice = response.payload.get("choices", [{}])[0]
        message = choice.get("message", {})
        self._Log(
            {
                "phase": "response",
                "request_id": request.requestId,
                "raw_response": message,
                "raw_output": response.rawOutput,
                "wire_response": response.payload,
                "stream": response.stream,
            }
        )

    def _LogError(self, request: OpenAIRequest, error: BaseException) -> None:
        self._Log(
            {
                "phase": "error",
                "request_id": request.requestId,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        )

    def _Log(self, record: Mapping[str, Any]) -> None:
        if self.debugLogPath is None:
            return
        try:
            self.debugLogPath.parent.mkdir(parents=True, exist_ok=True)
            with self.debugLogPath.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(record, ensure_ascii=False, default=str) + "\n"
                )
        except OSError:
            return

    def _Forget(self, request: OpenAIRequest) -> None:
        with self._activeLock:
            self._active.discard(request)
        self._QueueDoneIfDrained()

    def _QueueDone(self) -> None:
        with self._activeLock:
            if self._doneQueued:
                return
            self._doneQueued = True
        self._queue.put(_DONE)

    def _QueueDoneIfDrained(self) -> None:
        with self._activeLock:
            if (
                not self._finishRequested
                or self._active
                or self._doneQueued
            ):
                return
            self._doneQueued = True
        self._Log({"phase": "finish_drained"})
        self._queue.put(_DONE)

    def _DrainQueue(self, error: str) -> None:
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return
            if isinstance(item, OpenAIRequest):
                self.fail(item, error)


# The task-facing name emphasizes ownership and remains stable if the
# implementation file is renamed later.
KVBenchEndpoint = OpenAIEndpoint

__all__ = [
    "EndpointError",
    "EndpointResponse",
    "KVBenchEndpoint",
    "OpenAIEndpoint",
    "OpenAIRequest",
]
