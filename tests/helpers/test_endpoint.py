"""Tests for endpoint."""

import json
import threading
from http.client import HTTPConnection

import pytest

from helpers.backends import ModelAdapter
from helpers.endpoint import KVBenchEndpoint

pytestmark = pytest.mark.usefixtures("fakeTaskTokenizer")


@pytest.fixture
def endpoint(monkeypatch, tmp_path):
    renderCalls = []

    def render(messages, *, modelPath, tools=None, thinking=None):
        renderCalls.append(
            {
                "messages": messages,
                "modelPath": modelPath,
                "tools": tools,
                "thinking": thinking,
            }
        )
        return "rendered prompt"

    def parse(output, payload, *, modelPath):
        if output == "tool-output":
            return "", "internal reasoning", [{
                "id": "call-1",
                "type": "function",
                "function": {"name": "bash", "arguments": '{"command":"ls"}'},
            }]
        return output, None, []

    monkeypatch.setattr(ModelAdapter, "render_chat", render)
    monkeypatch.setattr(ModelAdapter, "parse_tool_calls", parse)
    server = KVBenchEndpoint(
        modelPath="/models/test",
        host="127.0.0.1",
        debugLogPath=tmp_path / "llm.jsonl",
    ).start()
    yield server, renderCalls
    server.stop()


def _post(endpoint, payload):
    connection = HTTPConnection("127.0.0.1", endpoint.port, timeout=3)
    connection.request(
        "POST",
        "/v1/chat/completions",
        body=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )
    response = connection.getresponse()
    body = response.read()
    connection.close()
    return response.status, response.getheader("Content-Type"), body


def _serve_one(endpoint, payload):
    result = {}

    def call():
        result["value"] = _post(endpoint, payload)

    thread = threading.Thread(target=call)
    thread.start()
    request = endpoint.wait_for_request(timeout=2)
    assert request is not None
    return thread, request, result


def test_endpoint_health(endpoint):
    server, _calls = endpoint
    connection = HTTPConnection("127.0.0.1", server.port, timeout=3)
    connection.request("GET", "/health")
    response = connection.getresponse()
    assert response.status == 200
    assert json.loads(response.read()) == {"status": "ok"}
    connection.close()


def test_endpoint_optional_bearer_authentication(tmp_path):
    server = KVBenchEndpoint(
        modelPath="/models/test",
        host="127.0.0.1",
        apiKey="provider-secret",
    ).start()
    try:
        connection = HTTPConnection("127.0.0.1", server.port, timeout=3)
        connection.request("GET", "/health")
        response = connection.getresponse()
        assert response.status == 401
        response.read()
        connection.close()

        connection = HTTPConnection("127.0.0.1", server.port, timeout=3)
        connection.request(
            "GET",
            "/health",
            headers={"Authorization": "Bearer provider-secret"},
        )
        response = connection.getresponse()
        assert response.status == 200
        assert json.loads(response.read()) == {"status": "ok"}
        connection.close()
    finally:
        server.stop()


def test_endpoint_renders_and_queues_without_skill_prefix(endpoint):
    server, renderCalls = endpoint
    payload = {
        "model": "vllm/test",
        "messages": [{"role": "system", "content": "system"}, {"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "bash"}}],
    }
    thread, request, result = _serve_one(server, payload)
    assert request.prompt == "rendered prompt"
    assert renderCalls[0]["tools"] == payload["tools"]
    assert "system_prefix" not in renderCalls[0]
    assert request.payload == payload
    server.respond(request, "hello")
    thread.join(timeout=2)
    assert result["value"][0] == 200
    body = json.loads(result["value"][2])
    assert body["choices"][0]["message"]["content"] == "hello"
    assert body["choices"][0]["finish_reason"] == "stop"


def test_endpoint_tool_calls_and_reasoning(endpoint):
    server, _calls = endpoint
    thread, request, result = _serve_one(
        server,
        {"model": "vllm/test", "messages": [{"role": "user", "content": "run ls"}]},
    )
    server.respond(request, "tool-output")
    thread.join(timeout=2)
    body = json.loads(result["value"][2])
    message = body["choices"][0]["message"]
    assert message["content"] == ""
    assert message["reasoning_content"] == "internal reasoning"
    assert message["tool_calls"][0]["function"]["name"] == "bash"
    assert body["choices"][0]["finish_reason"] == "tool_calls"


def test_endpoint_uses_native_finish_and_stop_reason(endpoint):
    server, _calls = endpoint
    thread, request, result = _serve_one(
        server,
        {"model": "vllm/test", "messages": [{"role": "user", "content": "hi"}]},
    )
    server.respond(
        request,
        "partial output",
        finishReason="length",
        stopReason=123,
    )
    thread.join(timeout=2)
    body = json.loads(result["value"][2])
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["choices"][0]["stop_reason"] == 123


def test_endpoint_stream_true_returns_sse(endpoint):
    server, _calls = endpoint
    thread, request, result = _serve_one(
        server,
        {
            "model": "vllm/test",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    server.respond(request, "hello")
    thread.join(timeout=2)
    status, contentType, body = result["value"]
    assert status == 200
    assert contentType == "text/event-stream"
    assert b"chat.completion.chunk" in body
    assert body.endswith(b"data: [DONE]\n\n")


def test_endpoint_stream_sends_heartbeat_while_generation_is_pending(endpoint):
    server, _calls = endpoint
    server.sseHeartbeatSec = 0.05
    connection = HTTPConnection("127.0.0.1", server.port, timeout=3)
    connection.request(
        "POST",
        "/v1/chat/completions",
        body=json.dumps(
            {
                "model": "vllm/test",
                "messages": [{"role": "user", "content": "wait"}],
                "stream": True,
            }
        ),
        headers={"Content-Type": "application/json"},
    )
    response = connection.getresponse()
    assert response.status == 200
    assert response.getheader("X-Accel-Buffering") == "no"
    assert response.fp.readline().startswith(b": kvbench keep-alive")
    assert response.fp.readline() == b"\n"

    request = server.wait_for_request(timeout=2)
    assert request is not None
    server.respond(request, "hello")
    body = response.read()
    connection.close()
    assert b"chat.completion.chunk" in body
    assert body.endswith(b"data: [DONE]\n\n")


def test_endpoint_malformed_request_is_4xx(endpoint):
    server, _calls = endpoint
    status, _contentType, body = _post(server, {"model": "vllm/test"})
    assert status == 400
    assert "messages must be a list" in json.loads(body)["error"]["message"]


def test_endpoint_logs_raw_request_and_response(endpoint, tmp_path):
    server, _calls = endpoint
    thread, request, result = _serve_one(
        server,
        {
            "model": "vllm/test",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.2,
        },
    )
    server.respond(request, "raw output")
    thread.join(timeout=2)
    assert result["value"][0] == 200
    records = [json.loads(line) for line in (tmp_path / "llm.jsonl").read_text().splitlines()]
    assert [record["phase"] for record in records] == [
        "request",
        "response_wait_started",
        "response_wait_finished",
        "response",
    ]
    assert records[0]["unsupported_generation_fields"] == ["temperature"]
    assert records[3]["raw_output"] == "raw output"


def test_endpoint_finish_drains_queued_but_not_inflight(endpoint):
    server, _calls = endpoint
    inflight = server._MakeRequest({"model": "vllm/test", "messages": []})
    queued = server._MakeRequest({"model": "vllm/test", "messages": []})
    server._Enqueue(inflight)
    server._Enqueue(queued)

    assert server.wait_for_request(timeout=1) is inflight
    server.finish("BenchFlow exited")

    assert not inflight.responseFuture.done()
    assert queued.responseFuture.done()

    server.respond(inflight, "raw output")
    assert server.wait_for_request(timeout=1) is None

