import pytest
import threading

from methods import A3
from core.Result import NumOutputTokensKey, TotalTimeKey, TtftKey


def test_a3_is_a_single_thin_method():
    method = A3(maxNewTokens=4, maxModelLen=128, recompRatio=0.25)

    assert method.Label == "a3"
    assert method.gpuNums == 1
    assert method.maxCaseBatchSize == 1
    assert method.recompRatio == 0.25


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"gpuNums": 2}, ValueError),
        ({"maxNewTokens": 0}, ValueError),
        ({"maxModelLen": 0}, ValueError),
        ({"recompRatio": -0.1}, ValueError),
        ({"recompRatio": 1.1}, ValueError),
    ],
)
def test_a3_rejects_invalid_configuration(kwargs, error):
    with pytest.raises(error):
        A3(**kwargs)


def test_a3_prepare_run_reset_protocol_without_model(monkeypatch):
    method = A3(maxNewTokens=4, maxModelLen=128)
    requests = []

    def fake_request(payload):
        requests.append(payload)
        if payload["op"] == "prepare":
            return {"ok": True, "cases": len(payload["chunks"])}
        if payload["op"] == "run":
            return {
                "ok": True,
                "results": [
                    {
                        "text": "answer",
                        "ttft": 0.1,
                        "total_time": 0.2,
                        "num_output_tokens": 1,
                        "metadata": {
                            "reuse_ratio": 0.75,
                            "recompute_ratio": 0.25,
                        },
                    }
                ],
            }
        if payload["op"] == "reset":
            return {"ok": True}
        raise AssertionError(payload)

    monkeypatch.setattr(method, "_request", fake_request)
    method.Prepare([["document"]])
    result = method.Run(["document Question:?"])[0]

    assert requests[0] == {"op": "prepare", "chunks": [["document"]]}
    assert requests[1] == {
        "op": "run",
        "prompts": ["document Question:?"],
        "chunks": [["document"]],
    }
    assert result.output == "answer"
    assert result.performance == {
        TtftKey: 0.1,
        TotalTimeKey: 0.2,
        NumOutputTokensKey: 1,
    }
    assert result.metadata["reuse_ratio"] == 0.75

    method.Reset()
    assert method._prepared is None


def test_a3_run_requires_matching_prepare(monkeypatch):
    method = A3()
    monkeypatch.setattr(method, "_request", lambda payload: {"ok": True, "cases": 1})
    method.Prepare([["document"]])

    with pytest.raises(ValueError, match="case count mismatch"):
        method.Run(["document?", "document?"])


def test_a3_result_rejects_missing_timing():
    with pytest.raises(RuntimeError, match="invalid timing"):
        A3._result({"text": "answer"})


class _FakePipe:
    def __init__(self, line=""):
        self.line = line
        self.writes = []

    def write(self, value):
        self.writes.append(value)

    def flush(self):
        pass

    def readline(self):
        return self.line

    def close(self):
        pass


class _BlockingPipe(_FakePipe):
    def __init__(self):
        super().__init__()
        self.released = threading.Event()

    def readline(self):
        self.released.wait()
        return ""


class _FakeProcess:
    def __init__(self, stdout):
        self.stdin = _FakePipe()
        self.stdout = stdout
        self.stderr = _FakePipe()
        self.killed = False

    def poll(self):
        return None

    def kill(self):
        self.killed = True
        if hasattr(self.stdout, "released"):
            self.stdout.released.set()

    def wait(self, timeout=None):
        return 0


def test_a3_request_rejects_bad_worker_json():
    method = A3()
    method._proc = _FakeProcess(_FakePipe("not-json\n"))

    with pytest.raises(RuntimeError, match="invalid JSON"):
        method._request({"op": "ping"})


def test_a3_request_kills_worker_on_timeout():
    method = A3(requestTimeout=0.01)
    method._proc = _FakeProcess(_BlockingPipe())

    with pytest.raises(TimeoutError, match="worker timeout"):
        method._request({"op": "ping"})
    assert method._proc is None
