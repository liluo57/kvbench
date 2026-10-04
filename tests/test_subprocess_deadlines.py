import importlib
import subprocess
import sys
import time

import pytest


@pytest.fixture(params=[("methods.A3Repo.Method", "A3Repo", "_request"),
                        ("methods.Cacheblend.Method", "CacheblendRepo", "_Request")])
def helper(request):
    module_name, class_name, request_name = request.param
    module = importlib.import_module(module_name)
    method = getattr(module, class_name)(startTimeout=0.2)
    yield module, method, request_name
    if method._proc is not None:
        if method._proc.poll() is None:
            method._proc.kill()
        method._proc.wait(timeout=3)
    method.Close()


@pytest.mark.parametrize("payload", [{"op": "run"}, {"ids": [0] * 100000}])
def test_request_deadline_covers_silent_response_and_blocked_stdin(helper, payload):
    _, method, request_name = helper
    method._proc = subprocess.Popen(
        [sys.executable, "-u", "-c", "import time; time.sleep(30)"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        getattr(method, request_name)(payload, timeout=0.2)
    assert time.monotonic() - start < 2
    method._proc.wait(timeout=3)


@pytest.mark.parametrize("output", ["", "partial line without newline"])
def test_startup_deadline_covers_silent_and_partial_output(helper, monkeypatch, output):
    module, method, _ = helper
    popen = subprocess.Popen

    def quiet_worker(command, **kwargs):
        script = f"import sys,time; sys.stdout.write({output!r}); sys.stdout.flush(); time.sleep(30)"
        return popen([sys.executable, "-u", "-c", script], **kwargs)

    monkeypatch.setattr(module.subprocess, "Popen", quiet_worker)
    method.gpuIds = [0]
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        method._startWorker()
    assert time.monotonic() - start < 2
    method._proc.wait(timeout=3)


def test_line_reader_preserves_split_utf8_and_multiple_buffered_lines(helper):
    _, method, _ = helper
    wire = "二\n三\n".encode("utf-8")
    script = (
        f"import os,time; data={wire!r}; "
        "os.write(1,data[:1]); time.sleep(0.02); os.write(1,data[1:])"
    )
    method._proc = subprocess.Popen(
        [sys.executable, "-u", "-c", script],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    deadline = time.monotonic() + 2
    assert method._ReadlineUntil(deadline) == "二\n"
    assert method._ReadlineUntil(deadline) == "三\n"
