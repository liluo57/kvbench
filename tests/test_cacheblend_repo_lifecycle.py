from methods.CacheblendRepo import CacheblendRepo


class _BrokenStream:
    def close(self):
        raise BrokenPipeError("closed peer")


class _ClosedProcess:
    def __init__(self):
        self.stdin = _BrokenStream()
        self.stdout = _BrokenStream()
        self.stderr = _BrokenStream()
        self.killed = False

    def poll(self):
        return 1

    def wait(self, timeout=None):
        return 1

    def kill(self):
        self.killed = True


def test_close_suppresses_broken_pipe_from_dead_helper():
    method = CacheblendRepo()
    proc = _ClosedProcess()
    method._proc = proc
    method.Close()
    assert method._proc is None
    assert proc.stdin is None
    assert proc.stdout is None
    assert proc.stderr is None


def test_stderr_tail_uses_already_drained_lines():
    method = CacheblendRepo()
    method._stderrTail.extend(["first\n", "CUDA out of memory\n"])
    assert method._StderrTail().endswith("CUDA out of memory\n")
