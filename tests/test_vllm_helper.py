import sys
import types

from helpers.backends.VllmHelper import GenerateBatch


class _SamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _Completion:
    text = "generated"
    token_ids = [11, 12]
    finish_reason = "length"
    stop_reason = None


class _RequestOutput:
    def __init__(self, request_id):
        self.request_id = request_id
        self.outputs = [_Completion()]
        self.finished = True
        self.num_cached_tokens = 3


class _Engine:
    def __init__(self):
        self.requests = []
        self.stepped = False

    def add_request(self, request_id, prompt, sampling_params):
        self.requests.append((request_id, prompt, sampling_params))

    def has_unfinished_requests(self):
        return bool(self.requests) and not self.stepped

    def step(self):
        self.stepped = True
        return [_RequestOutput(request_id) for request_id, _p, _s in self.requests]


def test_generate_batch_returns_native_stop_metadata(monkeypatch):
    fake_vllm = types.SimpleNamespace(SamplingParams=_SamplingParams)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    engine = _Engine()
    llm = types.SimpleNamespace(llm_engine=engine)

    result = GenerateBatch(llm, ["prompt"], maxNewTokens=10)[0]

    assert result.text == "generated"
    assert result.numTokens == 2
    assert result.numCached == 3
    assert result.finishReason == "length"
    assert result.stopReason is None
    assert engine.requests[0][2].kwargs == {"temperature": 0, "max_tokens": 10}
