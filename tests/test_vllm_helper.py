import sys
import types

from helpers.backends.VllmHelper import GenerateBatch
from methods.FullPrefill import FullPrefillVllm


class _SamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _Completion:
    text = "generated"
    token_ids = [11, 12]
    finish_reason = "length"
    stop_reason = None


class _SpecialCompletion(_Completion):
    token_ids = [11, 99, 12, 2]
    finish_reason = "stop"
    stop_reason = 2


class _Tokenizer:
    eos_token_id = 2
    all_special_ids = [2, 99]


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
    llm = types.SimpleNamespace(llm_engine=engine, get_tokenizer=lambda: _Tokenizer())

    prompt_ids = [7, 8, 9]
    result = GenerateBatch(llm, [prompt_ids], maxNewTokens=10)[0]

    assert result.text == "generated"
    assert result.numTokens == 2
    assert result.numCached == 3
    assert result.finishReason == "length"
    assert result.stopReason is None
    assert result.outputIds == [11, 12]
    assert engine.requests[0][1] == {"prompt_token_ids": prompt_ids}
    assert engine.requests[0][2].kwargs == {"temperature": 0, "max_tokens": 10}


def test_generate_batch_preserves_interior_specials_and_removes_only_terminal_stop(monkeypatch):
    fake_vllm = types.SimpleNamespace(SamplingParams=_SamplingParams)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    engine = _Engine()
    def step():
        engine.stepped = True
        return [
            types.SimpleNamespace(
                request_id=request_id,
                outputs=[_SpecialCompletion()],
                finished=True,
                num_cached_tokens=0,
            )
            for request_id, _prompt, _sampling in engine.requests
        ]

    engine.step = step
    llm = types.SimpleNamespace(llm_engine=engine, get_tokenizer=lambda: _Tokenizer())

    result = GenerateBatch(llm, [[7]], maxNewTokens=10)[0]

    assert result.outputIds == [11, 99, 12]
    assert result.numTokens == 4
    assert result.finishReason == "stop"
    assert result.stopReason == 2


def test_full_prefill_vllm_reads_max_num_seqs_from_config(monkeypatch):
    monkeypatch.setattr(
        "methods.FullPrefill.Get",
        lambda key, default=None: (
            {"MaxNumSeqs": 32}
            if key == "FullPrefillVllm"
            else default
        ),
    )

    method = FullPrefillVllm()

    assert method.maxNumSeqs == 32


def test_full_prefill_vllm_reports_native_prefix_hits(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(SamplingParams=_SamplingParams))
    method = object.__new__(FullPrefillVllm)
    method.llm = types.SimpleNamespace(llm_engine=_Engine(), get_tokenizer=lambda: _Tokenizer())
    method.samplingConfig = {"mode": "greedy"}

    result = method.Run([[7, 8, 9]], maxNewTokens=2)[0]

    assert result.metadata["num_cached_tokens"] == 3
    assert result.metadata["reuse_ratio"] == 1.0
    assert method.method_metrics == ("reuse_ratio",)
