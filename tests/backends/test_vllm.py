"""Tests for vllm."""

import sys
import types

from helpers.backends.VllmHelper import GenerateBatch
from tests.support.vllm import _Engine, _SamplingParams, _SpecialCompletion, _Tokenizer


def test_concurrent_batch_preserves_order_and_measures_each_first_token(clock, monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(SamplingParams=_SamplingParams))
    engine = _Engine()
    requests = iter([(1, False), (0, True), (1, True)])

    def step():
        clock.advance(2)
        index, finished = next(requests)
        request_id = engine.requests[index][0]
        engine.stepped = index == 1 and finished
        completion = types.SimpleNamespace(text=f"answer-{index}", token_ids=[10 + index],
            finish_reason="length", stop_reason=None)
        return [types.SimpleNamespace(request_id=request_id, outputs=[completion], finished=finished,
            num_cached_tokens=0)]

    engine.step = step
    llm = types.SimpleNamespace(llm_engine=engine, get_tokenizer=lambda: _Tokenizer())
    results = GenerateBatch(llm, [[101], [102]], maxNewTokens=1, onlineStart=-1.0)
    assert [r.text for r in results] == ["answer-0", "answer-1"]
    assert [r.ttft for r in results] == [5.0, 3.0]
    assert [r.totalTime for r in results] == [3.0, 3.0]


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
