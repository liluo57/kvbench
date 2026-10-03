"""Regression tests for the end-to-end online TTFT boundary."""

import time
from types import SimpleNamespace

import pytest

from helpers.backends.VllmHelper import GenerateBatch
from methods.A3Repo import A3Repo
from methods.CacheClip import CacheClip
from methods import CacheblendRepo, HypicMethod
from methods.DependencyAnalysis import DependencyAnalysisMethod
from methods.FullPrefill import FullPrefillTransformer
from methods.Naive import NaiveTransformer
from methods.ProphetKV import ProphetKV
from methods.Cacheblend.Worker import CacheBlendWorker


class _EncodingDelayGenerator:
    tokenizer = SimpleNamespace(all_special_ids=[99])

    def Encode(self, text, **_kwargs):
        raise AssertionError("Method inputs must already be token IDs")

    def OutputTokenIds(self, token_ids):
        return [token for token in token_ids if token != 99]

    def Generate(self, ids, **kwargs):
        assert all(isinstance(token, int) for token in ids)
        time.sleep(0.03)
        if kwargs.get("returnCache"):
            return "answer", 0.03, 0.04, 1, object(), [2]
        if kwargs.get("returnTokenIds"):
            return "answer", 0.03, 0.04, 1, [2]
        return "answer", 0.03, 0.04, 1


def test_full_prefill_consumes_token_ids_without_encoding():
    method = object.__new__(FullPrefillTransformer)
    method._gen = _EncodingDelayGenerator()

    result = method.Run([[41, 42]])[0]

    assert result.performance["ttft"] >= 0.025
    assert result.output_ids == [2]


def test_naive_miss_consumes_token_ids_without_encoding():
    method = object.__new__(NaiveTransformer)
    method._gen = _EncodingDelayGenerator()
    method._states = [{"segments": [], "caches": [], "ids": []}]

    result = method.Run([[41, 42]])[0]

    assert result.performance["ttft"] >= 0.025
    assert result.output_ids == [2]


def test_dependency_analysis_generation_consumes_token_ids():
    method = object.__new__(DependencyAnalysisMethod)
    method._gen = SimpleNamespace(
        Generate=lambda ids, *args, **kwargs: (
            "answer", 0.03, 0.04, 1, None, [2]
        ) if all(isinstance(token, int) for token in ids) else None,
        OutputTokenIds=lambda token_ids: list(token_ids),
    )
    method._chunks = [[]]
    method._Analyze = lambda *args: {}

    result = method.Run([[41, 42]])[0]

    assert result.performance["ttft"] >= 0.025
    assert result.output_ids == [2]


def test_prophet_fallback_consumes_token_ids_without_encoding():
    class Runtime:
        layers = []

        @staticmethod
        def encode(_text, **_kwargs):
            raise AssertionError("Method inputs must already be token IDs")

        @staticmethod
        def full_generate(ids):
            assert ids == [41, 42]
            return "answer", 0.03, 0.04, 1, [2]

    method = object.__new__(ProphetKV)
    method._runtime = Runtime()
    method._states = [SimpleNamespace(chunks=[], caches=[])]
    method.debug = False

    result = method.Run([[41, 42]])[0]

    assert result.performance["ttft"] >= 0.025
    assert result.output_ids == [2]


def test_cacheclip_zero_match_ttft_includes_full_id_generation():
    class Generator:
        model = SimpleNamespace(device="cpu")

        @staticmethod
        def Generate(ids, pastKeyValues=None, **kwargs):
            assert ids == [41, 42]
            assert pastKeyValues is None
            time.sleep(0.02)
            return "answer", 0.03, 0.04, 1, [2]

    method = object.__new__(CacheClip)
    method._gen = Generator()
    method._auxiliary = object()
    method._states = [{"document_ids": [[7, 8]], "primary": []}]
    method.requireReuse = False

    result = method.Run([[41, 42]])[0]

    assert result.performance["ttft"] >= 0.025
    assert result.output_ids == [2]


class _HypicTimingEngine:
    def generate(self, *, input_ids, pic_segments, sampling_params, stream):
        delay = 0.05 if sampling_params["max_new_tokens"] <= 4 else 0.03
        time.sleep(delay)
        yield {
            "text": None,
            "output_ids": [1],
            "meta_info": {"prompt_tokens": 1, "completion_tokens": 1},
        }


def test_hypic_prepare_is_excluded_but_online_matching_is_included(monkeypatch):
    import importlib
    hypic_module = importlib.import_module("methods.hypic.Method")

    method = object.__new__(HypicMethod)
    method.engine = _HypicTimingEngine()
    method.fullPrefill = False
    method.modelPath = ""
    method.picMode = "addition"
    method.samplingConfig = {}
    method._states = []
    method._systemIds, method._tailIds = [], [3]
    method._terminalStopIds, method._tokenizer = [], None

    monkeypatch.setattr(hypic_module, "arch_family", lambda _path: "llama")

    def delayed_no_match(_chunks, prompt):
        time.sleep(0.02)
        return [(None, prompt)]

    monkeypatch.setattr(hypic_module, "ComposeInterleavedReuse", delayed_no_match)

    method.Prepare([[[7, 8]]])
    result = method.Run([[41, 42]])[0]

    # Prepare sleeps for about 50 ms, but only the online 20 ms match plus the
    # 30 ms first-token delay belongs to this TTFT.
    assert 0.045 <= result.performance["ttft"] < 0.08


def test_subprocess_adapter_adds_parent_interval_once():
    method = object.__new__(CacheblendRepo)
    method.recompRatio = 0.15
    response = {
        "text": "answer",
        "ttft": 0.03,
        "generation_start": 1.02,
        "num_tokens": 1,
        "total_time": 0.04,
        "n_input": 1,
    }

    result = method._Result(response, full=False, onlineStart=1.0)

    assert result.performance["ttft"] == pytest.approx(0.05)


def test_cacheblend_worker_uses_submission_to_first_token_interval():
    worker = object.__new__(CacheBlendWorker)
    worker.args = SimpleNamespace(max_new_tokens=1)
    worker.terminalStopIds = []
    worker._SamplingParams = lambda _count: object()
    worker._endRetainOutput = lambda: None

    class Llm:
        @staticmethod
        def generate(**_kwargs):
            now = time.time()
            return [
                SimpleNamespace(
                    metrics=SimpleNamespace(
                        arrival_time=now,
                        first_scheduled_time=now + 0.01,
                        first_token_time=now + 0.03,
                    ),
                    outputs=[SimpleNamespace(token_ids=[1], text="answer")],
                )
            ]

    worker.llm = Llm()

    response = worker._Generate([1])

    assert response["ttft"] >= 0.02


def test_a3_adapter_adds_parent_interval_once():
    method = object.__new__(A3Repo)
    method.recompRatio = 0.15
    method.repoRoot = "/repo"
    response = {
        "text": "answer",
        "ttft": 0.03,
        "generation_start": 1.02,
        "num_tokens": 1,
        "total_time": 0.04,
        "n_input": 1,
        "reuse_ratio": 0.5,
    }

    result = method._result(response, full=False, onlineStart=1.0)

    assert result.performance["ttft"] == pytest.approx(0.05)


def test_vllm_backend_ttft_is_not_divided_by_batch_size(monkeypatch):
    class Completion:
        text = "answer"
        token_ids = [1]

    class Output:
        def __init__(self, request_id):
            self.request_id = request_id
            self.outputs = [Completion()]
            self.finished = True
            self.num_cached_tokens = 0

    class Engine:
        def __init__(self):
            self.requests = []
            self.done = False

        def add_request(self, request_id, prompt, sampling_params):
            self.requests.append((request_id, prompt, sampling_params))

        def has_unfinished_requests(self):
            return bool(self.requests) and not self.done

        def step(self):
            time.sleep(0.02)
            self.done = True
            return [Output(item[0]) for item in self.requests]

    import sys
    import types

    class SamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(SamplingParams=SamplingParams))
    engine = Engine()
    llm = SimpleNamespace(
        llm_engine=engine,
        get_tokenizer=lambda: SimpleNamespace(eos_token_id=2),
    )

    results = GenerateBatch(
        llm,
        [[101], [102]],
        maxNewTokens=1,
        onlineStart=time.perf_counter() - 0.02,
    )

    assert len(results) == 2
    assert all(item.ttft >= 0.035 for item in results)


def test_transformers_batch_helper_does_not_amortize_ttft():
    from helpers.backends.TransformersHelper import TransformersGenerator

    generator = object.__new__(TransformersGenerator)
    generator.Generate = lambda ids, **_kwargs: (
        f"answer-{ids[0]}",
        0.01 * ids[0],
        0.02 * ids[0],
        ids[0],
    )

    results = generator.GenerateBatch([[2], [5]])

    assert [item[1] for item in results] == [0.02, 0.05]
