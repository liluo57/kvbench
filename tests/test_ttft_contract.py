"""Regression tests for the end-to-end online TTFT boundary."""

import time
from types import SimpleNamespace

import pytest

from helpers.backends.VllmHelper import GenerateBatch
from methods.A3Repo import A3Repo
from methods.CacheClip import CacheClip
from methods.CacheblendRepo import CacheblendRepo
from methods.DependencyAnalysis import DependencyAnalysisMethod
from methods.FullPrefill import FullPrefillTransformer
from methods.Hypic import HypicMethod
from methods.Naive import NaiveTransformer
from methods.ProphetKV import ProphetKV
from helpers.cacheblend_repo.CacheblendRepoHelper import CacheBlendWorker


class _EncodingDelayGenerator:
    def Encode(self, text, **_kwargs):
        time.sleep(0.02)
        return [1]

    def Generate(self, _ids, **_kwargs):
        return "answer", 0.03, 0.04, 1


def test_full_prefill_ttft_includes_request_tokenization():
    method = object.__new__(FullPrefillTransformer)
    method._gen = _EncodingDelayGenerator()

    result = method.Run(["prompt"])[0]

    # The fake backend contributes 30 ms and Encode contributes about 20 ms.
    assert result.performance["ttft"] >= 0.045


def test_naive_miss_includes_request_tokenization():
    method = object.__new__(NaiveTransformer)
    method._gen = _EncodingDelayGenerator()
    method._states = [{"segments": [], "caches": [], "ids": []}]

    result = method.Run(["prompt"])[0]

    assert result.performance["ttft"] >= 0.045


def test_dependency_analysis_generation_keeps_tokenization_in_ttft():
    method = object.__new__(DependencyAnalysisMethod)
    method._gen = SimpleNamespace(
        Encode=_EncodingDelayGenerator().Encode,
        Generate=lambda *args, **kwargs: (
            "answer",
            0.03,
            0.04,
            1,
            None,
            [],
        ),
    )
    method._chunks = [[]]
    method._Analyze = lambda *args: {}

    result = method.Run(["prompt"])[0]

    assert result.performance["ttft"] >= 0.045


def test_prophet_fallback_keeps_request_encoding_in_ttft():
    class Runtime:
        layers = []

        @staticmethod
        def encode(_text, **_kwargs):
            time.sleep(0.02)
            return [1]

        @staticmethod
        def full_generate(_ids):
            return "answer", 0.03, 0.04, 1

    method = object.__new__(ProphetKV)
    method._runtime = Runtime()
    method._states = [SimpleNamespace(chunks=[], caches=[])]
    method.debug = False

    result = method.Run(["prompt"])[0]

    assert result.performance["ttft"] >= 0.045


def test_cacheclip_fallback_includes_matching_path_and_prompt_encoding():
    class Generator(_EncodingDelayGenerator):
        model = SimpleNamespace(device="cpu")

    method = object.__new__(CacheClip)
    method._gen = Generator()
    method._auxiliary = object()
    method._states = [{"chunks": ["cached"]}]
    method.requireReuse = False

    result = method.Run(["no match"])[0]

    assert result.performance["ttft"] >= 0.045


class _HypicTimingEngine:
    def generate(self, _prompt, sampling_params, stream):
        delay = 0.05 if sampling_params["max_new_tokens"] <= 4 else 0.03
        time.sleep(delay)
        yield {
            "text": None,
            "output_ids": [1],
            "meta_info": {"prompt_tokens": 1, "completion_tokens": 1},
        }


def test_hypic_prepare_is_excluded_but_online_matching_is_included(monkeypatch):
    import methods.Hypic as hypic_module

    method = object.__new__(HypicMethod)
    method.engine = _HypicTimingEngine()
    method.fullPrefill = False
    method.modelPath = ""
    method.separator = "<<PIC_SEP>>"
    method.picMode = "addition"
    method.samplingConfig = {}
    method._states = []

    monkeypatch.setattr(hypic_module, "arch_family", lambda _path: "llama")

    def delayed_no_match(_chunks, prompt):
        time.sleep(0.02)
        return [(None, prompt)]

    monkeypatch.setattr(hypic_module, "ComposeInterleavedReuse", delayed_no_match)

    method.Prepare([["offline chunk"]])
    result = method.Run(["online prompt"])[0]

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
    llm = SimpleNamespace(llm_engine=engine)

    results = GenerateBatch(
        llm,
        ["a", "b"],
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
