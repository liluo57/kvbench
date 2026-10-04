import importlib
import time
from types import SimpleNamespace

import pytest

from core import Config
from core.Sampling import SglangSamplingParams
from methods import HypicMethod, LinearKVMethod

adapter = importlib.import_module('methods.hypic.Method')


@pytest.fixture(autouse=True)
def config(monkeypatch):
    monkeypatch.setattr(adapter, 'Get', lambda key, default=None: {'MaxMambaCacheSize': 32})


class FakeEngine:
    def __init__(self):
        self.calls = []
        self.closed = False
        self.flushes = 0
        self.output = [31, 99, 32, 99]
        self.finishReason = None

    def generate(self, *, input_ids, pic_segments, sampling_params, stream):
        self.calls.append((list(input_ids), pic_segments, sampling_params))
        cached = min(7, len(input_ids)) if sampling_params['max_new_tokens'] > 1 else 0
        yield dict(text='wrong text', output_ids=[31], meta_info={'cached_tokens': cached})
        yield dict(text='wrong text', output_ids=self.output,
                   meta_info={'prompt_tokens': len(input_ids), 'cached_tokens': cached,
                              'completion_tokens': len(self.output),
                              'finish_reason': self.finishReason})

    def flush_cache(self):
        self.flushes += 1
        return SimpleNamespace(success=True)

    def shutdown(self):
        self.closed = True


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        raise AssertionError('Prepared, online and output IDs must never be re-encoded')

    def decode(self, ids, skip_special_tokens=True):
        return '/'.join(map(str, ids))


def method(kind=HypicMethod, **kwargs):
    obj = kind(**kwargs)
    obj.modelPath = ''
    obj.engine = FakeEngine()
    obj._tokenizer = FakeTokenizer()
    obj._systemIds = [100]
    obj._tailIds = [101]
    obj._terminalStopIds = [99]
    return obj


@pytest.mark.parametrize('kind', [HypicMethod, LinearKVMethod])
def test_token_ids_survive_reordering_fresh_gaps_and_special_tokens(kind):
    obj = method(kind)
    obj.Prepare([[[1, 99, 2], [3, 4]]])
    ids = [10, 3, 4, 11, 1, 99, 2, 12]
    result = obj.Run([ids], maxNewTokens=8)[0]
    sent, segments, sampling = obj.engine.calls[-1]
    assert sent == ids
    assert segments == [(0, 1), (1, 3), (3, 4), (4, 7), (7, 8)]
    assert sampling == SglangSamplingParams(obj.samplingConfig, 8)
    assert result.output_ids == [31, 99, 32]
    assert result.output == '31/99/32'
    assert result.metadata['matched_prepared_segments'] == 2
    assert result.metadata['n_input'] == len(ids)
    assert result.metadata['num_cached_tokens'] == 7
    assert result.metadata['reuse_ratio'] == 7 / len(ids)
    assert result.performance['num_output_tokens'] == 4
    assert 0 <= result.performance['ttft'] <= result.performance['total_time']


def test_hypic_uses_independent_priming_without_rewriting_online_chat(monkeypatch):
    obj = method()
    monkeypatch.setattr(adapter, 'arch_family', lambda path: 'qwen3_5')
    obj.Prepare([[[1, 2], [3, 4]]])
    assert [call[:2] for call in obj.engine.calls] == [
        ([100, 1, 2, 101], [(0, 1), (1, 3), (3, 4)]),
        ([100, 3, 4, 101], [(0, 1), (1, 3), (3, 4)]),
    ]
    chat = [151644, 20, 3, 4, 151645, 151644, 21]
    obj.Run([chat])
    assert obj.engine.calls[-1][0] == chat


def test_linearkv_warms_one_segment_without_system_or_tail():
    obj = method(LinearKVMethod)
    obj.Prepare([[[1, 2], [3, 4]]])
    assert [call[:2] for call in obj.engine.calls] == [
        ([1, 2], [(0, 2)]), ([3, 4], [(0, 2)])
    ]


def test_backend_reported_stop_id_is_stripped_without_removing_interior_special_ids():
    obj = method()
    obj.engine.output = [31, 99, 32, 77]
    obj.engine.finishReason = {'type': 'stop', 'matched': 77}
    result = obj.Run([[1, 2]], retainOutput=[True])[0]
    assert result.output_ids == [31, 99, 32]
    assert result.performance['num_output_tokens'] == 4
    assert obj._states == [[[31, 99, 32]]]


@pytest.mark.parametrize('kind', [HypicMethod, LinearKVMethod])
def test_output_retention_reuses_authoritative_ids_and_excludes_warmup_timing(kind):
    obj = method(kind)
    obj.Prepare([[[1, 2]]])
    original = obj._Warmup
    def delayed(ids):
        time.sleep(0.02)
        original(ids)
    obj._Warmup = delayed
    start = time.perf_counter()
    result = obj.Run([[1, 2, 10]], retainOutput=[True], maxNewTokens=8)[0]
    assert time.perf_counter() - start >= result.performance['total_time'] + 0.015
    assert obj._states == [[[1, 2], [31, 99, 32]]]
    assert result.output_ids == [31, 99, 32]
    obj.Run([[31, 99, 32, 10]])
    assert obj.engine.calls[-1][:2] == ([31, 99, 32, 10], [(0, 3), (3, 4)])


def test_no_match_and_full_prefill_keep_exact_ids():
    for obj in (method(), method(fullPrefill=True)):
        obj.Prepare([[[1, 2]]])
        obj.Run([[5, 99, 6]])
        assert obj.engine.calls[-1][0] == [5, 99, 6]
        assert obj.engine.calls[-1][1] == (None if obj.fullPrefill else [(0, 3)])


def test_repeated_chunk_occurrences_are_distinct_segments():
    obj = method()
    obj.Prepare([[[1, 2]]])
    result = obj.Run([[1, 2, 3, 1, 2, 4]])[0]
    assert obj.engine.calls[-1][1] == [(0, 2), (2, 3), (3, 5), (5, 6)]
    assert result.metadata['matched_prepared_segments'] == 2


def test_prepare_copies_ids_reset_flushes_and_close_shuts_down():
    obj = method()
    chunks = [[1, 2]]
    obj.Prepare([chunks])
    chunks[0].append(3)
    assert obj._states == [[[1, 2]]]
    engine = obj.engine
    obj.Reset()
    assert not obj._states and engine.flushes == 1
    obj.Close()
    obj.Close()
    assert engine.closed and obj.engine is None


@pytest.mark.parametrize('ids', ['abc', [True], [-1], [1.5]])
def test_rejects_non_token_inputs(ids):
    with pytest.raises(TypeError, match='token IDs'):
        method().Run([ids])


@pytest.mark.parametrize('kind', [HypicMethod, LinearKVMethod])
def test_initialize_uses_base_gpu_validation(kind):
    with pytest.raises(ValueError, match='requires exactly 1 GPU'):
        kind().Initialize([])


@pytest.mark.parametrize('ratio', [-0.1, 1.1, float('nan'), True])
def test_linearkv_rejects_invalid_ratio(ratio):
    with pytest.raises(ValueError, match='recompRatio'):
        LinearKVMethod(recompRatio=ratio)


def test_options_cap_state_pool_and_keep_radix_control():
    options = adapter._BuildHypicEngineKwargs('/model', [0], dtype='bfloat16',
        maxModelLen=100, memFractionStatic=.8, picMode='linearkv',
        fullPrefill=False, recompRatio=.2)
    assert options['pic_epic_ratio'] == .2
    assert options['max_mamba_cache_size'] == 32
    options = adapter._BuildHypicEngineKwargs('/model', [0], dtype='bfloat16',
        maxModelLen=100, memFractionStatic=.8, picMode='addition', fullPrefill=True)
    assert not options['pic_enable'] and not options['disable_radix_cache']


def test_method_defaults():
    assert HypicMethod().picMode == 'transition_rope_recompute'
    assert LinearKVMethod().Label == 'linearkv'
