"""Tests for hypic."""

import importlib
import importlib.util
import subprocess
import sys
from array import array
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from core.Sampling import SglangSamplingParams
from methods import HypicMethod, LinearKVMethod
from methods.hypic.backend.kvbench_hypic.srt.pic.policy import (
    POLICIES,
    PICCompose,
    PICPolicy,
    PICStateInit,
    resolve_policy,
)
from methods.hypic.backend.kvbench_hypic.srt.pic.segmenter import validate_segments
from methods.hypic.backend.kvbench_hypic.srt.pic.selector import (
    merge_online_positions,
    select_epic_positions,
)

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
    result = obj.Run([[1, 2]], retainOutput=[False])[0]
    assert result.output_ids == [31, 99, 32]
    assert result.performance['num_output_tokens'] == 4
    assert obj._states == [[]]


@pytest.mark.parametrize('kind', [HypicMethod, LinearKVMethod])
@pytest.mark.parametrize('full_prefill', [False, True])
@pytest.mark.parametrize('retain', [[True], [True, False]])
def test_output_retention_fails_before_generation_or_state_mutation(kind, full_prefill, retain):
    obj = method(kind, fullPrefill=full_prefill)
    obj.Prepare([[[1, 2]]])
    calls_before = list(obj.engine.calls)
    states_before = [[list(chunk) for chunk in chunks] for chunks in obj._states]
    with pytest.raises(NotImplementedError, match='native output KV retention'):
        obj.Run([[1, 2, 10]], retainOutput=retain, maxNewTokens=8)
    assert obj.engine.calls == calls_before
    assert obj._states == states_before


@pytest.mark.parametrize('kind', [HypicMethod, LinearKVMethod])
def test_output_retention_defaults_to_false_for_missing_or_false_flags(kind):
    for retain in (None, [], [False], [False, True]):
        obj = method(kind)
        result = obj.Run([[1, 2]], retainOutput=retain)[0]
        assert result.output_ids == [31, 99, 32]
        assert obj._states == [[]]


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


def test_explicit_segments_cover_special_token_ids_exactly():
    validate_segments([151644, 3, 151645, 4], [(0, 1), (1, 3), (3, 4)])


@pytest.mark.parametrize('ids,segments', [
    ([], []), ([[1, 2]], [(0, 2)]), ([True], [(0, 1)]),
    ([1, 2], [(1, 2)]), ([1, 2], [(0, 1)]),
    ([1, 2], [(0, 1), (0, 2)]), ([1, 2], [(0, 3)]),
    ([1, 2], [(0, 0), (0, 2)]), ([1, 2], [(False, 2)]),
])
def test_rejects_missing_overlapping_empty_or_invalid_ranges(ids, segments):
    with pytest.raises(ValueError):
        validate_segments(ids, segments)


def test_four_aliases_map_to_expected_attrs():
    assert POLICIES["addition"] == PICPolicy(PICCompose.ADDITION, rope=False, recompute=False)
    assert POLICIES["transition"] == PICPolicy(PICCompose.TRANSITION, rope=False, recompute=False)
    assert POLICIES["transition_rope"] == PICPolicy(PICCompose.TRANSITION, rope=True, recompute=False)
    assert POLICIES["transition_rope_recompute"] == PICPolicy(
        PICCompose.TRANSITION, rope=True, recompute=True
    )


def test_resolve_policy_known_and_unknown():
    assert resolve_policy("transition_rope") is POLICIES["transition_rope"]
    try:
        resolve_policy("bogus")
    except KeyError:
        pass
    else:
        raise AssertionError("unknown mode should raise KeyError")


def test_linearkv_policy_does_not_use_transition():
    policy = POLICIES["linearkv"]
    assert policy.state_init is PICStateInit.LAST_BLOCK
    assert policy.compose is PICCompose.ADDITION  # legacy field only
    assert policy.is_linearkv
    assert not policy.uses_transition
    assert policy.rope
    assert policy.selector == "epic"


def test_epic_positions_are_chunk_local_and_prompt_ordered():
    hits = [
        (20, 27, b"c2"),
        (0, 8, b"c1"),
        (40, 41, b"c3"),
    ]
    selected = select_epic_positions(hits, tokens_per_chunk=2)
    assert selected[(0, 8)] == [0, 1]
    assert selected[(20, 27)] == [20, 21]
    assert selected[(40, 41)] == [40]
    assert merge_online_positions([(8, 20), (27, 40)], selected) == [
        0,
        1,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
    ]


def test_epic_ratio_selects_chunk_prefix_and_keeps_short_chunks_nonempty():
    hits = [(0, 10, b"c1"), (20, 24, b"c2"), (40, 41, b"c3")]
    selected = select_epic_positions(hits, ratio=0.2)
    assert selected[(0, 10)] == [0, 1]
    assert selected[(20, 24)] == [20]
    assert selected[(40, 41)] == [40]


def test_epic_selector_rejects_both_budget_forms():
    try:
        select_epic_positions([(0, 8, b"c1")], tokens_per_chunk=2, ratio=0.2)
    except ValueError as exc:
        assert "mutually exclusive" in str(exc)
    else:
        raise AssertionError("EPIC selector accepted both budget forms")


class CpuAllocator:
    device = "cpu"

    def free(self, _slots):
        pass


class CpuReqToTokenPool:
    def __init__(self):
        self.mamba_allocator = CpuAllocator()
        self.req_to_token = torch.zeros((1, 32), dtype=torch.int64)


def _run_real_schedule_path(monkeypatch, mode):
    from methods.hypic.backend.kvbench_hypic.srt.managers import schedule_batch
    from methods.hypic.backend.kvbench_hypic.srt.pic import picache
    from methods.hypic.backend.kvbench_hypic.srt.sampling.sampling_params import (
        SamplingParams,
    )

    server_args = SimpleNamespace(
        pic_segment_min_tokens=-1,
        pic_seam_sink=2,
        pic_epic_tokens_per_chunk=None,
        pic_epic_ratio=0.25,
        enable_mamba_extra_buffer=lambda: False,
    )
    monkeypatch.setattr(picache, "get_global_server_args", lambda: server_args)
    monkeypatch.setattr(
        schedule_batch,
        "get_global_server_args",
        lambda: server_args,
    )
    monkeypatch.setattr(schedule_batch, "is_pin_memory_available", lambda _device: False)
    # Keep the real ScheduleBatch.prepare_for_extend path, replacing only the
    # GPU allocator boundary with CPU tensors.
    monkeypatch.setattr(
        schedule_batch,
        "alloc_for_extend",
        lambda batch: (
            torch.empty(batch.extend_num_tokens, dtype=torch.int64),
            torch.tensor([0]),
            torch.tensor([0]),
        ),
    )
    monkeypatch.setattr(
        schedule_batch,
        "SamplingBatchInfo",
        SimpleNamespace(from_schedule_batch=lambda *_args: None),
    )

    req_pool = CpuReqToTokenPool()
    cache = picache.PICache(
        req_pool,
        CpuAllocator(),
        mamba_pool=None,
        page_size=1,
        disable=False,
        pic_mode=mode,
    )
    prompt_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for start, slot, mamba_slot in ((0, 20, 1), (4, 24, 2)):
        token_ids = torch.tensor(prompt_ids[start : start + 4])
        cache.inject_received_segment(
            picache.segment_hash(token_ids),
            token_ids,
            torch.tensor([slot, slot + 1, slot + 2, slot + 3]),
            mamba_slot,
        )

    req = schedule_batch.Req(
        "cpu-metric-test",
        "",
        array("q", prompt_ids),
        SamplingParams(max_new_tokens=8),
    )
    req.pic_segments = [(0, 4), (4, 8), (8, 10)]
    req.init_next_round_input(tree_cache=cache)
    req.fill_len = len(prompt_ids)

    batch = schedule_batch.ScheduleBatch(
        [req],
        tree_cache=cache,
        req_to_token_pool=req_pool,
        model_config=SimpleNamespace(is_encoder_decoder=False, vocab_size=64),
        device="cpu",
    )
    batch.prepare_for_extend()
    return req


@pytest.mark.native
@pytest.mark.usefixtures("native")
def test_hypic_seam_hits_count_as_matched_tokens_after_real_schedule_prepare(monkeypatch):
    req = _run_real_schedule_path(monkeypatch, "transition_rope_recompute")

    # Two 4-token hits, with two seam rows removed from prefix_indices and
    # replayed online. The raw matched-input count remains eight.
    assert len(req.prefix_indices) == 6
    assert req.pic_hit_tokens_removed_from_prefix == 2
    assert req.cached_tokens == 8
    assert req.cached_tokens_device == 8


@pytest.mark.native
@pytest.mark.usefixtures("native")
def test_linearkv_repair_rows_count_as_matched_tokens_after_real_schedule_prepare(
    monkeypatch,
):
    req = _run_real_schedule_path(monkeypatch, "linearkv")

    # LinearKV keeps repaired rows in prefix_indices, so both matched chunks
    # contribute all eight tokens despite two selector positions being replayed.
    assert len(req.prefix_indices) == 8
    assert sum(map(len, req.pic_linearkv_selected_positions.values())) == 2
    assert req.cached_tokens == 8
    assert req.cached_tokens_device == 8


@pytest.fixture(scope='module')
def native():
    from methods.hypic.Build import _build_directory
    if not (_build_directory() / 'kvbench_hypic_ops.so').is_file():
        pytest.skip('Run setup.py --hypic to test the optional native backend')
    from methods.hypic.backend.kvbench_hypic.srt.managers.io_struct import (
        GenerateReqInput,
    )
    from methods.hypic.backend.kvbench_hypic.srt.managers.tokenizer_manager import (
        TokenizerManager,
    )
    from methods.hypic.backend.kvbench_hypic.srt.sampling.sampling_params import (
        SamplingParams,
    )
    return GenerateReqInput, TokenizerManager, SamplingParams


@pytest.mark.native
@pytest.mark.usefixtures("native")
def test_normalization_and_scheduler_message_preserve_ids_and_ranges(native):
    GenerateReqInput, TokenizerManager, SamplingParams = native
    ids = [151644, 3, 151645, 4]
    ranges = [(0, 1), (1, 3), (3, 4)]
    request = GenerateReqInput(input_ids=ids, pic_segments=ranges,
                               sampling_params={'max_new_tokens': 4})
    request.normalize_batch_and_arguments()
    manager = object.__new__(TokenizerManager)
    manager.preferred_sampling_params = None
    manager.sampling_params_class = SamplingParams
    manager.tokenizer = SimpleNamespace(encode=lambda *a, **k: pytest.fail('IDs were re-tokenized'))
    manager.model_config = SimpleNamespace(vocab_size=200000)
    manager.server_args = SimpleNamespace(disaggregation_transfer_backend='mooncake')
    manager.rid_to_state = {request.rid: SimpleNamespace(time_stats=SimpleNamespace(
        set_tokenize_finish_time=lambda: None))}
    message = manager._create_tokenized_object(request, None, request.input_ids)
    assert list(message.input_ids) == ids
    assert message.pic_segments == ranges


@pytest.mark.parametrize('kwargs', [
    {'input_ids': [[1, 2]], 'pic_segments': [(0, 2)]},
    {'input_ids': [1, 2], 'pic_segments': [(0, 2)], 'sampling_params': {'n': 2}},
])
@pytest.mark.native
@pytest.mark.usefixtures("native")
def test_pic_rejects_batch_or_parallel_sample_normalization(native, kwargs):
    with pytest.raises(ValueError):
        native[0](**kwargs).normalize_batch_and_arguments()


@pytest.mark.native
@pytest.mark.usefixtures("native")
def test_rope_theta_can_remain_in_the_transformers_v4_config(native):
    from methods.hypic.backend.kvbench_hypic.srt.utils.hf_transformers.common import (
        get_rope_config,
    )
    config = SimpleNamespace(rope_parameters={'rope_type': 'default'}, rope_theta=1000000)
    assert get_rope_config(config) == (1000000, {'rope_type': 'default'})


class RecordingAllocator:
    def __init__(self):
        self.freed = []

    def free(self, slots):
        self.freed.extend(int(slot) for slot in slots.tolist())


def _make_cache(monkeypatch):
    from methods.hypic.backend.kvbench_hypic.srt.pic import picache

    monkeypatch.setattr(
        picache, "get_global_server_args", lambda: SimpleNamespace(pic_segment_min_tokens=-1)
    )
    pool = SimpleNamespace(
        mamba_allocator=RecordingAllocator(),
        req_to_token=torch.zeros((1, 8), dtype=torch.int64),
    )
    kv_allocator = RecordingAllocator()
    cache = picache.PICache(
        pool,
        kv_allocator,
        mamba_pool=None,
        page_size=1,
        disable=False,
        pic_mode="transition_rope_recompute",
    )
    return cache, pool, kv_allocator


@pytest.mark.native
@pytest.mark.usefixtures("native")
def test_finished_request_does_not_free_slots_already_transferred_to_picache(monkeypatch):
    from methods.hypic.backend.kvbench_hypic.srt.mem_cache.base_prefix_cache import (
        EvictParams,
    )

    cache, pool, kv_allocator = _make_cache(monkeypatch)
    req = SimpleNamespace(
        pic_miss_segments=[(0, 2), (2, 4)],
        pic_segments=[(0, 2), (2, 4)],
        origin_input_ids=[11, 12, 21, 22],
        pic_miss_segment_slots={
            (0, 2): (torch.tensor([100, 101]), torch.tensor([10, 11]), 7),
            (2, 4): (torch.tensor([102, 103]), None, 8),
        },
        pic_segment_entries={},
        pic_cache_owned_miss_segments=set(),
        pic_freed_miss_segments=set(),
        pic_linearkv_skip_cache=False,
        req_pool_idx=0,
        mamba_pool_idx=None,
        kv_committed_len=4,
    )

    # The normal streaming path publishes cacheable misses after prefill, then
    # cache_finished_req calls cache_unfinished_req again on final completion.
    cache.add_inflight(num_tokens=6, num_mamba_slots=2)
    cache.cache_unfinished_req(req)
    entry = next(iter(cache._entries.values()))
    assert entry.full_kv_slots.tolist() == [10, 11]
    assert entry.lock_ref == 1

    cache.cache_finished_req(req)

    assert entry.seg_hash in cache._entries
    assert entry.full_kv_slots.tolist() == [10, 11]
    assert entry.lock_ref == 0
    # Only private request slots (including the uncached last segment) are
    # returned. The public KV and Mamba slots still owned by the entry survive.
    assert sorted(kv_allocator.freed) == [100, 101, 102, 103]
    assert pool.mamba_allocator.freed == [8]

    # The transferred slots are eventually freed exactly once when the now
    # evictable entry is removed.
    cache.evict(EvictParams(num_tokens=2, mamba_num=1))
    assert sorted(kv_allocator.freed) == [10, 11, 100, 101, 102, 103]
    assert sorted(pool.mamba_allocator.freed) == [7, 8]


def test_constructors_do_not_import_or_initialize_gpu_backend():
    root = Path(__file__).resolve().parents[2]
    subprocess.run([sys.executable, '-B', '-c', '''
import sys
from methods import HypicMethod, LinearKVMethod
HypicMethod()
LinearKVMethod()
assert not any(name.startswith('methods.hypic.backend.') for name in sys.modules)
import torch
assert not torch.cuda.is_initialized()
'''], cwd=root, check=True, capture_output=True, text=True)


def test_prepare_is_excluded_but_online_matching_is_included_in_ttft(clock, monkeypatch):
    obj = method()
    match = adapter.ComposeInterleavedReuse

    def lookup(*args):
        clock.advance(0.02)
        return match(*args)

    def generate(**kwargs):
        clock.advance(0.05 if kwargs["sampling_params"]["max_new_tokens"] == 1 else 0.03)
        yield {"text": "answer", "output_ids": [31], "meta_info": {"completion_tokens": 1}}

    monkeypatch.setattr(adapter, "ComposeInterleavedReuse", lookup)
    obj.engine.generate = generate
    obj.Prepare([[[7, 8]]])
    assert clock.now == pytest.approx(0.05)
    result = obj.Run([[41, 42]], maxNewTokens=8)[0]
    assert result.performance["ttft"] == pytest.approx(0.05)
    assert result.performance["total_time"] == pytest.approx(0.05)


@pytest.mark.parametrize("kind", [HypicMethod, LinearKVMethod])
@pytest.mark.parametrize("operation", ["Prepare", "Run"])
def test_batch_limit_is_enforced_before_backend_work(kind, operation):
    obj = method(kind)
    data = [[[1]], [[2]]] if operation == "Prepare" else [[1], [2]]
    with pytest.raises(ValueError, match="one case"):
        getattr(obj, operation)(data)
    assert obj.engine.calls == []


def test_empty_backend_stream_fails_and_failed_flush_is_visible():
    obj = method()
    obj.engine.generate = lambda **kwargs: iter(())
    with pytest.raises(RuntimeError, match="no generation output"):
        obj.Run([[1]])
    obj.engine.flush_cache = lambda: SimpleNamespace(success=False, message="flush failed")
    with pytest.raises(RuntimeError, match="flush failed"):
        obj.Reset()
