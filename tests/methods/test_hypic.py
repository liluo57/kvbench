"""Tests for hypic."""

import importlib
import dataclasses
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
    cacheblend_value_scores,
    prophet_attention_scores,
    select_initial_positions,
    select_scored_positions,
    merge_online_positions,
    select_epic_positions,
)

from tests.support.hypic_runtime import runtime_definition as _runtime_definition

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
    assert LinearKVMethod().Label == 'linearkv(epic,r=0.2)'


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
    # contribute all eight tokens despite only the second chunk being repaired.
    assert len(req.prefix_indices) == 8
    assert req.pic_linearkv_selected_positions == {(4, 8): [4]}
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


def _forward_batch_class():
    return _runtime_definition('model_executor/forward_batch_info.py', 'ForwardBatch',
                               ('pic_policy',), include_fields=True)


@pytest.mark.parametrize('no_copy', [False, True])
def test_real_eager_batch_copy_preserves_selector_probe(no_copy):
    batch_class = _forward_batch_class()
    registry_class = _runtime_definition('model_executor/cuda_graph_buffer_registry.py',
                                          'CudaGraphBufferRegistry', ('extract_buffer',))
    eager_class = _runtime_definition('model_executor/runner/eager_runner.py', 'EagerRunner', ('load_batch',))
    registry = registry_class()
    registry._slots = {}
    registry.fill_from = lambda *_args, **_kwargs: None
    eager = eager_class()
    eager._eager_registry = registry
    eager_class.load_batch.__globals__.update(replace=dataclasses.replace,
        envs=SimpleNamespace(SGLANG_EAGER_INPUT_NO_COPY=SimpleNamespace(get=lambda: no_copy)))
    ids = torch.arange(3)
    batch = batch_class(forward_mode='extend', batch_size=1, input_ids=ids,
        req_pool_indices=torch.tensor([0]), seq_lens=torch.tensor([3]), out_cache_loc=ids,
        seq_lens_sum=3)
    collector = object()
    batch.pic_selector_probe = collector
    copied = eager.load_batch(batch)
    assert copied is not batch
    assert copied.pic_selector_probe is collector


@pytest.mark.parametrize('selector', ['epic', 'cacheblend', 'prophetkv'])
def test_zero_ratio_uses_real_last_block_initialization_and_only_forwards_query(selector):
    backend_class = _runtime_definition('layers/attention/linear/gdn_backend.py', 'GDNAttnBackend',
        ('_initialize_linearkv_last_block_state', '_persist_linearkv_independent_state',
         'forward_extend_pic_linearkv'))
    backend_class._initialize_linearkv_last_block_state.__globals__['_diag_dump'] = SimpleNamespace(
        dump_ssm_state=lambda *_args: None)
    backend = backend_class()
    temporal = torch.randn(5, 2, 2)
    conv = torch.randn(5, 2, 3)
    tails = torch.randn(5, 2, 3)
    state = SimpleNamespace(temporal=temporal, conv=[conv])
    backend.req_to_token_pool = SimpleNamespace(mamba2_layer_cache=lambda _: state,
                                                mamba2_conv_tails_cache=lambda _: [tails])
    backend.forward_metadata = SimpleNamespace(mamba_cache_indices=torch.tensor([1]))
    hits = [(2, 4, b'b'), (0, 2, b'a')]  # Deliberately not in context order.
    selected = select_initial_positions(hits, selector, ratio=0.)
    positions = merge_online_positions([(4, 6)], selected)
    full_ids = torch.tensor([11, 12, 21, 22, 31, 32])
    batch = _forward_batch_class()(forward_mode='extend', batch_size=1,
        input_ids=full_ids[positions], positions=torch.tensor(positions),
        req_pool_indices=torch.tensor([0]), seq_lens=torch.tensor([6]),
        out_cache_loc=torch.tensor(positions), seq_lens_sum=6, pic_mode='linearkv',
        pic_hit_segments=[hits], pic_hit_mamba_slots=[{b'b': 3, b'a': 2}],
        pic_linearkv_selected_positions=[selected], pic_linearkv_independent_prefill=[False])
    initial_state = temporal[3].clone()
    initial_tail = tails[3].clone()
    cached_temporal, cached_tails = temporal[[2, 3]].clone(), tails[[2, 3]].clone()
    observed = []

    def reference_step(ssm, conv_state, token):
        key = torch.tensor([1., (int(token) % 5 + 1) / 5])
        key /= key.norm()
        value = torch.tensor([int(token) / 100, -int(token) / 200])
        ssm = .9 * ssm
        ssm = ssm + .3 * torch.outer(key, value - key @ ssm)
        conv_state = torch.cat([conv_state[:, 1:], value[:, None]], dim=1)
        return ssm, conv_state

    def extend(**kwargs):
        # This is the numerical kernel boundary, after the actual production
        # initializer has restored the cached state/tail pair.
        torch.testing.assert_close(temporal[1], initial_state)
        torch.testing.assert_close(conv[1], initial_tail)
        observed.extend(kwargs['forward_batch'].input_ids.tolist())
        ssm, conv_state = temporal[1].clone(), conv[1].clone()
        for token in kwargs['forward_batch'].input_ids:
            ssm, conv_state = reference_step(ssm, conv_state, token)
        temporal[1], conv[1] = ssm, conv_state
        return ssm

    backend.forward_extend = extend
    backend.forward_extend_pic_linearkv(SimpleNamespace(layer_id=0), batch,
        mixed_qkv=torch.empty(2, 1), a=torch.empty(2), b=torch.empty(2))
    expected, expected_tail = initial_state, initial_tail
    for token in [31, 32]:
        expected, expected_tail = reference_step(expected, expected_tail, token)
    assert selected == {} and positions == [4, 5] and observed == [31, 32]
    torch.testing.assert_close(temporal[1], expected)
    torch.testing.assert_close(conv[1], expected_tail)
    torch.testing.assert_close(temporal[[2, 3]], cached_temporal)
    torch.testing.assert_close(tails[[2, 3]], cached_tails)



@pytest.mark.parametrize('selector', ['epic', 'cacheblend', 'prophetkv', 'CacheBlend'])
def test_linearkv_selector_and_ratio_are_forwarded_without_initializing_backend(selector):
    obj = LinearKVMethod(selector=selector, recompRatio=.15)
    options = adapter._BuildHypicEngineKwargs('/model', [0], dtype='bfloat16',
        maxModelLen=100, memFractionStatic=.8, picMode=obj.picMode,
        fullPrefill=False, recompRatio=obj.recompRatio, selector=obj.selector,
        selectorLayer=obj.selectorLayer)
    assert options['pic_linearkv_selector'] == selector.lower()
    assert options['pic_epic_ratio'] == .15
    assert obj.Label == f'linearkv({selector.lower()},r=0.15)'
    if selector.lower() != 'epic':
        assert options['disable_overlap_schedule']


@pytest.mark.parametrize('kwargs', [dict(selector='unknown'), dict(selector=None),
    dict(selectorLayer=2), dict(selector='prophetkv', selectorLayer=2),
    dict(selector='cacheblend', selectorLayer=-1), dict(selector='cacheblend', selectorLayer=True)])
def test_linearkv_rejects_invalid_selector_settings(kwargs):
    with pytest.raises(ValueError):
        LinearKVMethod(**kwargs)


def test_selector_layer_and_user_tag_are_preserved():
    obj = LinearKVMethod(selector='cacheblend', selectorLayer=7, tag='probe')
    assert obj.selectorLayer == 7 and obj.Label == 'linearkv(probe)'
    assert LinearKVMethod(fullPrefill=True).Label == 'linearkv(full_prefill)'


def test_linearkv_reports_actual_selection_independently_from_cache_hit_coverage():
    obj = method(LinearKVMethod, selector='prophetkv', recompRatio=.2)
    selection = dict(selector='prophetkv', num_recomputed_tokens=1,
                     num_matched_tokens=7, recompute_ratio=1 / 7)

    def generate(**kwargs):
        yield dict(output_ids=[31], meta_info=dict(completion_tokens=1, cached_tokens=7,
                   linearkv_selection=[selection]))

    obj.engine.generate = generate
    result = obj.Run([[1, 2, 3, 4, 5, 6, 7, 8]], maxNewTokens=2)[0]
    assert result.metadata['reuse_ratio'] == 7 / 8
    assert result.metadata['recompute_ratio'] == 1 / 7
    assert result.metadata['target_recompute_ratio'] == .2
    assert result.metadata['num_recomputed_tokens'] == 1


@pytest.mark.parametrize('ratio,count', [(0, 0), (.01, 1), (.5, 2), (1, 4)])
def test_scored_selection_uses_global_budget_stable_ties_and_original_positions(ratio, count):
    hits = [(10, 12, b'b'), (2, 4, b'a')]
    selected = select_scored_positions(hits, torch.tensor([1., 8., 8., 2.]), ratio)
    positions = merge_online_positions([], selected)
    assert positions == sorted([3, 10, 11, 2][:count])


def test_cacheblend_scores_measure_value_drift_instead_of_value_magnitude():
    old = torch.tensor([[[100., 0.]], [[1., 0.]], [[5., 0.]]])
    current = torch.tensor([[[100., 0.]], [[1., 3.]], [[6., 0.]]])
    torch.testing.assert_close(cacheblend_value_scores(current, old), torch.tensor([0., 9., 1.]))


@pytest.mark.parametrize('heads,kv_heads', [(4, 4), (4, 2), (8, 1)])
@pytest.mark.parametrize('block_size', [1, 3, 16])
def test_prophet_scores_match_explicit_gqa_attention_reference(heads, kv_heads, block_size):
    query = torch.randn(7, heads, 8)
    keys = torch.randn(11, kv_heads, 8)
    repeated = keys.repeat_interleave(heads // kv_heads, dim=1)
    logits = torch.einsum('qhd,chd->hqc', query, repeated) / (8 ** .5)
    expected = logits.softmax(-1).mean(0).sum(0)
    torch.testing.assert_close(prophet_attention_scores(query, keys, block_size), expected)


def test_scored_selector_rejects_misaligned_or_nonfinite_scores():
    for scores in (torch.zeros(2), torch.tensor([float('nan')])):
        with pytest.raises(ValueError):
            select_scored_positions([(0, 1, b'a')], scores, .2)


def _selector_runtime_case(monkeypatch, selector, ratio=.5, selector_layer=None, fail=False):
    from methods.hypic.backend.kvbench_hypic.srt.pic import runtime_selector

    hits = [(1, 4, b'a'), (5, 8, b'b')]
    repair = select_initial_positions(hits, selector, ratio=ratio)
    req = SimpleNamespace(pic_hit_segments=hits, pic_miss_segments=[(0, 1), (4, 5), (8, 10)],
                          pic_segments=[(0, 1), (1, 4), (4, 5), (5, 8), (8, 10)],
                          pic_linearkv_selected_positions=repair)
    positions = torch.arange(10)
    online_positions = torch.tensor(merge_online_positions([(0, 1), (4, 5), (8, 10)], repair))
    meta = dict(local_hit=[(1, 4, torch.arange(1, 4), torch.arange(20, 23)),
                           (5, 8, torch.arange(5, 8), torch.arange(23, 26))],
                local_miss=[], repair=dict(positions=repair), seam=None, global_=None)
    original = _forward_batch_class()(forward_mode=SimpleNamespace(is_split_prefill=lambda: False), batch_size=1, input_ids=online_positions + 50, positions=online_positions,
        out_cache_loc=online_positions.clone(), mrope_positions=online_positions.repeat(3, 1),
        num_token_non_padded=None, input_embeds=None, replace_embeds=None, spec_info=None,
        req_pool_indices=torch.tensor([0]), pic_rope_meta=[meta], pic_mode='linearkv',
        seq_lens=torch.tensor([10]), seq_lens_cpu=torch.tensor([10]),
        orig_seq_lens=torch.tensor([10]), seq_lens_sum=10,
        extend_num_tokens=len(online_positions), extend_seq_lens=torch.tensor([len(online_positions)]),
        extend_seq_lens_cpu=[len(online_positions)],
        extend_prefix_lens=torch.tensor([6]), extend_prefix_lens_cpu=[6],
        extend_start_loc=torch.tensor([0]), num_token_non_padded_cpu=len(online_positions),
        mamba_clear_indices=None, mamba_cow_src_indices=None, mamba_cow_dst_indices=None)
    temporal = torch.randn(2, 4, 2, 3)
    conv = [torch.randn(2, 4, 2, 3)]
    tail = torch.randn_like(conv[0])
    transition = torch.randn_like(temporal)
    state = SimpleNamespace(temporal=temporal, conv=conv, conv_tails=[tail], transition=transition)
    mamba_pool = SimpleNamespace(mamba_cache=state)

    def clear(indices):
        temporal[:, indices] = 0
        conv[0][:, indices] = 0
        tail[:, indices] = 0
        transition[:, indices] = 0

    mamba_pool.clear_slots = clear
    pool = SimpleNamespace(mamba_pool=mamba_pool, get_mamba_indices=lambda _: torch.tensor([1]),
                           req_to_token=positions.unsqueeze(0).clone())
    keys = {layer: torch.randn(32, 2, 3) for layer in (3, 7)}
    values = {layer: torch.randn(32, 2, 3) for layer in (3, 7)}
    kv_pool = SimpleNamespace(get_key_buffer=lambda layer: keys[layer],
                              get_value_buffer=lambda layer: values[layer])
    backend = SimpleNamespace(req_to_token_pool=pool, token_to_kv_pool=kv_pool,
        full_attn_layers=[3, 7], _pic_rope_cos_sin_cache=True, _pic_rope_phase_b_cache='stale',
        linear_attn_backend=SimpleNamespace(_execute_deferred_mamba_cow_and_clear=lambda _: None))
    dispatcher_class = _runtime_definition('layers/attention/hybrid_linear_attn_backend.py',
        'HybridLinearAttnBackend', ('forward_extend',))
    backend._is_full_attn = lambda layer, _layer_id=None: layer.layer_id in (3, 7)
    backend._forward_extend_pic_linearkv = lambda *_args, **kwargs: kwargs['q']
    backend.full_attn_backend = SimpleNamespace(forward_extend=lambda q, *_args, **_kwargs: q)

    def relocate(layer, fb, rope_meta):
        for _, _, private, public in rope_meta[0]['local_hit']:
            keys[layer.layer_id][private] = keys[layer.layer_id][public]

    backend._pic_rope_phase_b = relocate
    # Prophet runs the two earlier fresh rows and the two terminal query rows.
    queries = {layer: torch.randn(4 if selector == 'prophetkv' else 10, 4, 3) for layer in (3, 7)}
    new_values = {layer: torch.randn(10, 2, 3) for layer in (3, 7)}
    observed = []

    def forward(fb):
        # Match the real eager runner's dataclass reconstruction boundary.
        fb = dataclasses.replace(fb)
        observed.append((fb.positions.clone(), fb.input_ids.clone(), fb.seq_lens.clone()))
        if selector == 'prophetkv':
            assert fb.pic_mode == 'linearkv'
            assert fb.extend_prefix_lens_cpu == [6]
            assert temporal[:, 1].any() and conv[0][:, 1].any()
            assert pool.req_to_token[0].tolist() == list(range(10))
            assert fb.positions.tolist() == [0, 4, 8, 9]
            assert fb.pic_linearkv_selected_positions == [{}]
            assert fb.pic_rope_meta[0]['repair']['positions'] == {}
        assert not fb.pic_linearkv_independent_prefill[0]
        temporal[:, 1] += 10
        conv[0][:, 1] += 20
        if fail:
            raise RuntimeError('probe failed')
        for layer_id in (3, 7):
            layer = SimpleNamespace(layer_id=layer_id, tp_k_head_num=2, tp_q_head_num=4, head_dim=3)
            dispatcher_class.forward_extend(backend, layer, fb,
                q=queries[layer_id].flatten(1), v=new_values[layer_id])
        raise AssertionError('selector probe must stop before the LM head')

    runner = SimpleNamespace(server_args=SimpleNamespace(pic_linearkv_selector=selector,
                             pic_linearkv_selector_layer=selector_layer, pic_epic_ratio=ratio),
                             attn_backend=backend, forward=forward)
    reductions = []

    def all_reduce(scores):
        reductions.append(scores.clone())
        return scores

    monkeypatch.setitem(sys.modules, 'methods.hypic.backend.kvbench_hypic.srt.distributed',
                        SimpleNamespace(tensor_model_parallel_all_reduce=all_reduce))
    return SimpleNamespace(module=runtime_selector, batch=SimpleNamespace(reqs=[req]),
        original=original, runner=runner, state=state, pool=pool, keys=keys, values=values,
        queries=queries, new_values=new_values, observed=observed, reductions=reductions)


@pytest.mark.parametrize('selector,selector_layer', [('cacheblend', None), ('cacheblend', 7), ('prophetkv', None)])
def test_dynamic_selector_probes_then_packs_only_selected_and_fresh_rows(monkeypatch, selector, selector_layer):
    case = _selector_runtime_case(monkeypatch, selector, selector_layer=selector_layer)
    before_state = case.state.temporal.clone()
    before_conv = case.state.conv[0].clone()
    before_tails = case.state.conv_tails[0].clone()
    before_transition = case.state.transition.clone()
    before_mapping = case.pool.req_to_token.clone()
    public_keys = {layer: keys[20:26].clone() for layer, keys in case.keys.items()}
    public_values = {layer: values[20:26].clone() for layer, values in case.values.items()}
    result = case.module.prepare_linearkv_selection(case.batch, case.original, case.runner)
    cached_positions = [1, 2, 3, 5, 6, 7]
    if selector == 'cacheblend':
        layer = 3 if selector_layer is None else selector_layer
        expected_scores = (case.new_values[layer][cached_positions] - public_values[layer]).square().sum((1, 2))
        assert case.observed[0][0].tolist() == list(range(10))
    else:
        layer_scores = []
        for layer in (3, 7):
            expanded = public_keys[layer].repeat_interleave(2, dim=1)
            logits = torch.einsum('qhd,chd->hqc', case.queries[layer][2:], expanded) / (3 ** .5)
            layer_scores.append(logits.softmax(-1).mean(0).sum(0))
        expected_scores = torch.stack(layer_scores).mean(0)
        assert case.observed[0][0].tolist() == [0, 4, 8, 9]
        assert case.observed[0][2].tolist() == [10]
    expected_indices = sorted(range(6), key=lambda index: (-float(expected_scores[index]), index))[:3]
    expected = sorted([0, 4, 8, 9] + [cached_positions[index] for index in expected_indices])
    assert result.positions.tolist() == expected
    assert result.input_ids.tolist() == [50 + position for position in expected]
    assert result.out_cache_loc.tolist() == expected
    assert result.extend_seq_lens_cpu == [7] and result.num_token_non_padded_cpu == 7
    assert result.seq_lens.tolist() == [10] and result.extend_prefix_lens_cpu == [6]
    assert result.mrope_positions.tolist() == [expected] * 3
    assert result.pic_rope_meta[0]['repair']['positions'] == case.batch.reqs[0].pic_linearkv_selected_positions
    assert case.original.positions.tolist() == list(range(10))
    torch.testing.assert_close(case.state.temporal, before_state)
    torch.testing.assert_close(case.state.conv[0], before_conv)
    torch.testing.assert_close(case.state.conv_tails[0], before_tails)
    torch.testing.assert_close(case.state.transition, before_transition)
    torch.testing.assert_close(case.pool.req_to_token, before_mapping)
    for layer in (3, 7):
        torch.testing.assert_close(case.keys[layer][20:26], public_keys[layer])
        torch.testing.assert_close(case.values[layer][20:26], public_values[layer])
    assert len(case.reductions) == 1
    torch.testing.assert_close(case.reductions[0], expected_scores)
    assert case.module.selection_metadata(case.batch.reqs[0], selector)['recompute_ratio'] == .5


@pytest.mark.parametrize('selector', ['cacheblend', 'prophetkv'])
def test_probe_failure_restores_state_and_query_slot_mapping(monkeypatch, selector):
    case = _selector_runtime_case(monkeypatch, selector, fail=True)
    temporal = case.state.temporal.clone()
    conv = case.state.conv[0].clone()
    tails = case.state.conv_tails[0].clone()
    transition = case.state.transition.clone()
    mapping = case.pool.req_to_token.clone()
    with pytest.raises(RuntimeError, match='probe failed'):
        case.module.prepare_linearkv_selection(case.batch, case.original, case.runner)
    torch.testing.assert_close(case.state.temporal, temporal)
    torch.testing.assert_close(case.state.conv[0], conv)
    torch.testing.assert_close(case.state.conv_tails[0], tails)
    torch.testing.assert_close(case.state.transition, transition)
    torch.testing.assert_close(case.pool.req_to_token, mapping)


@pytest.mark.parametrize('selector', ['cacheblend', 'prophetkv'])
@pytest.mark.parametrize('ratio', [0., 1.])
def test_endpoint_budgets_skip_unnecessary_dynamic_probes(monkeypatch, selector, ratio):
    case = _selector_runtime_case(monkeypatch, selector, ratio=ratio)
    assert case.module.prepare_linearkv_selection(case.batch, case.original, case.runner) is case.original
    assert case.observed == [] and case.reductions == []
    expected = [0, 4, 8, 9] if ratio == 0 else list(range(10))
    assert case.original.positions.tolist() == expected
    assert case.module.selection_metadata(case.batch.reqs[0], selector)['recompute_ratio'] == ratio


def test_global_rank_scores_determine_one_shared_selection(monkeypatch):
    case = _selector_runtime_case(monkeypatch, 'cacheblend')
    distributed = sys.modules['methods.hypic.backend.kvbench_hypic.srt.distributed']
    distributed.tensor_model_parallel_all_reduce = lambda scores: torch.arange(6, dtype=scores.dtype)
    result = case.module.prepare_linearkv_selection(case.batch, case.original, case.runner)
    assert result.positions.tolist() == [0, 4, 5, 6, 7, 8, 9]


def test_prophet_query_padding_does_not_change_scores(monkeypatch):
    case = _selector_runtime_case(monkeypatch, 'prophetkv')
    expected_layers = []
    for layer in (3, 7):
        keys = case.keys[layer][20:26].repeat_interleave(2, dim=1)
        logits = torch.einsum('qhd,chd->hqc', case.queries[layer][2:], keys) / (3 ** .5)
        expected_layers.append(logits.softmax(-1).mean(0).sum(0))
        case.queries[layer] = torch.cat([case.queries[layer], torch.full((1, 4, 3), 100.)])
    case.module.prepare_linearkv_selection(case.batch, case.original, case.runner)
    torch.testing.assert_close(case.reductions[0], torch.stack(expected_layers).mean(0))


def test_cacheblend_rejects_a_non_attention_probe_layer_before_execution(monkeypatch):
    case = _selector_runtime_case(monkeypatch, 'cacheblend', selector_layer=2)
    with pytest.raises(ValueError, match='full-attention layer'):
        case.module.prepare_linearkv_selection(case.batch, case.original, case.runner)
    assert case.observed == []


def test_dynamic_selector_skips_probe_when_no_prepared_chunk_hits(monkeypatch):
    case = _selector_runtime_case(monkeypatch, 'prophetkv')
    case.batch.reqs[0].pic_hit_segments = []
    assert case.module.prepare_linearkv_selection(case.batch, case.original, case.runner) is case.original
    assert case.observed == []


@pytest.mark.parametrize('ratio', [0., .2, 1.])
def test_scheduled_epic_skips_position_zero_chunk_only_for_partial_repair(ratio):
    hits = [(0, 10, b'first'), (10, 20, b'second')]
    selected = select_initial_positions(hits, 'epic', ratio=ratio)
    if ratio == 1:
        assert selected == {(0, 10): list(range(10)), (10, 20): list(range(10, 20))}
    elif ratio == 0:
        assert selected == {}
    else:
        assert selected == {(10, 20): [10, 11]}  # No budget redistribution.


@pytest.mark.parametrize('selector', ['cacheblend', 'prophetkv'])
def test_dynamic_probe_still_exposes_the_first_chunk(selector):
    assert select_initial_positions([(0, 10, b'first')], selector, ratio=.2) == {
        (0, 10): list(range(10))}


@pytest.mark.parametrize('options', [
    dict(pic_linearkv_selector='unknown'),
    dict(pic_linearkv_selector='epic', pic_linearkv_selector_layer=3),
    dict(pic_linearkv_selector_layer=-1),
    dict(pic_linearkv_selector_layer=True),
    dict(pic_epic_ratio=None),
    dict(pic_epic_ratio=float('nan')),
    dict(pic_epic_tokens_per_chunk=2),
    dict(pp_size=2), dict(dp_size=2), dict(enable_dp_attention=True),
    dict(enable_pdmux=True),
])
def test_dynamic_selector_server_constraints_reject_unsupported_execution(options):
    cls = _runtime_definition('server_args.py', 'ServerArgs', ('check_pic_constraints',))
    cls.check_pic_constraints.__globals__['math'] = __import__('math')
    args = cls()
    defaults = dict(pic_enable=True, model_path='Qwen3.6', pic_seam_sink=8,
        pic_mode='linearkv', pic_linearkv_selector='cacheblend',
        pic_linearkv_selector_layer=None, pic_epic_ratio=.2,
        pic_epic_tokens_per_chunk=None, pp_size=1, dp_size=1,
        enable_dp_attention=False, enable_pdmux=False, disable_overlap_schedule=False)
    defaults.update(options)
    args.__dict__.update(defaults)
    with pytest.raises(ValueError):
        args.check_pic_constraints()


def test_dynamic_selector_server_constraints_disable_scheduler_overlap():
    cls = _runtime_definition('server_args.py', 'ServerArgs', ('check_pic_constraints',))
    cls.check_pic_constraints.__globals__['math'] = __import__('math')
    args = cls()
    args.__dict__.update(pic_enable=True, model_path='Qwen3.6', pic_seam_sink=8,
        pic_mode='linearkv', pic_linearkv_selector='prophetkv',
        pic_linearkv_selector_layer=None, pic_epic_ratio=.2,
        pic_epic_tokens_per_chunk=None, pp_size=1, dp_size=1,
        enable_dp_attention=False, enable_pdmux=False, disable_overlap_schedule=False,
        chunked_prefill_size=-1, page_size=1, speculative_algorithm=None)
    args.check_pic_constraints()
    assert args.disable_overlap_schedule


def test_prophet_rejects_cached_terminal_query_before_probing(monkeypatch):
    case = _selector_runtime_case(monkeypatch, 'prophetkv')
    case.batch.reqs[0].pic_segments = [(0, 1), (1, 4), (4, 5), (5, 8)]
    with pytest.raises(ValueError, match='fresh terminal Query'):
        case.module.prepare_linearkv_selection(case.batch, case.original, case.runner)
    assert case.observed == []



@pytest.mark.parametrize('selector', ['cacheblend', 'prophetkv'])
@pytest.mark.parametrize('ratio', [0., .5, 1.])
def test_dynamic_selector_rejects_split_prefill_before_probe_or_state_changes(monkeypatch, selector, ratio):
    case = _selector_runtime_case(monkeypatch, selector, ratio=ratio)
    case.original.forward_mode = SimpleNamespace(is_split_prefill=lambda: True)
    state = case.state.temporal.clone()
    with pytest.raises(ValueError, match='split prefill'):
        case.module.prepare_linearkv_selection(case.batch, case.original, case.runner)
    assert case.observed == [] and case.reductions == []
    torch.testing.assert_close(case.state.temporal, state)


@pytest.mark.parametrize('selector', ['cacheblend', 'prophetkv'])
def test_worker_rejects_dynamic_split_prefill_before_building_batch(selector):
    cls = _runtime_definition('managers/tp_worker.py', 'TpModelWorker',
                              ('forward_batch_split_prefill',))
    worker = cls()
    worker.model_runner = SimpleNamespace(server_args=SimpleNamespace(pic_linearkv_selector=selector))
    batch = SimpleNamespace(pic_mode='linearkv', split_index=0)
    with pytest.raises(ValueError, match='split prefill'):
        worker.forward_batch_split_prefill(batch)


@pytest.mark.parametrize('selector', ['epic', 'cacheblend', 'prophetkv'])
def test_linearkv_preserves_prepared_prefix_boundaries_with_repeated_content(selector):
    obj = method(LinearKVMethod, selector=selector)
    chunks = [[1, 2], [1, 2], [1, 2, 1]]
    obj.Prepare([chunks])
    ids = sum(chunks, []) + [2, 1, 2]
    obj.Run([ids], maxNewTokens=1)
    sent, segments, _ = obj.engine.calls[-1]
    assert sent == ids
    assert segments == [(0, 2), (2, 4), (4, 7), (7, 10)]
