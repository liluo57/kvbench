from array import array
from types import SimpleNamespace

import torch

from methods.hypic.backend.kvbench_hypic.srt.managers import schedule_batch
from methods.hypic.backend.kvbench_hypic.srt.pic import picache
from methods.hypic.backend.kvbench_hypic.srt.sampling.sampling_params import (
    SamplingParams,
)


class CpuAllocator:
    device = "cpu"

    def free(self, _slots):
        pass


class CpuReqToTokenPool:
    def __init__(self):
        self.mamba_allocator = CpuAllocator()
        self.req_to_token = torch.zeros((1, 32), dtype=torch.int64)


def _run_real_schedule_path(monkeypatch, mode):
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


def test_hypic_seam_hits_count_as_matched_tokens_after_real_schedule_prepare(monkeypatch):
    req = _run_real_schedule_path(monkeypatch, "transition_rope_recompute")

    # Two 4-token hits, with two seam rows removed from prefix_indices and
    # replayed online. The raw matched-input count remains eight.
    assert len(req.prefix_indices) == 6
    assert req.pic_hit_tokens_removed_from_prefix == 2
    assert req.cached_tokens == 8
    assert req.cached_tokens_device == 8


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
