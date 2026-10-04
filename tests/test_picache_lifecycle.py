from types import SimpleNamespace

import torch

from methods.hypic.backend.kvbench_hypic.srt.mem_cache.base_prefix_cache import (
    EvictParams,
)
from methods.hypic.backend.kvbench_hypic.srt.pic import picache


class RecordingAllocator:
    def __init__(self):
        self.freed = []

    def free(self, slots):
        self.freed.extend(int(slot) for slot in slots.tolist())


def _make_cache(monkeypatch):
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


def test_finished_request_does_not_free_slots_already_transferred_to_picache(monkeypatch):
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
