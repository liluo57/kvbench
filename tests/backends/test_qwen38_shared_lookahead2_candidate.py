"""CPU-only tests for the opt-in root lookahead candidate.

Apply shared-lookahead2.patch after observe-single-sample.patch and
long-shared-admission-flag.patch before running this file. Pools are created with
__new__; mocked executors avoid model/GPU startup.
"""
from collections import deque
from concurrent.futures import Future

import torch


def _row(size, token):
    return {"chunks": [[token] * size]}


def _pool(monkeypatch, rows):
    from methods.CacheRepair.qwen38.vllm_pool import VllmPairPool

    class Executor:
        def __init__(self):
            self.calls = []

        def submit(self, fn, chunks, transport, verify):
            future = Future()
            self.calls.append((fn, chunks, transport, verify, future))
            return future

    pool = VllmPairPool.__new__(VllmPairPool)
    pool.groups = [(0, 1), (2, 3), (4, 5)]
    pool.device = torch.device("cuda", 0)  # Descriptor only; no CUDA context.
    pool.shared = True
    pool.prefetch_per_worker = 8
    pool.max_pending_pairs = 9
    pool.max_batch_examples = 1
    pool.max_batch_input_tokens = 8192
    pool.max_pending_input_tokens_per_worker = 49152
    pool.max_pending_input_tokens = 65536
    pool.scheduler = "least_work"
    pool.transport = "cuda"
    pool.dynamic_batching = False
    pool.defer_shared_on_long = False
    pool.pending = deque()
    pool.costs = {}
    pool.releases = []
    pool.cursor = 0
    pool.submitted = 0
    pool.verify_groups = set()
    pool.dispatchers = []
    pool.executors = [Executor(), Executor(), Executor()]
    pool.sequence = rows
    pool._root_empty_cache_wall_seconds = 0.0
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    return pool


def test_root_lookahead_is_opt_in_bounded_to_two_single_rows_and_keeps_order(monkeypatch):
    from methods.CacheRepair.qwen38.resume import fingerprint
    from methods.CacheRepair.qwen38.vllm_pool import _generate

    pool = _pool(monkeypatch, [_row(1000, i) for i in range(16)])
    assert pool._capacity(0) == 1
    assert pool._capacity(1) == pool._capacity(2) == 8

    pool.shared_prefetch_per_worker = 2
    pool.shared_pending_input_tokens = 16384
    assert pool._capacity(0) == 2
    pool._fill()

    root = [(key, future) for key, group, future in pool.pending if group == 0]
    assert len(root) == 2
    assert all(pool.costs[future] == 1000 for _, future in root)
    assert sum(pool.costs[future] for _, future in root) <= 16384
    assert all(fn is _generate for fn, *_ in pool.executors[0].calls)
    assert [key for key, _, _ in pool.pending] == [
        fingerprint(row["chunks"]) for row in pool.sequence[:pool.cursor]
    ]
    assert len(pool.pending) <= 9
    assert sum(pool.costs[future] for _, _, future in pool.pending) <= 65536


def test_row_larger_than_root_budget_skips_root_but_remote_queue_advances(monkeypatch):
    from methods.CacheRepair.qwen38.resume import fingerprint

    pool = _pool(monkeypatch, [_row(17000, 1), *[_row(1000, i) for i in range(1, 12)]])
    pool.scheduler = "round_robin"  # Root is first; fallback must keep remotes moving.
    pool.shared_prefetch_per_worker = 2
    pool.shared_pending_input_tokens = 16384
    pool._fill()

    assert pool.cursor > 0
    first_key = fingerprint(pool.sequence[0]["chunks"])
    first = next((group, future) for key, group, future in pool.pending if key == first_key)
    assert first[0] in (1, 2)
    assert pool.costs[first[1]] == 17000  # Remote worker/global credits still admit it.
    assert all(
        sum(pool.costs[future] for _, group, future in pool.pending if group == i) <= 16384
        for i in (0,)
    )
    assert [key for key, _, _ in pool.pending] == [
        fingerprint(row["chunks"]) for row in pool.sequence[:pool.cursor]
    ]
    assert sum(pool.costs[future] for _, _, future in pool.pending) <= 65536


def test_root_remaining_token_credit_excludes_row_without_blocking_remotes(monkeypatch):
    from methods.CacheRepair.qwen38.resume import fingerprint

    rows = [_row(14999, 0), _row(2000, 1), *[_row(500, i) for i in range(2, 10)]]
    pool = _pool(monkeypatch, rows)
    pool.scheduler = "round_robin"
    pool.shared_prefetch_per_worker = 2
    pool.shared_pending_input_tokens = 16384
    existing = Future()
    pool.pending.append((fingerprint(rows[0]["chunks"]), 0, existing))
    pool.costs[existing] = 15000
    pool.cursor = 1
    pool.submitted = 3  # Root would be the next round-robin candidate.

    pool._fill()

    row1_key = fingerprint(rows[1]["chunks"])
    row1_group = next(group for key, group, _ in pool.pending if key == row1_key)
    assert row1_group in (1, 2)
    root_tokens = sum(pool.costs[future] for _, group, future in pool.pending if group == 0)
    assert root_tokens <= 16384
    assert pool.cursor > 1
    assert [key for key, _, _ in pool.pending] == [
        fingerprint(row["chunks"]) for row in rows[:pool.cursor]
    ]
