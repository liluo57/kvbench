"""CPU-only tests for the opt-in earliest-finish scheduler candidate.

The device value below is metadata only; _fill uses mock executors and never
calls CUDA. The profile flag defaults off and the default reference config
uses include_incoming_work=False.
"""
from collections import deque
from concurrent.futures import Future

import torch

from methods.CacheRepair.qwen38.vllm_pool import VllmPairPool


class _Executor:
    def submit(self, *args):
        return Future()


def _pool(incoming_tokens, chunks, *, include_incoming_work, attention_cost_weight=1e-5, backlog=1000):
    pool = VllmPairPool.__new__(VllmPairPool)
    pool.groups = [(2, 3), (4, 5)]  # Both remote to this CPU-side fixture.
    pool.device = torch.device('cuda', 0)  # Metadata only; no CUDA operation.
    pool.prefetch_per_worker = 2
    pool.scheduler = 'least_work'
    pool.attention_cost_weight = attention_cost_weight
    pool.replica_work_scales = [1.0, 1.3]
    pool.include_incoming_work = include_incoming_work
    pool.pending = deque()
    pool.costs = {}
    pool.work_costs = {}
    pool.releases = []
    pool.cursor = pool.submitted = 0
    pool.transport = 'cuda'
    pool.verify_groups = set()
    pool.sequence = [{'chunks': chunks}]
    pool._check_releases = lambda wait=False: None
    pool.executors = [_Executor(), _Executor()]

    # Fast group 0 is already busy. Slow group 1 is idle. A 1.3 multiplier
    # models its lower service rate; admission remains available on both.
    busy = Future()
    pool.pending.append(('busy', 0, busy))
    pool.costs[busy] = backlog
    pool.work_costs[busy] = backlog
    assert sum(map(len, chunks)) == incoming_tokens
    return pool


def test_large_incoming_row_accounts_for_busy_fast_replica():
    tokens = 16_000
    chunks = [[0] * 8_000, [1] * 8_000]
    pool = _pool(tokens, chunks, include_incoming_work=True)

    pool._fill()

    # Backlog-only scaling selects idle slow group 1. Completion-time scoring
    # adds the same incoming row to both choices and selects busy fast group 0.
    assert pool.pending[-1][1] == 0
    assert pool.costs[pool.pending[-1][2]] == tokens
    assert sum(pool.costs.values()) == 1000 + tokens  # Token-credit semantics unchanged.


def test_small_incoming_row_still_prefers_idle_slow_replica():
    tokens = 100
    chunks = [[0] * 50, [1] * 50]
    pool = _pool(tokens, chunks, include_incoming_work=True)

    pool._fill()

    assert pool.pending[-1][1] == 1
    assert pool.costs[pool.pending[-1][2]] == tokens
    assert sum(pool.costs.values()) == 1000 + tokens


def test_unconfigured_flag_defaults_to_backlog_only_choice():
    tokens = 16_000
    chunks = [[0] * 8_000, [1] * 8_000]
    pool = _pool(tokens, chunks, include_incoming_work=False)
    del pool.include_incoming_work  # Constructor/default behavior via getattr(..., False).

    pool._fill()

    assert pool.pending[-1][1] == 1
    assert pool.costs[pool.pending[-1][2]] == tokens


def test_incoming_attention_square_term_changes_finish_choice():
    tokens = 10_000
    chunks = [[0] * tokens]

    token_only = _pool(tokens, chunks, include_incoming_work=True,
                       attention_cost_weight=0.0, backlog=3_200)
    token_only._fill()
    assert token_only.pending[-1][1] == 1  # 13,200 fast vs 13,000 slow.

    attention_aware = _pool(tokens, chunks, include_incoming_work=True,
                            attention_cost_weight=1e-5, backlog=3_200)
    attention_aware._fill()
    assert attention_aware.pending[-1][1] == 0  # 15,200 fast vs 15,600 slow.
    assert attention_aware.costs[attention_aware.pending[-1][2]] == tokens
    assert sum(attention_aware.costs.values()) == 3_200 + tokens
