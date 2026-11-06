"""Unapplied CPU-only tests for optional heterogeneous least-work scheduling.

Copy these cases into tests/backends/test_cacherepair_qwen38_vllm_pool.py only
when the companion candidate patch is deliberately applied. This draft has not
been run and performs no CUDA operations (torch.device('cuda', 0) is metadata).
"""
from collections import deque
from concurrent.futures import Future

import pytest
import torch

from methods.CacheRepair.qwen38.vllm_pool import VllmPairPool


def _pool_for_one_admission(weight=0.0, scales=None):
    pool = VllmPairPool.__new__(VllmPairPool)
    pool.groups = [(2, 3), (4, 5)]
    pool.device = torch.device('cuda', 0)
    pool.prefetch_per_worker = 2
    pool.scheduler = 'least_work'
    pool.attention_cost_weight = weight
    pool.replica_work_scales = scales
    pool.pending = deque()
    pool.costs = {}
    pool.work_costs = {}
    pool.releases = []
    pool.cursor = pool.submitted = 0
    pool.transport = 'cuda'
    pool.verify_groups = set()
    pool.sequence = [{'chunks': [[1] * 30, [2] * 20]}]
    pool._check_releases = lambda wait=False: None

    class Executor:
        def submit(self, *args):
            return Future()

    pool.executors = [Executor(), Executor()]
    return pool


def _set_two_pending(pool, cost0=100, cost1=200, work0=None, work1=None):
    group0, group1 = Future(), Future()
    pool.pending.extend([('old0', 0, group0), ('old1', 1, group1)])
    pool.costs.update({group0: cost0, group1: cost1})
    if work0 is not None:
        pool.work_costs[group0] = work0
    if work1 is not None:
        pool.work_costs[group1] = work1


def test_default_cost_path_is_historical_token_only():
    pool = _pool_for_one_admission(weight=0.0, scales=None)
    _set_two_pending(pool, work0=1000, work1=300)

    pool._fill()

    assert pool.pending[-1][1] == 0
    assert pool.costs[pool.pending[-1][2]] == 50


def test_attention_cost_changes_least_work_only_and_preserves_token_credits():
    pool = _pool_for_one_admission(weight=1e-5, scales=None)
    _set_two_pending(pool, work0=1000, work1=300)

    pool._fill()

    _, assigned_group, submitted = pool.pending[-1]
    assert assigned_group == 1
    assert pool.costs[submitted] == 50
    assert pool.work_costs[submitted] == pytest.approx(50 + 1e-5 * (50**2 + 30**2 + 20**2))
    assert sum(pool.costs.values()) == 350


def test_replica_scale_applies_even_when_attention_cost_weight_is_zero():
    pool = _pool_for_one_admission(weight=0.0, scales=[3.0, 0.1])
    # No work_costs entry: with weight zero, scaled work is scaled T.
    _set_two_pending(pool)

    pool._fill()

    _, assigned_group, submitted = pool.pending[-1]
    assert assigned_group == 1  # 200 * 0.1 < 100 * 3.0
    assert pool.costs[submitted] == 50
    assert submitted not in pool.work_costs
    assert sum(pool.costs.values()) == 350


def test_replica_scale_multiplies_attention_work_after_group_aggregation():
    pool = _pool_for_one_admission(weight=1e-5, scales=[1.0, 1.3])
    _set_two_pending(pool, cost0=100, cost1=100, work0=100, work1=80)

    pool._fill()

    # Group 0 has lower base work (100 vs 80*1.3), so group 0 is selected.
    assert pool.pending[-1][1] == 0
    assert pool.costs[pool.pending[-1][2]] == 50


@pytest.mark.parametrize('scales', [[1.0], [1.0, 0.0], [1.0, -0.2], [1.0, float('nan')], [1.0, float('inf')]])
def test_invalid_replica_scales_are_rejected_before_admission(scales):
    pool = _pool_for_one_admission(weight=0.0, scales=scales)

    with pytest.raises(ValueError, match='replica_work_scales'):
        pool._fill()


def test_work_cost_falls_back_for_legacy_new_allocated_pool():
    pool = VllmPairPool.__new__(VllmPairPool)
    future = Future()
    pool.costs = {future: 123}
    # Deliberately no work_costs, attention_cost_weight, or replica_work_scales.

    assert pool._work_cost(future) == 123


def test_set_sequence_clears_cost_maps_but_preserves_configured_scales():
    pool = VllmPairPool.__new__(VllmPairPool)
    pool.pending = deque()
    pool.costs = {Future(): 10}
    pool.work_costs = {Future(): 20}
    pool.replica_work_scales = [1.0, 1.3]
    pool._check_releases = lambda wait=False: None
    pool._fill = lambda: None

    pool.set_sequence([])

    assert not pool.costs
    assert not pool.work_costs
    assert pool.replica_work_scales == [1.0, 1.3]
