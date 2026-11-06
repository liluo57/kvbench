"""CPU-only scheduling comparison for the proposed long-pair refill gate.

This test is intended to run after applying
outputs/cacherepair-qwen38/single-sample-optimization-20261029/long-shared-admission-flag.patch.
It constructs VllmPairPool with __new__ and mock executors; it never initializes CUDA.
"""
from collections import deque
from concurrent.futures import Future

import torch


def _pool(monkeypatch):
    from methods.CacheRepair.qwen38.resume import fingerprint
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
    pool.device = torch.device("cuda", 0)  # Device descriptor only; no CUDA context.
    pool.prefetch_per_worker = 8
    pool.max_pending_pairs = 9
    pool.max_batch_examples = 1
    pool.max_batch_input_tokens = 8192
    pool.max_pending_input_tokens_per_worker = 49152
    pool.max_pending_input_tokens = 65536
    pool.scheduler = "least_work"
    pool.transport = "cuda"
    pool.shared = True
    pool.dynamic_batching = False
    pool.defer_shared_on_long = True
    pool.pending = deque()
    pool.costs = {}
    pool.releases = []
    pool.cursor = 1  # The currently consumed long sample is sequence row 0.
    pool.submitted = 1
    pool.verify_groups = set()
    pool.dispatchers = []
    pool.executors = [Executor(), Executor(), Executor()]
    pool.sequence = [
        {"chunks": [[i] * n, [i + 100]]}
        for i, n in enumerate([8192, 1200, 2000, 900, 1300, 1600, 800, 700, 1100, 1000, 1400, 1500])
    ]
    pool._fingerprint = fingerprint
    pool._root_empty_cache_wall_seconds = 0.0
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    return pool


def test_long_pair_refill_skips_shared_pair_then_default_refill_restores_it(monkeypatch):
    pool = _pool(monkeypatch)
    pool._fill(allow_shared=False)

    assert pool.cursor > 1
    assert all(group != 0 for _, group, _ in pool.pending)
    assert [key for key, _, _ in pool.pending] == [
        pool._fingerprint(row["chunks"]) for row in pool.sequence[1:pool.cursor]
    ]
    assert len(pool.pending) <= pool.max_pending_pairs
    assert all(sum(group == i for _, group, _ in pool.pending) <= 8 for i in (1, 2))
    assert sum(pool.costs[future] for _, _, future in pool.pending) <= pool.max_pending_input_tokens

    # Consuming one remote slot and calling the ordinary next refill may admit
    # the shared pair again: allow_shared is a per-refill gate, not persistent state.
    _, _, consumed = pool.pending.popleft()
    pool.costs.pop(consumed)
    pool._fill()
    assert any(group == 0 for _, group, _ in pool.pending)


def test_control_refill_without_long_gate_can_admit_shared_pair(monkeypatch):
    pool = _pool(monkeypatch)
    pool._fill()
    assert any(group == 0 for _, group, _ in pool.pending)
    assert [key for key, _, _ in pool.pending] == [
        pool._fingerprint(row["chunks"]) for row in pool.sequence[1:pool.cursor]
    ]


def test_get_gates_only_post_consume_refill_and_keeps_both_long_trim_guards(monkeypatch):
    import time

    from methods.CacheRepair.qwen38.resume import fingerprint
    from methods.CacheRepair.qwen38.vllm_pool import VllmPairPool

    class TokenCount:
        def numel(self):
            return 8193

    class Executor:
        def __init__(self):
            self.calls = 0

        def submit(self, *args):
            self.calls += 1
            future = Future()
            future.set_result(True)
            return future

    chunks = [[7] * 8192, [8]]
    packet = {
        "shards": [{"bytes": 0, "prefill_stats": {}}],
        "produced_unix": time.time(),
        "generation_seconds": 1.0,
        "export_seconds": 0.0,
        "compute_seconds": 1.0,
        "batch_id": "long-row",
    }
    current = Future()
    current.set_result(packet)
    pool = VllmPairPool.__new__(VllmPairPool)
    pool.groups = [(0, 1), (2, 3), (4, 5)]
    pool.device = torch.device("cuda", 0)
    pool.shared = True
    pool.defer_shared_on_long = True
    pool.transport = "cuda"
    pool.pending = deque([(fingerprint(chunks), 2, current)])
    pool.costs = {current: 8193}
    pool.sequence = [{"chunks": chunks}]
    pool.cursor = 1
    pool.executors = [Executor(), Executor(), Executor()]
    pool.stream = None
    pool.log_path = None
    pool.releases = []
    # Compatible with the optional observation patch's consume-side timer field.
    pool._root_empty_cache_wall_seconds = 0.0
    def consume(packet, group, consumed_chunks):
        pool._copy_event_host_wait_seconds = 0.0
        return {"tokens": TokenCount()}, 0.0
    pool._consume = consume
    refill_arguments = []
    pool._fill = lambda **kwargs: refill_arguments.append(kwargs)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    result = pool.get(chunks)

    assert result["tokens"].numel() == 8193
    assert refill_arguments == [{"allow_shared": False}]
    # One producer trim before consume and one safety trim after refill remain.
    assert pool.executors[0].calls == 2
    assert pool.last_event["shared_safety_wait_seconds"] >= 0
    for field in (
        "root_empty_cache_wall_seconds",
        "long_pretrim_wait_seconds",
        "long_pretrim_trim_rpc_wall_seconds",
        "future_result_wait_seconds",
        "consume_copy_event_host_wait_seconds",
        "long_safety_wait_seconds",
        "long_safety_trim_rpc_wall_seconds",
    ):
        assert pool.last_event[field] >= 0


def test_round_robin_gate_skips_shared_turn_without_stalling_remote_refill(monkeypatch):
    pool = _pool(monkeypatch)
    pool.scheduler = "round_robin"
    pool.submitted = 3  # Next turn would be shared pair 0.
    pool._fill(allow_shared=False)
    assert pool.cursor > 1
    assert all(group != 0 for _, group, _ in pool.pending)


def test_dispatchers_exist_only_when_dynamic_batching_is_enabled(monkeypatch):
    from concurrent.futures import Future
    from types import SimpleNamespace

    from methods.CacheRepair.qwen38 import vllm_pool as module

    class Executor:
        def __init__(self, *args, **kwargs):
            pass

        def submit(self, fn, *args, **kwargs):
            future = Future()
            future.set_result({"pid": 1, "physical_devices": args[1]} if fn is module._load_engine else True)
            return future

        def shutdown(self, *args, **kwargs):
            pass

    class Target:
        rotary_emb = object()

        @staticmethod
        def get_input_embeddings():
            return SimpleNamespace(weight=SimpleNamespace(device=torch.device("cuda", 0)))

    monkeypatch.setattr(module, "ProcessPoolExecutor", Executor)
    monkeypatch.setattr(module.multiprocessing, "get_context", lambda method: object())
    monkeypatch.setattr(torch.cuda.memory, "get_allocator_backend", lambda: "native")
    monkeypatch.setattr(torch.cuda, "Stream", lambda device=None: object())
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5")

    common = dict(target=Target(), geometry=object(), groups=[(0, 1), (2, 3), (4, 5)], model_path="mock")
    reference = module.VllmPairPool(**common)
    assert reference.dispatchers == []
    reference.close()

    dynamic = module.VllmPairPool(**common, dynamic_batching=True)
    assert len(dynamic.dispatchers) == 3
    dynamic.close()
