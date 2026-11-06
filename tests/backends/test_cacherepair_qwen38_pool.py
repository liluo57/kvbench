"""Bounded supervision routing and unchanged official optimizer semantics."""
from concurrent.futures import Future
from types import SimpleNamespace

import torch

from methods.CacheRepair.qwen38 import pair_pool as module
from methods.CacheRepair.qwen38.train import run_upstream
from methods.CacheRepair.qwen38.model import RepairConfig
from methods.CacheRepair.qwen38.supervision import make_pair
from tests.backends.test_cacherepair_qwen38_native import native
import pytest


def test_three_replicas_preserve_request_order_and_bound_prefetch(monkeypatch):
    geometry = SimpleNamespace(to_dict=lambda: {"target": "same"})
    target = SimpleNamespace(get_input_embeddings=lambda: torch.nn.Embedding(2, 2))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    packet = lambda chunks: {"tokens": torch.tensor([sum(chunks, [])])}
    monkeypatch.setattr(module, "make_pair", lambda target, chunks, **kw: packet(chunks))
    monkeypatch.setattr(module, "_worker_pair", lambda chunks: (packet(chunks), .1))
    monkeypatch.setattr(module, "_worker_ready", lambda: {"geometry": geometry.to_dict()})

    class Executor:
        def __init__(self, **kw):
            pass
        def submit(self, fn, *args):
            future = Future()
            future.set_result(fn(*args))
            return future
        def shutdown(self, **kw):
            pass

    monkeypatch.setattr(module, "ProcessPoolExecutor", Executor)
    pool = module.PairPool(target, geometry, [(0, 1), (2, 3), (4, 5)], model_path="unused")
    rows = [{"chunks": [[i], [i + 1]]} for i in range(12)]
    pool.set_sequence(rows)
    actors = []
    for row in rows:
        assert len(pool.pending) <= 6
        assert torch.equal(pool.get(row["chunks"])["tokens"], packet(row["chunks"])["tokens"])
        actors.append(pool.last_event["replica"])
    assert actors == [0, 1, 2] * 4
    pool.set_sequence(rows)
    other = [[20], [21]]
    assert torch.equal(pool.get(other)["tokens"], packet(other)["tokens"])
    pool.close()


def test_prefetch_adapter_keeps_original_training_weights_and_logs(native, tmp_path):
    target, _ = native
    rows = [{"chunks": [[1, 2, 3][:1 + i % 3], [4, 5]]} for i in range(9)]
    options = dict(model_id="tiny-qwen", repair_config=RepairConfig(d_b=16, n_heads=2, n_blocks=1, d_seg=2),
                   stats_examples=2, device="cpu", validate_train=False)
    run_upstream(target, rows, rows[:2], tmp_path / "baseline", **options)

    class Pool:
        last_event = {"compute_seconds": .1}
        phases = []
        def set_sequence(self, rows):
            self.phases.append(list(rows))
        def get(self, chunks):
            from methods.CacheRepair.qwen38.target import TargetGeometry
            return make_pair(target, chunks, geometry=TargetGeometry.from_target(target, target_model_id="tiny-qwen"))

    pool = Pool()
    run_upstream(target, rows, rows[:2], tmp_path / "pooled", pair_pool=pool, **options)
    assert len(pool.phases) >= 3  # Statistics, shuffled epoch, validation.
    assert (tmp_path / "pooled/loss.jsonl").read_bytes() == (tmp_path / "baseline/loss.jsonl").read_bytes()
    from safetensors.torch import load_file
    actual = load_file(str(tmp_path / "pooled/checkpoint-3/model.safetensors"))
    expected = load_file(str(tmp_path / "baseline/checkpoint-3/model.safetensors"))
    assert all(torch.equal(actual[key], expected[key]) for key in actual)


def test_replica_local_fp64_statistics_resume_matches_official(native, tmp_path, monkeypatch):
    from methods.CacheRepair.qwen38.resume import Restart, TrainingInterrupted
    from methods.CacheRepair.qwen38.target import TargetGeometry
    from methods.CacheRepair.qwen38.train import _bind
    from methods.CacheRepair.backend.cacherepair import train as official
    target, _ = native
    target.requires_grad_(False).eval()
    geometry = TargetGeometry.from_target(target, target_model_id="tiny-qwen")
    rows = [{"chunks": [[i + 1, 2, 3], [4, 5]]} for i in range(7)]
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    pool = module.PairPool(target, geometry, [(0, 1)], model_path="unused", chunk_batch_size=2)
    restart = Restart(tmp_path, {"train_size": 7}, target, "tiny-qwen")
    restart.requested = "SIGTERM"
    with pytest.raises(TrainingInterrupted):
        pool.statistics(rows, restart, block_size=2)
    restart.requested = None
    sums, count = pool.statistics(rows, restart, block_size=2)
    actual = {name: (value / count).clamp_min(1e-12).sqrt().float() for name, value in sums.items()}
    expected = _bind(official.compute_stats, {**official.__dict__, "make_pair":
        lambda target, chunks: make_pair(target, chunks, geometry=geometry, chunk_batch_size=2)})(target, rows)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)
    assert count == 35
    pool.close()


def test_local_rms_replica_runs_independently_of_coordinator(native, tmp_path, monkeypatch):
    import threading
    from methods.CacheRepair.qwen38.experimental import pipeline as implementation
    from methods.CacheRepair.qwen38.resume import Restart
    from methods.CacheRepair.qwen38.target import TargetGeometry
    from methods.CacheRepair.qwen38.train import _bind
    from methods.CacheRepair.backend.cacherepair import train as official
    target, _ = native
    target.requires_grad_(False).eval()
    geometry = TargetGeometry.from_target(target, target_model_id="tiny-qwen")
    rows = [{"chunks": [[i + 1, 2], [3, 4]]} for i in range(7)]
    identities = []

    class Pipeline:
        def __init__(self, *args, depth):
            self.depth = depth
        def set_sequence(self, rows):
            pass
        def get(self, chunks):
            identities.append(threading.get_ident())
            return make_pair(target, chunks, geometry=geometry)
        def close(self):
            pass

    monkeypatch.setattr(implementation, "NativePipeline", Pipeline)
    monkeypatch.setattr(torch.cuda, "set_device", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    pool = module.PairPool(target, geometry, [(0, 1)], model_path="unused", pack_chunks=True, pipeline_depth=3)
    try:
        sums, count = pool.statistics(rows, Restart(tmp_path, {}, target, "tiny-qwen"), block_size=2)
    finally:
        pool.close()
    assert identities and threading.get_ident() not in identities
    expected = _bind(official.compute_stats, {**official.__dict__, "make_pair":
        lambda target, chunks: make_pair(target, chunks, geometry=geometry)})(target, rows)
    for key in expected:
        torch.testing.assert_close((sums[key] / count).sqrt().float(), expected[key], rtol=0, atol=0)


def test_packed_prefill_resets_attention_convolution_and_recurrence(native):
    """CPU reference varlen kernels check all three independent boundaries."""
    from methods.CacheRepair.qwen38.target import TargetGeometry
    target, _ = native
    target.requires_grad_(False).eval()
    geometry = TargetGeometry.from_target(target, target_model_id="tiny-qwen")
    chunks = [[1, 2], [3, 4, 5, 6], [7], [8, 9, 10]]
    expected = make_pair(target, chunks, geometry=geometry)
    saved = []
    for layer in target.model.layers:
        if not hasattr(layer, "linear_attn"):
            continue
        module = layer.linear_attn
        original = module.chunk_gated_delta_rule
        def conv(*, x, weight, bias, activation, seq_idx=None):
            spans = [0, x.shape[-1]] if seq_idx is None else [0, *(
                (seq_idx[0, 1:] != seq_idx[0, :-1]).nonzero().flatten() + 1).tolist(), x.shape[-1]]
            parts = []
            for begin, end in zip(spans[:-1], spans[1:]):
                value = torch.nn.functional.conv1d(x[:, :, begin:end], weight[:, None], bias,
                    padding=weight.shape[-1] - 1, groups=weight.shape[0])[:, :, :end - begin]
                parts.append(torch.nn.functional.silu(value))
            return torch.cat(parts, -1)
        def gdn(q, k, v, g, beta, cu_seqlens=None, cu_seqlens_cpu=None, _original=original, **kw):
            if cu_seqlens is None:
                return _original(q, k, v, g, beta, **kw)
            parts, states = [], []
            for begin, end in zip(cu_seqlens_cpu[:-1], cu_seqlens_cpu[1:]):
                out, state = _original(q[:, begin:end], k[:, begin:end], v[:, begin:end],
                                       g[:, begin:end], beta[:, begin:end], **kw)
                parts.append(out)
                states.append(state[0])
            return torch.cat(parts, 1), torch.stack(states)
        module.causal_conv1d_fn, module.chunk_gated_delta_rule = conv, gdn
        saved.append((module, conv, gdn))
    actual = make_pair(target, chunks, geometry=geometry, pack_chunks=True)
    for key in ("stale", "joint"):
        torch.testing.assert_close(actual[key], expected[key], rtol=1e-5, atol=1e-6)
    for module, conv, gdn in saved:
        assert module.causal_conv1d_fn is conv and module.chunk_gated_delta_rule is gdn
