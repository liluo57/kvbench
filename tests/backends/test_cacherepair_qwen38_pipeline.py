"""Native pipeline ordering, request isolation, and wrapper restoration."""
import torch

from methods.CacheRepair.qwen38.experimental.pipeline import NativePipeline
from methods.CacheRepair.qwen38.supervision import make_pair
from methods.CacheRepair.qwen38.target import TargetGeometry
from tests.backends.test_cacherepair_qwen38_native import native


def reference_varlen(target):
    for layer in target.model.layers:
        if not hasattr(layer, "linear_attn"):
            continue
        module = layer.linear_attn
        original = module.chunk_gated_delta_rule

        def conv(*, x, weight, bias, activation, seq_idx=None):
            spans = [0, x.shape[-1]] if seq_idx is None else [0, *(
                (seq_idx[0, 1:] != seq_idx[0, :-1]).nonzero().flatten() + 1).tolist(), x.shape[-1]]
            pieces = [torch.nn.functional.silu(torch.nn.functional.conv1d(x[:, :, a:b], weight[:, None], bias,
                padding=weight.shape[-1] - 1, groups=weight.shape[0])[:, :, :b-a])
                for a, b in zip(spans[:-1], spans[1:])]
            return torch.cat(pieces, -1)

        def gdn(q, k, v, g, beta, cu_seqlens=None, cu_seqlens_cpu=None, _original=original, **kw):
            if cu_seqlens is None:
                return _original(q, k, v, g, beta, **kw)
            pieces, states = [], []
            for a, b in zip(cu_seqlens_cpu[:-1], cu_seqlens_cpu[1:]):
                out, state = _original(q[:, a:b], k[:, a:b], v[:, a:b], g[:, a:b], beta[:, a:b], **kw)
                pieces.append(out)
                states.append(state[0])
            return torch.cat(pieces, 1), torch.stack(states)

        module.causal_conv1d_fn, module.chunk_gated_delta_rule = conv, gdn


def test_native_concurrent_pairs_keep_boundaries_order_and_restore(native):
    torch.set_num_threads(1)
    target, _ = native
    target.requires_grad_(False).eval()
    reference_varlen(target)
    geometry = TargetGeometry.from_target(target, target_model_id="tiny-qwen")
    rows = [{"chunks": [[1, 2][:1+i%2], [3, 4, 5], [6, 7, 8, 9][:1+i%4]]} for i in range(9)]
    expected = [make_pair(target, row["chunks"], geometry=geometry, pack_chunks=True) for row in rows]
    original_boundary = target.model.layers[2].forward
    original = [(layer.linear_attn, layer.linear_attn.causal_conv1d_fn, layer.linear_attn.chunk_gated_delta_rule)
                for layer in target.model.layers if hasattr(layer, "linear_attn")]
    pipeline = NativePipeline(target, geometry, (), split=2, depth=2, record=True)
    try:
        pipeline.set_sequence(rows)
        for row, reference in zip(rows, expected):
            actual = pipeline.get(row["chunks"])
            assert len(pipeline.pending) <= 2
            for key in reference:
                torch.testing.assert_close(actual[key], reference[key], rtol=0, atol=0)
        assert len({event["thread"] for event in pipeline.events}) == 2
        assert sum(event["stale"] for event in pipeline.events) == len(rows)
    finally:
        pipeline.close()
    assert target.model.layers[2].forward == original_boundary
    for module, conv, gdn in original:
        assert module.causal_conv1d_fn is conv and module.chunk_gated_delta_rule is gdn


def test_lookahead_pipeline_preserves_source_order(native):
    torch.set_num_threads(1)
    target, _ = native
    target.requires_grad_(False).eval()
    reference_varlen(target)
    geometry = TargetGeometry.from_target(target, target_model_id="tiny-qwen")
    rows = [{"chunks": [[i+1, 2], [3, 4, 5], [6]]} for i in range(7)]
    expected = [make_pair(target, row["chunks"], geometry=geometry, pack_chunks=True) for row in rows]
    pipeline = NativePipeline(target, geometry, (), split=2, depth=3)
    try:
        for i, row in enumerate(rows):
            actual = pipeline.get_with_lookahead(row["chunks"], [v["chunks"] for v in rows[i+1:i+3]])
            for key in actual:
                torch.testing.assert_close(actual[key], expected[i][key], rtol=0, atol=0)
    finally:
        pipeline.close()
