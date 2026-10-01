from types import SimpleNamespace

import torch

from methods.Cacheblend.backend.cacheblend_ops import (
    RetainedOutputCollector,
    capture_kv,
    indexed_causal_mask,
)


def test_native_capture_retains_decode_tokens_without_collecting_prefill():
    attention = SimpleNamespace(hack_kv=None)
    collector = RetainedOutputCollector([SimpleNamespace(self_attn=attention)])
    metadata = {"collect": False, "retain_collector": collector}
    key = torch.tensor([[1.0, 2.0]])
    value = torch.tensor([[3.0, 4.0]])

    capture_kv(attention, key, value, 0, metadata)
    assert attention.hack_kv is None
    assert collector.finish() is None

    capture_kv(attention, key, value, -1, metadata)
    key.add_(10)
    value.add_(10)
    capture_kv(attention, key, value, -1, metadata)

    retained = collector.finish()
    assert metadata["collect"] is False
    assert torch.equal(retained[0][0], torch.tensor([[1.0, 2.0], [11.0, 12.0]]))
    assert torch.equal(retained[0][1], torch.tensor([[3.0, 4.0], [13.0, 14.0]]))


def test_native_collect_owns_kv_and_does_not_capture_ordinary_decode():
    attention = SimpleNamespace(hack_kv=None)
    metadata = {"collect": True}
    key = torch.tensor([[1.0]])
    value = torch.tensor([[2.0]])
    capture_kv(attention, key, value, 0, metadata)
    key.zero_()
    value.zero_()
    assert attention.hack_kv[0].item() == 1
    assert attention.hack_kv[1].item() == 2

    metadata["collect"] = False
    capture_kv(attention, key, value, -1, metadata)
    assert attention.hack_kv[0].item() == 1


def test_sparse_mask_uses_absolute_positions_for_interleaved_repair():
    metadata = {
        "imp_indices": torch.tensor([1, 5, 12]),
        "org_seq_len": 13,
        "kv_cache_dtype": torch.float32,
        "_num_kv_heads": 2,
        "_num_queries_per_kv": 4,
    }
    mask = indexed_causal_mask(metadata, lambda: None)
    assert mask.shape == (1, 2, 4, 3, 13)
    assert mask.stride(-2) == 16
    for row, position in enumerate([1, 5, 12]):
        assert torch.all(mask[..., row, :position + 1] == 0)
        assert torch.all(mask[..., row, position + 1:] == torch.finfo(torch.float32).min)
