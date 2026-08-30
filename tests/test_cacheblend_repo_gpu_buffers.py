from types import SimpleNamespace

import torch

from helpers.cacheblend_repo.CacheblendRepoHelper import CacheBlendWorker


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return list(range(len(text)))


def _worker():
    worker = CacheBlendWorker.__new__(CacheBlendWorker)
    worker.args = SimpleNamespace(max_model_len=16)
    worker._torch = torch
    worker.layers = [SimpleNamespace(), SimpleNamespace()]
    worker.engine = SimpleNamespace(
        model=SimpleNamespace(old_kvs=[[None, None], [None, None]])
    )
    worker.tokenizer = _Tokenizer()
    worker._assembledKv = []
    worker._assembledCapacity = 0
    kv = [
        [torch.ones(2, 3), torch.ones(2, 3)],
        [torch.ones(2, 3), torch.ones(2, 3)],
    ]
    worker._chunkIds = {"AB": [1, 2]}
    worker._chunkKv = {"AB": kv}
    return worker


def test_reserve_allocates_and_reuses_device_local_assembly_buffers():
    worker = _worker()

    report = worker.Reserve([[[True, "AB"], [False, "xyz"]]])

    assert report == {"ok": True, "capacity": 5, "requested": 5}
    assert len(worker._assembledKv) == 2
    assert worker._assembledKv[0][0].shape == (5, 3)
    assert worker._assembledKv[0][0].device == worker._chunkKv["AB"][0][0].device
    first_buffer = worker._assembledKv[0][0]

    worker.Reserve([[[True, "AB"]]])
    assert worker._assembledKv[0][0] is first_buffer


def test_reserve_rejects_prompt_beyond_model_limit():
    worker = _worker()
    worker.args.max_model_len = 4

    try:
        worker.Reserve([[[True, "AB"], [False, "xyz"]]])
    except ValueError as exc:
        assert "exceeds max_model_len" in str(exc)
    else:
        raise AssertionError("oversized reserve unexpectedly succeeded")
