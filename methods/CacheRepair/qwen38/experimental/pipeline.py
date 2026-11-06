"""Two independent native forwards overlap across the two resident halves.

The original HF target and kernels are used without arithmetic changes. Each
request has its own native cache and CUDA streams. GDN boundaries are thread
local; stage locks protect use of each half while allowing different halves
to execute concurrently. Outputs are returned in source order.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, ExitStack
from functools import partial
import threading
import time
import types

import torch

from ..cache import canonicalize_target_kv, pack_target_kv
from ..packed import packed_stale
from ..resume import fingerprint
from ..target import text_backbone


class _TargetProxy:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def __getattr__(self, name):
        return getattr(self.pipeline.target, name)

    def __call__(self, *args, **kwargs):
        return self.pipeline.forward(*args, **kwargs)


class NativePipeline:
    def __init__(self, target, geometry, gpu_ids, *, split=None, depth=2, record=False):
        self.target, self.geometry, self.gpu_ids = target, geometry, tuple(gpu_ids)
        self.backbone = text_backbone(target)
        self.split = split if split is not None else len(self.backbone.layers) // 2
        self.depth, self.record = depth, record
        self.local = threading.local()
        self.locks = [threading.Lock(), threading.Lock()]
        self.events, self.saved = [], []
        self.executor = ThreadPoolExecutor(max_workers=depth, thread_name_prefix="qwen38-pipeline")
        self.pending, self.rows, self.cursor = deque(), [], 0
        self.proxy = _TargetProxy(self)
        namespace = {**packed_stale.__wrapped__.__globals__, "chunk_boundaries": self.boundaries}
        raw = packed_stale.__wrapped__
        self.stale = torch.no_grad()(types.FunctionType(raw.__code__, namespace, raw.__name__, raw.__defaults__))
        boundary = self.backbone.layers[self.split]
        self.saved_boundary = boundary.forward
        boundary.forward = partial(self._second_stage, self.saved_boundary)
        for layer in self.backbone.layers:
            if not hasattr(layer, "linear_attn"):
                continue
            module = layer.linear_attn
            self.saved.append((module, module.causal_conv1d_fn, module.chunk_gated_delta_rule))
            module.causal_conv1d_fn = partial(self._conv, module.causal_conv1d_fn)
            module.chunk_gated_delta_rule = partial(self._gdn, module.chunk_gated_delta_rule)

    @contextmanager
    def streams(self):
        with ExitStack() as stack:
            for gpu in self.gpu_ids:
                stack.enter_context(torch.cuda.device(gpu))
                if not hasattr(self.local, "streams"):
                    self.local.streams = {}
                if gpu not in self.local.streams:
                    self.local.streams[gpu] = torch.cuda.Stream(device=gpu)
                stream = self.local.streams[gpu]
                stack.enter_context(torch.cuda.stream(stream))
            yield

    @contextmanager
    def boundaries(self, target, ids, lengths):
        cumulative_cpu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
        tables = {}
        for layer in self.backbone.layers:
            if not hasattr(layer, "linear_attn"):
                continue
            device = layer.linear_attn.in_proj_qkv.weight.device
            if device not in tables:
                tables[device] = (ids.to(device=device, dtype=torch.int32), cumulative_cpu.to(device))
        self.local.boundaries = tables, cumulative_cpu
        try:
            yield
        finally:
            self.local.boundaries = None

    def _conv(self, original, **kwargs):
        info = getattr(self.local, "boundaries", None)
        if info is not None:
            kwargs["seq_idx"] = info[0][kwargs["x"].device][0]
        return original(**kwargs)

    def _gdn(self, original, *args, **kwargs):
        info = getattr(self.local, "boundaries", None)
        if info is not None:
            kwargs.update(cu_seqlens=info[0][args[0].device][1], cu_seqlens_cpu=info[1])
        return original(*args, **kwargs)

    def _second_stage(self, original, *args, **kwargs):
        if not getattr(self.local, "in_forward", False):
            return original(*args, **kwargs)
        self.locks[1].acquire()
        self.local.held[1] = True
        # The producer stream retains ownership of its queued kernels/tensors.
        # Native cross-device copies establish dependencies for this request.
        self.local.held[0] = False
        self.locks[0].release()
        return original(*args, **kwargs)

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        self.locks[0].acquire()
        self.local.held, self.local.in_forward = [True, False], True
        started = time.perf_counter()
        try:
            return self.target(*args, **kwargs)
        finally:
            if self.record:
                self.events.append({"thread": threading.get_ident(), "begin": started,
                                    "end": time.perf_counter(), "stale": getattr(self.local, "boundaries", None) is not None})
            for i in (1, 0):
                if self.local.held[i]:
                    self.local.held[i] = False
                    self.locks[i].release()
            self.local.in_forward = False

    @torch.no_grad()
    def make_pair(self, chunks):
        with self.streams():
            stale, tokens, chunk_ids = self.stale(self.proxy, chunks, self.geometry)
            positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
            output = self.forward(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                                  position_ids=positions, use_cache=True, logits_to_keep=1)
            native = pack_target_kv(output.past_key_values, self.geometry, device=tokens.device)
            del output
            joint = canonicalize_target_kv(native, positions, self.backbone.rotary_emb)
            # Publish only completed tensors: consumers use their own stream.
            for stream in getattr(self.local, "streams", {}).values():
                stream.synchronize()
            return dict(stale=stale, joint=joint, tokens=tokens, chunk_ids=chunk_ids)

    def set_sequence(self, rows):
        for _, future in self.pending:
            if not future.cancel():
                future.result()
        self.pending.clear()
        self.rows, self.cursor = rows, 0
        self._fill()

    def _fill(self):
        while len(self.pending) < self.depth and self.cursor < len(self.rows):
            chunks = self.rows[self.cursor]["chunks"]
            self.pending.append((fingerprint(chunks), self.executor.submit(self.make_pair, chunks)))
            self.cursor += 1

    def get(self, chunks):
        key = fingerprint(chunks)
        if not self.pending or self.pending[0][0] != key:
            self.set_sequence([{"chunks": chunks}])
        _, future = self.pending.popleft()
        result = future.result()
        self._fill()
        return self.publish(result)

    @staticmethod
    def publish(result):
        # Autograd can retain frozen teacher inputs beyond this callback.
        # Protect storage allocated on a producer stream until its consumer
        # stream has also finished with it.
        for value in result.values():
            if value.is_cuda:
                value.record_stream(torch.cuda.current_stream(value.device))
        return result

    def get_with_lookahead(self, chunks, lookahead=()):
        """Keep the next requests running between serialized IPC callbacks."""
        candidates = [chunks, *lookahead]
        if not self.pending or self.pending[0][0] != fingerprint(chunks):
            self.set_sequence([])
        expected = [fingerprint(item) for item in candidates[:self.depth]]
        if [key for key, _ in self.pending] != expected[:len(self.pending)]:
            self.set_sequence([])
        # Preserve request positions even when adjacent examples have exactly
        # the same chunks; content equality does not remove a microbatch.
        for item in candidates[len(self.pending):self.depth]:
            self.pending.append((fingerprint(item), self.executor.submit(self.make_pair, item)))
        _, future = self.pending.popleft()
        return self.publish(future.result())

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.pending.clear()
        self.backbone.layers[self.split].forward = self.saved_boundary
        for module, conv, gdn in self.saved:
            module.causal_conv1d_fn, module.chunk_gated_delta_rule = conv, gdn
