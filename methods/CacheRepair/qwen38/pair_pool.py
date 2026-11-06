"""Bounded prefetch of frozen-target supervision across two-GPU replicas.

One replica lives in the training process; remote replicas run in spawn workers.
Only FP32 CPU supervision crosses process boundaries. The repairer's original
four-example optimizer loop remains in one process and keeps its exact order.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import json
import multiprocessing
from pathlib import Path
import time

import torch

from .resume import fingerprint, statistics_accumulator
from .supervision import make_pair
from .target import TargetGeometry

_WORKER = None
_PIPELINE = None


def _worker_load(model_path, gpu_ids, attention_backend, chunk_batch_size, pack_chunks, pipeline_depth=0):
    global _WORKER, _PIPELINE
    from .train import load_native_target
    torch.set_num_threads(1)
    torch.cuda.set_device(gpu_ids[0])
    target = load_native_target(model_path, gpu_ids, attention_backend=attention_backend, require_fast=True)
    geometry = TargetGeometry.from_target(target, target_model_id="Qwen/Qwen3.8-27B")
    _WORKER = (target, geometry, tuple(gpu_ids), chunk_batch_size, pack_chunks)
    if pipeline_depth:
        from .experimental.pipeline import NativePipeline
        _PIPELINE = NativePipeline(target, geometry, gpu_ids, depth=pipeline_depth)


def _worker_ready():
    return {"gpus": _WORKER[2], "geometry": _WORKER[1].to_dict()}


def _worker_pair(chunks, lookahead=()):
    target, geometry, gpu_ids, chunk_batch_size, pack_chunks = _WORKER
    if _PIPELINE is None:
        for gpu in gpu_ids:
            torch.cuda.synchronize(gpu)
    started = time.perf_counter()
    pair = _PIPELINE.get_with_lookahead(chunks, lookahead) if _PIPELINE else make_pair(
        target, chunks, geometry=geometry, chunk_batch_size=chunk_batch_size, pack_chunks=pack_chunks)
    result = {name: value.cpu() for name, value in pair.items()}
    if _PIPELINE is None:
        for gpu in gpu_ids:
            torch.cuda.synchronize(gpu)
    return result, time.perf_counter() - started


def _statistics(target, geometry, gpu_ids, chunk_batch_size, pack_chunks, rows, pipeline=None):
    from ..backend.cacherepair import train as official
    if pipeline is not None:
        pipeline.set_sequence(rows)
    pairs = lambda target, chunks: pipeline.get(chunks) if pipeline is not None else make_pair(
        target, chunks, geometry=geometry, chunk_batch_size=chunk_batch_size, pack_chunks=pack_chunks)
    accumulate = statistics_accumulator(official.compute_stats, {
        **official.__dict__, "make_pair": pairs, "print": lambda *args, **kwargs: None})
    started = time.perf_counter()
    sums, count = accumulate(target, rows)
    for gpu in gpu_ids:
        torch.cuda.synchronize(gpu)
    return sums, count, time.perf_counter() - started


def _worker_statistics(rows):
    return _statistics(*_WORKER, rows, pipeline=_PIPELINE)


class PairPool:
    def __init__(self, target, geometry, groups, *, model_path, chunk_batch_size=8,
                 attention_backend="sdpa", log_path=None, prefetch_per_worker=2, pack_chunks=False, pipeline_depth=0):
        groups = [tuple(group) for group in groups]
        flattened = [gpu for group in groups for gpu in group]
        if not groups or any(len(group) != 2 for group in groups) or len(set(flattened)) != len(flattened):
            raise ValueError("Supervision groups must be disjoint GPU pairs")
        self.target, self.geometry, self.groups = target, geometry, groups
        self.chunk_batch_size, self.log_path = chunk_batch_size, Path(log_path) if log_path else None
        self.pack_chunks = pack_chunks
        self.pipeline_depth, self.pipeline = pipeline_depth, None
        self.statistics_executor = None
        if pipeline_depth:
            if pipeline_depth < 2 or not pack_chunks:
                raise ValueError("Native pipeline requires packed chunks and depth >=2")
            from .experimental.pipeline import NativePipeline
            self.pipeline = NativePipeline(target, geometry, groups[0], depth=pipeline_depth)
            self.statistics_executor = ThreadPoolExecutor(max_workers=1,
                initializer=torch.cuda.set_device, initargs=(groups[0][0],), thread_name_prefix="qwen38-rms")
        self.prefetch = max(1, prefetch_per_worker) * len(groups)
        self.executors = [None]
        self.pending, self.sequence, self.cursor, self.submitted = deque(), [], 0, 0
        self.last_event = None
        try:
            for group in groups[1:]:
                self.executors.append(ProcessPoolExecutor(max_workers=1,
                    mp_context=multiprocessing.get_context("spawn"), initializer=_worker_load,
                    initargs=(model_path, group, attention_backend, chunk_batch_size, pack_chunks, pipeline_depth)))
            readiness = [executor.submit(_worker_ready) for executor in self.executors[1:]]
            for future in readiness:
                if future.result()["geometry"] != geometry.to_dict():
                    raise ValueError("Supervision replica target geometry differs")
        except BaseException:
            self.close()
            raise

    def set_sequence(self, rows):
        # Switching phases cannot accidentally consume a prefetched training
        # pair as validation supervision. Running work is drained explicitly.
        for _, _, future, _ in self.pending:
            if future is not None:
                if not future.cancel():
                    future.result()
        self.pending.clear()
        if self.pipeline is not None:
            self.pipeline.set_sequence([])
        self.sequence = rows
        self.cursor, self.submitted = 0, 0
        self._fill()

    def set_training(self, enabled):
        if self.pipeline is not None:
            self.pipeline.set_sequence([])
            # Three longest inputs plus backward fit in the measured L40
            # preflight. Keep the root as deep as the remote pipelines so it
            # does not starve the single optimizer's ordered consumption.
            self.pipeline.depth = min(3, self.pipeline_depth) if enabled else self.pipeline_depth

    def _fill(self):
        while len(self.pending) < self.prefetch and self.cursor < len(self.sequence):
            row = self.sequence[self.cursor]
            chunks = row["chunks"]
            group = self.submitted % len(self.groups)
            executor = self.executors[group]
            lookahead = [self.sequence[index]["chunks"] for index in range(
                self.cursor + len(self.groups), min(len(self.sequence),
                self.cursor + len(self.groups) * max(1, self.pipeline_depth)), len(self.groups))]
            future = (executor.submit(_worker_pair, chunks, lookahead) if self.pipeline_depth else
                      executor.submit(_worker_pair, chunks)) if executor is not None else None
            self.pending.append((fingerprint(chunks), group, future, lookahead))
            self.cursor += 1
            self.submitted += 1

    def get(self, chunks):
        key = fingerprint(chunks)
        if not self.pending or self.pending[0][0] != key:
            self.set_sequence([{"chunks": chunks}])
        _, group, future, lookahead = self.pending.popleft()
        started = time.perf_counter()
        if future is None:
            pair = self.pipeline.get_with_lookahead(chunks, lookahead) if self.pipeline is not None else make_pair(
                self.target, chunks, geometry=self.geometry, chunk_batch_size=self.chunk_batch_size, pack_chunks=self.pack_chunks)
            if self.pipeline is None:
                for gpu in self.groups[0]:
                    torch.cuda.synchronize(gpu)
            compute_seconds = time.perf_counter() - started
        else:
            pair, compute_seconds = future.result()
            device = self.target.get_input_embeddings().weight.device
            pair = {name: value.to(device) for name, value in pair.items()}
        self.last_event = {"pair_key": key, "replica": group, "gpus": self.groups[group],
                           "compute_seconds": compute_seconds, "fetch_seconds": time.perf_counter() - started,
                           "tokens": pair["tokens"].numel()}
        if self.log_path is not None:
            with self.log_path.open("a") as stream:
                stream.write(json.dumps(self.last_event) + "\n")
        self._fill()
        return pair

    def statistics(self, rows, restart, block_size=None):
        """Keep gigabyte KV pairs on each replica; transfer only FP64 sums.

        Source rows are partitioned into contiguous blocks. Blocks are merged
        in source order, with the original token weights and FP64 accumulators.
        A restart commits the original next-row cursor and accumulation format.
        """
        self.set_sequence([])
        if block_size is None:
            block_size = 100 if self.pipeline_depth else 20
        sums, count, start = restart.restore_statistics(rows)
        blocks = deque()
        cursor, submitted = start, 0

        def fill():
            nonlocal cursor, submitted
            while len(blocks) < len(self.groups) * 2 and cursor < len(rows):
                end = min(len(rows), cursor + block_size)
                group = submitted % len(self.groups)
                executor = self.executors[group]
                if executor is not None:
                    future = executor.submit(_worker_statistics, rows[cursor:end])
                elif self.statistics_executor is not None:
                    future = self.statistics_executor.submit(_statistics, self.target, self.geometry, self.groups[0],
                        self.chunk_batch_size, self.pack_chunks, rows[cursor:end], pipeline=self.pipeline)
                else:
                    future = None
                blocks.append((cursor, end, group, future))
                cursor, submitted = end, submitted + 1

        fill()
        while blocks:
            begin, end, group, future = blocks.popleft()
            result = future.result() if future is not None else _statistics(
                self.target, self.geometry, self.groups[0], self.chunk_batch_size, self.pack_chunks, rows[begin:end],
                pipeline=self.pipeline)
            partial, tokens, elapsed = result
            for name, value in partial.items():
                sums[name] = sums.get(name, 0) + value
            count += tokens
            if self.log_path is not None:
                with self.log_path.open("a") as stream:
                    stream.write(json.dumps({"phase": "statistics", "replica": group, "gpus": self.groups[group],
                        "begin": begin, "end": end, "tokens": tokens, "compute_seconds": elapsed}) + "\n")
            restart.after_statistics(end, sums, count)
            if (end - start) % 100 == 0 or end == len(rows):
                print(json.dumps({"statistics_examples": end, "total": len(rows),
                                  "replicas": len(self.groups)}), flush=True)
            fill()
        return sums, count

    def close(self):
        self.pending.clear()
        if self.statistics_executor is not None:
            self.statistics_executor.shutdown(wait=True, cancel_futures=True)
        for executor in self.executors:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
        if self.pipeline is not None:
            self.pipeline.close()
