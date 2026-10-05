"""Tests for scheduler."""

import multiprocessing as mp
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.engine.Scheduler import Scheduler


def _partial_event_worker(*args):
    event_queue = args[7]
    event_queue.put({"payload": b"x" * (16 * 1024 * 1024)})
    time.sleep(30)


@pytest.mark.skipif("fork" not in mp.get_all_start_methods(), reason="POSIX crash regression")
def test_scheduler_can_drain_queue_after_producer_dies_mid_frame(tmp_path, monkeypatch):
    import importlib
    module = importlib.import_module("core.engine.Scheduler")
    monkeypatch.setattr(module, "WorkerMain", _partial_event_worker)
    ctx = SimpleNamespace(
        methods=[SimpleNamespace(gpuNums=1, Label="fake")], freeGpus=[0],
        workerSerial=0, outputDir=Path(tmp_path), mpContext=mp.get_context("fork"),
        effectiveBatchSizes=[1], workers={},
    )
    reporter = SimpleNamespace(addLog=lambda *args: None, recordEvent=lambda event: None)
    engine = SimpleNamespace(metrics=lambda: [], recordAllSamples=False, initializeTimeout=10)
    scheduler = Scheduler(ctx, reporter, engine)
    scheduler.spawnWorker(0)
    worker = next(iter(ctx.workers.values()))
    try:
        assert worker.eventQueue._writer.closed
        assert worker.eventQueue._reader.poll(5)
        worker.process.kill()
        worker.process.join(5)
        assert not worker.process.is_alive()
        # Without closing the coordinator's writer this read waits forever.
        start = time.monotonic()
        scheduler.drainEvents(ctx)
        assert time.monotonic() - start < 1
    finally:
        if worker.process.is_alive():
            worker.process.kill()
        worker.process.join(5)
        worker.connection.close()
        worker.eventQueue.close()
        worker.eventQueue._reader.close()
