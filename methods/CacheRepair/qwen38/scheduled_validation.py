"""Durable, in-process holdout schedule; no target reload or optimizer writes."""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import random
import time

import torch

from .resume import atomic_json, fingerprint


@contextmanager
def evaluation_state(model, gpu_ids=()):
    mode, rng = model.training, random.getstate()
    try:
        with torch.random.fork_rng(devices=list(gpu_ids)):
            with torch.no_grad():
                model.eval()
                yield
    finally:
        random.setstate(rng)
        model.train(mode)


class ScheduledValidation:
    def __init__(self, output, rows, plan, *, clock=None):
        self.output, self.rows, self.plan, self.clock = Path(output), rows, plan, clock or time.time
        if not plan.get('enabled') or len(rows) != plan.get('hourly_validation_examples', 500):
            raise ValueError('Scheduled validation requires the complete fixed holdout')
        self.interval = plan.get('hourly_interval_seconds', 3600)
        if self.interval <= 0:
            raise ValueError('Validation interval must be positive')
        self.path = self.output / 'validation-schedule.json'
        self.identity = fingerprint({'rows': rows, 'plan': plan})
        self.state = (json.loads(self.path.read_text()) if self.path.exists() else
                      dict(identity=self.identity, next_due=None, baseline_done=False))
        if self.state['identity'] != self.identity:
            raise ValueError('Scheduled validation holdout or plan changed on resume')

    def due(self):
        return self.state['next_due'] is not None and self.clock() >= self.state['next_due']

    def record(self, step, kind, result, seconds):
        if result['examples'] != len(self.rows):
            raise ValueError('Incomplete scheduled MSE validation')
        folder = self.output / 'scheduled-validation'
        folder.mkdir(exist_ok=True)
        path = folder / f'{kind}-step-{step:08d}.json'
        atomic_json(path, dict(status='completed', kind=kind, step=step,
            validation=result, validation_rows_sha256=fingerprint(self.rows),
            finished_unix=self.clock(), seconds=seconds))
        self.state.update(next_due=self.clock()+self.interval, baseline_done=True,
                          latest=str(path.resolve()), latest_step=step)
        atomic_json(self.path, self.state)

    def run(self, model, target, pair_function, diagnostics, *, step, kind, gpu_ids=()):
        started = self.clock()
        with evaluation_state(model, gpu_ids):
            result = diagnostics(model, target, self.rows, pair_function)
        self.record(step, kind, result, self.clock()-started)
        return result


def request_epoch_quality(output, checkpoint, training, plan):
    """Publish a retryable request only after deployment checkpoint validation."""
    if not plan or not plan.get('enabled') or not plan.get('epoch_quality', {}).get('enabled'):
        return False
    output, checkpoint = Path(output), Path(checkpoint)
    step = training['checkpoint_step']
    done = output.parent / 'quality-evaluations' / f'step-{step:08d}' / 'completed.json'
    if done.exists():
        result = json.loads(done.read_text())
        if result.get('status') != 'completed' or result.get('checkpoint') != str(checkpoint.resolve()):
            raise ValueError('Quality completion checkpoint differs')
        return False
    atomic_json(output / 'epoch-quality-request.json', dict(checkpoint=str(checkpoint.resolve()),
        step=step, epoch=training['checkpoint_epoch'], evaluation_plan_sha256=fingerprint(plan)))
    return True
