"""Compare Full/TRR/Hybrid on the same frozen ten-workload KVBench inputs.

Pass --config for a run-local YAML snapshot and --output for new artifacts.
The trusted local frozen input pickle is supplied explicitly with --frozen.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import Config

# Spawned Engine workers re-execute this module before unpickling Methods.
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--config', type=Path, default=ROOT/'config.yaml')
_known, _ = _parser.parse_known_args()
Config.DefaultConfigPath = _known.config.resolve()

from core.Task import Task
from core.engine import Engine
from methods import HybridCacheRepair, HypicMethod
from metrics import TTFTMetric, ThroughputMetric


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(',', ':'), sort_keys=True).encode()).hexdigest()


class FrozenTask(Task):
    def Cases(self):
        yield from copy.deepcopy(self.cases)

    def Evaluate(self, result, metadata):
        return self.original.Evaluate(result, metadata)


class _TrustedFrozenInputs(pickle.Unpickler):
    def find_class(self, module, name):
        if module in ('__main__', '__mp_main__') and name == 'FrozenTask':
            return FrozenTask
        return super().find_class(module, name)


class Record:
    _supportsCaseResultHooks = True

    def SampleResult(self, case, result, scores):
        import os
        directory = Path(os.environ['KVBENCH_HYBRID_PREDICTIONS']) / self.tag
        directory.mkdir(parents=True, exist_ok=True)
        row = dict(case_id=case.workflow.case_id, input_sha256=case.metadata['experiment_input_sha256'],
                   output=result.output, output_ids=result.output_ids, scores=scores,
                   performance=result.performance, result_metadata=result.metadata)
        with (directory / (case.metadata['experiment_task']+'.jsonl')).open('a') as stream:
            stream.write(json.dumps(row, ensure_ascii=False)+'\n')
        return row


class RecordedHybrid(Record, HybridCacheRepair):
    pass


class RecordedHypic(Record, HypicMethod):
    def Initialize(self, gpuIds):
        # Same native backend/options as Hybrid, without its scheduler adapter.
        import os
        from core.Method import Method
        from helpers.backends.ModelAdapter import _tokenizer
        from helpers.backends.Tokenization import TerminalStopIdsFromTokenizer
        from methods.hypic.Method import _BuildHypicEngineKwargs, _QWEN35_SYSTEM, _WARMUP_TAIL
        Method.Initialize(self, gpuIds)
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpuIds))
        from methods.hypic.backend.kvbench_hypic.srt.entrypoints.engine import Engine as NativeEngine
        self._tokenizer = _tokenizer(self.modelPath)
        self._terminalStopIds = TerminalStopIdsFromTokenizer(self._tokenizer, self.samplingConfig)
        self._systemIds = self._tokenizer.encode(_QWEN35_SYSTEM, add_special_tokens=False)
        self._tailIds = self._tokenizer.encode(_WARMUP_TAIL, add_special_tokens=False)
        options = _BuildHypicEngineKwargs(self.modelPath, gpuIds, dtype=self.dtype,
            maxModelLen=self.maxModelLen, memFractionStatic=self.memFractionStatic,
            picMode=self.picMode, fullPrefill=self.fullPrefill, seamTokens=self.seamTokens)
        options.update(disable_overlap_schedule=True, disable_cuda_graph=True,
            linear_attn_prefill_backend='triton', linear_attn_decode_backend='triton')
        self.engine = NativeEngine(**options)


def load_tasks(path, samples, allow_fewer=False, start_indices=None, task_filter=None):
    raw = path.read_bytes()
    tasks = _TrustedFrozenInputs(io.BytesIO(raw)).load()
    reference = json.loads(path.with_name('inputs.json').read_text())
    if len(tasks) != 10 or len(reference['tasks']) != 10:
        raise ValueError('Expected exactly ten frozen workloads')
    selected = []
    selected_tasks = []
    start_indices = start_indices or {}
    for task, entry in zip(tasks, reference['tasks']):
        if task.Label != entry['task'] or task.maxNewTokens != entry['max_new_tokens']:
            raise ValueError('Frozen workload/budget differs from its manifest')
        for case, expected_id, expected in zip(task.cases, entry['case_ids'], entry['input_sha256'], strict=True):
            actual = digest(dict(prepare=case.input.prepare_input, run=case.input.run_input))
            if case.workflow.case_id != expected_id or actual != expected:
                raise ValueError('Frozen token IDs or case ID changed')
        if task_filter is not None and task.Label not in task_filter:
            continue
        start = int(start_indices.get(task.Label, 0))
        if start < 0 or start >= len(task.cases):
            raise ValueError(f'Invalid start index for {task.Label}: {start}')
        stop = min(start + samples, len(task.cases))
        if stop - start < samples and not allow_fewer:
            raise ValueError(f'{task.Label} has too few frozen cases')
        task.cases = task.cases[start:stop]
        for case in task.cases:
            case.metadata['experiment_task'] = task.Label
            case.metadata['experiment_input_sha256'] = digest(dict(prepare=case.input.prepare_input, run=case.input.run_input))
        selected_tasks.append(task)
        selected.append(dict(task=task.Label, count=len(task.cases), max_new_tokens=task.maxNewTokens,
            case_ids=[c.workflow.case_id for c in task.cases],
            input_sha256=[c.metadata['experiment_input_sha256'] for c in task.cases],
            input_lengths=[len(c.input.run_input) for c in task.cases]))
    return selected_tasks, dict(frozen_pickle_sha256=hashlib.sha256(raw).hexdigest(), tasks=selected)


def main():
    import os
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--frozen', type=Path, required=True, help='Trusted local KVBench frozen_tasks.pkl')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=64)
    parser.add_argument('--allow-fewer-samples', action='store_true',
        help='Use all available frozen cases when a task has fewer than --samples')
    parser.add_argument('--task-filter', default=None,
        help='Optional comma-separated subset of frozen task names')
    parser.add_argument('--sample-starts', type=Path, default=None,
        help='Optional JSON object mapping task names to starting case offsets')
    parser.add_argument('--residual-alpha', type=float, default=None,
        help='Scale the denormalized KV residual; default: HybridCacheRepair.ResidualAlpha or 1')
    parser.add_argument('--arms', default='full,trr,stage_b',
        help='Comma-separated full,trr,stage_b,cacherepair; cacherepair uses the configured current checkpoint')
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    arms = args.arms.split(',')
    if len(set(arms)) != len(arms) or not set(arms) <= {'full', 'trr', 'stage_b', 'cacherepair'} or not arms:
        parser.error('Specify distinct supported arms')
    torch.set_num_threads(1)
    task_filter = set(args.task_filter.split(',')) if args.task_filter else None
    start_indices = json.loads(args.sample_starts.read_text()) if args.sample_starts else None
    tasks, selection = load_tasks(args.frozen, args.samples, args.allow_fewer_samples,
                                  start_indices, task_filter)
    if not tasks:
        parser.error('Task filter selected no tasks')
    cfg = Config.LoadConfig()
    experiment = cfg.get('Experiment', {})
    common = dict(gpuNums=experiment.get('gpuNums', 2), memFractionStatic=experiment.get('memFractionStatic', .8), seamTokens=8)
    if args.preflight:
        tasks = [task for task in tasks if task.name in ('musique', 'niah_shuffle')]
        for task in tasks:
            task.cases = task.cases[:1]
        selected_names = {task.Label for task in tasks}
        selection['tasks'] = [{**entry, 'count':1, 'case_ids':entry['case_ids'][:1],
            'input_sha256':entry['input_sha256'][:1], 'input_lengths':entry['input_lengths'][:1]}
            for entry in selection['tasks'] if entry['task'] in selected_names]
        output = args.output/'preflight'
        available = cfg['Engine'].get('AvailableGpuIds')
        if isinstance(available, list):
            cfg['Engine']['AvailableGpuIds'] = available[:common['gpuNums']]
    else:
        output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if (output/'engine_report.json').exists():
        raise FileExistsError('Refusing to overwrite a completed run')
    cfg['Engine']['OutputRoot'] = str((output/'engine').resolve())
    os.environ['KVBENCH_HYBRID_PREDICTIONS'] = str((output/'predictions').resolve())
    (output/'selection.json').write_text(json.dumps(selection, indent=2)+'\n')
    hybrid_arm = 'cacherepair' if 'cacherepair' in arms else 'stage_b'
    hybrid = RecordedHybrid(tag=hybrid_arm, residualAlpha=args.residual_alpha, **common)
    methods = [hybrid]
    if not args.preflight:
        constructors = {
            'full': lambda: RecordedHypic(tag='full', fullPrefill=True, **common),
            'trr': lambda: RecordedHypic(tag='trr', picMode='transition_rope_recompute', **common),
            'stage_b': lambda: RecordedHybrid(tag='stage_b', residualAlpha=args.residual_alpha, **common),
            'cacherepair': lambda: RecordedHybrid(tag='cacherepair', residualAlpha=args.residual_alpha, **common),
        }
        methods = [constructors[arm]() for arm in arms]
    selection['arms'] = [hybrid_arm] if args.preflight else arms
    (output/'selection.json').write_text(json.dumps(selection, indent=2)+'\n')
    tasks.sort(key=lambda task: ('govreport', 'multinews', 'triviaqa', 'niah_shuffle', 'cwe_shuffle',
        'vt_shuffle', 'musique', 'samsum', '2wikimultihopqa', 'hotpotqa').index(task.name))
    print(json.dumps({'tasks':[t.Label for t in tasks], 'samples':args.samples,
                      'methods':[m.Label for m in methods], 'model':Config.ModelPath()}), flush=True)
    report = Engine().Evaluate(tasks, methods, metrics=[TTFTMetric(), ThroughputMetric()])
    (output/'engine_report.json').write_text(json.dumps(report, indent=2)+'\n')
    if report['status'] != 'completed' or report['failures']:
        raise RuntimeError(f'Incomplete evaluation: {report["output_dir"]}')
    print(json.dumps({'status':report['status'], 'output':report['output_dir']}), flush=True)


if __name__ == '__main__':
    main()
