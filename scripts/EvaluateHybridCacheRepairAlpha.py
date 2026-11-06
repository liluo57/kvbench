"""Resumable, audited 10-task residual-strength sweep on epoch 3 and 6.

Each experiment generates all 640 cases, including alpha=0 and alpha=1. Three
independent TP=2 services use disjoint pairs of GPUs. Full/TRR are reused only
after their input selection and prediction hashes are recorded and checked.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.SummarizeHybridCacheRepair import identity, summarize, write_summary

PYTHON = '/data/lyh/.local/cpython-3.13.9/bin/python'
TRAINING = ROOT / 'outputs/cacherepair-qwen38/full-training-repaired-v5/training'
BASE = ROOT / 'outputs/hybrid-cacherepair-fulltrain-step75000-10task-scheduled/paired-results'
TEMPLATE = BASE.parent / 'eval.yaml'
FROZEN = ROOT / 'outputs/hypic-ablation-64-20261014/frozen_tasks.pkl'
ALPHAS = (0., .25, .5, .75, 1., 1.25)
PAIRS = ((0, 1), (2, 3), (4, 5))
SOURCES = ('methods/HybridCacheRepair/Method.py', 'methods/HybridCacheRepair/Runtime.py',
    'methods/HybridCacheRepair/__init__.py', 'methods/CacheRepair/qwen38/model.py',
    'methods/CacheRepair/qwen38/cache.py', 'scripts/EvaluateHybridCacheRepair.py',
    'scripts/SummarizeHybridCacheRepair.py', 'scripts/EvaluateHybridCacheRepairAlpha.py')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path, value):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def experiment_key(epoch, alpha):
    return f'epoch-{epoch}-alpha-{alpha:g}'


def baseline_selection():
    selection = read(BASE / 'selection.json')
    assert read(BASE / 'completion-audit.json')['status'] == 'passed'
    assert read(BASE / 'comparison.json')['complete']
    assert sha(FROZEN) == selection['frozen_pickle_sha256']
    assert len(selection['tasks']) == 10
    assert all(t['count'] == 64 for t in selection['tasks'])
    return selection


def initialize(job):
    selection = baseline_selection()
    proof = {'source_sha256': {s: sha(ROOT / s) for s in SOURCES},
        'frozen_inputs': str(FROZEN), 'frozen_sha256': sha(FROZEN),
        'baseline': str(BASE), 'baseline_selection_sha256': sha(BASE / 'selection.json'),
        'baseline_prediction_sha256': {str(p.relative_to(BASE)): sha(p)
            for arm in ('full', 'trr') for p in (BASE / 'predictions' / arm).glob('*.jsonl')},
        'checkpoints': {str(e): {'path': str(TRAINING / f'checkpoint-{e * 12500}'),
            'sha256': {n: sha(TRAINING / f'checkpoint-{e * 12500}' / n)
            for n in ('model.safetensors', 'config.json', 'target-provenance.json')}} for e in (3, 6)},
        'alphas': list(ALPHAS), 'epochs': [3, 6], 'samples_per_task': 64,
        'gpu_pairs': PAIRS, 'all_twelve_combinations_regenerated': True,
        'formula': 'canonical_stale_fp32 + alpha * (delta_norm_fp32 * sigma_delta_fp32)',
        'gdn': 'hypic_trr', 'alpha_zero': 'same Hybrid path with zero KV residual',
        'generation_budgets': {t['task']: t['max_new_tokens'] for t in selection['tasks']}}
    path = job / 'provenance.json'
    if path.exists():
        # Reject changed source/config/checkpoints when resuming a sweep.
        assert read(path) == json.loads(json.dumps(proof)), 'Sweep provenance changed'
    else:
        atomic(path, proof)
        (job / 'selection.json').write_bytes((BASE / 'selection.json').read_bytes())
    return proof


def verify_provenance(proof):
    for name, expected in proof['source_sha256'].items():
        assert sha(ROOT / name) == expected, name
    for ck in proof['checkpoints'].values():
        for name, expected in ck['sha256'].items():
            assert sha(Path(ck['path']) / name) == expected, (ck['path'], name)
    for name, expected in proof['baseline_prediction_sha256'].items():
        assert sha(BASE / name) == expected, name
    assert sha(BASE / 'selection.json') == proof['baseline_selection_sha256']
    assert sha(FROZEN) == proof['frozen_sha256']


def pair_available(pair):
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used',
        '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True)
    memory = {int(i): int(m) for i, m in (line.split(',') for line in result.stdout.splitlines())}
    return all(memory[g] < 512 for g in pair)


def recover_completed(job, state, proof):
    """Adopt completed attempts after a controller restart; never duplicate live work."""
    status = read(job / 'status.json') if (job / 'status.json').exists() else {}
    for entry in status.get('active', []):
        command_path = Path(f'/proc/{entry["pid"]}/cmdline')
        if command_path.exists():
            arguments = command_path.read_bytes().split(b'\0')
            if str(entry['output']).encode() in arguments and any(b'EvaluateHybridCacheRepair.py' in a for a in arguments):
                raise RuntimeError(f'Previous evaluation still running: PID {entry["pid"]}, {entry["output"]}')
    for epoch in (3, 6):
        for alpha in ALPHAS:
            key = experiment_key(epoch, alpha)
            if key in state['completed']:
                audit = read(Path(state['completed'][key]) / 'completion-audit.json')
                assert audit['status'] == 'passed' and audit['epoch'] == epoch and audit['alpha'] == alpha
                for task, expected in audit['prediction_sha256'].items():
                    assert sha(Path(state['completed'][key]) / f'predictions/cacherepair/{task}.jsonl') == expected
                continue
            attempts = sorted((job / key).glob('attempt-*'), key=lambda p: int(p.name.split('-')[-1]), reverse=True)
            if attempts:
                state['attempts'][key] = max(state['attempts'].get(key, 0), int(attempts[0].name.split('-')[-1]))
            for path in attempts:
                report_path = path / 'engine_report.json'
                if not report_path.exists():
                    continue
                report = read(report_path)
                if report['status'] != 'completed' or report['failures']:
                    continue
                audit_run(path, epoch, alpha, proof)
                state['completed'][key] = str(path)
                break


def audit_run(directory, epoch, alpha, proof):
    selection = read(directory / 'selection.json')
    expected = baseline_selection()
    assert selection['arms'] == ['cacherepair']
    assert selection['tasks'] == expected['tasks']
    assert selection['frozen_pickle_sha256'] == expected['frozen_pickle_sha256']
    summary = summarize(directory)
    engine = read(directory / 'engine_report.json')
    reports = {r['task']: r for r in engine['runs']}
    assert len(reports) == 10 and all(r['cases'] == 64 for r in reports.values())
    hashes = {}
    for entry in summary['tasks']:
        task = entry['task']
        path = directory / 'predictions/cacherepair' / (task + '.jsonl')
        hashes[task] = sha(path)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == 64
        trr = {identity(r['case_id']): r for r in map(json.loads,
            (BASE / f'predictions/trr/{task}.jsonl').read_text().splitlines())}
        for row in rows:
            meta = row['result_metadata']
            assert meta['residual_alpha'] == alpha
            assert meta['repairer_invoked'] and meta['gdn_state_source'] == 'hypic_trr'
            assert meta['repairer_input'] == 'independent_causal_prefill'
            assert Path(meta['repairer_path']) == Path(proof['checkpoints'][str(epoch)]['path'])
            reference = trr[identity(row['case_id'])]['result_metadata']
            for field in ('n_input', 'num_cached_tokens', 'pic_mode', 'matched_prepared_segments'):
                assert meta[field] == reference[field], (task, row['case_id'], field)
        for metric, score in entry['arms']['cacherepair']['scores'].items():
            assert math.isclose(reports[task]['task_metrics'][metric]['mean'], score, abs_tol=1e-12, rel_tol=0)
    events = []
    for path in (directory / 'engine').glob('*/logs/*hybrid*/*-instance.log'):
        for line in path.read_text().splitlines():
            if '"event": "hybrid_repair"' in line:
                events.append(json.loads(line))
    assert len(events) == 640 and all(e['residual_alpha'] == alpha for e in events)
    verify_provenance(proof)
    write_summary(directory, summary)
    atomic(directory / 'completion-audit.json', {'status': 'passed', 'epoch': epoch,
        'alpha': alpha, 'task_groups': 10, 'cases': 640, 'repair_events': len(events),
        'prediction_sha256': hashes, 'fixed_inputs_and_cache_routes_verified': True,
        'engine_scores_equal_recorded_scores': True, 'finished_unix': time.time()})


def output_rows(directory, task):
    return {identity(r['case_id']): r for r in map(json.loads,
        (directory / f'predictions/cacherepair/{task}.jsonl').read_text().splitlines())}


def final_report(job, completed, proof):
    verify_provenance(proof)
    baseline = read(BASE / 'comparison.json')
    results = {key: read(Path(path) / 'comparison.json') for key, path in completed.items()}
    reproducibility = {'alpha_zero_epoch3_vs_epoch6': {}, 'alpha_one_vs_previous_evaluation': {}}
    for task in baseline['tasks']:
        name = task['task']
        zeros = [output_rows(Path(completed[experiment_key(e, 0.)]), name) for e in (3, 6)]
        equal = sum(zeros[0][k]['output_ids'] == zeros[1][k]['output_ids'] for k in zeros[0])
        reproducibility['alpha_zero_epoch3_vs_epoch6'][name] = {'identical_output_cases': equal, 'cases': 64}
        for e in (3, 6):
            previous = ROOT / f'outputs/hybrid-cacherepair-fulltrain-step{e*12500}-10task-scheduled/paired-results'
            old, new = output_rows(previous, name), output_rows(Path(completed[experiment_key(e, 1.)]), name)
            reproducibility['alpha_one_vs_previous_evaluation'][f'epoch-{e}/{name}'] = {
                'identical_output_cases': sum(old[k]['output_ids'] == new[k]['output_ids'] for k in old), 'cases': 64}
    atomic(job / 'reproducibility.json', reproducibility)
    all_rows = {}
    lines = ['# CacheRepair residual alpha: epoch 3 and epoch 6', '',
        '64 identical frozen cases per task; all 12 checkpoint/alpha combinations regenerated.',
        'Quality: 0–100. Full/TRR reused from the audited comparison.',
        'KV = stale + alpha * residual; GDN remains Hypic TRR at every alpha.', '']
    columns = ['Full', 'TRR', *[f'alpha={a:g}' for a in ALPHAS]]
    for epoch in (3, 6):
        rows = []
        indexes = {a: {t['task']: t for t in results[experiment_key(epoch, a)]['tasks']} for a in ALPHAS}
        for task in baseline['tasks']:
            name, metric = task['task'], task['primary']
            quality = [task['arms'][a]['scores'][metric] for a in ('full', 'trr')]
            ttft = [task['arms'][a]['ttft_mean_s'] for a in ('full', 'trr')]
            for a in ALPHAS:
                t = indexes[a][name]
                assert t['primary'] == metric and t['arms']['cacherepair']['n'] == 64
                quality.append(t['arms']['cacherepair']['scores'][metric])
                ttft.append(t['arms']['cacherepair']['ttft_mean_s'])
            rows.append({'task': name, 'metric': metric, 'quality': dict(zip(columns, quality)),
                'ttft_mean_s': dict(zip(columns, ttft))})
        all_rows[str(epoch)] = rows
        lines += [f'## Epoch {epoch} (step {epoch*12500})', '',
            '| Task | ' + ' | '.join(columns) + ' |', '|---|' + '---:|' * 8]
        lines += ['| ' + r['task'] + ' | ' + ' | '.join(f'{100*r["quality"][c]:.2f}' for c in columns) + ' |' for r in rows]
        lines += ['']
        for field, scale, suffix in [('quality', 100, 'quality'), ('ttft_mean_s', 1, 'ttft')]:
            with (job / f'epoch-{epoch}-{suffix}.csv').open('w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['task', *columns])
                w.writerows([[r['task'], *[f'{scale*r[field][c]:.2f}' for c in columns]] for r in rows])
    lines += ['Metrics: accuracy for NIAH/CWE/VT; RougeL for SAMSum/GovReport/MultiNews; F1 for the remaining tasks.',
        'TTFT CSVs include online work; preparation is excluded. No average is computed across heterogeneous quality metrics.']
    (job / 'comparison.md').write_text('\n'.join(lines) + '\n')
    atomic(job / 'comparison.json', {'status': 'completed', 'quality_scale': '0..1',
        'epochs': all_rows, 'sources': completed, 'samples_per_task': 64,
        'new_predictions': 7680, 'score_columns_per_epoch': 8, 'finished_unix': time.time()})
    atomic(job / 'completion-audit.json', {'status': 'passed', 'combinations': 12,
        'tasks_per_combination': 10, 'cases_per_task': 64, 'new_predictions': 7680,
        'all_sources_and_checkpoint_hashes_verified': True,
        'all_case_inputs_and_engine_scores_verified': True, 'finished_unix': time.time()})
    print('\n'.join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    job = args.output.resolve()
    job.mkdir(parents=True, exist_ok=True)
    lock = (job / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    proof = initialize(job)
    atomic(job / 'process.json', {'pid': os.getpid(), 'started_unix': time.time(), 'argv': sys.argv})
    if args.prepare_only:
        return
    state_path = job / 'state.json'
    state = read(state_path) if state_path.exists() else {'completed': {}, 'attempts': {}}
    recover_completed(job, state, proof)
    atomic(state_path, state)
    completed = state['completed']
    if (job / 'completion-audit.json').exists():
        assert len(completed) == 12
        return
    # Run the unit-strength and zero-strength controls early in the experiment.
    queue = [(e, a) for a in (1., 0., .25, .5, .75, 1.25) for e in (3, 6)
        if experiment_key(e, a) not in completed]
    active = {}
    environment = {k: v for k, v in os.environ.items()
        if k not in ('PYTORCH_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF', 'CUDA_VISIBLE_DEVICES')}
    environment.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
    with (job / 'events.jsonl').open('a', buffering=1) as events:
        while queue or active:
            for pair, run in list(active.items()):
                code = run['process'].poll()
                if code is None:
                    continue
                run['log'].close()
                epoch, alpha = run['epoch'], run['alpha']
                key = experiment_key(epoch, alpha)
                error = None
                try:
                    assert code == 0, f'Inference exit code {code}'
                    audit_run(run['output'], epoch, alpha, proof)
                except Exception as exc:
                    error = f'{type(exc).__name__}: {exc}'
                if error:
                    events.write(json.dumps({'event': 'attempt_failed', 'key': key,
                        'output': str(run['output']), 'error': error, 'time': time.time()}) + '\n')
                    if state['attempts'][key] >= 2:
                        raise RuntimeError(f'{key} failed twice: {error}; see {run["output"]}')
                    queue.append((epoch, alpha))
                else:
                    completed[key] = str(run['output'])
                    atomic(state_path, state)
                    events.write(json.dumps({'event': 'combination_completed', 'key': key, 'time': time.time()}) + '\n')
                del active[pair]
            for pair in PAIRS:
                if not queue or pair in active or not pair_available(pair):
                    continue
                verify_provenance(proof)
                epoch, alpha = queue.pop(0)
                key = experiment_key(epoch, alpha)
                attempt = state['attempts'].get(key, 0) + 1
                state['attempts'][key] = attempt
                directory = job / key / f'attempt-{attempt}'
                directory.mkdir(parents=True, exist_ok=False)
                config = yaml.safe_load(TEMPLATE.read_text())
                config['HybridCacheRepair'].update(RepairerPath=proof['checkpoints'][str(epoch)]['path'], ResidualAlpha=alpha)
                config['Engine']['AvailableGpuIds'] = list(pair)
                config['Engine']['OutputRoot'] = str(directory / 'engine')
                config['Engine']['GpuReleaseTimeoutSec'] = 120
                config['Experiment'].update(checkpoint_step=epoch*12500, residual_alpha=alpha,
                    checkpoint=proof['checkpoints'][str(epoch)]['path'])
                (directory / 'eval.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
                log = (directory / 'evaluation.log').open('x')
                command = [PYTHON, '-u', str(ROOT / 'scripts/EvaluateHybridCacheRepair.py'),
                    '--config', str(directory / 'eval.yaml'), '--frozen', str(FROZEN),
                    '--output', str(directory), '--samples', '64', '--arms', 'cacherepair',
                    '--residual-alpha', str(alpha)]
                process = subprocess.Popen(command, cwd=ROOT, env=environment,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                active[pair] = {'process': process, 'log': log, 'epoch': epoch,
                    'alpha': alpha, 'output': directory}
                atomic(state_path, state)
                events.write(json.dumps({'event': 'launched', 'key': key, 'gpus': pair,
                    'pid': process.pid, 'attempt': attempt, 'time': time.time()}) + '\n')
            atomic(job / 'status.json', {'status': 'running', 'completed': len(completed),
                'total': 12, 'active': [{'epoch': r['epoch'], 'alpha': r['alpha'],
                    'pid': r['process'].pid, 'output': str(r['output']), 'gpus': pair}
                    for pair, r in active.items()], 'pending': len(queue), 'updated_unix': time.time()})
            if queue or active:
                time.sleep(30)
    final_report(job, completed, proof)
    atomic(job / 'status.json', {'status': 'completed', 'completed': 12, 'total': 12,
        'active': [], 'pending': 0, 'finished_unix': time.time()})


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        raise
