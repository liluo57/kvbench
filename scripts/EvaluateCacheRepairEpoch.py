"""Scheduled 10x64 Hybrid quality evaluation with verified reused Full/TRR."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_quality_inputs(plan):
    quality = plan['epoch_quality']
    for path, expected in quality['files_sha256'].items():
        if sha(path) != expected:
            raise ValueError(f'Quality evaluation input/reference changed: {path}')
    base = Path(quality['baseline_results'])
    for name, expected in quality['runtime_source_sha256'].items():
        if sha(ROOT/name) != expected:
            raise ValueError(f'Quality runtime changed after preparation: {name}')
    selection = read(base/'selection.json')
    if (selection['frozen_pickle_sha256'] != sha(quality['frozen_inputs'])
            or len(selection['tasks']) != 10
            or any(t['count'] != quality['samples_per_task'] for t in selection['tasks'])):
        raise ValueError('Quality selection differs from frozen 10-task protocol')
    return quality, base, selection


def assemble(base, inference, checkpoint, output):
    from scripts.SummarizeHybridCacheRepair import summarize, write_summary
    selection = read(base/'selection.json')
    current = read(inference/'selection.json')
    if any(current[k] != selection[k] for k in ['tasks','frozen_pickle_sha256']):
        raise ValueError('New quality input selection differs from Full/TRR')
    engine = read(inference/'engine_report.json')
    if engine['status'] != 'completed' or engine['failures']:
        raise ValueError('Incomplete epoch quality evaluation')
    output.mkdir(exist_ok=False)
    selection['arms'] = ['full','trr','cacherepair']
    reports = [r for r in read(base/'engine_report.json')['runs']
               if r['method'] in ('hypic(full)','hypic(trr)')] + engine['runs']
    if len(reports) != 30 or len({(r['method'],r['task']) for r in reports}) != 30:
        raise ValueError('Expected 30 distinct task/method groups')
    for arm in selection['arms']:
        dest = output/'predictions'/arm
        dest.mkdir(parents=True)
        source = inference if arm == 'cacherepair' else base
        for entry in selection['tasks']:
            name = entry['task']+'.jsonl'
            shutil.copyfile(source/'predictions'/arm/name, dest/name)
    for entry in selection['tasks']:
        task = entry['task']
        trr = {json.dumps(r['case_id'],sort_keys=True): r for r in
               map(json.loads,(output/'predictions/trr'/f'{task}.jsonl').read_text().splitlines())}
        for row in map(json.loads,(output/'predictions/cacherepair'/f'{task}.jsonl').read_text().splitlines()):
            meta = row['result_metadata']
            reference = trr[json.dumps(row['case_id'],sort_keys=True)]['result_metadata']
            if Path(meta['repairer_path']).resolve() != checkpoint.resolve() or meta['gdn_state_source'] != 'hypic_trr':
                raise ValueError('Incorrect Hybrid checkpoint/GDN route')
            for key in ['n_input','num_cached_tokens','pic_mode','matched_prepared_segments']:
                if meta[key] != reference[key]:
                    raise ValueError(f'Hybrid/TRR cache protocol differs: {task}/{key}')
    (output/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    (output/'engine_report.json').write_text(json.dumps(dict(status='completed',failures=[],runs=reports),indent=2)+'\n')
    summary = summarize(output)
    if not summary['complete']:
        raise ValueError('Incomplete paired quality results')
    labels=dict(full='hypic(full)',trr='hypic(trr)',cacherepair='hybrid_cacherepair(cacherepair)')
    indexed={(r['method'],r['task']):r for r in reports}
    for task in summary['tasks']:
        for arm,label in labels.items():
            report=indexed[label,task['task']]
            if report['cases']!=task['arms'][arm]['n']:
                raise ValueError('Engine/prediction case count differs')
            for metric,score in task['arms'][arm]['scores'].items():
                if not math.isclose(report['task_metrics'][metric]['mean'],score,abs_tol=1e-12,rel_tol=0):
                    raise ValueError('Engine/prediction quality scores differ')
    write_summary(output,summary)
    lines=['| Task | Full | TRR | CacheRepair |','|---|---:|---:|---:|']
    for task in summary['tasks']:
        values=[100*task['arms'][arm]['scores'][task['primary']] for arm in selection['arms']]
        lines.append('| '+task['task']+' | '+' | '.join(f'{v:.2f}' for v in values)+' |')
    (output/'quality.md').write_text('\n'.join(lines)+'\n')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--request',type=Path,required=True)
    p.add_argument('--job',type=Path,required=True)
    p.add_argument('--physical-gpus',required=True)
    a=p.parse_args()
    from methods.CacheRepair.qwen38.resume import atomic_json,fingerprint
    plan,request=read(a.plan),read(a.request)
    if request['evaluation_plan_sha256'] != fingerprint(plan):
        raise ValueError('Quality request evaluation plan differs')
    quality,base,_=verify_quality_inputs(plan)
    checkpoint=Path(request['checkpoint'])
    if not (checkpoint/'validation.json').is_file():
        raise ValueError('Epoch checkpoint MSE/reload validation is not committed')
    destination=a.job/'quality-evaluations'/f"step-{request['step']:08d}"
    destination.mkdir(parents=True,exist_ok=True)
    done=destination/'completed.json'
    if done.exists():
        if read(done)['checkpoint'] != str(checkpoint.resolve()):
            raise ValueError('Existing quality checkpoint differs')
        return
    attempt=destination/f'attempt-{time.time_ns()}'
    attempt.mkdir()
    initial_hash=sha(checkpoint/'model.safetensors')
    cfg=yaml.safe_load(Path(quality['config_template']).read_text())
    cfg['HybridCacheRepair'].update(RepairerPath=str(checkpoint.resolve()),ResidualAlpha=1.)
    cfg['Engine']['AvailableGpuIds']=list(map(int,a.physical_gpus.split(',')))
    cfg['Engine']['OutputRoot']=str((attempt/'inference/engine').resolve())
    cfg['Experiment'].update(checkpoint=str(checkpoint.resolve()),checkpoint_step=request['step'])
    config=attempt/'eval.yaml'
    config.write_text(yaml.safe_dump(cfg,sort_keys=False))
    subprocess.run([quality['python'],'-u',str(ROOT/'scripts/EvaluateHybridCacheRepair.py'),
        '--config',str(config),'--frozen',quality['frozen_inputs'],'--output',str(attempt/'inference'),
        '--samples',str(quality['samples_per_task']),'--arms','cacherepair'],cwd=ROOT,check=True)
    summary=assemble(base,attempt/'inference',checkpoint,attempt/'paired-results')
    verify_quality_inputs(plan)
    if sha(checkpoint/'model.safetensors') != initial_hash:
        raise ValueError('Quality checkpoint changed during inference')
    atomic_json(done,dict(status='completed',checkpoint=str(checkpoint.resolve()),checkpoint_sha256=initial_hash,
        step=request['step'],epoch=request['epoch'],tasks=len(summary['tasks']),samples_per_task=quality['samples_per_task'],
        paired_results=str((attempt/'paired-results').resolve()),finished_unix=time.time()))


if __name__=='__main__':
    main()
