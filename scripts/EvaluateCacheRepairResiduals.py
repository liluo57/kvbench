"""Memory-bounded native KV residual validation of a fixed repairer checkpoint.

Three disjoint two-GPU native target replicas evaluate the unchanged 500 rows.
The serving repairer's exact FP32 efficient attention is reused. A head hook
reads normalized predictions before stale addition, so metrics do not inherit
extra subtract-stale rounding. All statistics accumulate in FP64 in token tiles.
"""
from __future__ import annotations

import argparse
import collections
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCES = ['scripts/EvaluateCacheRepairResiduals.py', 'methods/HybridCacheRepair/Runtime.py',
    'methods/CacheRepair/qwen38/model.py', 'methods/CacheRepair/qwen38/train.py',
    'methods/CacheRepair/qwen38/supervision.py', 'methods/CacheRepair/qwen38/packed.py']


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path, value):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def measure_pair(model, pair, *, tile=256, efficient=True, check_dense=False):
    import torch
    from methods.HybridCacheRepair.Runtime import streamed_repair
    accum = torch.zeros(6, device=pair['stale'].device, dtype=torch.float64)
    captured, dense_pieces = {}, []
    count = pair['stale'].numel()

    def capture(module, arguments, output):
        assert not captured
        captured['prediction'] = output

    def write(start, end, corrected):
        pred = captured.pop('prediction').reshape(1, end-start, *model.target_config.kv_shape).float()
        true = pair['joint'][:, start:end] - pair['stale'][:, start:end]
        norm_true = true / model.sigma_delta
        raw_pred = pred * model.sigma_delta
        # Match FP32 residual subtraction; use FP64 squares, products and sums.
        norm_error = (pred - norm_true).double()
        raw_error = (raw_pred - true).double()
        p, t = raw_pred.double(), true.double()
        accum.add_(torch.stack((norm_error.square().sum(), norm_true.double().square().sum(),
            raw_error.square().sum(), t.square().sum(), p.square().sum(), (p*t).sum())))
        if check_dense:
            dense_pieces.append(pred.detach().clone())

    hook = model.v_head.register_forward_hook(capture)
    try:
        with torch.no_grad():
            streamed_repair(model, pair['tokens'], pair['chunk_ids'],
                lambda a,b: pair['stale'][:, a:b], write, tile=tile, efficient=efficient)
    finally:
        hook.remove()
    assert not captured
    parity = None
    if check_dense:
        with torch.no_grad():
            expected = model(pair['stale'], pair['tokens'], pair['chunk_ids']).float()
        actual = torch.cat(dense_pieces, dim=1)
        torch.testing.assert_close(actual, expected, rtol=3e-4, atol=1e-4)
        parity = (actual-expected).abs().max().item()
    values = accum.cpu().tolist()
    assert all(math.isfinite(x) for x in values)
    return {'elements':count, 'tokens':pair['tokens'].numel(),
        **dict(zip(('normalized_error_ss','normalized_true_ss','raw_error_ss','raw_true_ss','raw_pred_ss','raw_dot'), values)),
        'normalized_mse':values[0]/count,'raw_mse':values[2]/count,
        'dense_max_abs_error':parity}


def summarize(rows):
    n = len(rows)
    totals = {k:sum(r[k] for r in rows) for k in ('elements','tokens','normalized_error_ss',
        'normalized_true_ss','raw_error_ss','raw_true_ss','raw_pred_ss','raw_dot')}
    nt, rt, rp = totals['normalized_true_ss'], totals['raw_true_ss'], totals['raw_pred_ss']
    norm_mean = sum(r['normalized_mse'] for r in rows)/n
    norm_zero = sum(r['normalized_true_ss']/r['elements'] for r in rows)/n
    return {'examples':n,'tokens':totals['tokens'],
        'normalized_mse_example_mean':norm_mean,'normalized_zero_baseline_example_mean':norm_zero,
        'normalized_g_mse_example_mean':1.-norm_mean/norm_zero if norm_zero else None,
        'normalized_mse_element_mean':totals['normalized_error_ss']/totals['elements'],
        'raw_mse_element_mean':totals['raw_error_ss']/totals['elements'],
        'raw_zero_baseline_element_mean':rt/totals['elements'],
        'raw_g_mse':1.-totals['raw_error_ss']/rt if rt else None,
        'raw_cosine_pooled':totals['raw_dot']/math.sqrt(rp*rt) if rp and rt else None,
        'predicted_to_true_raw_l2':math.sqrt(rp/rt) if rt else None,
        'raw_g_mse_by_alpha':{str(a):1.-(a*a*rp+rt-2*a*totals['raw_dot'])/rt
            for a in (0.,.25,.5,.75,1.,1.25)} if rt else None}


def worker(args):
    import gc
    import torch
    from methods.CacheRepair.qwen38.model import load_repairer
    from methods.CacheRepair.qwen38.sources import local_checkpoint_spec
    from methods.CacheRepair.qwen38.target import TargetGeometry
    from methods.CacheRepair.qwen38.train import load_native_target
    from methods.CacheRepair.qwen38.supervision import make_pair
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    torch.cuda.set_device(0)
    proof = read(args.output/'provenance.json')
    assert sha(args.checkpoint/'model.safetensors') == proof['checkpoint_sha256']
    assert sha(args.dataset/'validation.jsonl') == proof['validation_sha256']
    rows = [json.loads(line) for line in (args.dataset/'validation.jsonl').read_text().splitlines()]
    target = load_native_target(str(args.model), (0,1), attention_backend='sdpa', require_fast=True)
    geometry = TargetGeometry.from_target(target, target_model_id='Qwen/Qwen3.8-27B')
    torch.set_default_dtype(torch.float32)
    model = load_repairer(args.checkpoint,target,target_model_id='Qwen/Qwen3.8-27B').requires_grad_(False).eval()
    assert all(p.dtype == torch.float32 for p in model.parameters())
    norm_digest = hashlib.sha256(model.sigma_delta.cpu().numpy().tobytes()+model.sigma_stale.cpu().numpy().tobytes()).hexdigest()

    def pair(row):
        return make_pair(target,row['chunks'],geometry=geometry,chunk_batch_size=8,pack_chunks=False)

    if args.worker == 0:
        for label, index in [('short',min(range(500),key=lambda i:sum(map(len,rows[i]['chunks'])))),
                ('long',max(range(500),key=lambda i:sum(map(len,rows[i]['chunks']))))]:
            p = pair(rows[index])
            result = measure_pair(model,p,check_dense=label=='short')
            del p
            gc.collect()
            print(json.dumps({'preflight':label,'row':index,**result}),flush=True)
        atomic(args.output/'preflight.json',{'status':'passed','max_tokens':16384,
            'target_chunk_batch_size':8,'target_packed_chunks':False,'repairer_efficient_fp32':True,
            'normalization_digest':norm_digest,'gpu_peak_allocated':{str(g):torch.cuda.max_memory_allocated(g) for g in (0,1)}})
    path = args.output/f'worker-{args.worker}-cases.jsonl'
    existing = [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    done = {r['row'] for r in existing}
    assert len(done) == len(existing)
    with path.open('a',buffering=1) as stream:
        for i,row in enumerate(rows):
            if i%3 != args.worker or i in done:
                continue
            started = time.perf_counter()
            p = pair(row)
            measured = measure_pair(model,p)
            del p
            result = {'row':i,'example_id':row['example_id'],'task':row['task'],**measured,
                'seconds':time.perf_counter()-started}
            stream.write(json.dumps(result)+'\n')
            done.add(i)
            atomic(args.output/f'worker-{args.worker}-status.json',{'status':'running','examples':len(done),'last_row':i})
            if len(done)%10==0:
                print(json.dumps({'worker':args.worker,'examples':len(done),'last_row':i}),flush=True)
    assert hashlib.sha256(model.sigma_delta.cpu().numpy().tobytes()+model.sigma_stale.cpu().numpy().tobytes()).hexdigest()==norm_digest
    assert sha(args.checkpoint/'model.safetensors')==proof['checkpoint_sha256']
    atomic(args.output/f'worker-{args.worker}-status.json',{'status':'completed','examples':len(done),'normalization_unchanged':True})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('model','checkpoint','dataset','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--worker',type=int,choices=(0,1,2))
    args=parser.parse_args()
    for name in ('model','checkpoint','dataset','output'):
        setattr(args,name,getattr(args,name).resolve())
    if args.worker is not None:
        worker(args)
        return
    args.output.mkdir(parents=True,exist_ok=True)
    lock=(args.output/'controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (args.output/'completion-audit.json').exists():
        return
    proof={'checkpoint':str(args.checkpoint),'checkpoint_sha256':sha(args.checkpoint/'model.safetensors'),
        'validation_sha256':sha(args.dataset/'validation.jsonl'),
        'source_sha256':{name:sha(ROOT/name) for name in SOURCES},
        'teacher':'native HF/FLA BF16, true independent causal chunk prefill, batch 8, no packed concatenation',
        'repairer':'FP32, exact efficient SDPA, token tiles 256, original checkpoint RMS unchanged',
        'normalized_mse':'FP64 sum of squared FP32 normalized errors; mean over examples',
        'raw_g_mse_and_cosine':'Pooled across raw canonical KV elements; FP64 dot products/norms',
        'alpha':1.,'gpu_pairs':[[0,1],[2,3],[4,5]]}
    pp=args.output/'provenance.json'
    if pp.exists():
        assert read(pp)==proof
        prior=read(args.output/'process.json') if (args.output/'process.json').exists() else {}
        for child in prior.get('children',[]):
            path=Path(f'/proc/{child["pid"]}/cmdline')
            if path.exists() and str(args.output).encode() in path.read_bytes().split(b'\0'):
                raise RuntimeError('Previous validation child still active; refusing duplicate evaluation')
    else:
        atomic(pp,proof)
    manifest=read(args.dataset/'dataset.json')
    assert manifest['files_sha256']['validation.jsonl']==proof['validation_sha256']
    children=[]
    def launch(index):
        environment={**os.environ,'CUDA_VISIBLE_DEVICES':','.join(map(str,(2*index,2*index+1))),
            'PYTORCH_ALLOC_CONF':'backend:cudaMallocAsync','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
            'TOKENIZERS_PARALLELISM':'false'}
        log=(args.output/f'worker-{index}.log').open('a')
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'--model',str(args.model),
            '--checkpoint',str(args.checkpoint),'--dataset',str(args.dataset),'--output',str(args.output),'--worker',str(index)]
        proc=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT)
        children.append((proc,log))
        atomic(args.output/'process.json',{'pid':os.getpid(),'children':[{'pid':p.pid} for p,l in children]})
    launch(0)
    while not (args.output/'preflight.json').exists():
        if children[0][0].poll() is not None:
            raise RuntimeError('Preflight failed; see worker-0.log')
        time.sleep(10)
    launch(1)
    launch(2)
    while any(p.poll() is None for p,l in children):
        completed=sum(len(p.read_text().splitlines()) for p in args.output.glob('worker-*-cases.jsonl'))
        atomic(args.output/'status.json',{'status':'running','examples':completed,'total':500,'updated_unix':time.time()})
        time.sleep(30)
    for p,l in children:
        l.close()
        assert p.returncode==0, f'Validation worker exit {p.returncode}'
    rows=[json.loads(s) for p in args.output.glob('worker-*-cases.jsonl') for s in p.read_text().splitlines()]
    assert len(rows)==500 and {r['row'] for r in rows}==set(range(500))
    source=[json.loads(s) for s in (args.dataset/'validation.jsonl').read_text().splitlines()]
    for r in rows:
        assert r['example_id']==source[r['row']]['example_id'] and r['task']==source[r['row']]['task']
    by_task=collections.defaultdict(list)
    for r in rows:
        by_task[r['task']].append(r)
    assert len(by_task)==10 and all(len(v)==50 for v in by_task.values())
    result={'status':'completed','checkpoint':str(args.checkpoint),'checkpoint_epoch':3,
        'overall':summarize(rows),'tasks':{k:summarize(v) for k,v in sorted(by_task.items())},'finished_unix':time.time()}
    for name,expected in proof['source_sha256'].items():
        assert sha(ROOT/name)==expected,name
    assert sha(args.checkpoint/'model.safetensors')==proof['checkpoint_sha256']
    atomic(args.output/'validation.json',result)
    columns=['normalized_mse_example_mean','normalized_zero_baseline_example_mean','raw_mse_element_mean',
        'raw_g_mse','raw_cosine_pooled','predicted_to_true_raw_l2']
    with (args.output/'validation.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['task','examples',*columns])
        for name,m in [*result['tasks'].items(),('ALL',result['overall'])]:
            w.writerow([name,m['examples'],*[m[c] for c in columns]])
    lines=['# Epoch 3: raw 10x50 validation','',
        '| Task | n | Normalized MSE | Zero baseline | Raw G_MSE | Raw cosine |',
        '|---|---:|---:|---:|---:|---:|']
    for name,m in [*result['tasks'].items(),('ALL',result['overall'])]:
        lines.append(f'| {name} | {m["examples"]} | {m["normalized_mse_example_mean"]:.6f} | {m["normalized_zero_baseline_example_mean"]:.6f} | {100*m["raw_g_mse"]:.2f}% | {m["raw_cosine_pooled"]:.6f} |')
    lines+=['','Normalized MSE and zero baseline are per-example means, matching the old validation aggregation.',
        'Raw G_MSE and cosine pool all KV elements within each row group. All rows are raw source inputs, not generation benchmarks.',
        'The original checkpoint RMS is unchanged. This evaluation performs no optimizer updates.','']
    (args.output/'report.md').write_text('\n'.join(lines))
    atomic(args.output/'completion-audit.json',{'status':'passed','examples':500,'tasks':10,'per_task':50,
        'all_inputs_and_checkpoint_hashes_verified':True,'original_rms_unchanged':True,'no_training':True})
    atomic(args.output/'status.json',{'status':'completed','examples':500,'total':500,'finished_unix':time.time()})
    print('\n'.join(lines),flush=True)


if __name__=='__main__':
    main()
