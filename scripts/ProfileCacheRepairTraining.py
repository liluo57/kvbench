"""Disposable real-data optimizer trials. Never save weights, RMS or state."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import threading
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--backend',choices=['native','vllm'],default='vllm')
    p.add_argument('--groups',default='0,1;2,3;4,5')
    p.add_argument('--examples',type=int,default=24)
    p.add_argument('--passes',type=int,default=2)
    p.add_argument('--memory-utilization',type=float,default=.72)
    p.add_argument('--max-batched-tokens',type=int,default=32768)
    p.add_argument('--transport',choices=['cuda','host'],default='cuda')
    p.add_argument('--serial',action='store_true')
    p.add_argument('--prefetch-per-worker',type=int,default=1)
    p.add_argument('--scheduler',choices=['round_robin','least_work'],default='round_robin')
    p.add_argument('--tile',type=int,default=256)
    p.add_argument('--pack-chunks',action='store_true',
        help='Use the native packed stale-context forward')
    p.add_argument('--pipeline-depth',type=int,default=0,
        help='Native layer-pipeline depth; requires --pack-chunks')
    p.add_argument('--rows',type=Path)
    p.add_argument('--operator-profile',action='store_true')
    p.add_argument('--audit-kv-first-pass',action='store_true',
        help='Hash every producer/consumer BF16 capture during warmup, outside measured passes')
    p.add_argument('--optimized-sweep',action='store_true')
    p.add_argument('--queue-sweep',action='store_true')
    p.add_argument('--max-pending-pairs',type=int)
    p.add_argument('--batching-sweep',action='store_true')
    p.add_argument('--single-sample-sweep',action='store_true',
        help='Compare reference vs deferred shared refill on the same vLLM engines')
    p.add_argument('--bounded-root-speculative-sweep',action='store_true',
        help='ABBA profile: baseline root queue vs one extra root-only bounded slot')
    p.add_argument('--cost-scheduler-sweep',action='store_true',
        help='Compare T-only, attention-work, and diagnostic heterogeneous least-work costs')
    p.add_argument('--finish-scheduler-sweep',action='store_true',
        help='Compare backlog-only reference with heterogeneous estimated-finish-time scheduling')
    p.add_argument('--max-batch-examples',type=int,default=1)
    p.add_argument('--max-batch-input-tokens',type=int,default=8192)
    p.add_argument('--remote-kv-cache-gib',type=float,default=1)
    p.add_argument('--remote-max-seqs',type=int,default=32)
    p.add_argument('--dynamic-batching',action='store_true')
    p.add_argument('--worker-pending-tokens',type=int)
    p.add_argument('--global-pending-tokens',type=int)
    p.add_argument('--sweep',action='store_true',help='Reuse targets for controlled GPU allocation, transport and tiling comparisons')
    a=p.parse_args()
    if a.backend=='vllm' and (a.pack_chunks or a.pipeline_depth):
        p.error('vLLM does not use native packed chunks or the native layer pipeline')
    if a.pipeline_depth and (a.pipeline_depth<2 or not a.pack_chunks):
        p.error('Native pipeline requires --pack-chunks and depth >= 2')
    a.output.mkdir(parents=True,exist_ok=False)
    import torch
    from methods.CacheRepair.qwen38.vllm_target import VllmTarget
    from methods.CacheRepair.qwen38.vllm_pool import VllmPairPool
    from methods.CacheRepair.qwen38.target import TargetGeometry
    from methods.CacheRepair.qwen38.model import load_repairer
    from methods.CacheRepair.qwen38.long_training import configure,loss
    from methods.CacheRepair.backend.cacherepair.train import learning_rate
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    ckpt=a.checkpoint/'model.safetensors'
    before=hashlib.sha256(ckpt.read_bytes()).hexdigest()
    source_paths=[Path(__file__),ROOT/'methods/CacheRepair/qwen38/long_training.py']
    if a.backend=='vllm':
        source_paths.extend([ROOT/'methods/CacheRepair/qwen38/vllm_pool.py',
            ROOT/'methods/CacheRepair/qwen38/vllm_target.py'])
    else:
        source_paths.extend([ROOT/'methods/CacheRepair/qwen38/pair_pool.py',
            ROOT/'methods/CacheRepair/qwen38/supervision.py',
            ROOT/'methods/CacheRepair/qwen38/experimental/pipeline.py'])
    source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    if a.rows:
        rows=[json.loads(s) for s in a.rows.read_text().splitlines() if s.strip()][:a.examples]
    else:
        manifest=json.loads((a.dataset/'dataset.json').read_text())
        count=manifest.get('sizes',{}).get('train')
        if count is None:
            with (a.dataset/'train.jsonl').open() as f:count=sum(1 for s in f if s.strip())
        order=list(range(count))
        random.Random(7*1_000_003).shuffle(order)
        selected=set(order[:a.examples]);found={}
        with (a.dataset/'train.jsonl').open() as f:
            for i,line in enumerate(f):
                if i in selected:found[i]=json.loads(line)
                if len(found)==len(selected):break
        rows=[found[i] for i in order[:a.examples]]
    (a.output/'selected-rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    stop=threading.Event()
    samples=[]
    phase={'name':'load','time':time.time()}
    def monitor():
        while not stop.is_set():
            out=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,utilization.memory,memory.used,power.draw','--format=csv,noheader,nounits'],text=True)
            entry=dict(time=time.time(),phase=phase['name'],gpus=[[float(x) for x in line.split(',')] for line in out.strip().splitlines()])
            samples.append(entry)
            with (a.output/'gpu-samples.jsonl').open('a') as f:f.write(json.dumps(entry)+'\n')
            stop.wait(1)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    pool=None
    events=[]
    try:
        if a.backend=='vllm':
            target=VllmTarget(a.model,'cuda:0')
        else:
            from methods.CacheRepair.qwen38.train import load_native_target
            target=load_native_target(a.model,(0,1),attention_backend='sdpa',require_fast=True)
        geometry=TargetGeometry.from_target(target,target_model_id='Qwen/Qwen3.8-27B')
        model=load_repairer(a.checkpoint,target,target_model_id='Qwen/Qwen3.8-27B').train()
        configure(model,tile=a.tile)
        parameters=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(parameters,lr=1e-4,betas=(.9,.95),eps=1e-8,weight_decay=.01)
        rms_before={name:getattr(model,name).clone() for name in ('sigma_stale','sigma_delta')}
        groups=[tuple(map(int,x.split(','))) for x in a.groups.split(';')]
        if a.backend=='vllm':
            pool=VllmPairPool(target,geometry,groups,model_path=a.model,
                memory_utilization=a.memory_utilization,max_batched_tokens=a.max_batched_tokens,
                log_path=a.output/'pairs.jsonl',transport=a.transport,verify_first=True,prefetch_per_worker=a.prefetch_per_worker,scheduler=a.scheduler,max_pending_pairs=a.max_pending_pairs,
                max_batch_examples=a.max_batch_examples,max_batch_input_tokens=a.max_batch_input_tokens,
                remote_kv_cache_bytes=int(a.remote_kv_cache_gib*1024**3),remote_max_num_seqs=a.remote_max_seqs,dynamic_batching=a.dynamic_batching,
                max_pending_input_tokens_per_worker=a.worker_pending_tokens,max_pending_input_tokens=a.global_pending_tokens)
        else:
            from methods.CacheRepair.qwen38.pair_pool import PairPool
            pool=PairPool(target,geometry,groups,model_path=a.model,chunk_batch_size=8,
                attention_backend='sdpa',pack_chunks=a.pack_chunks,pipeline_depth=a.pipeline_depth,
                log_path=a.output/'pairs.jsonl')
        torch.cuda.synchronize(0)
        variants=[('default',a.prefetch_per_worker,a.scheduler,a.transport,a.tile)]
        if a.sweep or a.optimized_sweep or a.queue_sweep or a.batching_sweep or a.single_sample_sweep or a.cost_scheduler_sweep or a.finish_scheduler_sweep:
            if a.backend!='vllm' or groups!=[(0,1),(2,3),(4,5)]:
                raise ValueError('Sweep requires three vLLM pairs on all six GPUs')
            variants=[('shared_rr',1,'round_robin','cuda',256),
                ('shared_balanced',4,'least_work','cuda',256),
                ('dedicated_rr',1,'round_robin','cuda',256),
                ('dedicated_balanced',4,'least_work','cuda',256),
                ('dedicated_host',4,'least_work','host',256),
                ('dedicated_tile1024',4,'least_work','cuda',1024)]
        if a.optimized_sweep:
            variants=[('shared_balanced_async',4,'least_work','cuda',256),
                ('shared_tile1024_async',4,'least_work','cuda',1024),
                ('dedicated_balanced_async',4,'least_work','cuda',256),
                ('dedicated_host_async',4,'least_work','host',256),
                ('dedicated_tile1024_async',4,'least_work','cuda',1024)]
        if a.queue_sweep:
            variants=[('shared_window9',8,'least_work','cuda',256),
                ('dedicated_window9',8,'least_work','cuda',256)]
        if a.batching_sweep:
            variants=[('cache4_batch1',8,'least_work','cuda',256),
                ('cache4_batch4_16k',8,'least_work','cuda',256),
                ('cache4_batch8_32k',8,'least_work','cuda',256)]
        if a.single_sample_sweep:
            if a.sweep or a.optimized_sweep or a.queue_sweep or a.batching_sweep or a.cost_scheduler_sweep or a.bounded_root_speculative_sweep:
                raise ValueError('--single-sample-sweep cannot be combined with another sweep')
            if a.backend!='vllm' or groups!=[(0,1),(2,3),(4,5)] or a.max_batch_examples!=1 or a.dynamic_batching:
                raise ValueError('--single-sample-sweep requires three vLLM pairs and single-sample generation')
            same=(a.prefetch_per_worker,a.scheduler,a.transport,a.tile)
            variants=[('single_reference',*same),('long_refill_gate',*same),
                ('shared_lookahead2',*same)]
        if a.bounded_root_speculative_sweep:
            other_sweeps=(a.sweep,a.optimized_sweep,a.queue_sweep,a.batching_sweep,
                a.single_sample_sweep,a.cost_scheduler_sweep,a.finish_scheduler_sweep)
            if any(other_sweeps):
                raise ValueError('--bounded-root-speculative-sweep cannot be combined with another sweep')
            if (a.backend!='vllm' or groups!=[(0,1),(2,3),(4,5)] or a.serial
                or a.max_batch_examples!=1 or a.dynamic_batching
                or a.max_pending_pairs is None or a.max_pending_pairs<1):
                raise ValueError('--bounded-root-speculative-sweep requires three queued vLLM pairs, single-sample generation, and a finite pending-row cap')
            same=(a.prefetch_per_worker,a.scheduler,a.transport,a.tile)
            variants=[('root_spec_A1',*same),('root_spec_B1',*same),
                ('root_spec_B2',*same),('root_spec_A2',*same)]
        if a.cost_scheduler_sweep:
            other_sweeps=(a.sweep,a.optimized_sweep,a.queue_sweep,a.batching_sweep,a.single_sample_sweep)
            if any(other_sweeps):
                raise ValueError('--cost-scheduler-sweep cannot be combined with another sweep')
            if (a.backend!='vllm' or groups!=[(0,1),(2,3),(4,5)] or a.serial
                or a.max_batch_examples!=1 or a.dynamic_batching):
                raise ValueError('--cost-scheduler-sweep requires three vLLM pairs, queued scheduling, and single-sample generation')
            same=(a.prefetch_per_worker,'least_work',a.transport,a.tile)
            variants=[('cost_reference',*same),('attention_cost',*same),('heterogeneous_cost',*same)]
        if a.finish_scheduler_sweep:
            other_sweeps=(a.sweep,a.optimized_sweep,a.queue_sweep,a.batching_sweep,a.single_sample_sweep,a.cost_scheduler_sweep)
            if any(other_sweeps):
                raise ValueError('--finish-scheduler-sweep cannot be combined with another sweep')
            if (a.backend!='vllm' or groups!=[(0,1),(2,3),(4,5)] or a.serial
                or a.max_batch_examples!=1 or a.dynamic_batching):
                raise ValueError('--finish-scheduler-sweep requires three vLLM pairs, queued scheduling, and single-sample generation')
            same=(a.prefetch_per_worker,'least_work',a.transport,a.tile)
            variants=[('finish_reference',*same),('earliest_finish',*same)]
        for variant,prefetch,scheduler,transport,tile in variants:
            phase['name']=variant+':setup'
            if variant in ('dedicated_rr','dedicated_balanced_async','dedicated_window9'):
                from methods.CacheRepair.qwen38.vllm_pool import _shutdown
                pool.set_sequence([])
                if pool.dispatchers:
                    pool.dispatchers.pop(0).close()
                executor=pool.executors.pop(0)
                executor.submit(_shutdown).result()
                executor.shutdown(wait=True)
                pool.groups.pop(0)
                pool.verify_groups.clear()
                pool.shared=False
            scheduler_cost_config=None
            if a.backend=='vllm':
                # Reset every shared/long-gate tuning knob before selecting a variant.
                pool.defer_shared_on_long=False
                pool.shared_prefetch_per_worker=1
                pool.shared_pending_input_tokens=None
                pool.max_pending_pairs=a.max_pending_pairs
                if variant=='long_refill_gate':
                    pool.defer_shared_on_long=True
                if variant=='shared_lookahead2':
                    pool.shared_prefetch_per_worker=2
                    pool.shared_pending_input_tokens=16384
                if variant in ('root_spec_B1','root_spec_B2'):
                    # One extra global row credit belongs to the root pair;
                    # the existing global/remote token budgets remain fixed.
                    pool.shared_prefetch_per_worker=2
                    pool.shared_pending_input_tokens=16384
                    pool.max_pending_pairs=a.max_pending_pairs+1
                pool.attention_cost_weight=0.0
                pool.replica_work_scales=None
                pool.include_incoming_work=False
                if a.cost_scheduler_sweep:
                    attention_weight={'cost_reference':0.0,'attention_cost':1e-5,
                        'heterogeneous_cost':1e-5}[variant]
                    replica_scales={'cost_reference':None,'attention_cost':None,
                        'heterogeneous_cost':[1.0,1.0,1.3]}[variant]
                    pool.attention_cost_weight=attention_weight
                    pool.replica_work_scales=replica_scales
                    scheduler_cost_config=dict(scheduler='least_work',
                        attention_cost_weight=attention_weight,replica_work_scales=replica_scales,
                        work_formula='T' if attention_weight==0 else 'T + c*(T*T + sum(chunk_len*chunk_len))',
                        credit_formula='T')
                if a.finish_scheduler_sweep:
                    finish_config={'finish_reference':(0.0,None,False),
                        'earliest_finish':(1e-5,[1.0,1.0,1.3],True)}[variant]
                    pool.attention_cost_weight,pool.replica_work_scales,pool.include_incoming_work=finish_config
                    scheduler_cost_config=dict(scheduler='least_work',
                        attention_cost_weight=finish_config[0],replica_work_scales=finish_config[1],
                        include_incoming_work=finish_config[2],
                        work_formula='backlog only' if not finish_config[2] else '(queued + incoming T + c*(T*T + sum(chunk_len*chunk_len))) * group_scale',
                        credit_formula='T')
                pool.prefetch_per_worker=prefetch
                pool.capacities=[pool._capacity(i) for i in range(len(pool.groups))]
                pool.scheduler=scheduler
                pool.transport=transport
                if a.batching_sweep:
                    pool.max_batch_examples={'cache4_batch1':1,'cache4_batch4_16k':4,'cache4_batch8_32k':8}[variant]
                    pool.max_batch_input_tokens=32768 if variant=='cache4_batch8_32k' else 16384
            configure(model,tile=tile)
            if a.sweep or a.optimized_sweep or a.queue_sweep or a.batching_sweep or a.single_sample_sweep or a.cost_scheduler_sweep or a.finish_scheduler_sweep or a.bounded_root_speculative_sweep:
                from safetensors.torch import load_file
                optimizer.zero_grad(set_to_none=True)
                model.load_state_dict(load_file(str(ckpt)),strict=True)
                del optimizer
                optimizer=torch.optim.AdamW(parameters,lr=1e-4,betas=(.9,.95),eps=1e-8,weight_decay=.01)
                torch.cuda.empty_cache()
            for epoch in range(a.passes):
                phase['name']=f'{variant}:pass-{epoch+1}'
                if a.backend=='vllm':pool.verify_all=a.audit_kv_first_pass and epoch==0
                for gpu in [0]:torch.cuda.reset_peak_memory_stats(gpu)
                start=time.perf_counter()
                if not a.serial:pool.set_sequence(rows)
                for begin in range(0,len(rows),4):
                    batch=rows[begin:begin+4]
                    optimizer.zero_grad(set_to_none=True)
                    for row in batch:
                        fetch=time.perf_counter()
                        pair=pool.get(row['chunks'])
                        torch.cuda.current_stream(0).synchronize()
                        fetched=time.perf_counter()
                        fwdstart=torch.cuda.Event(enable_timing=True);fwdend=torch.cuda.Event(enable_timing=True)
                        bwdend=torch.cuda.Event(enable_timing=True)
                        fwdstart.record()
                        value=loss(model,**pair)
                        fwdend.record()
                        (value/len(batch)).backward()
                        bwdend.record();bwdend.synchronize()
                        entry={**pool.last_event, 'variant':variant,'pass_number':epoch+1,'index':len(events)%len(rows),
                            'scheduler_cost_config':scheduler_cost_config,
                            'fetch_seconds':fetched-fetch,'forward_seconds':fwdstart.elapsed_time(fwdend)/1000,
                            'backward_seconds':fwdend.elapsed_time(bwdend)/1000,'loss':float(value.detach()),'allocated_bytes':torch.cuda.memory_allocated(0),'reserved_bytes':torch.cuda.memory_reserved(0),'peak_allocated_bytes':torch.cuda.max_memory_allocated(0)}
                        if hasattr(pool,'last_stage_events'):
                            e0,e1,e2=pool.last_stage_events
                            entry.update(copy_gpu_seconds=e0.elapsed_time(e1)/1000,
                                canonical_gpu_seconds=e1.elapsed_time(e2)/1000)
                        events.append(entry)
                        with (a.output/'samples.jsonl').open('a') as f:f.write(json.dumps(entry)+'\n')
                        print(json.dumps(entry),flush=True)
                        del pair,value
                    step=epoch*((len(rows)+3)//4)+begin//4+1
                    for group in optimizer.param_groups:group['lr']=learning_rate(step,327762)/3
                    torch.nn.utils.clip_grad_norm_(parameters,1.)
                    optimizer.step()
                    if not torch.stack([torch.isfinite(p).all() for p in parameters]).all().item():raise FloatingPointError('Nonfinite parameter')
                torch.cuda.synchronize(0)
                elapsed=time.perf_counter()-start
                entry=dict(variant=variant,pass_number=epoch+1,scheduler_cost_config=scheduler_cost_config,
                    elapsed_seconds=elapsed,tokens=sum(sum(map(len,r['chunks'])) for r in rows),
                    tokens_per_second=sum(sum(map(len,r['chunks'])) for r in rows)/elapsed,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(0),peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
                print(json.dumps({'pass_summary':entry}),flush=True)
                with (a.output/'passes.jsonl').open('a') as f:f.write(json.dumps(entry)+'\n')
        if a.operator_profile and a.backend=='vllm':
            from methods.CacheRepair.qwen38.vllm_pool import _profile_generate
            phase['name']='operator-profile'
            pool.set_sequence([])
            row=max(rows,key=lambda r:sum(map(len,r['chunks'])))
            packet,worker_ops=pool.executors[0].submit(_profile_generate,row['chunks'],pool.transport).result()
            pair,_=pool._consume(packet,0,row['chunks'])
            torch.cuda.current_stream(0).synchronize()
            optimizer.zero_grad(set_to_none=True)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA]) as prof:
                probe_loss=loss(model,**pair)
                probe_loss.backward()
                torch.cuda.synchronize(0)
            repair_ops=[dict(name=o.key,calls=o.count,self_cuda_us=o.self_device_time_total,
                total_cuda_us=o.device_time_total,self_cpu_us=o.self_cpu_time_total)
                for o in sorted(prof.key_averages(),key=lambda o:o.self_device_time_total,reverse=True)[:40]]
            (a.output/'operator-profile.json').write_text(json.dumps(dict(tokens=sum(map(len,row['chunks'])),
                target_ranks=worker_ops,repairer=repair_ops),indent=2)+'\n')
            del pair,probe_loss
            optimizer.zero_grad(set_to_none=True)
        assert all(torch.equal(getattr(model,k),v) for k,v in rms_before.items())
        measured_events=[event for event in events if event['pass_number']>1] or events
        result=dict(status='completed',optimizer_updates=len(variants)*a.passes*((len(rows)+3)//4),
            checkpoint_written=False,optimizer_state_written=False,rms_unchanged=True,
            source_sha256=source_hashes,
            source_unchanged=all(hashlib.sha256(p.read_bytes()).hexdigest()==source_hashes[str(p)] for p in source_paths),
            options={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
            runtime_environment={key:os.environ.get(key) for key in ('NCCL_PROTO','NCCL_ALGO','PYTORCH_ALLOC_CONF')},
            scheduler_cost_variants=([
                dict(variant='cost_reference',attention_cost_weight=0.0,replica_work_scales=None),
                dict(variant='attention_cost',attention_cost_weight=1e-5,replica_work_scales=None),
                dict(variant='heterogeneous_cost',attention_cost_weight=1e-5,replica_work_scales=[1.0,1.0,1.3])
            ] if a.cost_scheduler_sweep else None),
            finish_scheduler_variants=([
                dict(variant='finish_reference',attention_cost_weight=0.0,replica_work_scales=None,include_incoming_work=False),
                dict(variant='earliest_finish',attention_cost_weight=1e-5,replica_work_scales=[1.0,1.0,1.3],include_incoming_work=True)
            ] if a.finish_scheduler_sweep else None),
            bounded_root_speculative_policy=({
                'order':['root_spec_A1','root_spec_B1','root_spec_B2','root_spec_A2'],
                'A':{'root_capacity':1,'global_pending_rows':a.max_pending_pairs,
                    'root_input_token_cap':None},
                'B':{'root_capacity':2,'global_pending_rows':a.max_pending_pairs+1,
                    'root_input_token_cap':16384,
                    'remote_prefetch_per_worker':a.prefetch_per_worker,
                    'worker_pending_input_tokens':a.worker_pending_tokens,
                    'global_pending_input_tokens':a.global_pending_tokens},
                'batching':'single sample; no dynamic batching',
                'optimizer':'fresh AdamW and restored identical checkpoint weights per arm'
            } if a.bounded_root_speculative_sweep else None),
            source_checkpoint_sha256_before=before,
            source_checkpoint_sha256_after=hashlib.sha256(ckpt.read_bytes()).hexdigest(),
            average={key:statistics.mean(e[key] for e in measured_events) if all(key in e for e in measured_events) else None for key in
                ('fetch_seconds','wait_seconds','transfer_seconds','generation_seconds','export_seconds','forward_seconds','backward_seconds')})
        assert result['source_checkpoint_sha256_after']==before
        (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    except BaseException as error:
        (a.output/'failure.json').write_text(json.dumps({'error':repr(error)},indent=2)+'\n')
        raise
    finally:
        phase['name']='shutdown'
        if pool:pool.close()
        stop.set();thread.join(timeout=5)

if __name__=='__main__':main()
