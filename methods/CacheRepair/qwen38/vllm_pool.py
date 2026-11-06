"""Ordered asynchronous vLLM TP2 supervision, with bounded BF16 CUDA IPC.

All replicas, including the pair sharing GPU0 with the repairer, run in separate
processes. Only metadata crosses CPU IPC. The training process owns canonical
FP32 supervision. Producer buffers remain live until GPU copies complete and
explicit acknowledgement is received. No supervision is written to disk.
"""
from __future__ import annotations
from collections import deque
from concurrent.futures import ProcessPoolExecutor, Future
import threading
import json
import math
import multiprocessing
import os
from pathlib import Path
import time
import uuid
import torch
from .cache import canonicalize_target_kv, reposition_target_kv
from .resume import fingerprint
from .vllm_capture import CaptureWorkerExtension


def remap_cuda_descriptor(descriptor, device):
    # Version-checked descriptor, not a pickled CUDA tensor: vLLM workers see
    # pair-local CUDA ordinals while the consumer sees the whole allocation.
    rebuild, args = descriptor
    from torch.multiprocessing.reductions import rebuild_cuda_tensor
    if rebuild is not rebuild_cuda_tensor or len(args) != 15:
        raise RuntimeError('Unsupported PyTorch CUDA IPC descriptor')
    args = list(args)
    args[6] = int(device)
    return rebuild, tuple(args)


class TrainingCaptureWorker(CaptureWorkerExtension):
    def cr_capture_begin(self):
        from vllm.forward_context import get_forward_context
        import re
        self._cr_capture, self._cr_prompts = {}, {}
        self._cr_batch_spans = None
        self._cr_batch_stats = dict(batches=0,max_tokens=0,max_requests=0)
        if not hasattr(self, '_cr_hooks'):
            self._cr_hooks = []
            for name, module in self.model_runner.model.named_modules():
                if not (name.endswith('.self_attn.attn') and hasattr(module, 'num_kv_heads')):
                    continue
                layer = int(re.search(r'\.layers\.(\d+)\.', '.'+name).group(1))
                def capture(module, args, layer=layer):
                    if self._cr_capture is None:
                        return
                    if self._cr_batch_spans is None:
                        ctx = get_forward_context()
                        metadata = ctx.attn_metadata[module.layer_name]
                        starts = metadata.query_start_loc.cpu().tolist()
                        batch = self.model_runner.input_batch
                        self._cr_batch_stats['batches']+=1
                        self._cr_batch_stats['max_tokens']=max(self._cr_batch_stats['max_tokens'],starts[-1])
                        self._cr_batch_stats['max_requests']=max(self._cr_batch_stats['max_requests'],len(batch.req_ids))
                        spans = []
                        for row, request in enumerate(batch.req_ids):
                            start, end = starts[row:row+2]
                            if (int(batch.num_computed_tokens_cpu[row]) != 0
                                or end-start != int(batch.num_prompt_tokens[row])):
                                raise RuntimeError('Capture requires complete uncached prompt prefill')
                            if request in self._cr_prompts:
                                raise RuntimeError('Repeated request in capture')
                            self._cr_prompts[request] = batch.token_ids_cpu[row,:end-start].tolist()
                            spans.append((request,start,end))
                        self._cr_batch_spans = spans
                    k, v = args[1:3]
                    k = k.reshape(-1,module.num_kv_heads,module.head_size)
                    v = v.reshape_as(k)
                    for request,start,end in self._cr_batch_spans:
                        values = self._cr_capture.setdefault(request,{})
                        if layer in values:
                            raise RuntimeError('Repeated layer/request capture')
                        values[layer] = torch.stack((k[start:end],v[start:end]),dim=-2).detach()
                    # A generate call can require several complete-prefill
                    # batches. Reset metadata after the final attention layer.
                    if layer == 63:
                        self._cr_batch_spans = None
                self._cr_hooks.append(module.register_forward_pre_hook(capture))
            if len(self._cr_hooks)!=16:
                raise ValueError('Expected 16 full-attention hooks')
        return {'hooks':len(self._cr_hooks)}

    def cr_training_trim(self, rank):
        from vllm.distributed import get_tensor_model_parallel_rank
        if get_tensor_model_parallel_rank()==rank:
            torch.cuda.empty_cache()
        return True

    def cr_training_profile_begin(self):
        self._cr_profiler=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA],record_shapes=False)
        self._cr_profiler.__enter__()
        return True

    def cr_training_profile_end(self):
        self._cr_profiler.__exit__(None,None,None)
        ops=self._cr_profiler.key_averages()
        result=[dict(name=o.key,calls=o.count,self_cuda_us=o.self_device_time_total,
            total_cuda_us=o.device_time_total,self_cpu_us=o.self_cpu_time_total)
            for o in sorted(ops,key=lambda o:o.self_device_time_total,reverse=True)[:40]]
        self._cr_profiler=None
        return result

    def cr_training_export(self, batch_id, request_ids, expected_prompts, transport='cuda', verify=False, partition_lengths=None):
        from vllm.distributed import get_tensor_model_parallel_rank
        from torch.multiprocessing.reductions import reduce_tensor
        rank = get_tensor_model_parallel_rank()
        layers = sorted(next(iter(self._cr_capture.values())))
        if len(layers) != 16:
            raise ValueError('Expected 16 attention layers')
        pieces = []
        for request, expected in zip(request_ids, expected_prompts, strict=True):
            matches = [k for k in self._cr_capture if k == request or k.startswith(request+'-')]
            if len(matches) != 1 or self._cr_prompts[matches[0]] != expected:
                raise ValueError('Captured request/token mismatch')
            value = self._cr_capture.pop(matches[0])
            if sorted(value) != layers:
                raise ValueError('Incomplete captured KV')
            pieces.append(torch.stack([value[layer] for layer in layers], dim=1))
        if self._cr_capture:
            raise ValueError('Unexpected additional capture requests')
        packed = torch.cat(pieces, dim=0).contiguous()
        self._cr_capture = None
        self._cr_prompts = {}
        # Each row gets a view/IPC reference into one packed allocation. The
        # allocation stays live until the last row has been acknowledged.
        if not hasattr(self, '_cr_exports'):
            self._cr_exports = {}
        lengths=partition_lengths or [packed.shape[0]]
        if sum(lengths)!=packed.shape[0] or any(length<1 for length in lengths):
            raise ValueError('Invalid batched KV partition lengths')
        parts=[]
        offset=0
        for index,length in enumerate(lengths):
            view=packed[offset:offset+length]
            offset+=length
            row_id=f'{batch_id}:{index}' if partition_lengths is not None else batch_id
            if transport=='cuda':
                self._cr_exports[row_id]=view
                payload=reduce_tensor(view)
            elif transport=='host':
                from multiprocessing import shared_memory
                shm=shared_memory.SharedMemory(create=True,size=view.numel()*view.element_size())
                cpu=torch.frombuffer(shm.buf,dtype=view.dtype).reshape(view.shape)
                cpu.copy_(view)
                self._cr_exports[row_id]=(shm,cpu)
                payload=dict(name=shm.name,shape=list(view.shape))
            else:
                raise ValueError('Unknown transfer route')
            digest=None
            if verify:
                import hashlib
                digest=hashlib.sha256(view.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
            parts.append(dict(rank=rank,layers=layers,payload=payload,sha256=digest,shape=list(view.shape),
                bytes=view.numel()*view.element_size(),prefill_stats=dict(self._cr_batch_stats)))
        return {'parts':parts} if partition_lengths is not None else parts[0]

    def cr_training_release(self, batch_id, trim_rank=None):
        value = self._cr_exports.pop(batch_id)
        if isinstance(value, tuple):
            shm, cpu = value
            del cpu, value
            shm.close()
            shm.unlink()
        else:
            del value
        if trim_rank is not None:
            from vllm.distributed import get_tensor_model_parallel_rank
            if get_tensor_model_parallel_rank()==trim_rank:
                torch.cuda.empty_cache()
        return True


_ENGINE = None

def _load_engine(model_path, physical_devices, options):
    global _ENGINE
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, physical_devices))
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
    os.environ['OMP_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    from vllm import LLM
    _ENGINE = LLM(model=model_path, tensor_parallel_size=2, dtype='bfloat16',
        kv_cache_dtype='auto', quantization=None, max_model_len=16385,
        mamba_ssm_cache_dtype='float32', max_num_seqs=options.get('max_num_seqs',32),
        max_num_batched_tokens=options['max_batched_tokens'],
        gpu_memory_utilization=options['memory_utilization'], kv_cache_memory_bytes=options.get('kv_cache_bytes',1024**3),
        enforce_eager=True, compilation_config={'mode': 0}, language_model_only=True,
        enable_prefix_caching=False, enable_chunked_prefill=False,
        async_scheduling=False, seed=7, distributed_executor_backend='mp',
        worker_extension_cls='methods.CacheRepair.qwen38.vllm_pool.TrainingCaptureWorker')
    return dict(pid=os.getpid(), physical_devices=physical_devices)


def _generate_many(rows, transport, verify=False):
    from vllm import SamplingParams
    groups=[[ *chunks, [token for chunk in chunks for token in chunk]] for chunks in rows]
    prompts=[prompt for group in groups for prompt in group]
    lengths=[sum(map(len,group)) for group in groups]
    started=time.perf_counter()
    batch_id=uuid.uuid4().hex
    _ENGINE.collective_rpc('cr_capture_begin')
    outputs=_ENGINE.generate([{'prompt_token_ids':prompt} for prompt in prompts],
        SamplingParams(temperature=0,max_tokens=1,ignore_eos=True),use_tqdm=False)
    generated=time.perf_counter()
    shards=_ENGINE.collective_rpc('cr_training_export',args=(batch_id,
        [o.request_id for o in outputs],prompts,transport,verify,lengths))
    ended=time.perf_counter()
    result=[]
    offset=0
    for index,group in enumerate(groups):
        weight=lengths[index]/sum(lengths)
        result.append(dict(batch_id=f'{batch_id}:{index}',shards=[rank['parts'][index] for rank in shards],
            producer_batch_id=batch_id,producer_batch_examples=len(rows),producer_batch_tokens=sum(lengths)//2,
            batch_generation_seconds=generated-started,batch_export_seconds=ended-generated,
            produced_unix=time.time(),generation_seconds=(generated-started)*weight,
            export_seconds=(ended-generated)*weight,compute_seconds=(ended-started)*weight,
            first_token_ids=[o.outputs[0].token_ids[0] for o in outputs[offset:offset+len(group)]]))
        offset+=len(group)
    return result


def _generate(chunks, transport, verify=False):
    return _generate_many([chunks],transport,verify)[0]


class _BatchItem:
    def __init__(self,future,index):
        self.future,self.index=future,index
    def done(self):return self.future.done()
    def result(self):return self.future.result()[self.index]


class _BatchDispatcher:
    """Coalesce waiting rows when the actor is free, under parent token credits."""
    def __init__(self,executor,settings):
        self.executor,self.settings=executor,settings
        self.queue=deque();self.condition=threading.Condition();self.stopping=False;self.failure=None
        self.thread=threading.Thread(target=self._run,daemon=True,name='qwen38-vllm-batcher')
        self.thread.start()

    def submit(self,chunks,transport,verify):
        future=Future()
        with self.condition:
            if self.failure:raise self.failure
            if self.stopping:raise RuntimeError('Batch dispatcher is closed')
            self.queue.append((chunks,transport,verify,future))
            self.condition.notify()
        return future

    def _run(self):
        while True:
            with self.condition:
                while not self.queue and not self.stopping:self.condition.wait()
                if not self.queue:return
                # One bounded delay only at an empty->nonempty transition;
                # most coalescing happens while the previous forward runs.
                maximum,budget=self.settings()
                self.condition.wait_for(lambda:self.stopping or len(self.queue)>=maximum,timeout=.005)
                selected=[];tokens=0
                while self.queue and len(selected)<maximum:
                    item=self.queue[0];length=sum(map(len,item[0]))
                    if selected and (tokens+length>budget or item[1]!=selected[0][1]):break
                    selected.append(self.queue.popleft());tokens+=length
            try:
                packets=self.executor.submit(_generate_many,[item[0] for item in selected],
                    selected[0][1],any(item[2] for item in selected)).result()
                if len(packets)!=len(selected):raise ValueError('Missing batched row result')
                for item,packet in zip(selected,packets,strict=True):item[3].set_result(packet)
            except BaseException as error:
                with self.condition:
                    self.failure=error
                    for item in selected:
                        if not item[3].done():item[3].set_exception(error)
                    while self.queue:self.queue.popleft()[3].set_exception(error)
                return

    def close(self):
        with self.condition:
            self.stopping=True;self.condition.notify_all()
        self.thread.join()


def _trim(rank):
    return _ENGINE.collective_rpc('cr_training_trim',args=(rank,))


def _profile_generate(chunks, transport):
    _ENGINE.collective_rpc('cr_training_profile_begin')
    packet=_generate(chunks,transport)
    ops=_ENGINE.collective_rpc('cr_training_profile_end')
    return packet,ops


def _release(batch_id, trim_rank=None):
    return _ENGINE.collective_rpc('cr_training_release', args=(batch_id,trim_rank))


def _shutdown():
    global _ENGINE
    if _ENGINE is not None:
        _ENGINE.llm_engine.engine_core.shutdown()
        _ENGINE = None


def canonical_pair(native, chunks, rotary, *, tile=256):
    """Existing CR coordinates/rounding, with bounded FP32 intermediates."""
    from .cache import _tables, _rotate
    length = sum(map(len, chunks))
    if native.shape[:2] != (1, 2*length):
        raise ValueError('Captured token count differs')
    stale = torch.empty((1,length,*native.shape[2:]),device=native.device,dtype=torch.float32)
    joint = torch.empty_like(stale)
    def convert(source, destination, positions, offset=0):
        local_cos,local_sin = _tables(source,positions,rotary)
        if offset:
            cos,sin = _tables(source,positions+offset,rotary)
            norm = local_cos.square()+local_sin.square()
            shift_cos = (cos*local_cos+sin*local_sin)/norm
            shift_sin = (sin*local_cos-cos*local_sin)/norm
        else:
            cos,sin = local_cos,local_sin
        norm=cos.square()+sin.square()
        inverse_cos,inverse_sin=cos/norm,-sin/norm
        for a in range(0,source.shape[1],tile):
            b=min(source.shape[1],a+tile)
            key=source[:,a:b,...,0,:]
            if offset:
                key=_rotate(key,shift_cos[:,a:b],shift_sin[:,a:b]).to(source.dtype)
            destination[:,a:b,...,0,:].copy_(_rotate(key,inverse_cos[:,a:b],inverse_sin[:,a:b]))
            destination[:,a:b,...,1,:].copy_(source[:,a:b,...,1,:])
    offset=0
    for chunk in chunks:
        positions=torch.arange(len(chunk),device=native.device).unsqueeze(0)
        convert(native[:,offset:offset+len(chunk)],stale[:,offset:offset+len(chunk)],positions,offset)
        offset+=len(chunk)
    convert(native[:,length:],joint,torch.arange(length,device=native.device).unsqueeze(0))
    return dict(stale=stale,joint=joint,
        tokens=torch.tensor([[t for c in chunks for t in c]], device=native.device),
        chunk_ids=torch.tensor([[i for i,c in enumerate(chunks) for _ in c]], device=native.device))


def select_batch(rows,start,max_examples,token_budget):
    """Contiguous independent samples; a single long row is never clipped."""
    selected=[];tokens=0
    for row in rows[start:start+max_examples]:
        length=sum(map(len,row['chunks']))
        if selected and tokens+length>token_budget:break
        selected.append(row);tokens+=length
    return selected


class VllmPairPool:
    asynchronous = True
    def __init__(self, target, geometry, groups, *, model_path, log_path=None,
                 memory_utilization=.72, max_batched_tokens=32768, transport='cuda', verify_first=False, prefetch_per_worker=1, scheduler='round_robin', max_pending_pairs=None, max_batch_examples=1, max_batch_input_tokens=8192, remote_kv_cache_bytes=1024**3, remote_max_num_seqs=32, dynamic_batching=False, max_pending_input_tokens_per_worker=None, max_pending_input_tokens=None):
        self.groups = [tuple(g) for g in groups]
        flattened = [g for group in self.groups for g in group]
        if not self.groups or any(len(g)!=2 for g in self.groups) or len(set(flattened))!=len(flattened):
            raise ValueError('vLLM requires disjoint GPU pairs')
        if transport not in ('cuda', 'host'):
            raise ValueError('Unknown KV transfer route')
        if transport == 'cuda' and torch.cuda.memory.get_allocator_backend() != 'native':
            raise ValueError('CUDA IPC requires PYTORCH_ALLOC_CONF=backend:native')
        if prefetch_per_worker<1 or scheduler not in ('round_robin','least_work'):
            raise ValueError('Invalid vLLM prefetch/scheduler')
        if max_pending_pairs is not None and max_pending_pairs<1:
            raise ValueError('Pending pair limit must be positive')
        if max_batch_examples<1 or max_batch_input_tokens<1 or remote_kv_cache_bytes<1:
            raise ValueError('Batch and cache budgets must be positive')
        self.max_batch_examples,self.max_batch_input_tokens=max_batch_examples,max_batch_input_tokens
        if any(value is not None and value<1 for value in (max_pending_input_tokens_per_worker,max_pending_input_tokens)):
            raise ValueError('Pending token budgets must be positive')
        self.max_pending_input_tokens_per_worker=max_pending_input_tokens_per_worker
        self.max_pending_input_tokens=max_pending_input_tokens
        self.remote_kv_cache_bytes=remote_kv_cache_bytes
        self.remote_max_num_seqs=remote_max_num_seqs
        self.dynamic_batching=dynamic_batching
        self.dispatchers=[]
        self.max_pending_pairs=max_pending_pairs
        self.prefetch_per_worker,self.scheduler=prefetch_per_worker,scheduler
        self.costs={}
        # Optional least-work estimates; token credits stay in self.costs.
        self.work_costs={}
        self.replica_work_scales=None
        self.verify_groups = set(range(len(self.groups))) if verify_first else set()
        self.verify_first = verify_first
        self.device = target.get_input_embeddings().weight.device
        self.geometry, self.rotary = geometry, target.rotary_emb
        self.shared = any(self.device.index in group for group in self.groups)
        self.capacities=[self._capacity(i) for i in range(len(self.groups))]
        visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        devices = visible.split(',') if visible else list(map(str, range(torch.cuda.device_count())))
        self.log_path = Path(log_path) if log_path else None
        self.transport = transport
        self.executors, self.pending = [], deque()
        self.sequence, self.cursor, self.submitted = [], 0, 0
        self.last_event = None
        self.releases = []
        # Wall time accumulated by root-side allocator clears since the last
        # consumed pair; includes initial/set_sequence and refill calls.
        self._root_empty_cache_wall_seconds = 0.0
        self.stream = torch.cuda.Stream(device=self.device)
        try:
            # Initialize sequentially: each engine profiles memory on its pair.
            for group in self.groups:
                executor = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn'))
                self.executors.append(executor)
                executor.submit(_load_engine, model_path, [devices[i] for i in group],
                    dict(memory_utilization=memory_utilization, max_batched_tokens=min(32768,max_batched_tokens) if self.device.index in group else max_batched_tokens,
                         kv_cache_bytes=1024**3 if self.device.index in group else remote_kv_cache_bytes,
                         max_num_seqs=32 if self.device.index in group else remote_max_num_seqs)).result()
            if self.dynamic_batching:
                for executor,group in zip(self.executors,self.groups,strict=True):
                    shared=self.device.index in group
                    self.dispatchers.append(_BatchDispatcher(executor,
                        lambda shared=shared:(1 if shared else self.max_batch_examples,self.max_batch_input_tokens)))
        except BaseException:
            self.close()
            raise

    def _log(self, event):
        if self.log_path:
            with self.log_path.open('a') as stream:
                stream.write(json.dumps(event)+'\n')

    def _trim_rank(self, group):
        return self.groups[group].index(self.device.index) if self.device.index in self.groups[group] else None

    def _capacity(self, group):
        """Maximum queued rows for a pair; root admission is independently opt-in."""
        if self.device.index in self.groups[group]:
            return getattr(self,'shared_prefetch_per_worker',1)
        return self.prefetch_per_worker

    def _empty_root_cache(self):
        """Clear the training process allocator and account for host wall time."""
        started=time.perf_counter()
        torch.cuda.empty_cache()
        self._root_empty_cache_wall_seconds = getattr(self, '_root_empty_cache_wall_seconds', 0.0) + time.perf_counter()-started

    def _queue_release(self,group,batch_id):
        self.releases.append(self.executors[group].submit(_release,batch_id,self._trim_rank(group)))

    def _check_releases(self,wait=False):
        outstanding=[]
        for future in self.releases:
            if wait or future.done():
                future.result()
            else:
                outstanding.append(future)
        self.releases=outstanding

    def _work_cost(self, future):
        # Pools built with __new__ may not define this optional map.
        return getattr(self, 'work_costs', {}).get(future, self.costs[future])

    def _fill(self, *, allow_shared=True):
        attention_weight=getattr(self, 'attention_cost_weight', 0.0)
        if not math.isfinite(attention_weight) or attention_weight < 0:
            raise ValueError('attention_cost_weight must be finite and nonnegative')
        scales=getattr(self, 'replica_work_scales', None)
        if scales is not None:
            if (not isinstance(scales, (list, tuple)) or len(scales) != len(self.groups)
                or any(isinstance(s, bool) or not isinstance(s, (int, float))
                    or not math.isfinite(s) or s <= 0 for s in scales)):
                raise ValueError('replica_work_scales must contain one positive finite number per group')
        self._check_releases()
        # Deeper queues are for replicas with enough independent memory. Work
        # remains consumed in exactly the original sample/optimizer order.
        limit=sum(self._capacity(i) for i in range(len(self.groups)))
        if getattr(self,'max_pending_pairs',None) is not None:
            limit=min(limit,self.max_pending_pairs)
        while self.cursor<len(self.sequence) and len(self.pending)<limit:
            counts=[sum(g==i for _,g,_ in self.pending) for i in range(len(self.groups))]
            next_tokens=sum(map(len,self.sequence[self.cursor]['chunks']))
            queued=[sum(self.costs[f] for _,g,f in self.pending if g==i) for i in range(len(self.groups))]
            global_budget=getattr(self,'max_pending_input_tokens',None)
            worker_budget=getattr(self,'max_pending_input_tokens_per_worker',None)
            shared_budget=getattr(self,'shared_pending_input_tokens',None)
            if global_budget is not None and sum(queued)+next_tokens>global_budget:break
            candidates=[i for i,c in enumerate(counts) if c<self._capacity(i)
                and (worker_budget is None or queued[i]+next_tokens<=worker_budget)
                and (allow_shared or self.device.index not in self.groups[i])
                and (self.device.index not in self.groups[i] or shared_budget is None
                    or queued[i]+next_tokens<=shared_budget)]
            if not candidates:break
            if self.scheduler=='round_robin':
                group=self.submitted%len(self.groups)
                if group not in candidates:
                    root_budget_reject=(self.device.index in self.groups[group]
                        and shared_budget is not None and queued[group]+next_tokens>shared_budget)
                    if (allow_shared and not root_budget_reject) or self.device.index not in self.groups[group]:break
                    # Skip only a root candidate excluded by this refill's
                    # long gate or root token credit; retain remote admission.
                    start=group
                    group=next((i for i in sorted(candidates,key=lambda i:(i-start)%len(self.groups))),None)
                    if group is None:break
            else:
                if attention_weight:
                    work=[sum(self._work_cost(f) for _,g,f in self.pending if g==i and not f.done())
                        for i in range(len(self.groups))]
                else:
                    # At weight 0 retain the existing token-only work estimate.
                    work=[sum(self.costs[f] for _,g,f in self.pending if g==i and not f.done())
                        for i in range(len(self.groups))]
                if scales is not None:
                    if getattr(self, 'include_incoming_work', False):
                        # Compare estimated completion work, not only backlog.
                        # The incoming row's attention estimate mirrors work_costs.
                        incoming_work=next_tokens
                        if attention_weight:
                            lengths=[len(chunk) for chunk in self.sequence[self.cursor]['chunks']]
                            incoming_work += attention_weight*(next_tokens*next_tokens + sum(n*n for n in lengths))
                        work=[(work[i]+incoming_work)*scales[i] for i in range(len(self.groups))]
                    else:
                        # Preserve historical backlog-only heterogeneous selection by default.
                        work=[work[i]*scales[i] for i in range(len(self.groups))]
                group=min(candidates,key=lambda i:(work[i],counts[i],i))
            chunks=self.sequence[self.cursor]['chunks']
            if self.device.index in self.groups[group]:
                # Independent processes cannot reuse each other's idle blocks.
                self._empty_root_cache()
            verify=group in self.verify_groups or getattr(self, 'verify_all', False)
            self.verify_groups.discard(group)
            cap=self._capacity(group)
            max_rows=min(getattr(self,'max_batch_examples',1),cap-counts[group],limit-len(self.pending))
            if self.device.index in self.groups[group] or getattr(self,'dynamic_batching',False):max_rows=1
            budget=getattr(self,'max_batch_input_tokens',8192)
            if worker_budget is not None:budget=min(budget,worker_budget-queued[group])
            if global_budget is not None:budget=min(budget,global_budget-sum(queued))
            selected=select_batch(self.sequence,self.cursor,max_rows,budget)
            if self.device.index in self.groups[group] and shared_budget is not None:
                selected_tokens=sum(sum(map(len,row['chunks'])) for row in selected)
                if queued[group]+selected_tokens>shared_budget:
                    raise RuntimeError('Selected root rows exceed shared pending-token budget')
            if getattr(self,'dynamic_batching',False):
                futures=[self.dispatchers[group].submit(chunks,self.transport,verify)]
            elif len(selected)==1:
                futures=[self.executors[group].submit(_generate,chunks,self.transport,verify)]
            else:
                combined=self.executors[group].submit(_generate_many,[row['chunks'] for row in selected],self.transport,verify)
                futures=[_BatchItem(combined,i) for i in range(len(selected))]
            for row,future in zip(selected,futures,strict=True):
                self.pending.append((fingerprint(row['chunks']),group,future))
                token_cost=sum(map(len,row['chunks']))
                self.costs[future]=token_cost
                if attention_weight:
                    lengths=[len(chunk) for chunk in row['chunks']]
                    attention_work=token_cost*token_cost+sum(length*length for length in lengths)
                    work_costs=getattr(self, 'work_costs', None)
                    if work_costs is None:
                        work_costs={}
                        self.work_costs=work_costs
                    work_costs[future]=token_cost+attention_weight*attention_work
                self.cursor+=1
                self.submitted+=1

    def set_sequence(self, rows):
        for _, group, future in self.pending:
            packet = future.result()
            # Open/release exported CUDA references so counters aren't leaked.
            self._consume(packet, group, None)
        self.pending.clear()
        self.costs.clear()
        getattr(self, 'work_costs', {}).clear()
        self._check_releases(wait=True)
        self.sequence, self.cursor, self.submitted = rows, 0, 0
        self._fill()

    def set_training(self, enabled):
        pass

    def _consume(self, packet, group, chunks):
        from multiprocessing import shared_memory
        local, shared = [], []
        self._copy_event_host_wait_seconds=0.0
        if chunks is None:
            if self.transport == 'cuda':
                for shard in packet['shards']:
                    rebuild,args=remap_cuda_descriptor(shard['payload'],self.groups[group][shard['rank']])
                    discarded=rebuild(*args)
                    del discarded
            self._queue_release(group,packet['batch_id'])
            return None,0.
        start = time.perf_counter()
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            copy_started=torch.cuda.Event(enable_timing=True)
            copy_started.record(self.stream)
            first=packet['shards'][0]['shape']
            native=torch.empty((1,first[0],first[1],4,2,first[-1]),dtype=torch.bfloat16,device=self.device)
            for shard in sorted(packet['shards'], key=lambda s:s['rank']):
                if shard['layers'] != list(self.geometry.full_attention_layer_indices):
                    raise ValueError('vLLM layer mapping differs from target')
                if self.transport == 'cuda':
                    rebuild, args = remap_cuda_descriptor(shard['payload'], self.groups[group][shard['rank']])
                    source = rebuild(*args)
                    # Always clone the colocated rank too: producer lifetime
                    # ends after this copy, before the repairer consumes it.
                    dest = native[0,:,:,shard['rank']*2:(shard['rank']+1)*2]
                    dest.copy_(source, non_blocking=True)
                else:
                    payload = shard['payload']
                    shm = shared_memory.SharedMemory(name=payload['name'])
                    shared.append(shm)
                    source = torch.frombuffer(shm.buf, dtype=torch.bfloat16).reshape(payload['shape'])
                    dest = native[0,:,:,shard['rank']*2:(shard['rank']+1)*2]
                    dest.copy_(source,non_blocking=True)
                local.append(dest)
                del source
            copied = torch.cuda.Event(enable_timing=True)
            copied.record(self.stream)
            pair = canonical_pair(native, chunks, self.rotary)
            done = torch.cuda.Event(enable_timing=True)
            done.record(self.stream)
        self.last_stage_events=(copy_started,copied,done)
        copy_wait_started=time.perf_counter()
        copied.synchronize()
        self._copy_event_host_wait_seconds=time.perf_counter()-copy_wait_started
        for shard, dest in zip(sorted(packet['shards'], key=lambda s:s['rank']), local, strict=True):
            if shard.get('sha256'):
                import hashlib
                actual=hashlib.sha256(dest.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
                if actual != shard['sha256']:
                    raise ValueError('CUDA IPC changed captured BF16 KV bytes')
        transfer_seconds = time.perf_counter()-start
        # No producer allocation is freed while DMA is still reading it.
        self._queue_release(group,packet['batch_id'])
        for shm in shared:
            shm.close()
        torch.cuda.current_stream(self.device).wait_event(done)
        # The default stream consumes tensors allocated by the copy stream.
        if pair:
            for value in pair.values():
                value.record_stream(torch.cuda.current_stream(self.device))
        return pair, transfer_seconds

    def get(self, chunks):
        self._long_pretrim_wait_seconds=0.0
        self._long_pretrim_trim_rpc_seconds=0.0
        self._long_safety_wait_seconds=0.0
        self._long_safety_trim_rpc_seconds=0.0
        self._copy_event_host_wait_seconds=0.0
        if self.shared:
            self._empty_root_cache()
        key = fingerprint(chunks)
        if not self.pending or self.pending[0][0]!=key:
            self.set_sequence([{'chunks':chunks}])
        started = time.perf_counter()
        if self.shared and sum(map(len,chunks))>=8192:
            shared_group=next(i for i,g in enumerate(self.groups) if self.device.index in g)
            pretrim_wait_started=time.perf_counter()
            for _,g,pending in self.pending:
                if g==shared_group:
                    pending.result()
            self._long_pretrim_wait_seconds=time.perf_counter()-pretrim_wait_started
            pretrim_trim_started=time.perf_counter()
            self.executors[shared_group].submit(_trim,self._trim_rank(shared_group)).result()
            self._long_pretrim_trim_rpc_seconds=time.perf_counter()-pretrim_trim_started
        _, group, future = self.pending.popleft()
        self.costs.pop(future,None)
        getattr(self, 'work_costs', {}).pop(future,None)
        future_wait_started=time.perf_counter()
        packet = future.result()
        future_result_wait_seconds=time.perf_counter()-future_wait_started
        waited = time.perf_counter()-started
        pair, transfer_seconds = self._consume(packet, group, chunks)
        self.last_event = dict(replica=group, gpus=self.groups[group], tokens=pair['tokens'].numel(),
            pair_key=key, wait_seconds=waited, producer_ready_wait_seconds=time.time()-packet['produced_unix']-transfer_seconds, transfer_seconds=transfer_seconds,
            fetch_seconds=time.perf_counter()-started, **{k:packet[k] for k in
                ('generation_seconds','export_seconds','compute_seconds')},
            transport_bytes=sum(s['bytes'] for s in packet['shards']), transport=self.transport, transport_verified=all(s.get('sha256') for s in packet['shards']),
            producer_batch_examples=packet.get('producer_batch_examples',1),
            producer_batch_id=packet.get('producer_batch_id',packet['batch_id']),
            producer_batch_tokens=packet.get('producer_batch_tokens',pair['tokens'].numel()),
            first_token_ids=packet.get('first_token_ids'),
            capture_sha256=[shard.get('sha256') for shard in sorted(packet['shards'], key=lambda shard:shard.get('rank',0))],
            prefill_stats=packet['shards'][0].get('prefill_stats'))
        # Opt-in, per-pool candidate: defer only this refill's shared-GPU
        # admission after a long pair. All existing memory guards still run.
        defer_shared = (getattr(self,'defer_shared_on_long',False)
            and pair['tokens'].numel()>=8192)
        self._fill(allow_shared=not defer_shared)
        safety_started=time.perf_counter()
        if self.shared and pair['tokens'].numel()>=8192:
            # Avoid concurrent peak memory from a long 27B prefill and long
            # repairer backward in independent processes on the same L40.
            shared_group=next(i for i,g in enumerate(self.groups) if self.device.index in g)
            safety_wait_started=time.perf_counter()
            for _,group,pending in self.pending:
                if group==shared_group:
                    pending.result()
            self._long_safety_wait_seconds=time.perf_counter()-safety_wait_started
            safety_trim_started=time.perf_counter()
            self.executors[shared_group].submit(_trim,self._trim_rank(shared_group)).result()
            self._long_safety_trim_rpc_seconds=time.perf_counter()-safety_trim_started
        self.last_event['shared_safety_wait_seconds']=time.perf_counter()-safety_started
        self.last_event['fetch_seconds']=time.perf_counter()-started
        self.last_event.update(pending_examples=len(self.pending),pending_input_tokens=sum(self.costs[f] for _,_,f in self.pending),
            ready_examples=sum(f.done() for _,_,f in self.pending),
            ready_input_tokens=sum(self.costs[f] for _,_,f in self.pending if f.done()),
            # New fields are host wall seconds. Existing wait/fetch/safety fields
            # retain their historical scopes and are intentionally unchanged.
            root_empty_cache_wall_seconds=self._root_empty_cache_wall_seconds,
            long_pretrim_wait_seconds=self._long_pretrim_wait_seconds,
            long_pretrim_trim_rpc_wall_seconds=self._long_pretrim_trim_rpc_seconds,
            future_result_wait_seconds=future_result_wait_seconds,
            consume_copy_event_host_wait_seconds=self._copy_event_host_wait_seconds,
            long_safety_wait_seconds=self._long_safety_wait_seconds,
            long_safety_trim_rpc_wall_seconds=self._long_safety_trim_rpc_seconds)
        self._root_empty_cache_wall_seconds=0.0
        self._log(self.last_event)
        return pair

    def close(self):
        error = None
        try:
            if self.pending:
                self.set_sequence([])
        except BaseException as caught:
            error = caught
        for dispatcher in getattr(self,'dispatchers',[]):dispatcher.close()
        for executor in self.executors:
            try:
                executor.submit(_shutdown).result(timeout=90)
            except BaseException as caught:
                error = error or caught
            finally:
                executor.shutdown(wait=True, cancel_futures=True)
        self.executors.clear()
        if error:
            raise error
