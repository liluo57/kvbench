"""Geometry, coordinate protocol and CUDA-device mapping for vLLM supervision."""
from types import SimpleNamespace
import pytest
import torch
from methods.CacheRepair.qwen38.cache import canonicalize_target_kv,reposition_target_kv
from methods.CacheRepair.qwen38.target import TargetGeometry
from methods.CacheRepair.qwen38.vllm_pool import canonical_pair,remap_cuda_descriptor
from tests.backends.test_cacherepair_qwen38_native import native


def test_embedding_only_target_has_native_geometry(native):
    target,_=native
    facade=SimpleNamespace(_cr_vllm_target=True,config=SimpleNamespace(text_config=target.config),
        rotary_emb=target.model.rotary_emb,model_id='tiny',get_input_embeddings=target.get_input_embeddings)
    assert TargetGeometry.from_target(facade)==TargetGeometry.from_target(target,target_model_id='tiny')
    facade.get_input_embeddings=lambda:torch.nn.Embedding(4,5)
    with pytest.raises(ValueError,match='embedding'):
        TargetGeometry.from_target(facade)


@pytest.mark.parametrize("tile", [1, 127, 256])
def test_vllm_chunks_preserve_bf16_rerotation_and_chunk_ids(native,tile):
    target,_=native
    rotary=target.model.rotary_emb
    chunks=[list(range(1,135)),[2,3,4],list(range(1,302))]
    length=sum(map(len,chunks))
    raw=torch.randn(1,2*length,2,2,2,16).bfloat16()
    actual=canonical_pair(raw,chunks,rotary,tile=tile)
    stale=[]
    offset=0
    for chunk in chunks:
        positions=torch.arange(len(chunk))[None]
        kv=raw[:,offset:offset+len(chunk)]
        if offset:kv=reposition_target_kv(kv,positions,positions+offset,rotary)
        stale.append(canonicalize_target_kv(kv,positions+offset,rotary))
        offset+=len(chunk)
    assert torch.equal(actual['stale'],torch.cat(stale,dim=1))
    assert torch.equal(actual['joint'],canonicalize_target_kv(raw[:,length:],torch.arange(length)[None],rotary))
    assert actual['tokens'].tolist()==[sum(chunks,[])]
    assert actual['chunk_ids'].tolist()==[[i for i,c in enumerate(chunks) for _ in c]]
    with pytest.raises(ValueError,match='token count'):
        canonical_pair(raw[:,:-1],chunks,rotary)


def test_cuda_ipc_pair_local_ordinal_is_remapped_before_rebuild():
    from torch.multiprocessing.reductions import rebuild_cuda_tensor
    args=tuple(range(15))
    rebuild,mapped=remap_cuda_descriptor((rebuild_cuda_tensor,args),5)
    assert rebuild is rebuild_cuda_tensor
    assert mapped[:6]==args[:6] and mapped[6]==5 and mapped[7:]==args[7:]
    with pytest.raises(RuntimeError):remap_cuda_descriptor((lambda:None,args),3)
    with pytest.raises(RuntimeError):remap_cuda_descriptor((rebuild_cuda_tensor,args[:-1]),3)


def test_prefetch_balances_outstanding_work_without_reordering_rows():
    from collections import deque
    from concurrent.futures import Future
    from methods.CacheRepair.qwen38.vllm_pool import VllmPairPool
    from methods.CacheRepair.qwen38.resume import fingerprint
    pool=VllmPairPool.__new__(VllmPairPool)
    pool.groups=[(2,3),(4,5)]
    pool.device=torch.device('cuda',0)
    pool.prefetch_per_worker=4
    pool.scheduler='least_work'
    pool.pending=deque();pool.costs={};pool.releases=[];pool.cursor=pool.submitted=0
    pool.transport='cuda';pool.verify_groups={0,1}
    pool.sequence=[{'chunks':[[i]*n,[i+1]]} for i,n in enumerate([2,1,16,6,1,16,2,6,12,6,1,16])]
    calls=[]
    class Executor:
        def submit(self,fn,chunks,transport,verify):
            future=Future();calls.append((chunks,verify));return future
    pool.executors=[Executor(),Executor()]
    pool._fill()
    assert len(pool.pending)==8
    assert [key for key,_,_ in pool.pending]==[fingerprint(r['chunks']) for r in pool.sequence[:8]]
    assert [g for _,g,_ in list(pool.pending)[:3]]==[0,1,1]
    assert all(sum(g==i for _,g,_ in pool.pending)<=4 for i in range(2))
    assert sum(verify for _,verify in calls)==2
    # A consumed pair frees one bounded slot; no checkpoint/epoch reset should
    # repeat expensive payload hashes in measured steady-state passes.
    _,_,future=pool.pending.popleft();pool.costs.pop(future)
    pool._fill()
    assert len(pool.pending)==8 and not calls[-1][1]


def test_vllm_embedding_facade_keeps_optimizer_and_resume_protocol(native,tmp_path):
    from methods.CacheRepair.qwen38.vllm_target import VllmTarget
    from methods.CacheRepair.qwen38.train import run_upstream
    from methods.CacheRepair.qwen38.model import RepairConfig
    from methods.CacheRepair.qwen38.supervision import make_pair
    from safetensors.torch import load_file
    target,_=native
    target.requires_grad_(False).eval()
    facade=VllmTarget.__new__(VllmTarget)
    torch.nn.Module.__init__(facade)
    facade.config=SimpleNamespace(text_config=target.config)
    facade.model_id='tiny'
    facade.rotary_emb=target.model.rotary_emb
    facade.embed_tokens=target.get_input_embeddings()
    geometry=TargetGeometry.from_target(target,target_model_id='tiny')
    class Pool:
        asynchronous=True
        last_event={'compute_seconds':.01}
        def set_sequence(self,rows):pass
        def get(self,chunks):return make_pair(target,chunks,geometry=geometry)
    rows=[{'chunks':[[i+1,2],[3,4]]} for i in range(9)]
    options=dict(model_id='tiny',repair_config=RepairConfig(d_b=16,n_heads=2,n_blocks=1,d_seg=2),
        stats_examples=2,device='cpu',validate_train=False,pair_pool=Pool(),checkpoint_steps=1,
        backend_spec={'supervision_backend':'vllm'})
    run_upstream(facade,rows,rows[:2],tmp_path/'complete',**options)
    result=run_upstream(facade,rows,rows[:2],tmp_path/'resumed',stop_after_steps=1,**options)
    assert result['interrupted']
    run_upstream(facade,rows,rows[:2],tmp_path/'resumed',resume_from='auto',**options)
    expected=load_file(str(tmp_path/'complete/checkpoint-3/model.safetensors'))
    actual=load_file(str(tmp_path/'resumed/checkpoint-3/model.safetensors'))
    assert all(torch.equal(actual[k],expected[k]) for k in expected)


def test_colocated_replica_is_shallow_while_remote_queues_are_deep(monkeypatch):
    from collections import deque
    from concurrent.futures import Future
    from methods.CacheRepair.qwen38.vllm_pool import VllmPairPool
    pool=VllmPairPool.__new__(VllmPairPool)
    pool.groups=[(0,1),(2,3),(4,5)];pool.device=torch.device('cuda',0)
    pool.prefetch_per_worker=4;pool.scheduler='least_work';pool.transport='cuda'
    pool.pending=deque();pool.costs={};pool.releases=[];pool.cursor=pool.submitted=0;pool.verify_groups={0,1,2}
    pool.sequence=[{'chunks':[[i]*n,[i+1]]} for i,n in enumerate([2,1,16,6,1,16,2,6,12,6,1,16])]
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    class Executor:
        def submit(self,*args):return Future()
    pool.executors=[Executor(),Executor(),Executor()]
    pool._fill()
    assert len(pool.pending)==9
    assert sum(g==0 for _,g,_ in pool.pending)==1
    assert all(sum(g==i for _,g,_ in pool.pending)<=4 for i in [1,2])


def test_release_ack_cannot_block_training_behind_queued_prefills():
    from concurrent.futures import Future
    from methods.CacheRepair.qwen38.vllm_pool import VllmPairPool
    pool=VllmPairPool.__new__(VllmPairPool)
    pool.groups=[(2,3)];pool.device=torch.device('cuda',0);pool.releases=[]
    class Pending(Future):
        def result(self,*args,**kwargs):
            assert self.done(),'Release acknowledgement blocked the consumer'
            return super().result(*args,**kwargs)
    ack=Pending()
    class Executor:
        def submit(self,*args):return ack
    pool.executors=[Executor()]
    pool._queue_release(0,'pair-id')
    pool._check_releases()
    assert pool.releases==[ack]
    ack.set_result(True)
    pool._check_releases(wait=True)
    assert pool.releases==[]


def test_global_window_does_not_force_equal_per_replica_slot_counts(monkeypatch):
    from collections import deque
    from concurrent.futures import Future
    from methods.CacheRepair.qwen38.vllm_pool import VllmPairPool
    from methods.CacheRepair.qwen38.resume import fingerprint
    pool=VllmPairPool.__new__(VllmPairPool)
    pool.groups=[(0,1),(2,3),(4,5)];pool.device=torch.device('cuda',0)
    pool.prefetch_per_worker=8;pool.max_pending_pairs=9
    pool.scheduler='least_work';pool.transport='cuda'
    pool.pending=deque();pool.costs={};pool.releases=[];pool.cursor=pool.submitted=0;pool.verify_groups=set()
    pool.sequence=[{'chunks':[[i]*n,[i+1]]} for i,n in enumerate([2053,876,16384,6372,200,16384,2465,6463,11856,6524,805,16384])]
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    class Executor:
        def submit(self,*args):return Future()
    pool.executors=[Executor(),Executor(),Executor()]
    pool._fill()
    assert len(pool.pending)==9
    assert sum(g==0 for _,g,_ in pool.pending)==1
    assert max(sum(g==i for _,g,_ in pool.pending) for i in [1,2])>4
    assert [k for k,_,_ in pool.pending]==[fingerprint(r['chunks']) for r in pool.sequence[:9]]


def test_batch_selection_respects_tokens_and_keeps_a_long_row_whole():
    from methods.CacheRepair.qwen38.vllm_pool import select_batch
    rows=[{'chunks':[[i]*n,[i+1]]} for i,n in enumerate([3,4,20,2])]
    assert select_batch(rows,0,8,10)==rows[:2]
    assert select_batch(rows,2,8,10)==rows[2:3]
    assert select_batch(rows,0,1,100)==rows[:1]


def test_batch_items_keep_one_generation_result_in_original_row_order():
    from concurrent.futures import Future
    from methods.CacheRepair.qwen38.vllm_pool import _BatchItem
    future=Future();a,b=_BatchItem(future,0),_BatchItem(future,1)
    assert not a.done() and not b.done()
    future.set_result([{'row':0},{'row':1}])
    assert a.done() and b.done() and a.result()['row']==0 and b.result()['row']==1


def test_batched_export_views_have_independent_release_lifetimes(monkeypatch):
    import sys
    from methods.CacheRepair.qwen38.vllm_pool import TrainingCaptureWorker
    monkeypatch.setitem(sys.modules,'vllm.distributed',SimpleNamespace(get_tensor_model_parallel_rank=lambda:0))
    monkeypatch.setattr(torch.multiprocessing.reductions,'reduce_tensor',lambda t:dict(offset=t.storage_offset(),shape=list(t.shape)))
    worker=TrainingCaptureWorker()
    worker._cr_capture={key:{l:torch.full((n,2,2,256),value,dtype=torch.bfloat16) for l in range(3,64,4)}
        for key,n,value in [('a',2,1),('b',1,2)]}
    worker._cr_prompts={'a':[1,2],'b':[3]}
    worker._cr_batch_stats={'batches':1,'max_tokens':3,'max_requests':2}
    exported=worker.cr_training_export('batch',['a','b'],[[1,2],[3]],partition_lengths=[2,1])
    assert [p['shape'][0] for p in exported['parts']]==[2,1]
    first,second=worker._cr_exports['batch:0'],worker._cr_exports['batch:1']
    assert first.untyped_storage().data_ptr()==second.untyped_storage().data_ptr()
    assert first.storage_offset()==0 and second.storage_offset()==first.numel()
    worker.cr_training_release('batch:0')
    assert 'batch:1' in worker._cr_exports and (worker._cr_exports['batch:1']==2).all()
    worker.cr_training_release('batch:1')
    assert not worker._cr_exports


def test_pending_token_credits_include_finished_but_unconsumed_kv(monkeypatch):
    from collections import deque
    from concurrent.futures import Future
    from methods.CacheRepair.qwen38.vllm_pool import VllmPairPool
    pool=VllmPairPool.__new__(VllmPairPool)
    pool.groups=[(2,3),(4,5)];pool.device=torch.device('cuda',0)
    pool.prefetch_per_worker=8;pool.max_pending_pairs=9;pool.max_batch_examples=8;pool.max_batch_input_tokens=32
    pool.max_pending_input_tokens_per_worker=10;pool.max_pending_input_tokens=10
    pool.scheduler='least_work';pool.transport='cuda'
    pool.pending=deque();pool.costs={};pool.releases=[];pool.cursor=pool.submitted=0;pool.verify_groups=set()
    pool.sequence=[{'chunks':[[i]*5,[i+1]]} for i in range(4)]
    class Executor:
        def submit(self,*args):return Future()
    pool.executors=[Executor(),Executor()]
    pool._fill()
    assert len(pool.pending)==1 and pool.cursor==1
    _,_,future=pool.pending[0];future.set_result({})
    pool._fill()
    assert len(pool.pending)==1 and pool.cursor==1
    pool.pending.popleft();pool.costs.pop(future);pool._fill()
    assert len(pool.pending)==1 and pool.cursor==2


def test_generate_many_keeps_fresh_prompt_groups_and_splits_rank_results(monkeypatch):
    import sys
    from methods.CacheRepair.qwen38 import vllm_pool as module
    calls=[]
    class Engine:
        def generate(self,prompts,params,**kwargs):
            calls.append(prompts)
            return [SimpleNamespace(request_id=f'r{i}',outputs=[SimpleNamespace(token_ids=[100+i])]) for i in range(len(prompts))]
        def collective_rpc(self,method,args=()):
            if method=='cr_capture_begin':return True
            batch_id,ids,prompts,transport,verify,lengths=args
            assert lengths==[6,4] and ids==['r0','r1','r2','r3','r4','r5']
            return [{'parts':[{'rank':rank,'row':i} for i in range(2)]} for rank in range(2)]
    monkeypatch.setitem(sys.modules,'vllm',SimpleNamespace(SamplingParams=lambda **kwargs:kwargs))
    monkeypatch.setattr(module,'_ENGINE',Engine())
    packets=module._generate_many([[[1],[2,3]],[[4],[5]]],'cuda')
    assert [p['prompt_token_ids'] for p in calls[0]]==[[1],[2,3],[1,2,3],[4],[5],[4,5]]
    assert packets[0]['first_token_ids']==[100,101,102] and packets[1]['first_token_ids']==[103,104,105]
    assert [s['row'] for s in packets[0]['shards']]==[0,0]
    assert [s['row'] for s in packets[1]['shards']]==[1,1]
    assert packets[0]['batch_id']!=packets[1]['batch_id']
    assert sum(p['generation_seconds'] for p in packets)==pytest.approx(packets[0]['batch_generation_seconds'])


def test_dynamic_dispatcher_coalesces_rows_arriving_during_forward():
    from concurrent.futures import Future
    import threading
    from methods.CacheRepair.qwen38.vllm_pool import _BatchDispatcher
    started=threading.Event();first=Future();calls=[]
    class Executor:
        def submit(self,fn,rows,transport,verify):
            calls.append(rows)
            if len(calls)==1:
                started.set();return first
            future=Future();future.set_result([{'id':r[0][0]} for r in rows]);return future
    dispatcher=_BatchDispatcher(Executor(),lambda:(4,1000))
    try:
        a=dispatcher.submit([[0],[1]],'cuda',False)
        assert started.wait(2)
        pending=[dispatcher.submit([[i],[i+1]],'cuda',False) for i in range(1,5)]
        first.set_result([{'id':0}])
        assert a.result(timeout=2)['id']==0
        assert [f.result(timeout=2)['id'] for f in pending]==[1,2,3,4]
        assert list(map(len,calls))==[1,4]
    finally:dispatcher.close()
