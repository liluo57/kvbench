"""Compare old single-row vLLM supervision with enlarged-cache cross-row batches.

KV stays in GPU memory. Only metrics and fixed input IDs are written. No optimizer.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def compare(a,b,sigma):
    import torch
    if any(not torch.equal(a[k],b[k]) for k in ('tokens','chunk_ids')):
        raise ValueError('Batched tokens/chunk boundaries differ')
    sums=torch.zeros(3,5,device='cuda:0',dtype=torch.float64)
    maximum=torch.zeros(3,device='cuda:0',dtype=torch.float64)
    normalized=torch.zeros(2,device='cuda:0',dtype=torch.float64)
    for start in range(0,a['tokens'].shape[1],256):
        sa,ja=a['stale'][:,start:start+256],a['joint'][:,start:start+256]
        sb,jb=b['stale'][:,start:start+256],b['joint'][:,start:start+256]
        for i,(x,y) in enumerate(((sa,sb),(ja,jb),(ja-sa,jb-sb))):
            x,y=x.double(),y.double();error=y-x
            sums[i]+=torch.stack((x.square().sum(),y.square().sum(),error.square().sum(),(x*y).sum(),x.new_tensor(x.numel())))
            maximum[i]=torch.maximum(maximum[i],error.abs().max())
        truth=(ja-sa)/sigma;change=((jb-sb)-(ja-sa))/sigma
        normalized+=torch.stack((truth.double().square().sum(),change.double().square().sum()))
    out={}
    for name,v,mx in zip(['stale','joint','delta'],sums.cpu().tolist(),maximum.cpu().tolist(),strict=True):
        aa,bb,ee,ab,n=v
        out[name]=dict(bitwise_equal=mx==0,max_abs=mx,relative_l2=(ee/max(aa,1e-30))**.5,
            cosine=ab/max((aa*bb)**.5,1e-30),rmse=(ee/n)**.5)
    norm=normalized.cpu().tolist()
    out['normalized_delta_label_error_to_zero_error']=norm[1]/max(norm[0],1e-30)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--audit-rows',type=Path,required=True)
    p.add_argument('--train-rows',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    import torch
    from safetensors import safe_open
    from methods.CacheRepair.qwen38.vllm_target import VllmTarget
    from methods.CacheRepair.qwen38.target import TargetGeometry
    from methods.CacheRepair.qwen38.vllm_pool import VllmPairPool,_generate,_generate_many
    torch.set_num_threads(1);torch.cuda.set_device(0)
    target=VllmTarget(a.model,'cuda:0');geo=TargetGeometry.from_target(target)
    with safe_open(a.checkpoint/'model.safetensors',framework='pt',device='cpu') as f:
        sigma=f.get_tensor('sigma_delta').cuda()
    full=[json.loads(s) for s in a.audit_rows.read_text().splitlines() if s.strip()]
    train=[json.loads(s) for s in a.train_rows.read_text().splitlines() if s.strip()]
    short=[train[i] for i in [0,1,4,6,10,12,13,20]]
    baseline=candidate=None
    try:
        baseline=VllmPairPool(target,geo,[(2,3)],model_path=a.model,remote_kv_cache_bytes=1024**3)
        candidate=VllmPairPool(target,geo,[(4,5)],model_path=a.model,remote_kv_cache_bytes=4*1024**3,max_batched_tokens=65536)
        reference=[];reference_meta=[]
        for row in [*full,*short]:
            packet=baseline.executors[0].submit(_generate,row['chunks'],'cuda',True).result()
            pair,_=baseline._consume(packet,0,row['chunks'])
            torch.cuda.current_stream(0).synchronize()
            reference.append(pair)
            reference_meta.append(dict(first_token_ids=packet['first_token_ids'],stats=packet['shards'][0]['prefill_stats']))
        results=[];offset=0
        for group in [full,short]:
            packets=candidate.executors[0].submit(_generate_many,[r['chunks'] for r in group],'cuda',True).result()
            for row,packet in zip(group,packets,strict=True):
                pair,_=candidate._consume(packet,0,row['chunks'])
                torch.cuda.current_stream(0).synchronize()
                metrics=compare(reference[offset],pair,sigma)
                ids=reference_meta[offset]['first_token_ids'];new_ids=packet['first_token_ids']
                entry=dict(index=offset,tokens=pair['tokens'].numel(),batch_examples=len(group),
                    first_token_matches=sum(x==y for x,y in zip(ids,new_ids,strict=True)),first_token_total=len(ids),
                    baseline_prefill=reference_meta[offset]['stats'],candidate_prefill=packet['shards'][0]['prefill_stats'],**metrics)
                results.append(entry);print(json.dumps(entry),flush=True)
                del pair;reference[offset]=None;offset+=1
        baseline._check_releases(wait=True);candidate._check_releases(wait=True)
        report=dict(status='completed',optimizer_updates=0,kv_saved=False,checkpoint_written=False,
            input_tokens=sum(r['tokens'] for r in results),rows=results,
            first_token_matches=sum(r['first_token_matches'] for r in results),first_token_total=sum(r['first_token_total'] for r in results),
            baseline=dict(cache_gib=1,max_batched_tokens=32768,batch_examples=1),
            candidate=dict(cache_gib=4,max_batched_tokens=65536,batch_examples=[4,8]),
            note='Batching is checked for numerical drift; bitwise equality to the old batching is reported, never assumed.')
        (a.output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    finally:
        if baseline:baseline.close()
        if candidate:candidate.close()

if __name__=='__main__':main()
