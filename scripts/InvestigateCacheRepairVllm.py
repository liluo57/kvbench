"""Read-only 27B forward, generation, trace, and one-variable controls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    import torch
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('backend',choices=['native','vllm','vllm-controls','native-controls','native-consistency'])
    p.add_argument('--model',required=True)
    p.add_argument('--rows',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tokens',type=int,default=32)
    p.add_argument('--control-modes',default='baseline,beta,norm,both,both_fla,strict,aligned_strict,aligned_conv')
    a=p.parse_args();torch.set_num_threads(1)
    rows=[json.loads(s) for s in a.rows.read_text().splitlines()]
    out=a.output/a.backend;out.mkdir(parents=True,exist_ok=False)
    if a.backend=='native-consistency':run_native_consistency(a,rows,out)
    elif a.backend=='native-controls':run_native_controls(a,rows,out)
    elif a.backend=='native':run_native(a,rows,out)
    else:run_vllm(a,rows,out)


def run_native(a,rows,out):
    import torch
    from methods.CacheRepair.qwen38.train import load_native_target
    from methods.CacheRepair.qwen38.target import text_backbone,TargetGeometry
    from methods.CacheRepair.qwen38.cache import pack_target_kv
    from methods.CacheRepair.qwen38.supervision import make_pair
    target=load_native_target(a.model,(0,1),attention_backend='sdpa',require_fast=True)
    backbone=text_backbone(target);geometry=TargetGeometry.from_target(target,target_model_id='Qwen/Qwen3.8-27B')
    trace={};active=[False]
    def save(name,value,full=False):
        if not active[0]:return
        if isinstance(value,tuple):value=value[0]
        if value.ndim>=3:value=value[0]
        trace[name]=value.detach()[:min(64,value.shape[0])].cpu() if not full else value.detach().cpu()
    backbone.embed_tokens.register_forward_hook(lambda m,i,o:save('embedding',o))
    for l,layer in enumerate(backbone.layers):
        for n in ['input_layernorm','post_attention_layernorm']:
            getattr(layer,n).register_forward_hook(lambda m,i,o,l=l,n=n:save(f'layer.{l}.{n}',o))
    layer=backbone.layers[0].linear_attn
    for n in ['in_proj_qkv','in_proj_z','in_proj_b','in_proj_a']:
        getattr(layer,n).register_forward_hook(lambda m,i,o,n=n:save('gdn0.'+n,o,True))
    layer.out_proj.register_forward_pre_hook(lambda m,i:save('gdn0.gated',i[0],True))
    layer.out_proj.register_forward_hook(lambda m,i,o:save('gdn0.output',o,True))
    original=layer.chunk_gated_delta_rule
    def delta(q,k,v,**kwargs):
        if active[0]:
            for key,value in [('q',q),('k',k),('v',v),('g',kwargs['g']),('beta',kwargs['beta'])]:
                trace['gdn0.'+key]=value.detach().cpu()
        result=original(q,k,v,**kwargs)
        if active[0]:trace['gdn0.core']=result[0].detach().cpu()
        return result
    layer.chunk_gated_delta_rule=delta
    results=[]
    with torch.no_grad():
        for i,row in enumerate(rows):
            tokens=torch.tensor([sum(row['chunks'],[])],device='cuda:0');length=tokens.shape[1]
            active[0]=True;trace.clear()
            output=target(input_ids=tokens,attention_mask=torch.ones_like(tokens),position_ids=torch.arange(length,device=tokens.device)[None],use_cache=True,logits_to_keep=1)
            active[0]=False
            raw=pack_target_kv(output.past_key_values,geometry).cpu()
            logits=output.logits[0,-1].float().cpu()
            torch.save({'raw_kv':raw,'logits':logits,'trace':dict(trace)},out/f'prefill-{i}.pt')
            sequence=[];past=output.past_key_values
            for j in range(a.tokens):
                token=int(output.logits[0,-1].argmax());sequence.append(token)
                if j+1<a.tokens:
                    output=target(input_ids=torch.tensor([[token]],device=tokens.device),
                        attention_mask=torch.ones(1,length+j+1,device=tokens.device,dtype=torch.long),
                        position_ids=torch.tensor([[length+j]],device=tokens.device),
                        past_key_values=past,use_cache=True,logits_to_keep=1)
                    past=output.past_key_values
            del output,past,raw
            # Compare generation against full-prefix teacher forcing as well.
            forced=torch.cat((tokens,torch.tensor([sequence],device=tokens.device)),dim=1)
            output=target(input_ids=forced,use_cache=False,logits_to_keep=a.tokens+1)
            torch.save(output.logits[0].float().cpu(),out/f'teacher-forced-{i}.pt')
            del output
            record=dict(row=i,tokens=length,generated_ids=sequence)
            results.append(record);print(json.dumps(record),flush=True)
        # First sample controls preserve actual tokens and use new caches.
        row=rows[0];tokens=torch.tensor([sum(row['chunks'],[])],device='cuda:0')
        for label,kwargs in [('repeat',{}),('no_mask',{'attention_mask':None}),('no_cache',{'use_cache':False})]:
            options=dict(input_ids=tokens,attention_mask=torch.ones_like(tokens),position_ids=torch.arange(tokens.shape[1],device=tokens.device)[None],use_cache=True,logits_to_keep=1)
            options.update(kwargs);result=target(**options)
            value={'logits':result.logits[0,-1].float().cpu()}
            if options['use_cache']:value['raw_kv']=pack_target_kv(result.past_key_values,geometry).cpu()
            torch.save(value,out/f'control-{label}.pt');del result
        # The native pair adapter must reproduce a direct official forward.
        pair=make_pair(target,row['chunks'],geometry=geometry,chunk_batch_size=8,pack_chunks=False)
        torch.save({k:v.cpu() for k,v in pair.items()},out/'control-pair.pt');del pair
        # Make only beta sigmoid FP32; keep weights, normalization and FLA.
        handles=[l.linear_attn.in_proj_b.register_forward_hook(lambda m,i,o:o.float())
                 for l in backbone.layers if hasattr(l,'linear_attn')]
        result=target(input_ids=tokens,attention_mask=torch.ones_like(tokens),use_cache=True,logits_to_keep=1)
        torch.save({'raw_kv':pack_target_kv(result.past_key_values,geometry).cpu(),
                    'logits':result.logits[0,-1].float().cpu()},out/'control-beta32.pt')
        for h in handles:h.remove()
    (out/'generation.json').write_text(json.dumps(results,indent=2)+'\n')
    torch.save({n:p.detach().cpu() for n,p in layer.named_parameters()},out/'layer0-weights.pt')


def run_vllm(a,rows,out):
    import torch
    from vllm import LLM,SamplingParams
    controls=a.backend=='vllm-controls'
    engine=LLM(model=a.model,tensor_parallel_size=2,dtype='bfloat16',mamba_ssm_cache_dtype='float32',
        kv_cache_dtype='auto',quantization=None,max_model_len=1024 if controls else 16449,max_num_seqs=8 if controls else 40,max_num_batched_tokens=4096 if controls else 32768,
        gpu_memory_utilization=.80,enforce_eager=True,compilation_config={'mode':0},language_model_only=True,
        enable_prefix_caching=False,enable_chunked_prefill=False,async_scheduling=False,seed=7,
        worker_extension_cls='methods.CacheRepair.qwen38.vllm_investigation.InvestigationWorker')
    results=[]
    for i,row in enumerate([] if controls else rows):
        tokens=sum(row['chunks'],[]);folder=out/f'prefill-{i}';folder.mkdir()
        engine.collective_rpc('cr_probe_begin')
        first=engine.generate([{'prompt_token_ids':tokens}],SamplingParams(temperature=0,max_tokens=1,ignore_eos=True),use_tqdm=False)[0]
        engine.collective_rpc('cr_probe_export',args=(str(folder.resolve()),))
        generated=engine.generate([{'prompt_token_ids':tokens}],SamplingParams(temperature=0,max_tokens=a.tokens,ignore_eos=True),use_tqdm=False)[0]
        record=dict(row=i,tokens=len(tokens),prefill_first_ids=list(first.outputs[0].token_ids),generated_ids=list(generated.outputs[0].token_ids),request_id=first.request_id)
        results.append(record);print(json.dumps(record),flush=True)
        (out/'generation.json').write_text(json.dumps(results,indent=2)+'\n')
        # All prefix positions share native generated history; retain vLLM's
        # scored prompt tokens and top-1 to avoid free-run divergence ambiguity.
        native=json.loads((a.output/'native/generation.json').read_text()) if (a.output/'native/generation.json').exists() else None
        if native:
            forced=tokens+native[i]['generated_ids']
            folder=out/f'teacher-forced-{i}';folder.mkdir()
            engine.collective_rpc('cr_teacher_begin',args=(a.tokens+1,))
            engine.generate([{'prompt_token_ids':forced}],SamplingParams(temperature=0,max_tokens=1,ignore_eos=True),use_tqdm=False)
            engine.collective_rpc('cr_teacher_export',args=(str(folder.resolve()),))
    (out/'generation.json').write_text(json.dumps(results,indent=2)+'\n')
    for mode in a.control_modes.split(','):
        engine.collective_rpc('cr_ablation',args=(mode,))
        folder=out/f'control-{mode}';folder.mkdir()
        engine.collective_rpc('cr_probe_begin')
        generated=engine.generate([{'prompt_token_ids':sum(rows[0]['chunks'],[])}],SamplingParams(temperature=0,max_tokens=1,ignore_eos=True),use_tqdm=False)[0]
        engine.collective_rpc('cr_probe_export',args=(str(folder.resolve()),))
        (folder/'request.json').write_text(json.dumps(dict(request_id=generated.request_id,token_ids=list(generated.outputs[0].token_ids))))


def run_native_controls(a,rows,out):
    import torch
    from methods.CacheRepair.qwen38.train import load_native_target
    from methods.CacheRepair.qwen38.supervision import make_pair
    from methods.CacheRepair.qwen38.target import TargetGeometry
    from methods.CacheRepair.qwen38.cache import pack_target_kv
    target=load_native_target(a.model,(0,1),attention_backend='sdpa',require_fast=True)
    geometry=TargetGeometry.from_target(target,target_model_id='Qwen/Qwen3.8-27B')
    with torch.no_grad():
        for i,row in enumerate(rows[:2]):
            pair=make_pair(target,row['chunks'],geometry=geometry,chunk_batch_size=1,pack_chunks=False)
            torch.save({k:v.cpu() for k,v in pair.items()},out/f'batch1-pair-{i}.pt');del pair
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
        row=rows[0];tokens=torch.tensor([sum(row['chunks'],[])],device='cuda:0')
        value=target(input_ids=tokens,attention_mask=torch.ones_like(tokens),use_cache=True,logits_to_keep=1)
        torch.save(dict(raw_kv=pack_target_kv(value.past_key_values,geometry).cpu(),logits=value.logits[0,-1].float().cpu()),out/'strict.pt')


def run_native_consistency(a,rows,out):
    import torch
    from methods.CacheRepair.qwen38.train import load_native_target
    from methods.CacheRepair.qwen38.target import text_backbone
    target=load_native_target(a.model,(0,1),attention_backend='sdpa',require_fast=True)
    sequence=json.loads((a.output/'native/generation.json').read_text())[0]['generated_ids']
    tokens=torch.tensor([sum(rows[0]['chunks'],[])],device='cuda:0');length=tokens.shape[1]
    settings=[('baseline',True,False,'sdpa'),('strict',False,False,'sdpa'),
              ('strict_beta32',False,True,'sdpa'),('eager',True,False,'eager')]
    with torch.no_grad():
        for label,reduced,beta32,attention in settings:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=reduced
            backbone=text_backbone(target);backbone.config._attn_implementation=attention
            handles=[l.linear_attn.in_proj_b.register_forward_hook(lambda m,i,o:o.float())
                     for l in backbone.layers if hasattr(l,'linear_attn')] if beta32 else []
            value=target(input_ids=tokens,attention_mask=torch.ones_like(tokens),use_cache=True,logits_to_keep=1)
            cached=[]
            for j in range(len(sequence)):
                cached.append(value.logits[0,-1].float().cpu())
                if j+1<len(sequence):
                    value=target(input_ids=torch.tensor([[sequence[j]]],device=tokens.device),
                        attention_mask=torch.ones(1,length+j+1,device=tokens.device,dtype=torch.long),
                        position_ids=torch.tensor([[length+j]],device=tokens.device),past_key_values=value.past_key_values,use_cache=True,logits_to_keep=1)
            del value
            forced=torch.cat((tokens,torch.tensor([sequence],device=tokens.device)),dim=1)
            full=target(input_ids=forced,use_cache=False,logits_to_keep=len(sequence)+1).logits[0,:-1].float().cpu()
            torch.save(dict(cached=torch.stack(cached),full=full,expected_ids=sequence),out/f'{label}.pt')
            for h in handles:h.remove()
            print(json.dumps(dict(control=label,top1_equal=int((torch.stack(cached).argmax(-1)==full.argmax(-1)).sum()))),flush=True)


if __name__=='__main__':main()
