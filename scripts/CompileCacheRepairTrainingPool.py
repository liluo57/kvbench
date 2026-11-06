"""Compile a clean merged pool using document chunks and first task prefixes.

Original CoRAG chunks stay unchanged. Training and validation must declare the
same document policy. Raw512 is available only as an explicit legacy option.
"""
from __future__ import annotations

import argparse
import collections
from concurrent.futures import ThreadPoolExecutor
import hashlib
from functools import lru_cache
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.MergeCacheRepairTrainingPool import iter_pool
from scripts.PrepareCacheRepairTaskValidation import chunks_for, token_hash, hash_file
from methods.CacheRepair.qwen38.decontam import window_hashes
from methods.CacheRepair.qwen38.sources import local_tokenizer_spec
from methods.CacheRepair.qwen38.document_data import POLICY, compile_documents, task_prefix

LOCAL=threading.local()


def setup_tokenizer(model, protected=None, chunk_policy='documents'):
    from transformers import AutoTokenizer
    LOCAL.tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
    # Different evidence contexts frequently reuse a source document. The
    # formatter only slices encoded lists, so cached token lists stay immutable.
    LOCAL.tokenizer.encode=lru_cache(maxsize=4096)(LOCAL.tokenizer.encode)
    LOCAL.protected=protected
    LOCAL.chunk_policy=chunk_policy


def compile_one(row):
    layout={}
    if 'chunks' in row:
        chunks=row['chunks']
        ids=[token for c in chunks for token in c]
        truncated=False
        layout={'chunk_policy':'legacy-corag-unchanged'}
    elif LOCAL.chunk_policy=='raw512':
        text='\n\n'.join(row['documents'])
        ids=LOCAL.tokenizer.encode(text,add_special_tokens=False,truncation=True,max_length=16385)
        truncated=len(ids)>16384;ids=ids[:16384]
        if len(ids)<2:return None,row
        chunks=chunks_for(ids)
    else:
        try:
            chunks,layout=compile_documents(row['documents'],row['source'],LOCAL.tokenizer)
        except ValueError as error:
            return None,{**row,'exclusion':str(error)}
        ids=[token for c in chunks for token in c]
        truncated=bool(layout['truncated_documents'] or layout['dropped_documents'])
    overlap=(not row['source'].endswith('_shuffle') and LOCAL.protected is not None
        and has_overlap(ids,LOCAL.protected))
    return {'chunks':chunks,'source':row['source'],'example_id':token_hash(ids)}, {
        k:v for k,v in row.items() if k not in ('documents','chunks','document_ids')
    }|layout|{'tokens':len(ids),'truncated_to_16384':truncated,'_heldout_overlap':overlap}


def heldout_index(validation,old,tokenizer):
    from tasks.Musique import MusiqueTask
    from scripts.EvaluateHybridCacheRepair import load_tasks
    texts=[]
    prefix=tokenizer.encode(MusiqueTask.prefixPrompt,add_special_tokens=False)
    for line in (old/'validation.jsonl').open():
        ids=[i for c in json.loads(line)['chunks'] for i in c]
        if ids[:len(prefix)]==prefix:ids=ids[len(prefix):]
        texts.append(ids)
    for line in (validation/'validation.jsonl').open():
        row=json.loads(line)
        if row['task'].endswith('_shuffle'):continue
        ids=[i for c in row['chunks'] for i in c]
        prefix=tokenizer.encode(task_prefix(row['task']),add_special_tokens=False)
        if prefix and ids[:len(prefix)]==prefix:ids=ids[len(prefix):]
        texts.append(ids)
    tasks,_=load_tasks(ROOT/'outputs/hypic-ablation-64-20261014/frozen_tasks.pkl',64)
    for task in tasks:
        if task.name.endswith('_shuffle'):continue
        prefix=tokenizer.encode(getattr(task.original,'prefixPrompt',''),add_special_tokens=False)
        for case in task.cases:
            for chunk in case.input.prepare_input:
                ids=list(chunk)
                if prefix and ids[:len(prefix)]==prefix:ids=ids[len(prefix):]
                texts.append(ids)
    keys=[];owners=[];offsets=[]
    for owner,ids in enumerate(texts):
        hashes=window_hashes(ids)
        keys.append(hashes);owners.append(np.full(len(hashes),owner,dtype=np.uint32))
        offsets.append(np.arange(len(hashes),dtype=np.uint32))
    raw=np.concatenate(keys);order=np.argsort(raw,kind='stable')
    return raw[order],np.concatenate(owners)[order],np.concatenate(offsets)[order],texts


def has_overlap(ids,index):
    keys,owners,offsets,texts=index
    if not len(keys):return False
    hashes=window_hashes(ids)
    loc=np.searchsorted(keys,hashes)
    safe=np.minimum(loc,len(keys)-1)
    hits=np.flatnonzero((loc<len(keys))&(keys[safe]==hashes))
    for position in hits:
        left=int(loc[position]);right=int(np.searchsorted(keys,hashes[position],side='right'))
        for at in range(left,right):
            owner,start=int(owners[at]),int(offsets[at])
            if texts[owner][start:start+50]==ids[position:position+50]:return True
    return False


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('pool','validation','old-data','model','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--chunk-policy',choices=('documents','raw512'),default='documents')
    a=p.parse_args()
    for name in ('pool','validation','old_data','model','output'):setattr(a,name,getattr(a,name).resolve())
    a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'completion-audit.json').exists():return
    while not (a.pool/'completion-audit.json').exists():time.sleep(10)
    pool_meta=json.loads((a.pool/'dataset.json').read_text())
    validation_meta=json.loads((a.validation/'dataset.json').read_text())
    if a.chunk_policy=='documents' and validation_meta.get('chunk_policy')!=POLICY:
        raise ValueError('Reformat validation with the shared document/prefix policy first')
    os.environ['TOKENIZERS_PARALLELISM']='false'
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    protected=heldout_index(a.validation,a.old_data,tokenizer)
    # Start from scratch in staging after an incomplete compiler invocation.
    tmp=a.output/'writing';tmp.mkdir(exist_ok=True)
    counts=collections.Counter();drops=collections.Counter();tokens=collections.Counter();seen=set();total=0
    actual_max_chunk=0;actual_max_length=0;actual_max_chunks=0
    manifest=tmp/'examples.manifest.jsonl';train=tmp/'train.jsonl'
    from collections import deque
    with ThreadPoolExecutor(a.workers,initializer=setup_tokenizer,
            initargs=(str(a.model),protected,a.chunk_policy)) as executor, train.open('w') as out, manifest.open('w') as meta, (tmp/'compilation-exclusions.jsonl').open('w') as excluded:
        pending=deque()
        def reject(reason,entry):
            drops[reason]+=1
            excluded.write(json.dumps({'reason':reason,**{k:entry[k] for k in
                ('pool_id','source','source_record_id','exclusion') if k in entry}})+'\n')
        def consume():
            nonlocal total,actual_max_chunk,actual_max_length,actual_max_chunks
            row,entry=pending.popleft().result()
            if row is None:reject('invalid_document_layout',entry);return
            overlap=entry.pop('_heldout_overlap')
            if row['example_id'] in seen:reject('duplicate_tokenized_input',entry);return
            if overlap:
                reject('heldout_verified_50_token_span',entry);return
            seen.add(row['example_id']);total+=1
            actual_max_chunk=max(actual_max_chunk,max(map(len,row['chunks'])))
            actual_max_length=max(actual_max_length,entry['tokens'])
            actual_max_chunks=max(actual_max_chunks,len(row['chunks']))
            counts[row['source']]+=1;tokens[row['source']]+=entry['tokens']
            out.write(json.dumps(row)+'\n');meta.write(json.dumps({**entry,'example_id':row['example_id']},ensure_ascii=False)+'\n')
            if total%5000==0:
                print(json.dumps({'retained':total,'sources':dict(counts),'drops':dict(drops)}),flush=True)
        for record in iter_pool(a.pool/'pool.sqlite'):
            pending.append(executor.submit(compile_one,record))
            if len(pending)>=a.workers*2:consume()
        while pending:consume()
    shutil.copyfile(a.validation/'validation.jsonl',tmp/'validation.jsonl')
    for path in tmp.iterdir():path.replace(a.output/path.name)
    tmp.rmdir()
    metadata={'format':'kvbench-qwen38-merged-tokenized-v1','tokenizer':local_tokenizer_spec(a.model),
        'sizes':{'train':total,'validation':500},'sources':dict(counts),'tokens_per_source':dict(tokens),
        'max_tokens':16384,'actual_max_tokens':actual_max_length,'max_chunk_tokens':actual_max_chunk,
        'chunk_policy':POLICY if a.chunk_policy=='documents' else 'legacy-raw512',
        'actual_max_chunks':actual_max_chunks,
        'legacy_policy':'Retained original CoRAG token chunks unchanged',
        'sampling':'Complete deduplicated union; no training-mixture weighting or downsampling applied',
        'pool':str(a.pool),'pool_records':pool_meta['records'],'compilation_exclusions':dict(drops),
        'decontamination':'Raw pool exact-document/context and heldout character anchors; final exact 50-token match verification for real input and CoRAG',
        'ruler_exemption':'Task templates/noise/vocabulary may share spans; complete inputs/target signatures and NIAH background independently held out',
        'files_sha256':{n:hash_file(a.output/n) for n in ('train.jsonl','validation.jsonl','examples.manifest.jsonl','compilation-exclusions.jsonl')},
        'validation_source_sha256':hash_file(a.validation/'validation.jsonl'),
        'builder_sha256':hash_file(Path(__file__)),
        'document_formatter_sha256':hash_file(ROOT/'methods/CacheRepair/qwen38/document_data.py')}
    (a.output/'dataset.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (a.output/'completion-audit.json').write_text(json.dumps({'status':'passed','train_records':total,
        'unique_token_inputs':len(seen),'validation_records':500,'sources':dict(counts),'no_training':True},indent=2)+'\n')
    print(json.dumps({'status':'completed','train_records':total,'sources':dict(counts),'drops':dict(drops)}),flush=True)


if __name__=='__main__':main()
