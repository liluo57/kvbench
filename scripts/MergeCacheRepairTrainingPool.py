"""Merge CoRAG, seven real train splits and fresh RULER into a deduplicated pool.

The pool stores unique normalized documents once and preserves distinct ordered
contexts: the same passage in different contexts has different joint KV labels.
It is a lossless source pool before token-budget sampling/tokenization, not an
implicit change to the original training job or its RMS.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time
import zipfile
import zlib

import ijson
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'.cache/cacherepair-data-tools')]
from methods.CacheRepair.qwen38.eval_inventory import normalize, text_hash
from scripts.PrepareCacheRepairTaskValidation import raw_text, source_archive, hash_file

REAL=('musique','2wikimultihopqa','samsum','hotpotqa','triviaqa','govreport','multinews')


def read(path):return json.loads(path.read_text())


def atomic(path,value):
    temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def train_rows(task,audit):
    if task in ('musique','2wikimultihopqa'):
        path=source_archive(task)
        with zipfile.ZipFile(path) as archive:
            member='data/musique_ans_v1.0_train.jsonl' if task=='musique' else 'data_ids/train.json'
            with archive.open(member) as f:
                rows=(json.loads(s) for s in f if s.strip()) if task=='musique' else ijson.items(f,'item')
                for i,row in enumerate(rows):yield row,str(path),i
    elif task=='samsum':
        path=audit/'raw-samsum/train.json'
        for i,row in enumerate(read(path)):yield row,str(path),i
    elif task=='govreport':
        for item in read(audit/'govreport-sources-verified.json'):
            if not item['url'].endswith('_train.jsonl'):continue
            path=Path(item['path']);assert hash_file(path)==item['sha256']
            for i,line in enumerate(path.open()):
                if line.strip():yield json.loads(line),str(path),i
    else:
        meta=read(ROOT/f'data/{task}/_metadata/source.json')
        hub=Path('/data/lyh/.cache/huggingface/hub')/('datasets--'+meta['Dataset'].replace('/','--'))/'snapshots'/meta['Revision']
        pattern={'hotpotqa':'distractor/train-*.parquet','triviaqa':'rc/train-*.parquet','multinews':'train.parquet'}[task]
        for path in sorted(hub.glob(pattern)):
            i=0
            for batch in pq.ParquetFile(path).iter_batches(batch_size=32,use_threads=False):
                for row in batch.to_pylist():yield row,str(path),i;i+=1


def documents(task,row):
    if task=='musique':return [p['title']+'\n'+p['paragraph_text'] for p in row['paragraphs']]
    if task=='2wikimultihopqa':return [t+'\n'+' '.join(s) for t,s in row['context']]
    if task=='hotpotqa':return [t+'\n'+' '.join(s) for t,s in zip(row['context']['title'],row['context']['sentences'],strict=True)]
    if task=='triviaqa':
        wiki=zip(row['entity_pages']['title'],row['entity_pages']['wiki_context'])
        web=zip(row['search_results']['title'],row['search_results']['search_context'])
        return [t+'\n'+b for t,b in [*wiki,*web] if b.strip()]
    if task=='multinews':return [s.strip() for s in row['document'].split('|||||') if s.strip()]
    if task=='samsum':return [row['dialogue']]
    if task=='govreport':return [raw_text(task,row)]
    if task.endswith('_shuffle'):return [row['input']]
    raise ValueError(task)


def content_identity(norm_docs):
    # Document/whitespace boundaries cannot create a different example identity.
    return hashlib.sha256(' '.join(norm_docs).encode()).hexdigest()


class Protection:
    def __init__(self):
        self.exact=set();self.queries=set();self.windows={};self.automaton=None

    def add(self,text,ref,*,spans=False):
        n=normalize(text)
        if not n:return
        self.exact.add(hashlib.sha256(n.encode()).hexdigest())
        if spans:
            for start in range(0,max(0,len(n)-255),128):
                self.windows.setdefault(n[start:start+256],ref)

    def finish(self):
        import ahocorasick
        self.automaton=ahocorasick.Automaton()
        for s,ref in self.windows.items():self.automaton.add_word(s,ref)
        self.automaton.make_automaton()

    def reason(self,norm):
        if hashlib.sha256(norm.encode()).hexdigest() in self.exact:return 'heldout_exact_document'
        # This is a verified normalized-character match, not a token or semantic
        # near-duplicate claim. The index is restricted to held-out source text.
        for end,ref in self.automaton.iter(norm):return 'heldout_256_character_span'
        return None


def strip_prefix(text,prefix):
    return text[len(prefix):] if text.startswith(prefix) else text


def protection(args,tokenizer):
    from tasks.Musique import MusiqueTask
    p=Protection();prefix=MusiqueTask.prefixPrompt
    # Full registered dev/test document inventory, excluding generic questions.
    inv=ROOT/'outputs/cacherepair-qwen38/full-training-v1/evaluation-inventory/contents.sqlite'
    with sqlite3.connect(f'file:{inv}?mode=ro',uri=True) as db:
        sql='SELECT DISTINCT c.hash,c.text,r.kind FROM contents c JOIN refs r ON c.hash=r.hash WHERE r.workload IN ('+','.join('?' for _ in REAL)+')'
        for digest,text,kind in db.execute(sql,REAL):
            if kind=='document':p.add(text,'registered-evaluation')
            elif kind=='question':p.queries.add(text_hash(text))
    for line in (args.old_data/'validation.jsonl').open():
        row=json.loads(line)
        for chunk in row['chunks']:
            p.add(strip_prefix(tokenizer.decode(chunk,skip_special_tokens=False),prefix),'old-validation',spans=True)
    for line in (args.validation/'validation.jsonl').open():
        row=json.loads(line)
        if row['task'].endswith('_shuffle'):continue
        text=tokenizer.decode([i for c in row['chunks'] for i in c],skip_special_tokens=False)
        p.add(text,'new-task-validation',spans=True)
    for row in map(json.loads,(args.validation/'examples.manifest.jsonl').read_text().splitlines()):
        if not row['task'].endswith('_shuffle'):p.exact.add(row['raw_text_sha256_normalized'])
    # The new dialogue/report/news validation splits were not in the old
    # registered test snapshots. Protect their selected complete source units.
    selected=collections.defaultdict(set)
    for r in map(json.loads,(args.validation/'examples.manifest.jsonl').read_text().splitlines()):
        if r['task'] in ('samsum','govreport','multinews'):
            selected[(r['task'],r['source_file'])].add(r['source_row'])
    for (task,file),indices in selected.items():
        path=Path(file)
        if path.suffix=='.json':
            rows=read(path)
            for i in indices:p.add(rows[i]['dialogue'],'new-samsum-validation',spans=True)
        elif task=='multinews':
            pf=pq.ParquetFile(path);start=0
            for g in range(pf.num_row_groups):
                stop=start+pf.metadata.row_group(g).num_rows
                wanted=[i for i in indices if start<=i<stop]
                if wanted:
                    table=pf.read_row_group(g)
                    for i in wanted:
                        for doc in documents('multinews',table.slice(i-start,1).to_pylist()[0]):p.add(doc,'new-news-validation',spans=True)
                start=stop
        else:
            for i,line in enumerate(path.open()):
                if i in indices:p.add(raw_text('govreport',json.loads(line)),'new-report-validation',spans=True)
    # Historical 64-case generation inputs: use content-only prepared chunks,
    # without task/chat instructions. Synthetic constants are handled separately.
    from scripts.EvaluateHybridCacheRepair import load_tasks
    tasks,_=load_tasks(ROOT/'outputs/hypic-ablation-64-20261014/frozen_tasks.pkl',64)
    for task in tasks:
        if task.name.endswith('_shuffle'):continue
        task_prefix=getattr(task.original,'prefixPrompt','')
        for case in task.cases:
            for chunk in case.input.prepare_input:
                text=tokenizer.decode(chunk,skip_special_tokens=False)
                p.add(strip_prefix(text,task_prefix),'historical-generation',spans=True)
    # MuSiQue dev/test seed-question ancestry is independent of composed Q IDs.
    with zipfile.ZipFile(source_archive('musique')) as archive:
        seed=json.loads(archive.read('data/dev_test_singlehop_questions_v1.0.json'))
    for values in seed.values():p.queries.update(text_hash(v['question']) for v in values)
    p.finish()
    return p


def iter_pool(path):
    """Yield retained old chunks or raw ordered documents for later tokenization."""
    db=sqlite3.connect(f'file:{path}?mode=ro',uri=True)
    try:
        for pid,source,payload in db.execute('SELECT id,source,payload FROM examples ORDER BY id'):
            row=json.loads(zlib.decompress(payload))
            if 'chunks' not in row:
                row['documents']=[zlib.decompress(db.execute('SELECT text FROM documents WHERE hash=?',(key,)).fetchone()[0]).decode()
                    for key in row['document_ids']]
            yield {'pool_id':pid,'source':source,**row}
    finally:db.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('old-data','validation','audit','ruler','output','model'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    for name in ('old_data','validation','audit','ruler','output','model'):setattr(args,name,getattr(args,name).resolve())
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'completion-audit.json').exists():return
    from transformers import AutoTokenizer
    from tasks.Musique import MusiqueTask
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    protected=protection(args,tokenizer)
    before={n:hash_file(args.old_data/n) for n in ('train.jsonl','validation.jsonl')}
    db=sqlite3.connect(args.output/'pool.sqlite')
    db.executescript('''PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;
        CREATE TABLE IF NOT EXISTS documents(hash TEXT PRIMARY KEY,text BLOB NOT NULL,bytes INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS examples(id INTEGER PRIMARY KEY,source TEXT NOT NULL,record_id TEXT NOT NULL,
            content_hash TEXT NOT NULL UNIQUE,group_id TEXT NOT NULL,payload BLOB NOT NULL);
        CREATE TABLE IF NOT EXISTS completed_sources(source TEXT PRIMARY KEY,report TEXT NOT NULL);
    ''')
    complete={s:json.loads(r) for s,r in db.execute('SELECT source,report FROM completed_sources')}
    seen={key for key, in db.execute('SELECT content_hash FROM examples')}
    stored={key for key, in db.execute('SELECT hash FROM documents')}
    doc_cache={};count=db.execute('SELECT count(*) FROM examples').fetchone()[0]
    rejection=(args.output/'rejections.jsonl').open('a',buffering=1)

    def insert(source,rid,docs,metadata,counts,chunks=None):
        nonlocal count
        counts['input_records']+=1
        if metadata.get('query') and text_hash(metadata['query']) in protected.queries:
            counts['excluded_query_or_musique_ancestry']+=1;return
        norms=[]
        for doc in docs:
            raw=doc.encode();rawkey=hashlib.sha256(raw).hexdigest()
            if rawkey not in doc_cache:
                normalized=normalize(doc)
                doc_cache[rawkey]=(normalized,protected.reason(normalized))
            n,reason=doc_cache[rawkey]
            if reason:
                counts[reason]+=1
                if counts[reason]<=50:rejection.write(json.dumps({'source':source,'record_id':rid,'reason':reason})+'\n')
                return
            if n:norms.append(n)
        if not norms:counts['empty_input']+=1;return
        key=content_identity(norms)
        if key in protected.exact:counts['heldout_exact_context']+=1;return
        if key in seen:counts['duplicate_normalized_input']+=1;return
        seen.add(key)
        payload={'source_record_id':rid,'source_split':'train',**metadata}
        payload.pop('query',None)
        if chunks is not None:
            payload['chunks']=chunks
        else:
            doc_ids=[]
            for doc,n in zip(docs,norms,strict=True):
                digest=hashlib.sha256(n.encode()).hexdigest();doc_ids.append(digest)
                if digest not in stored:
                    encoded=doc.encode();db.execute('INSERT INTO documents VALUES (?,?,?)',(digest,zlib.compress(encoded,1),len(encoded)));stored.add(digest)
            payload['document_ids']=doc_ids
        count+=1
        db.execute('INSERT INTO examples VALUES (?,?,?,?,?,?)',(count,source,rid,key,
            source+':'+rid,zlib.compress(json.dumps(payload,ensure_ascii=False).encode(),1)))
        counts['retained_records']+=1

    sources=['corag',*REAL,'ruler']
    for source in sources:
        if source in complete:continue
        # Transactional source commit: a failed source is recomputed, completed
        # source partitions are never appended twice on restart.
        db.execute('BEGIN')
        counts=collections.Counter();started=time.time()
        if source=='corag':
            manifests=[json.loads(s) for s in (args.old_data/'examples.manifest.jsonl').open()]
            manifests=[r for r in manifests if r['split']=='train']
            with (args.old_data/'train.jsonl').open() as stream:
                for i,(line,meta) in enumerate(zip(stream,manifests,strict=True)):
                    row=json.loads(line)
                    docs=[strip_prefix(tokenizer.decode(c,skip_special_tokens=False),MusiqueTask.prefixPrompt) for c in row['chunks']]
                    insert(source,meta['example_id'],docs,{'original_training_row':i,'query':meta['query'],
                        'corag_config':meta['config'],'source_file':str(args.old_data/'train.jsonl')},counts,chunks=row['chunks'])
                    if (i+1)%10000==0:print(json.dumps({'source':source,**counts}),flush=True)
        elif source=='ruler':
            while not (args.ruler/'completion-audit.json').exists():time.sleep(10)
            audit=read(args.ruler/'completion-audit.json');assert audit['status']=='passed'
            assert hash_file(args.ruler/'train.jsonl')==audit['train_sha256']
            for line in (args.ruler/'train.jsonl').open():
                row=json.loads(line)
                insert(row['task'],row['source_record_id'],[row['input']],
                    {'ruler_seed':row['seed'],'length_budget':row['length_budget'],'target_signature':row['target_signature']},counts)
        else:
            for i,(row,path,index) in enumerate(train_rows(source,args.audit)):
                rid=str(row.get('id',row.get('_id',row.get('question_id',f'{Path(path).name}:{index}'))))
                docs=[s for s in documents(source,row) if s.strip()]
                insert(source,rid,docs,{'source_file':path,'source_row':index,'query':row.get('question')},counts)
                if (i+1)%10000==0:print(json.dumps({'source':source,**counts}),flush=True)
        report={**dict(counts),'seconds':time.time()-started}
        db.execute('INSERT INTO completed_sources VALUES (?,?)',(source,json.dumps(report)))
        db.commit();complete[source]=report
        atomic(args.output/'status.json',{'status':'running','completed_sources':list(complete),'retained_records':count,'last_source':source,'statistics':complete})
        print(json.dumps({'completed_source':source,**report}),flush=True)
    rejection.close()
    # The official source supplies input variants under one TriviaQA question ID.
    # Keep distinct contexts, mark the same sampling group; identical input is removed.
    grouped={s:{'examples':n,'sampling_groups':g} for s,n,g in db.execute('SELECT source,count(*),count(DISTINCT group_id) FROM examples GROUP BY source')}
    docs=db.execute('SELECT count(*),coalesce(sum(bytes),0) FROM documents').fetchone()
    manifest=args.output/'train.records.jsonl'
    with manifest.open('w') as f:
        for row in db.execute('SELECT id,source,record_id,content_hash,group_id FROM examples ORDER BY id'):
            f.write(json.dumps(dict(zip(('pool_id','source','record_id','content_sha256','sampling_group'),row)))+'\n')
    db.execute('PRAGMA wal_checkpoint(TRUNCATE)');db.close()
    assert before=={n:hash_file(args.old_data/n) for n in before}
    metadata={'format':'kvbench-qwen38-merged-source-pool-v1','status':'completed',
        'records':count,'sources':grouped,'source_processing':complete,
        'unique_stored_documents':docs[0],'uncompressed_document_bytes':docs[1],
        'storage':'pool.sqlite: existing CoRAG token chunks plus unique raw documents and ordered context references',
        'not_yet_applied':'Training-token truncation/tokenization and mixture sampling; no optimizer/RMS/training launched',
        'suggested_training_max_tokens':16384,
        'deduplication':'Keep first normalized complete ordered context; store normalized documents once; shared documents in distinct contexts remain distinct KV-supervision examples',
        'validation_protection':{'registered_dev_test_document_hashes':len(protected.exact),
            'normalized_256_character_anchors':len(protected.windows),
            'source':'All seven registered evaluation-source document inventories; old 500 and new 500; historical 640 generation inputs',
            'scope':'Exact document/context and question-provenance checks; 256-character exact segment checks on selected held-out text; not an exhaustive semantic/near-duplicate guarantee'},
        'ruler_audit':str(args.ruler/'completion-audit.json'),
        'original_data_sha256_unchanged':before,'new_validation_sha256':hash_file(args.validation/'validation.jsonl'),
        'builder_sha256':hash_file(Path(__file__)),'manifest_sha256':hash_file(manifest)}
    atomic(args.output/'dataset.json',metadata)
    atomic(args.output/'completion-audit.json',{'status':'passed','records':count,'sources':grouped,
        'unique_content_keys':len(seen),'original_data_unchanged':True,'no_training':True})
    atomic(args.output/'status.json',{'status':'completed','records':count,'source_processing':complete})
    print(json.dumps({'status':'completed','records':count,'sources':grouped}),flush=True)


if __name__=='__main__':main()
