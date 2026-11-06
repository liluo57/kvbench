"""Generate new RULER training instances with held-out input/signature checks.

The pinned official generators are unchanged. NIAH uses an unused suffix of
the cached essay corpus in a private workspace, so its background is separated
from the existing <=8k benchmark prefix. Shared templates/word lists are allowed;
complete inputs and target signatures must be disjoint.
"""
from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import runpy
import subprocess
import sys

import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'.cache/cacherepair-data-tools')]
from scripts.PrepareDataset import _ruler_source, _ruler_environment
from scripts.PrepareCacheRepairTaskValidation import hash_file
from methods.CacheRepair.qwen38.eval_inventory import normalize, text_hash


def signature(task,row):
    values=sorted(normalize(v) for v in row['outputs'])
    if task=='niah':
        keys=re.findall(r'special magic numbers for (\S+) is:',row['input'])
        values+=sorted(normalize(v) for v in keys)
    return hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest()


def source_record_id(task,length,seed,row):
    # Native NIAH indices may repeat. Include the full input identity so that
    # distinct generated instances cannot become the same sampling group.
    return f'{task}:{length}:{seed}:{row["index"]}:{text_hash(row["input"])[:16]}'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',required=True)
    p.add_argument('--samples-per-length',type=int,default=500)
    a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'completion-audit.json').exists():return
    prior=json.loads((ROOT/'data/ruler/provenance.json').read_text())
    revision=prior['revision'];cache=ROOT/'.cache/ruler'/revision
    work=a.output/'private-source';work.mkdir(exist_ok=True)
    scripts=_ruler_source({'Revision':revision,'ArchiveSha256':prior['archive_sha256']},work,cache)
    assets=scripts/'data/synthetic/json'
    (assets/'english_words.json').write_bytes((cache/'english_words.json').read_bytes())
    # Original generators use a prefix, not random essay sampling. The full
    # historical pool is <=8192 tokens; start 100k words into a 500k-word corpus.
    all_words=re.sub(r'\s+',' ',json.loads((cache/'PaulGrahamEssays.json').read_text())['text']).split()
    assert len(all_words)>130000
    background=' '.join(all_words[100000:])
    (assets/'PaulGrahamEssays.json').write_text(json.dumps({'text':background}))
    blocked_inputs,blocked_signatures={},{}
    for task in ('niah','cwe','vt'):
        blocked_inputs[task],blocked_signatures[task]=set(),set()
        for path in (ROOT/'data/ruler').glob(f'{task}_len*.jsonl'):
            for line in path.read_text().splitlines():
                row=json.loads(line);blocked_inputs[task].add(text_hash(row['input']))
                blocked_signatures[task].add(signature(task,row))
    # Historical frozen workloads may predate the current on-disk benchmark
    # source files. Check their actual compiled instances as a separate source.
    from transformers import AutoTokenizer
    from scripts.EvaluateHybridCacheRepair import load_tasks
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    frozen_tasks,_=load_tasks(ROOT/'outputs/hypic-ablation-64-20261014/frozen_tasks.pkl',64)
    historical_niah=[]
    def flatten(value):
        return [item for part in value for item in flatten(part)] if isinstance(value,list) else [str(value)]
    for task in frozen_tasks:
        if not task.name.endswith('_shuffle'):continue
        name=task.name.split('_')[0]
        for case in task.cases:
            text=tokenizer.decode(case.input.run_input,skip_special_tokens=False)
            outputs=flatten(case.metadata.get('outputs',case.metadata.get('answers',[])))
            blocked_inputs[name].add(text_hash(text))
            needle=case.metadata.get('needle') if name=='niah' else None
            signature_input=needle if isinstance(needle,str) and needle else text
            blocked_signatures[name].add(signature(name,{'input':signature_input,'outputs':outputs}))
            if name=='niah':
                historical_niah.extend(tokenizer.decode(c,skip_special_tokens=False) for c in case.input.prepare_input)
    # Check the proposed NIAH background against all benchmark NIAH haystacks.
    sys.path.insert(0,str(ROOT/'.cache/cacherepair-data-tools'))
    import ahocorasick
    automaton=ahocorasick.Automaton()
    for path in (ROOT/'data/ruler').glob('niah_len*.jsonl'):
        for line in path.read_text().splitlines():
            s=normalize(json.loads(line)['input'])
            for start in range(0,max(0,len(s)-511),256):
                automaton.add_word(s[start:start+512],True)
    for text in historical_niah:
        s=normalize(text)
        for start in range(0,max(0,len(s)-511),256):automaton.add_word(s[start:start+512],True)
    automaton.make_automaton()
    assert next(automaton.iter(normalize(background)),None) is None,'Training NIAH corpus overlaps the held-out prefix'
    definitions=yaml.safe_load((scripts/'synthetic.yaml').read_text())
    constants=runpy.run_path(str(scripts/'data/synthetic/constants.py'))['TASKS']
    names={'niah':'niah_single_2','cwe':'cwe','vt':'vt'}
    env=_ruler_environment(cache)
    env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1')
    env['PYTHONPATH']=str(ROOT/'.cache/cacherepair-data-tools')+os.pathsep+env.get('PYTHONPATH','')
    jobs=[]
    for task in names:
        definition=definitions[names[task]];base=constants[definition['task']]
        for length in (4096,8192,16384):
            seed=700001+list(names).index(task)*10000+length
            dest=a.output/'generated'/task/str(length)
            cmd=[sys.executable,str(scripts/f'data/synthetic/{definition["task"]}.py'),
                '--save_dir',str(dest),'--save_name',names[task],'--subset','train',
                '--tokenizer_path',a.model,'--tokenizer_type','hf','--max_seq_length',str(length),
                '--tokens_to_generate',str(base['tokens_to_generate']),'--num_samples',str(a.samples_per_length+50),
                '--random_seed',str(seed),'--template',base['template']+base.get('answer_prefix','')]
            for k,v in definition['args'].items():cmd.extend([f'--{k}',str(v)])
            jobs.append((task,length,seed,dest,cmd))
    def run(job):
        task,length,seed,dest,cmd=job
        output=dest/names[task]/'train.jsonl'
        if not output.exists():
            log=a.output/f'{task}-{length}.log'
            with log.open('w') as f:subprocess.run(cmd,cwd=scripts,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        rows=[json.loads(s) for s in output.read_text().splitlines()]
        assert len(rows)==a.samples_per_length+50
        keep=[];seen=set()
        for row in rows:
            hashed=text_hash(row['input']);sig=signature(task,row)
            if hashed in blocked_inputs[task] or sig in blocked_signatures[task] or hashed in seen:continue
            keep.append({'task':task+'_shuffle','input':row['input'],'source_split':'generated-train',
                'source_record_id':source_record_id(task,length,seed,row),
                'seed':seed,'length_budget':length,'target_signature':sig,'input_sha256':hashed})
            seen.add(hashed)
            if len(keep)==a.samples_per_length:break
        assert len(keep)==a.samples_per_length,(task,length,len(keep))
        print(json.dumps({'completed':task,'length':length,'examples':len(keep)}),flush=True)
        return keep,{'task':task,'length':length,'seed':seed,'source_sha256':hash_file(output),
            'blocked_full_input_collisions':sum(text_hash(r['input']) in blocked_inputs[task] for r in rows),
            'blocked_signature_collisions':sum(signature(task,r) in blocked_signatures[task] for r in rows)}
    with concurrent.futures.ThreadPoolExecutor(3) as executor:results=list(executor.map(run,jobs))
    rows=[r for group,proof in results for r in group]
    assert len({r['input_sha256'] for r in rows})==len(rows)
    assert len({r['source_record_id'] for r in rows})==len(rows)
    path=a.output/'train.jsonl'
    with path.open('w') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    report={'status':'passed','examples':len(rows),'per_task':3*a.samples_per_length,
        'per_length':a.samples_per_length,'length_budgets':[4096,8192,16384],
        'official_source_revision':revision,'source_archive_sha256':prior['archive_sha256'],
        'unchanged_official_generator_code':True,'qwen_tokenizer':a.model,
        'niah_background_policy':'cached essay corpus suffix starting at word 100000; no sampled 512-character normalized anchor (stride 256) shared with current benchmark or historical frozen inputs',
        'niah_background_sha256':text_hash(background),'original_corpus_sha256':hash_file(cache/'PaulGrahamEssays.json'),
        'existing_test_and_validation_full_inputs_disjoint':True,'target_signatures_disjoint':True,
        'historical_frozen_cases_additionally_verified':192,
        'shared_constants':'Templates, noise filler and vocabulary are task definitions and may recur',
        'generation':[proof for group,proof in results],'train_sha256':hash_file(path)}
    (a.output/'completion-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':'completed','examples':len(rows),'output':str(a.output)}),flush=True)


if __name__=='__main__':main()
