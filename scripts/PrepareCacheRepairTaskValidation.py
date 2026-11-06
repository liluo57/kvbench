"""Compile 10x50 document/prefix KV residual validation without running models.

Uses cached, pinned source splits and avoids the historical generation-test IDs.
The source documents and task prefixes use the same formatter as training.
There is no generation workflow or answer-based training label here.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import heapq
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import sys
import zipfile

import ijson
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from methods.CacheRepair.qwen38.decontam import window_hashes
from methods.CacheRepair.qwen38.eval_inventory import text_hash
from methods.CacheRepair.qwen38.sources import local_tokenizer_spec
from methods.CacheRepair.qwen38.document_data import POLICY, compile_documents

TASKS = ('niah_shuffle', 'cwe_shuffle', 'vt_shuffle', 'musique', 'samsum',
    '2wikimultihopqa', 'govreport', 'hotpotqa', 'multinews', 'triviaqa')
SPLITS = {'musique': 'dev', '2wikimultihopqa': 'dev', 'samsum': 'val',
    'govreport': 'valid', 'hotpotqa': 'validation', 'multinews': 'validation', 'triviaqa': 'validation'}


def read(path):
    return json.loads(path.read_text())


def hash_file(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(8*1024*1024):
            h.update(block)
    return h.hexdigest()


def token_hash(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()


def source_archive(dataset):
    meta = read(ROOT / f'data/{dataset}/_metadata/source.json')
    a = meta['downloaded_artifacts'][0]
    path = ROOT / '.cache/dataset_sources' / hashlib.sha256(a['url'].encode()).hexdigest() / 'payload'
    assert path.stat().st_size == a['bytes'] and hash_file(path) == a['sha256']
    return path


def parquet_sources(dataset):
    meta = read(ROOT / f'data/{dataset}/_metadata/source.json')
    snapshot = Path('/data/lyh/.cache/huggingface/hub') / ('datasets--' + meta['Dataset'].replace('/', '--')) / 'snapshots' / meta['Revision']
    pattern = {'hotpotqa': 'distractor/validation-*.parquet', 'triviaqa': 'rc/validation-*.parquet',
        'multinews': 'validation.parquet'}[dataset]
    paths = sorted(snapshot.glob(pattern))
    assert paths, (dataset, snapshot)
    return paths


def raw_sources(task, audit):
    if task.endswith('_shuffle'):
        name = task.split('_')[0]
        for path in sorted((ROOT / 'data/ruler').glob(f'{name}_len*.jsonl')):
            for index, line in enumerate(path.open()):
                if line.strip():
                    yield json.loads(line), path, index
    elif task in ('musique', '2wikimultihopqa'):
        path = source_archive(task)
        with zipfile.ZipFile(path) as archive:
            member = 'data/musique_ans_v1.0_dev.jsonl' if task == 'musique' else 'data_ids/dev.json'
            with archive.open(member) as stream:
                rows = (json.loads(s) for s in stream if s.strip()) if task == 'musique' else ijson.items(stream, 'item')
                for index, row in enumerate(rows):
                    yield row, path, index
    elif task == 'samsum':
        path = audit / 'raw-samsum/val.json'
        for index, row in enumerate(read(path)):
            yield row, path, index
    elif task == 'govreport':
        for entry in read(audit / 'govreport-sources-verified.json'):
            if not entry['url'].endswith('_valid.jsonl'):
                continue
            path = Path(entry['path'])
            assert hash_file(path) == entry['sha256']
            for index, line in enumerate(path.open()):
                if line.strip():
                    yield json.loads(line), path, index
    else:
        for path in parquet_sources(task):
            index = 0
            for batch in pq.ParquetFile(path).iter_batches(batch_size=64, use_threads=False):
                for row in batch.to_pylist():
                    yield row, path, index
                    index += 1


def raw_text(task, row):
    if task.endswith('_shuffle'):
        return row['input'].replace('[INST]', '').replace('[/INST]', '').strip()
    if task == 'musique':
        return '\n\n'.join(p['title']+'\n'+p['paragraph_text'] for p in row['paragraphs'])
    if task == '2wikimultihopqa':
        return '\n\n'.join(t+'\n'+' '.join(s) for t, s in row['context'])
    if task == 'samsum':
        return row['dialogue']
    if task == 'hotpotqa':
        return '\n\n'.join(t+'\n'+' '.join(s) for t, s in zip(row['context']['title'], row['context']['sentences'], strict=True))
    if task == 'triviaqa':
        wiki = zip(row['entity_pages']['title'], row['entity_pages']['wiki_context'])
        web = zip(row['search_results']['title'], row['search_results']['search_context'])
        return '\n\n'.join(t+'\n'+s for t, s in [*wiki, *web] if s.strip())
    if task == 'multinews':
        return '\n\n'.join(s.strip() for s in row['document'].split('|||||') if s.strip())
    if task == 'govreport':
        from scripts.PrepareDataset import _govreport_sections
        if 'report' in row:
            sections = [s for section in row['report'] for s in _govreport_sections(section, keep_letter=False, depth=1)]
        else:
            sections = _govreport_sections(row['reports'], keep_letter=True, depth=0)
        return ' '.join(f'{s["title"]} {s["paragraphs"]}'.strip() for s in sections).replace('\n', ' ').strip()
    raise ValueError(task)


def record_id(task, row, path, index):
    if task.endswith('_shuffle'):
        return f'{path.name}:{row.get("index", index)}'
    if task == 'multinews':
        return f'{path.name}:{index}'  # Current KVBench MultiNews _id is empty.
    if task == 'govreport':
        return ('GAO_' if 'report' in row else 'CRS_') + str(row['id'])
    return str(row.get('id', row.get('_id', row.get('question_id', index))))


def ranked_candidates(task, audit, blocked, seed, count):
    """Uniform deterministic selection over unique source records/question IDs."""
    if task == 'triviaqa':
        # Read only question IDs first; avoid decoding every long evidence text.
        chosen = {}
        for path in parquet_sources(task):
            qids = pq.read_table(path, columns=['question_id']).column(0).to_pylist()
            for index, rid in enumerate(qids):
                if str(rid) in blocked:
                    continue
                rank = hashlib.sha256(f'{seed}:{task}:{rid}'.encode()).hexdigest()
                tie = hashlib.sha256(f'{seed}:{path.name}:{index}'.encode()).hexdigest()
                if rid not in chosen or tie < chosen[rid][1]:
                    chosen[rid] = rank, tie, path, index
        picked = sorted(chosen.values())[:count]
        groups = collections.defaultdict(list)
        for rank, tie, path, index in picked:
            groups[path].append((index, rank))
        for path, wanted in groups.items():
            pf, start = pq.ParquetFile(path), 0
            for group in range(pf.num_row_groups):
                stop = start + pf.metadata.row_group(group).num_rows
                selected = [(i, r) for i, r in wanted if start <= i < stop]
                if selected:
                    table = pf.read_row_group(group)
                    for index, rank in selected:
                        row = table.slice(index-start, 1).to_pylist()[0]
                        yield rank, row, path, index
                start = stop
        return
    heap, seen = [], set()
    for row, path, index in raw_sources(task, audit):
        rid = record_id(task, row, path, index)
        if rid in blocked or rid in seen:
            continue
        seen.add(rid)
        rank = int(hashlib.sha256(f'{seed}:{task}:{rid}'.encode()).hexdigest(), 16)
        entry = (-rank, rid, row, path, index)
        if len(heap) < count:
            heapq.heappush(heap, entry)
        elif rank < -heap[0][0]:
            heapq.heapreplace(heap, entry)
    for negative, rid, row, path, index in sorted(heap, reverse=True):
        yield f'{-negative:064x}', row, path, index


def chunks_for(ids, maximum=512):
    if len(ids) < 2:
        raise ValueError('Need at least two input tokens for nontrivial KV residuals')
    count = max(2, math.ceil(len(ids)/maximum))
    return [ids[i*len(ids)//count:(i+1)*len(ids)//count] for i in range(count)]


def screen_training_spans(candidates, train):
    """Index candidate raw tokens, then scan actual old training tokens once.

    Hash hits are verified against the actual 50 token IDs. No original CoRAG
    train rows are changed. Contaminated validation candidates are rejected.
    """
    keys, owners, offsets = [], [], []
    for owner, row in enumerate(candidates):
        prefix=row.get('prefix_tokens',0)
        hashes = window_hashes(row['ids'][prefix:])
        keys.append(hashes)
        owners.append(np.full(len(hashes), owner, dtype=np.uint16))
        offsets.append(np.arange(len(hashes), dtype=np.uint16)+prefix)
    raw = np.concatenate(keys)
    order = np.argsort(raw, kind='stable')
    key = raw[order]
    owner = np.concatenate(owners)[order]
    offset = np.concatenate(offsets)[order]
    rejected, evidence = set(), []
    with train.open() as stream:
        for index, line in enumerate(stream):
            row = json.loads(line)
            ids = [token for c in row['chunks'] for token in c]
            hashes = window_hashes(ids)
            loc = np.searchsorted(key, hashes)
            safe = np.minimum(loc, len(key)-1)
            hits = np.flatnonzero((loc < len(key)) & (key[safe] == hashes))
            for position in hits:
                left = int(loc[position])
                right = int(np.searchsorted(key, hashes[position], side='right'))
                for match in range(left, right):
                    who, start = int(owner[match]), int(offset[match])
                    if who not in rejected and candidates[who]['ids'][start:start+50] == ids[position:position+50]:
                        rejected.add(who)
                        evidence.append({'candidate': candidates[who]['example_id'], 'task': candidates[who]['task'],
                            'old_training_row': index, 'training_token_offset': int(position), 'validation_token_offset': start})
            if (index+1) % 10000 == 0:
                print(json.dumps({'screened_old_train': index+1, 'rejected_candidates': len(rejected)}), flush=True)
    return rejected, evidence


def reformat_existing(source, audit, output, model):
    """Keep the fixed 500 source IDs; change only their input compilation."""
    from transformers import AutoTokenizer
    from scripts.MergeCacheRepairTrainingPool import documents
    source, output=source.resolve(),output.resolve()
    if output.exists():raise FileExistsError(output)
    original=read(source/'dataset.json')
    for name in ('validation.jsonl','examples.manifest.jsonl'):
        assert hash_file(source/name)==original['files_sha256'][name]
    selected=[json.loads(s) for s in (source/'examples.manifest.jsonl').open()]
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
    compiled={}
    for task in TASKS:
        wanted={(r['source_file'],r['source_row']):i for i,r in enumerate(selected) if r['task']==task}
        found=set()
        for raw,path,index in raw_sources(task,audit):
            key=(str(path.resolve()),index)
            if key not in wanted:continue
            i=wanted[key]
            chunks,layout=compile_documents(documents(task,raw),task,tokenizer)
            ids=[token for chunk in chunks for token in chunk]
            compiled[i]=({'chunks':chunks,'task':task,'example_id':token_hash(ids)},
                {**selected[i],'previous_raw_example_id':selected[i]['example_id'],
                 'example_id':token_hash(ids),**layout})
            found.add(key)
            if len(found)==len(wanted):break
        assert len(found)==len(wanted),(task,len(found),len(wanted))
        print(json.dumps({'reformatted_task':task,'examples':len(found)}),flush=True)
    assert len(compiled)==500
    output.mkdir(parents=True)
    with (output/'validation.jsonl').open('x') as f,(output/'examples.manifest.jsonl').open('x') as mf:
        for i in range(500):
            row,entry=compiled[i];f.write(json.dumps(row)+'\n');mf.write(json.dumps(entry,ensure_ascii=False)+'\n')
    unique=len({r['example_id'] for r,m in compiled.values()});assert unique==500
    metadata={'format':'kvbench-qwen38-task-validation-documents-v1','validation_only':True,
        'tokenizer':local_tokenizer_spec(model),'sizes':{'validation':500},
        'chunk_policy':POLICY,'tasks':dict(collections.Counter(r['task'] for r,m in compiled.values())),
        'source_selection':'Same 500 source records, same order; no quality-based resampling',
        'source_dataset':str(source),'source_validation_sha256':hash_file(source/'validation.jsonl'),
        'source_manifest_sha256':hash_file(source/'examples.manifest.jsonl'),
        'decontamination':'Source pool protects these same raw source records; merged train receives a fresh token-span screen against this layout',
        'tokenizer_runtime':{n:importlib.metadata.version(n) for n in ('transformers','tokenizers')},
        'files_sha256':{n:hash_file(output/n) for n in ('validation.jsonl','examples.manifest.jsonl')},
        'builder_sha256':hash_file(Path(__file__)),
        'document_formatter_sha256':hash_file(ROOT/'methods/CacheRepair/qwen38/document_data.py')}
    (output/'dataset.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (output/'completion-audit.json').write_text(json.dumps({'status':'passed','examples':500,
        'unique_inputs':unique,'same_source_records':True,'chunk_policy':POLICY},indent=2)+'\n')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-tokens', type=int, default=16384)
    parser.add_argument('--per-task', type=int, default=50)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--reformat-existing',type=Path,
        help='Keep a previous fixed 500-source selection and apply document/prefix formatting')
    args = parser.parse_args()
    assert args.max_tokens == 16384 and args.per_task == 50
    assert not args.output.exists(), args.output
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    if args.reformat_existing:
        reformat_existing(args.reformat_existing,args.audit,args.output,args.model)
        return
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer_spec = local_tokenizer_spec(args.model)
    old = ROOT / 'outputs/cacherepair-qwen38/full-training-v1/data'
    old_hashes = {n: hash_file(old/n) for n in ('train.jsonl', 'validation.jsonl')}
    historical = read(args.audit / 'historical-test-selection.json')
    all_candidates, source_files = [], set()
    for task in TASKS:
        blocked = {str(x) for x in historical[task]['sample_ids'] if x not in (None, '')}
        if task.endswith('_shuffle'):
            prefix = task.split('_')[0]
            blocked = {f'{prefix}_len4096.jsonl:{i}' for i in range(64)}
        elif task == 'multinews':
            # Distinct official split: historical rows are test, new rows validation.
            blocked = set()
        seen_texts = set()
        count = 0
        for rank, row, path, index in ranked_candidates(task, args.audit, blocked, args.seed, 100):
            text = raw_text(task, row)
            if not text.strip() or text_hash(text) in seen_texts:
                continue
            seen_texts.add(text_hash(text))
            from scripts.MergeCacheRepairTrainingPool import documents
            try:chunks,layout=compile_documents(documents(task,row),task,tokenizer)
            except ValueError:continue
            ids=[i for c in chunks for i in c]
            truncated=bool(layout['truncated_documents'] or layout['dropped_documents'])
            rid = record_id(task, row, path, index)
            example_id = token_hash(ids)
            all_candidates.append({'task': task, 'ids': ids, 'chunks':chunks, **layout,'example_id': example_id,
                'source_record_id': rid, 'source_file': str(path.resolve()), 'source_row': index,
                'source_split': 'generated-unused-test-pool' if task.endswith('_shuffle') else SPLITS[task],
                'raw_text_sha256_normalized': text_hash(text), 'tokens': len(ids),
                'truncated_to_16384': truncated, 'rank': rank})
            source_files.add(path.resolve())
            count += 1
        assert count >= 50, (task, count)
        print(json.dumps({'candidate_task': task, 'candidates': count}), flush=True)
    rejected, evidence = screen_training_spans(all_candidates, old/'train.jsonl')
    selected, seen_inputs = [], set()
    for task in TASKS:
        bands = [[], [], []]
        for i, row in enumerate(all_candidates):
            if row['task'] == task and i not in rejected and row['example_id'] not in seen_inputs:
                band = 0 if row['tokens'] <= 2048 else 1 if row['tokens'] <= 8192 else 2
                bands[band].append(row)
        for band in bands:
            band.sort(key=lambda r: r['rank'], reverse=True)
        picked = []
        while len(picked) < 50:
            progress = False
            for band in bands:
                while band and band[-1]['example_id'] in seen_inputs:
                    band.pop()
                if band and len(picked) < 50:
                    row = band.pop()
                    seen_inputs.add(row['example_id'])
                    picked.append(row)
                    progress = True
            if not progress:
                raise RuntimeError(f'Insufficient clean {task} candidates: {len(picked)}; enlarge candidate pool')
        selected.extend(picked)
    random.Random(args.seed).shuffle(selected)
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output/'validation.jsonl').open('x') as stream, (args.output/'examples.manifest.jsonl').open('x') as manifest:
        for row in selected:
            chunks = row['chunks']
            assert [token for c in chunks for token in c] == row['ids'] and 2 <= len(chunks) <= 32
            stream.write(json.dumps({'chunks': chunks, 'task': row['task'], 'example_id': row['example_id']})+'\n')
            manifest.write(json.dumps({k:v for k,v in row.items() if k not in ('ids','chunks')}, ensure_ascii=False)+'\n')
    assert {n:hash_file(old/n) for n in old_hashes} == old_hashes
    counts = dict(collections.Counter(r['task'] for r in selected))
    assert len(selected) == 500 and all(counts[t] == 50 for t in TASKS)
    stats = {t: {'examples': counts[t], 'min_tokens': min(r['tokens'] for r in selected if r['task']==t),
        'median_tokens': float(np.median([r['tokens'] for r in selected if r['task']==t])),
        'max_tokens': max(r['tokens'] for r in selected if r['task']==t),
        'truncated_examples': sum(r['truncated_to_16384'] for r in selected if r['task']==t)} for t in TASKS}
    metadata = {'format': 'kvbench-qwen38-task-validation-v1', 'validation_only': True,
        'tokenizer': tokenizer_spec, 'seed': args.seed, 'sizes': {'validation': 500}, 'tasks': counts,
        'max_tokens': 16384, 'min_chunks': 2, 'max_chunks': 32, 'chunk_policy':POLICY,
        'policy': 'Source document boundaries and first task prefix; no generation workflow or gold-answer targets',
        'length_sampling': 'Round-robin available <=2k, 2k..8k, 8k..16k bands within each task; no duplication or padding to fill absent bands',
        'supervision': 'Native independent-chunk stale KV versus causal joint KV, same canonical residual definition as existing training',
        'ruler_policy': 'Unused generated source rows; historical first 64 length-4096 rows excluded; same synthetic background may recur',
        'trivia_policy': 'Unique question IDs; random deterministic choice of evidence variant before selecting questions',
        'decontamination': {'historical_generation_sample_ids_excluded': True,
            'verified_50_token_span_overlap_with_old_training': 0,
            'candidate_count': len(all_candidates), 'rejected_candidates': len(rejected),
            'rejection_evidence': evidence, 'scope': 'Actual old training token stream, including cross-chunk windows; hashed hits verified exactly',
            'not_claimed': 'No exhaustive semantic/paraphrase audit or source-document disjointness; new mixed train splits still need screening'},
        'previous_data_sha256_unchanged': old_hashes,
        'statistics': stats,
        'files_sha256': {n:hash_file(args.output/n) for n in ('validation.jsonl','examples.manifest.jsonl')},
        'source_files': [{'path':str(p),'bytes':p.stat().st_size,'sha256':hash_file(p)} for p in sorted(source_files)],
        'builder_sha256': hash_file(Path(__file__))}
    (args.output/'dataset.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False)+'\n')
    (args.output/'completion-audit.json').write_text(json.dumps({'status':'passed','examples':500,
        'tasks':counts,'unique_input_token_hashes':len(seen_inputs),'max_tokens':max(r['tokens'] for r in selected),
        'historical_ids_excluded':True,'no_verified_50_token_match_to_old_train':True,
        'original_training_and_validation_unchanged':True},indent=2)+'\n')
    print(json.dumps({'status':'completed','output':str(args.output.resolve()),'statistics':stats}),flush=True)


if __name__ == '__main__':
    main()
