"""Compare native and vLLM supervision on identical, already-tokenized rows.

Run native and vllm modes in separate processes, then compare. No optimizer,
checkpoint writes, RMS replacement, or production-backend switch occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['native', 'vllm', 'compare'])
    p.add_argument('--model', required=True)
    p.add_argument('--rows', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    rows = [json.loads(s) for s in a.rows.read_text().splitlines() if s.strip()]
    a.output.mkdir(parents=True, exist_ok=True)
    if a.mode == 'compare':
        torch.set_num_threads(8)
        from methods.CacheRepair.qwen38.vllm_audit import compare_runs
        result = compare_runs(a.output, rows)
        (a.output / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
        return
    folder = a.output / a.mode
    folder.mkdir(exist_ok=False)
    started = time.perf_counter()
    if a.mode == 'native':
        from methods.CacheRepair.qwen38.train import load_native_target
        from methods.CacheRepair.qwen38.supervision import make_pair
        from methods.CacheRepair.qwen38.target import TargetGeometry
        target = load_native_target(a.model, (0, 1), attention_backend='sdpa', require_fast=True)
        geometry = TargetGeometry.from_target(target, target_model_id='Qwen/Qwen3.8-27B')
        loaded = time.perf_counter()
        timings = []
        for i, row in enumerate(rows):
            torch.cuda.synchronize()
            t = time.perf_counter()
            pair = make_pair(target, row['chunks'], geometry=geometry, chunk_batch_size=8, pack_chunks=False)
            pair = {k: v.cpu() for k, v in pair.items()}
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - t)
            torch.save(pair, folder / f'pair-{i}.pt')
            print(json.dumps({'row': i, 'seconds': timings[-1]}), flush=True)
    else:
        from transformers import Qwen3_5Config
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
        from vllm import LLM, SamplingParams
        from methods.CacheRepair.qwen38.vllm_capture import assemble_shards
        from methods.CacheRepair.qwen38.cache import canonicalize_target_kv, reposition_target_kv

        cfg = Qwen3_5Config.from_pretrained(a.model).text_config
        rotary = Qwen3_5TextRotaryEmbedding(cfg).cpu()
        layers = [i for i, kind in enumerate(cfg.layer_types) if kind == 'full_attention']
        engine = LLM(model=a.model, tensor_parallel_size=2, dtype='bfloat16',
            kv_cache_dtype='auto', quantization=None, max_model_len=16385,
            mamba_ssm_cache_dtype='float32',
            max_num_seqs=40, max_num_batched_tokens=32768,
            gpu_memory_utilization=.80, enforce_eager=True,
            compilation_config={'mode': 0}, language_model_only=True,
            enable_prefix_caching=False, enable_chunked_prefill=False,
            async_scheduling=False, seed=7,
            worker_extension_cls='methods.CacheRepair.qwen38.vllm_capture.CaptureWorkerExtension')
        loaded = time.perf_counter()
        timings = []
        for i, row in enumerate(rows):
            chunks = row['chunks']
            prompts = chunks + [sum(chunks, [])]
            capture = folder / f'capture-{i}'
            capture.mkdir()
            t = time.perf_counter()
            engine.collective_rpc('cr_capture_begin')
            results = engine.generate([{'prompt_token_ids': x} for x in prompts],
                SamplingParams(temperature=0, max_tokens=1, ignore_eos=True), use_tqdm=False)
            engine.collective_rpc('cr_capture_export', args=(str(capture.resolve()),))
            shards = [torch.load(capture / f'rank-{rank}.pt', weights_only=True) for rank in range(2)]
            for rank in range(2):
                captured=torch.load(capture/f'prompts-rank-{rank}.pt',weights_only=True)
                for request,expected in zip(results,prompts,strict=True):
                    matches=[v for k,v in captured.items() if k==request.request_id or k.startswith(request.request_id+'-')]
                    if matches != [expected]:
                        raise ValueError('Captured request prompt token IDs differ')
            native = assemble_shards(shards, [x.request_id for x in results], layers=layers)
            stale, offset = [], 0
            for chunk, kv in zip(chunks, native[:-1], strict=True):
                positions = torch.arange(len(chunk)).unsqueeze(0)
                if offset:
                    kv = reposition_target_kv(kv, positions, positions + offset, rotary)
                stale.append(canonicalize_target_kv(kv, positions + offset, rotary))
                offset += len(chunk)
            tokens = torch.tensor([sum(chunks, [])])
            pair = dict(stale=torch.cat(stale, dim=1),
                joint=canonicalize_target_kv(native[-1], torch.arange(offset).unsqueeze(0), rotary),
                tokens=tokens, chunk_ids=torch.tensor([[j for j, c in enumerate(chunks) for _ in c]]))
            timings.append(time.perf_counter() - t)
            torch.save(pair, folder / f'pair-{i}.pt')
            for path in capture.glob('*.pt'):
                path.unlink()
            print(json.dumps({'row': i, 'seconds': timings[-1]}), flush=True)
    import importlib.metadata
    metadata = dict(backend=a.mode, rows_sha256=hashlib.sha256(a.rows.read_bytes()).hexdigest(),
        model=str(Path(a.model).resolve()), config_sha256=hashlib.sha256((Path(a.model)/'config.json').read_bytes()).hexdigest(),
        load_seconds=loaded-started, pair_seconds=timings,
        packages={x: importlib.metadata.version(x) for x in ['torch','transformers','vllm']},
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), optimizer_updates=0)
    if a.mode=='vllm':
        metadata['engine_settings']=dict(tensor_parallel_size=2,dtype='bfloat16',kv_cache_dtype='auto',
            mamba_ssm_cache_dtype='float32',enable_prefix_caching=False,enable_chunked_prefill=False,
            enforce_eager=True,async_scheduling=False)
    (folder / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
