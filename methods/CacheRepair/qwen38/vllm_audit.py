"""Strict backend migration audit. Approximate agreement never means unchanged."""
from __future__ import annotations

import json

import torch


def difference(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        raise ValueError('Backend output shape/dtype differs')
    a, b = a.double(), b.double()
    d = a - b
    return dict(bitwise_equal=torch.equal(a, b), max_abs=d.abs().max().item(),
        rmse=d.square().mean().sqrt().item(),
        relative_l2=(d.square().sum() / a.square().sum().clamp_min(1e-30)).sqrt().item(),
        cosine=(a.flatten() @ b.flatten() / (a.norm()*b.norm()).clamp_min(1e-30)).item())


def compare_runs(directory, rows):
    runs = [json.loads((directory / backend / 'run.json').read_text()) for backend in ['native','vllm']]
    for key in ['rows_sha256','model','config_sha256']:
        if runs[0][key] != runs[1][key]:
            raise ValueError(f'Backend provenance differs: {key}')
    results = []
    for i, row in enumerate(rows):
        a, b = [torch.load(directory / backend / f'pair-{i}.pt', weights_only=True) for backend in ['native','vllm']]
        if any(not torch.equal(a[k], b[k]) for k in ['tokens','chunk_ids']):
            raise ValueError('Token or document boundaries differ')
        values = {key: difference(a[key], b[key]) for key in ['stale','joint']}
        values['delta'] = difference(a['joint']-a['stale'], b['joint']-b['stale'])
        values['k_v'] = {key: {name:difference(a[key][...,j,:],b[key][...,j,:])
            for j,name in enumerate(['K','V'])} for key in ['stale','joint']}
        values['per_layer'] = [{key: difference(a[key][:,:,l], b[key][:,:,l])
            for key in ['stale','joint']} for l in range(a['stale'].shape[2])]
        results.append(dict(row=i, task=row.get('source',row.get('task')), tokens=a['tokens'].numel(), **values))
    exact = all(r[key]['bitwise_equal'] for r in results for key in ['stale','joint','delta'])
    return dict(status='passed' if exact else 'rejected_numerical_change',
        kv_bitwise_equal=exact, production_switch_allowed=exact, rows=results, runs=runs,
        note='No tolerance-based migration or automatic production switch; no optimizer updates.')
