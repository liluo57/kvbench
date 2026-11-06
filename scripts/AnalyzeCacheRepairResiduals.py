"""Recover pooled raw-residual MSE gains/cosine from saved holdout L2 statistics.

No target forward, checkpoint loading, or GPU allocation is required. The saved
diagnostics accumulate squared errors/norms over all tokens and KV coordinates.
The raw error was formed in FP32 before its FP64 sum; cosine reconstructed by
the norm identity therefore inherits that original floating-point precision.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


ALPHAS = (0., .25, .5, .75, 1., 1.25)


def residual_metrics(error_ratio, prediction_ratio):
    """e=||prediction-truth||/||truth||, r=||prediction||/||truth||.

    All norms must refer to the same concatenated, unnormalized residual vectors.
    Scalar prediction scaling gives G(alpha)=2*alpha*dot/true_ss-alpha²*pred_ss/true_ss.
    """
    if not all(math.isfinite(v) and v >= 0 for v in (error_ratio, prediction_ratio)):
        raise ValueError('Expected finite nonnegative L2 ratios')
    error_relative_ss, prediction_relative_ss = error_ratio**2, prediction_ratio**2
    dot_relative_ss = (1. + prediction_relative_ss - error_relative_ss) / 2.
    cosine = dot_relative_ss / prediction_ratio if prediction_ratio else None
    if cosine is not None and abs(cosine) > 1. + 1e-6:
        raise ValueError('Inconsistent error and prediction norms')
    if cosine is None and not math.isclose(error_ratio, 1., abs_tol=1e-6):
        raise ValueError('Zero prediction requires error norm equal to truth norm')
    if cosine is not None:
        cosine = max(-1., min(1., cosine))
    return {'g_mse_raw': 1. - error_relative_ss,
        'cosine_raw_pooled': cosine,
        'relative_error_l2': error_ratio,
        'predicted_to_true_l2': prediction_ratio,
        'dot_over_true_ss': dot_relative_ss,
        'optimal_nonnegative_alpha_raw_mse': max(0., dot_relative_ss / prediction_relative_ss) if prediction_relative_ss else None,
        'g_mse_raw_by_alpha': {f'{a:g}': 2*a*dot_relative_ss-a*a*prediction_relative_ss for a in ALPHAS}}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(training, output, dataset):
    manifest = json.loads((dataset / 'dataset.json').read_text())
    validation = dataset / 'validation.jsonl'
    validation_sha = sha(validation)
    assert validation_sha == manifest['files_sha256']['validation.jsonl']
    rows = [json.loads(s) for s in validation.read_text().splitlines() if s.strip()]
    tokens = sum(sum(map(len, row['chunks'])) for row in rows)
    assert len(rows) == 500
    records = []
    for epoch in range(1, 7):
        checkpoint = training / f'checkpoint-{epoch*12500}'
        source = checkpoint / 'validation.json'
        saved = json.loads(source.read_text())
        assert Path(saved['path']).resolve() == checkpoint.resolve() and saved['reload_exact']
        d = saved['validation']
        assert d['examples'] == len(rows) and d['tokens'] == tokens
        metrics = residual_metrics(d['relative_residual_l2_error'], d['predicted_to_true_residual_l2'])
        metrics['g_mse_normalized_example_mean'] = 1. - d['normalized_residual_mse'] / d['zero_residual_baseline_mse']
        layer_gain = [[1.-m/b for m, b in zip(ms, bs, strict=True)] for ms, bs in zip(
            d['per_attention_layer_k_v_normalized_mse'], d['per_attention_layer_k_v_zero_residual_mse'], strict=True)]
        records.append({'epoch': epoch, 'step': epoch*12500, 'examples': len(rows), 'tokens': tokens,
            **metrics, 'normalized_residual_mse': d['normalized_residual_mse'],
            'normalized_zero_residual_mse': d['zero_residual_baseline_mse'],
            'per_attention_layer_k_v_normalized_g_mse': layer_gain,
            'source': str(source.resolve()), 'validation_json_sha256': sha(source),
            'checkpoint_sha256': sha(checkpoint / 'model.safetensors')})
    output.mkdir(parents=True, exist_ok=False)
    result = {'status': 'completed', 'scope': 'Unchanged 500-example CoRAG holdout, not the KVBench ten-task test set',
        'prediction_alpha': 1., 'coordinates': 'Canonical FP32 full-attention KV, pre-materialization and pre-BF16-writeback',
        'aggregation_raw': 'All examples, tokens, attention layers, KV heads, K/V and dimensions concatenated; element-weighted',
        'aggregation_normalized': 'Mean per-example MSE after coordinate-wise division by sigma_delta, matching training diagnostics',
        'method': 'Algebraic reconstruction from already saved aggregate L2 ratios; no new forward or GPU usage',
        'formulas': {'G_MSE': '1 - e^2', 'cosine': '(1 + r^2 - e^2) / (2*r)',
            'definitions': 'e=||prediction-truth||/||truth||, r=||prediction||/||truth||',
            'G_MSE_alpha': 'alpha*(1+r^2-e^2) - alpha^2*r^2'},
        'precision': 'Saved error uses FP32 subtraction and FP64 sum; derived cosine inherits FP32 subtraction roundoff',
        'limitations': ['Pooled cosine is not a mean of per-example, per-token or per-head cosine similarities',
            'No assertion that every example or KV coordinate improves',
            'KV-space holdout improvement does not imply generation-quality improvement on the ten tasks'],
        'validation_data': str(validation.resolve()), 'validation_data_sha256': validation_sha,
        'statistics_source': str((Path(__file__).resolve().parents[1] / 'methods/CacheRepair/qwen38/train.py')),
        'statistics_source_sha256': sha(Path(__file__).resolve().parents[1] / 'methods/CacheRepair/qwen38/train.py'),
        'analysis_script_sha256': sha(Path(__file__)), 'epochs': records}
    (output / 'residual-metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    columns = ['epoch', 'step', 'g_mse_raw', 'cosine_raw_pooled', 'predicted_to_true_l2',
        'g_mse_normalized_example_mean', 'optimal_nonnegative_alpha_raw_mse']
    with (output / 'residual-metrics.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows([[r[c] for c in columns] for r in records])
    with (output / 'alpha-mse-gain.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', *[f'alpha={a:g}' for a in ALPHAS]])
        writer.writerows([[r['epoch'], *[r['g_mse_raw_by_alpha'][f'{a:g}'] for a in ALPHAS]] for r in records])
    lines = ['# Residual direction and relative MSE gain', '',
        f'Same {len(rows)} held-out examples / {tokens:,} tokens at all checkpoints. Alpha=1.',
        'Raw metrics use canonical KV before serving materialization and BF16 writeback.',
        'Recovered from saved global L2 statistics without any target/repairer forward or GPU allocation.', '',
        '| Epoch | Step | G_MSE (raw KV) | Pooled cosine (raw KV) | Pred/true L2 | G_MSE (normalized) |',
        '|---|---:|---:|---:|---:|---:|']
    for r in records:
        lines.append(f'| {r["epoch"]} | {r["step"]:,} | {100*r["g_mse_raw"]:.4f}% | {r["cosine_raw_pooled"]:.6f} | {r["predicted_to_true_l2"]:.6f} | {100*r["g_mse_normalized_example_mean"]:.4f}% |')
    lines += ['', 'Let e=||prediction-truth||/||truth||, r=||prediction||/||truth||.',
        'Then G_MSE=1-e² and cosine=(1+r²-e²)/(2r), by the squared-distance identity.',
        'Cosine uses concatenated validation vectors; it is not a mean of per-example cosines.',
        'Normalized MSE divides each coordinate by sigma_delta, then averages examples, as in training diagnostics.',
        'Raw MSE gain weights all KV elements equally and is the user-requested metric on the denormalized residual.', '',
        '## Expected raw KV MSE gain after scalar alpha', '',
        'These gains follow algebraically from the same holdout statistics. They are not generation-quality scores.', '',
        '| Epoch | ' + ' | '.join(f'alpha={a:g}' for a in ALPHAS) + ' |', '|---|' + '---:|'*6]
    for r in records:
        lines.append('| ' + str(r['epoch']) + ' | ' + ' | '.join(f'{100*r["g_mse_raw_by_alpha"][f"{a:g}"]:.2f}%' for a in ALPHAS) + ' |')
    lines += ['', 'Epoch 3/6 least-squares scalar alpha on this holdout: ' +
        ' / '.join(f'{records[e-1]["optimal_nonnegative_alpha_raw_mse"]:.6f}' for e in (3, 6)) + '.',
        'The learned correction has positive pooled alignment and reduces holdout KV error.',
        'This does not establish improvement for every example, token, layer/head, or downstream generation task.', '']
    (output / 'report.md').write_text('\n'.join(lines))
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', required=True, type=Path)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    analyze(args.training.resolve(), args.output.resolve(), args.dataset.resolve())
