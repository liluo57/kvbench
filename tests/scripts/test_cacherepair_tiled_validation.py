"""Tiled diagnostics preserve residuals and aggregate the requested metrics."""
import math

import pytest
import torch

from scripts.EvaluateCacheRepairResiduals import measure_pair, summarize
from tests.methods.test_cacherepair_qwen38 import geometry, repairer, activate


def test_tiled_metrics_match_dense_nonzero_predictions(repairer):
    activate(repairer)
    repairer.set_normalization(torch.rand_like(repairer.sigma_stale)+.2,
        torch.rand_like(repairer.sigma_delta)+.2)
    stale = torch.randn(1, 11, *repairer.target_config.kv_shape)
    pair = {'stale':stale,'joint':stale+torch.randn_like(stale),
        'tokens':torch.arange(11)[None],'chunk_ids':torch.tensor([[0]*3+[1]*8])}
    actual = measure_pair(repairer,pair,tile=4,efficient=False,check_dense=True)
    with torch.no_grad():
        prediction=repairer(pair['stale'],pair['tokens'],pair['chunk_ids'])
    truth=pair['joint']-pair['stale']
    normalized_error=(prediction-truth/repairer.sigma_delta).double()
    raw_pred=prediction*repairer.sigma_delta
    assert actual['normalized_mse']==pytest.approx(normalized_error.square().mean().item(),rel=2e-6)
    assert actual['raw_error_ss']==pytest.approx((raw_pred-truth).double().square().sum().item(),rel=2e-6)
    assert actual['raw_dot']==pytest.approx((raw_pred.double()*truth.double()).sum().item(),abs=2e-5)
    result=summarize([actual])
    cosine=torch.nn.functional.cosine_similarity(raw_pred.double().flatten(),truth.double().flatten(),dim=0).item()
    assert result['raw_cosine_pooled']==pytest.approx(cosine,abs=1e-6)


def test_example_mean_and_element_weighting_are_distinct():
    rows=[dict(elements=1,tokens=1,normalized_error_ss=1.,normalized_true_ss=4.,raw_error_ss=1.,
        raw_true_ss=4.,raw_pred_ss=1.,raw_dot=2.,normalized_mse=1.,raw_mse=1.),
        dict(elements=3,tokens=3,normalized_error_ss=12.,normalized_true_ss=12.,raw_error_ss=12.,
        raw_true_ss=12.,raw_pred_ss=0.,raw_dot=0.,normalized_mse=4.,raw_mse=4.)]
    r=summarize(rows)
    assert r['normalized_mse_example_mean']==2.5
    assert r['normalized_mse_element_mean']==3.25
    assert r['raw_g_mse']==1-13/16
    assert r['raw_cosine_pooled']==2/math.sqrt(16)
    assert r['raw_g_mse_by_alpha']['0.0']==0
