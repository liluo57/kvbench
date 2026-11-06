"""Recover vector alignment without storing the original dense KV tensors."""
import math

import pytest

from scripts.AnalyzeCacheRepairResiduals import ALPHAS, residual_metrics


@pytest.mark.parametrize('prediction', [[0., 0., 0.], [1., 2., 3.], [.5, 1., 1.5], [-1., -2., -3.], [2., -1., 0.], [3., 1., -1.]])
def test_saved_l2_ratios_recover_direct_vector_metrics(prediction):
    truth = [1., 2., 3.]
    true_ss = sum(t*t for t in truth)
    pred_ss = sum(p*p for p in prediction)
    error_ss = sum((p-t)**2 for p, t in zip(prediction, truth))
    dot = sum(p*t for p, t in zip(prediction, truth))
    result = residual_metrics(math.sqrt(error_ss/true_ss), math.sqrt(pred_ss/true_ss))
    assert result['g_mse_raw'] == pytest.approx(1.-error_ss/true_ss)
    if pred_ss:
        assert result['cosine_raw_pooled'] == pytest.approx(dot/math.sqrt(pred_ss*true_ss), abs=1e-14)
    else:
        assert result['cosine_raw_pooled'] is None
    for alpha in ALPHAS:
        direct = 1.-sum((alpha*p-t)**2 for p, t in zip(prediction, truth))/true_ss
        assert result['g_mse_raw_by_alpha'][f'{alpha:g}'] == pytest.approx(direct, abs=1e-14)


def test_pooled_cosine_is_not_mean_of_example_cosines():
    # A large aligned example dominates the pooled metric while an equally
    # numerous small anti-aligned example makes the unweighted mean cosine zero.
    result = residual_metrics(math.sqrt(4./101.), 1.)
    assert result['cosine_raw_pooled'] == pytest.approx(99./101.)
    assert result['cosine_raw_pooled'] != 0.


@pytest.mark.parametrize('e,r', [(-1., 1.), (1., -1.), (math.nan, 1.), (1., math.inf), (0., 0.), (3., 1.)])
def test_invalid_or_inconsistent_norms_rejected(e, r):
    with pytest.raises(ValueError):
        residual_metrics(e, r)
