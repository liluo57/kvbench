"""Long-prefix storage changes preserve predictions, gradients and diagnostics."""
import copy

import pytest
import torch

from methods.CacheRepair.backend.cacherepair.model import CacheRepair as OfficialRepairer
from methods.CacheRepair.qwen38 import long_training
from methods.CacheRepair.qwen38.model import chunk_causal_mask
from methods.CacheRepair.qwen38.train import validation_diagnostics
from tests.methods.test_cacherepair_qwen38 import geometry,repairer,activate


def inputs(model):
    stale=torch.randn(2,9,*model.target_config.kv_shape)
    joint=stale+torch.randn_like(stale)
    tokens=torch.randint(0,model.target_config.vocab_size,(2,9))
    chunks=torch.tensor([[0,0,0,1,1,2,2,2,2],[0,0,1,1,1,1,-1,-1,-1]])
    return dict(stale=stale,joint=joint,tokens=tokens,chunk_ids=chunks)


def test_aligned_additive_bias_preserves_chunk_and_padding_visibility():
    ids=torch.tensor([[0]*3+[1]*12+[-1]*2,[0]*8+[1]*9])
    bias=long_training.aligned_chunk_bias(ids,tile=3)
    assert bias.stride(-2)%16==0
    assert torch.equal(torch.isfinite(bias),chunk_causal_mask(ids))


def test_tiled_checkpointed_loss_and_all_parameter_gradients_match_dense(repairer):
    activate(repairer)
    repairer.set_normalization(torch.rand_like(repairer.sigma_stale)+.2,
        torch.rand_like(repairer.sigma_delta)+.2)
    other=copy.deepcopy(repairer);pair=inputs(repairer)
    before=set(repairer.state_dict())
    expected=OfficialRepairer.loss(repairer,**pair);expected.backward()
    long_training.configure(other,tile=4,efficient=False)
    actual=long_training.loss(other,**pair);actual.backward()
    torch.testing.assert_close(actual,expected,rtol=2e-6,atol=2e-7)
    assert before==set(other.state_dict())
    for (name,p),(name2,q) in zip(repairer.named_parameters(),other.named_parameters()):
        assert name==name2 and p.grad is not None and q.grad is not None
        torch.testing.assert_close(q.grad,p.grad,rtol=3e-4,atol=3e-6,msg=name)
    assert other.tok_embed.weight.grad is None


def test_tiled_forward_and_legacy_diagnostics_agree(repairer):
    activate(repairer);other=copy.deepcopy(repairer);pair=inputs(repairer)
    rows=[{'chunks':[[1,2],[3]]}]
    make=lambda target,chunks:pair
    with torch.no_grad():expected=repairer(pair['stale'],pair['tokens'],pair['chunk_ids'])
    baseline=validation_diagnostics(repairer,None,rows,make)
    long_training.configure(other,tile=4,efficient=False)
    with torch.no_grad():actual=long_training.forward(other,pair['stale'],pair['tokens'],pair['chunk_ids'])
    torch.testing.assert_close(actual,expected,rtol=3e-5,atol=1e-6)
    measured=validation_diagnostics(other,None,rows,make)
    for key in baseline:
        torch.testing.assert_close(torch.tensor(measured[key]),torch.tensor(baseline[key]),rtol=3e-5,atol=1e-6,msg=key)
