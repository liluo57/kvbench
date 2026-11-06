import pytest
import torch

from methods.CacheRepair.qwen38.vllm_audit import difference
from methods.CacheRepair.qwen38.vllm_capture import assemble_shards


def test_tp_capture_restores_original_head_and_layer_order():
    shards=[{'req':{layer:torch.full((3,2,2,256),rank*10+layer,dtype=torch.bfloat16)
        for layer in [3,7]}} for rank in range(2)]
    kv=assemble_shards(shards,['req'],layers=[3,7])[0]
    assert kv.shape==(1,3,2,4,2,256)
    assert (kv[:,:,0,:2]==3).all() and (kv[:,:,0,2:]==13).all()
    assert (kv[:,:,1,:2]==7).all() and (kv[:,:,1,2:]==17).all()
    with pytest.raises(ValueError):assemble_shards(shards[:1],['req'],layers=[3,7])
    with pytest.raises(ValueError):assemble_shards(shards,['wrong'],layers=[3,7])


def test_migration_never_calls_small_error_unchanged():
    a=torch.ones(4,4);b=a.clone();b[0,0]+=1e-5
    assert difference(a,a)['bitwise_equal']
    result=difference(a,b)
    assert not result['bitwise_equal'] and result['max_abs']>0
