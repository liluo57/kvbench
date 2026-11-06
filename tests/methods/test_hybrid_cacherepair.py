"""Hybrid ownership, TP embedding, native rotary and full-prefix tiled parity."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from methods import HybridCacheRepair
from methods.HybridCacheRepair import Runtime as runtime
from methods.CacheRepair.qwen38 import chunk_causal_mask, materialize_target_kv
from tests.methods.test_cacherepair_qwen38 import geometry, repairer, activate
from tests.methods.test_hypic import method


@pytest.mark.parametrize("alpha", [0., .25, .5, .75, 1., 1.25])
@pytest.mark.parametrize("length,tile", [(19, 4), (19, 64), (1, 1)])
def test_tiled_inference_matches_nonzero_full_model(repairer, length, tile, alpha):
    activate(repairer)
    repairer.set_normalization(torch.rand_like(repairer.sigma_stale)+.2, torch.rand_like(repairer.sigma_delta)+.2)
    stale = torch.randn(1, length, *repairer.target_config.kv_shape)
    tokens = torch.randint(0, 40, (1, length))
    chunks = torch.arange(length)[None] // 7
    writes, reads = [], []
    def read(start, end):
        reads.append((start, end))
        return stale[:, start:end]
    with torch.no_grad():
        expected = stale.float() + alpha * repairer(stale, tokens, chunks).float() * repairer.sigma_delta.float()
    runtime.streamed_repair(repairer, tokens, chunks, read,
        lambda start, end, value: writes.append(value), tile=tile, alpha=alpha)
    torch.testing.assert_close(torch.cat(writes, 1), expected, rtol=2e-5, atol=2e-6)
    assert max(end-start for start, end in reads) <= tile
    assert len(reads) == 2*((length+tile-1)//tile)
    if alpha == 0:
        assert torch.equal(torch.cat(writes, 1), stale)


def test_default_alpha_is_bitwise_equal_to_explicit_one(repairer):
    activate(repairer)
    stale = torch.randn(1, 9, *repairer.target_config.kv_shape)
    tokens, chunks = torch.arange(9)[None], torch.tensor([[0]*4+[1]*5])
    outputs = []
    for kwargs in ({}, {"alpha": 1.}):
        writes = []
        runtime.streamed_repair(repairer, tokens, chunks, lambda a, b: stale[:, a:b],
            lambda a, b, value: writes.append(value), tile=4, **kwargs)
        outputs.append(torch.cat(writes, 1))
    assert torch.equal(*outputs)


def test_aligned_additive_mask_is_exact_chunk_causality():
    chunks = torch.tensor([[0, 0, 1, 1, 2], [0, 1, 1, 2, 2]])
    mask = runtime.chunk_bias(chunks, tile=2)
    assert mask.shape == (2, 1, 5, 5) and mask.stride(-2) % 16 == 0
    assert torch.equal(mask == 0, chunk_causal_mask(chunks))
    assert torch.isneginf(mask[~chunk_causal_mask(chunks)]).all()
    with pytest.raises(ValueError, match="nonnegative"):
        runtime.chunk_bias(torch.tensor([[0, -1]]))


def test_logical_embedding_shards_reconstruct_without_registering_target(geometry):
    geometry = replace(geometry, vocab_size=8)
    full = torch.randn(8, geometry.hidden_size)
    ids = torch.tensor([[0, 4, 7, 3]])
    outputs = []
    for rank in range(2):
        native = nn.Embedding(4, geometry.hidden_size)
        native.weight.data.copy_(full[rank*4:(rank+1)*4])
        native.org_vocab_size, native.num_added_embeddings = 8, 0
        native.shard_indices = SimpleNamespace(org_vocab_start_index=rank*4, org_vocab_end_index=(rank+1)*4)
        embed = runtime.LogicalEmbedding(native, geometry, lambda x: x)
        assert embed.weight.shape == (8, geometry.hidden_size)
        assert not list(embed.parameters()) and not native.weight.requires_grad
        outputs.append(embed(ids))
    torch.testing.assert_close(sum(outputs), torch.nn.functional.embedding(ids, full), rtol=0, atol=0)


def rotary_view(geometry):
    dim, theta = 4, 1000
    frequency = 1. / (theta ** (torch.arange(0, dim, 2).float()/dim))
    angles = torch.arange(1024).float()[:, None]*frequency
    native = SimpleNamespace(rotary_dim=dim, head_size=geometry.head_dim, base=theta,
        is_neox_style=True, mrope_section=[1, 1, 0], cos_sin_cache=torch.cat((angles.cos(), angles.sin()), -1))
    spec = dict(rotary_dim=dim, parameters={"rope_theta": theta}, mrope_section=[1, 1, 0],
        inv_freq_fp32_sha256=hashlib.sha256(frequency.numpy().tobytes()).hexdigest())
    return runtime.NativeRotaryView(native, replace(geometry, target_rope_spec=spec))


def test_native_rotary_view_tracks_actual_table_and_preserves_nonrotary(geometry):
    rotary = rotary_view(geometry)
    canonical = torch.randn(1, 5, *geometry.kv_shape)
    positions = torch.tensor([[10, 11, 12, 13, 14]])
    native = materialize_target_kv(canonical, positions, rotary, dtype=torch.float32)
    restored = runtime.canonicalize_target_kv(native, positions, rotary)
    torch.testing.assert_close(restored, canonical)
    assert torch.equal(native[..., 1, :], canonical[..., 1, :])
    assert torch.equal(native[..., 0, 4:], canonical[..., 0, 4:])
    rotary.native.cos_sin_cache = rotary.native.cos_sin_cache.bfloat16()
    cos, _ = rotary(torch.empty(1, 5, 1), positions)
    assert torch.equal(cos[..., :2], rotary.native.cos_sin_cache[positions][..., :2].float())


@pytest.mark.parametrize("alpha", [0., .25, .5, .75, 1., 1.25])
def test_repair_writes_only_private_kv_and_keeps_source_immutable(repairer, monkeypatch, alpha):
    activate(repairer)
    g = repairer.target_config
    native_rope = rotary_view(g)
    original = torch.randn(1, 11, *g.kv_shape)
    tokens, chunk_ids = torch.arange(11)[None], torch.tensor([[0]*3+[1]*8])
    chunks = []
    for start, end in [(0, 3), (3, 11)]:
        positions = torch.arange(end-start)[None]
        chunks.append((start, end, materialize_target_kv(original[:, start:end], positions, native_rope, dtype=torch.float32)))
    before = [x[2].clone() for x in chunks]
    keys = {layer:torch.randn(40, g.n_kv_heads, g.head_dim) for layer in g.full_attention_layer_indices}
    values = {layer:torch.randn_like(keys[layer]) for layer in keys}
    public = {layer:(keys[layer][:20].clone(), values[layer][:20].clone()) for layer in keys}
    obj = object.__new__(runtime.HybridPrefill)
    obj.geometry, obj.device, obj.rank, obj.world, obj.tile = g, torch.device('cpu'), 0, 1, 4
    obj.model, obj.rotary = repairer, native_rope
    obj.alpha = alpha
    obj.kv = SimpleNamespace(dtype=torch.float32, get_key_buffer=keys.__getitem__, get_value_buffer=values.__getitem__)
    original_stream = runtime.streamed_repair
    monkeypatch.setattr(runtime, 'streamed_repair', lambda *a, **kw: original_stream(*a, **{**kw, 'efficient':False}))
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda *a: None)
    obj._repair(tokens[0].tolist(), chunks, torch.arange(20, 31))
    with torch.no_grad():
        corrected = original + alpha * repairer(original, tokens, chunk_ids) * repairer.sigma_delta
        expected = materialize_target_kv(corrected, torch.arange(11)[None], native_rope, dtype=torch.float32)
    for i, layer in enumerate(g.full_attention_layer_indices):
        torch.testing.assert_close(keys[layer][20:31], expected[0, :, i, :, 0], rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(values[layer][20:31], expected[0, :, i, :, 1], rtol=2e-5, atol=2e-6)
        assert torch.equal(keys[layer][:20], public[layer][0])
        assert torch.equal(values[layer][:20], public[layer][1])
    assert all(torch.equal(x[2], y) for x, y in zip(chunks, before))


def test_method_preserves_tokens_prepares_both_branches_and_rejects_retention(tmp_path):
    obj = method(HybridCacheRepair, repairerPath=tmp_path)
    obj.Prepare([[[1, 2, 3], [4, 5]]])
    assert len(obj.engine.calls) == 4
    assert obj.engine.calls[1][0] == [1, 2, 3, 101]
    result = obj.Run([[10, 4, 5, 11, 1, 2, 3, 12]], maxNewTokens=8)[0]
    assert obj.engine.calls[-1][0] == [10, 4, 5, 11, 1, 2, 3, 12]
    assert result.metadata['gdn_state_source'] == 'hypic_trr'
    assert result.metadata['residual_alpha'] == 1.
    assert result.output_ids == [31, 99, 32]
    with pytest.raises(NotImplementedError, match='retention'):
        obj.Run([[1, 2]], retainOutput=[True])
    obj.Reset()
    assert obj.engine.flushes == 1


def test_no_hit_metadata_does_not_claim_learned_repair(tmp_path):
    obj = method(HybridCacheRepair, repairerPath=tmp_path)
    generate = obj.engine.generate
    def uncached(**kwargs):
        for row in generate(**kwargs):
            row['meta_info']['cached_tokens'] = 0
            yield row
    obj.engine.generate = uncached
    obj.Prepare([[]])
    result = obj.Run([[10, 20]], maxNewTokens=8)[0]
    assert not result.metadata['repairer_invoked']
    assert result.metadata['attention_kv_source'] == result.metadata['gdn_state_source'] == 'full_prefill'


@pytest.mark.parametrize('kwargs', [{'fullPrefill':True}, {'picMode':'addition'}, {'dtype':'float16'}, {'kvTileTokens':0},
    {'residualAlpha':float('nan')}, {'residualAlpha':float('inf')}, {'residualAlpha':-1}, {'residualAlpha':True}])
def test_invalid_configuration_fails_without_loading_target(tmp_path, kwargs):
    with pytest.raises(ValueError):
        HybridCacheRepair(repairerPath=tmp_path, **kwargs)
