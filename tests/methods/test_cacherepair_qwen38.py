"""Repairer body tests, independent of installed target-model Transformers support."""

from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from safetensors.torch import load_file

from methods.CacheRepair.backend.cacherepair.model import CacheRepair as OfficialCacheRepair
from methods.CacheRepair.qwen38 import (
    QwenCacheRepair, RepairConfig, TargetGeometry, chunk_causal_mask,
    load_repairer, pack_target_kv, replace_target_kv, save_repairer,
)


@pytest.fixture
def geometry():
    return TargetGeometry(
        target_model_id="test-qwen", config_sha256="test-config",
        full_attention_layer_indices=(1, 4), num_hidden_layers=5,
        n_kv_heads=2, head_dim=8, hidden_size=24, n_attention_heads=4,
        vocab_size=40, max_position_embeddings=1024,
        target_rope_spec={"rotary_dim": 4, "inv_freq_fp32_sha256": "test-rope"},
    )


@pytest.fixture
def repairer(geometry):
    return QwenCacheRepair(
        geometry, RepairConfig(d_b=16, n_heads=2, n_blocks=2, d_seg=3),
        nn.Embedding(geometry.vocab_size, geometry.hidden_size),
    ).eval()


def activate(model):
    """Exercise actual attention, reinjection and output rather than zero gates."""
    with torch.no_grad():
        model.v_head.weight.normal_(std=0.1)
        model.v_head.bias.normal_(std=0.1)
        for block, projection in zip(model.blocks, model.stale_reinject):
            block.ada.normal_(std=0.2)
            projection.weight.normal_(std=0.1)
            projection.bias.normal_(std=0.1)


def inputs(model, batch=2, length=5):
    return (
        torch.randn(batch, length, *model.target_config.kv_shape),
        torch.randint(0, model.target_config.vocab_size, (batch, length)),
        torch.tensor([[0, 0, 1, 1, 2]]).expand(batch, -1).clone(),
    )


def test_m_shape_count_and_zero_identity(geometry):
    target = replace(
        geometry, full_attention_layer_indices=tuple(range(3, 64, 4)),
        num_hidden_layers=64, n_kv_heads=4, head_dim=256,
        hidden_size=5120, n_attention_heads=24,
    )
    # A small vocabulary tests the full M body without allocating target weights.
    model = QwenCacheRepair(target, RepairConfig(), nn.Embedding(40, 5120))
    assert sum(p.numel() for p in model.parameters()) == 54_017_664
    assert model.stale_enc.W.shape == (128, 256, 16)
    assert model.stale_enc.out.in_features == 2048
    assert model.v_head.out_features == 32768
    stale = torch.randn(1, 3, *target.kv_shape)
    tokens, chunks = torch.tensor([[1, 2, 3]]), torch.tensor([[0, 0, 1]])
    with torch.no_grad():
        features = model.stale_enc(stale.reshape(1, 3, 128, 256))
        assert features.shape == (1, 3, 640)
        assert torch.count_nonzero(model(stale, tokens, chunks)) == 0
        assert torch.equal(model.repair(stale, tokens, chunks), stale)
    for projection, block in zip(model.stale_reinject, model.blocks):
        assert torch.count_nonzero(projection.weight) == 0
        assert torch.count_nonzero(projection.bias) == 0
        assert torch.count_nonzero(block.ada) == 0
        assert not block.ln1.elementwise_affine and not block.ln2.elementwise_affine


def test_matches_official_network_with_nonzero_weights(repairer):
    activate(repairer)
    repairer.set_normalization(torch.rand_like(repairer.sigma_stale) + 0.1,
                               torch.rand_like(repairer.sigma_delta) + 0.1)
    official = OfficialCacheRepair(repairer.target_config, repairer.config, repairer.tok_embed).eval()
    result = official.load_state_dict(repairer.state_dict(), strict=False)
    assert result.missing_keys == ["tok_embed.weight"] and not result.unexpected_keys
    stale, tokens, chunks = inputs(repairer)
    with torch.no_grad():
        torch.testing.assert_close(repairer(stale, tokens, chunks), official(stale, tokens, chunks))
        torch.testing.assert_close(repairer.repair(stale, tokens, chunks), official.repair(stale, tokens, chunks))


def test_exact_chunk_mask():
    ids = torch.tensor([[0, 0, 1, -1], [0, 1, -1, -1]])
    mask = chunk_causal_mask(ids)
    assert mask.shape == (2, 1, 4, 4)
    assert torch.equal(mask[0, 0], torch.tensor([
        [1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 0, 1],
    ], dtype=torch.bool))
    assert not mask[1, 0, :2, 2:].any()


def test_later_chunks_invisible_same_chunk_bidirectional(repairer):
    activate(repairer)
    stale, tokens, chunks = inputs(repairer)
    with torch.no_grad():
        baseline = repairer(stale, tokens, chunks)
        changed = stale.clone()
        changed[:, 2:] = torch.randn_like(changed[:, 2:]) * 5
        later = repairer(changed, tokens, chunks)
        torch.testing.assert_close(later[:, :2], baseline[:, :2], rtol=0, atol=0)
        changed = stale.clone()
        changed[:, 1] = torch.randn_like(changed[:, 1]) * 5
        same_chunk = repairer(changed, tokens, chunks)
        assert not torch.allclose(same_chunk[:, 0], baseline[:, 0])


def test_padding_cannot_leak_and_has_finite_zero_output(repairer):
    activate(repairer)
    stale, tokens, chunks = inputs(repairer)
    chunks[1, 3:] = -1
    with torch.no_grad():
        baseline = repairer(stale, tokens, chunks)
        stale[1, 3:] = float("nan")
        tokens[1, 3:] = -100
        actual = repairer(stale, tokens, chunks)
        assert torch.isfinite(actual).all()
        assert torch.count_nonzero(actual[1, 3:]) == 0
        torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
        torch.testing.assert_close(actual[1:2, :3], repairer(stale[1:2, :3], tokens[1:2, :3], chunks[1:2, :3]))
        chunks[:] = -1
        assert torch.count_nonzero(repairer(stale, tokens, chunks)) == 0


def test_embedding_shared_frozen_and_not_moved_or_serialized(repairer):
    embedding = repairer.tok_embed
    assert not embedding.weight.requires_grad
    assert all(p is not embedding.weight for p in repairer.parameters())
    assert not any("embed" in key for key in repairer.state_dict())
    repairer.to(dtype=torch.bfloat16)
    assert repairer.tok_embed is embedding and embedding.weight.dtype == torch.float32


def test_coordinate_normalization_and_segment_order(repairer):
    stale, tokens, chunks = inputs(repairer)
    sigma = torch.arange(1, repairer.sigma_stale.numel() + 1).reshape_as(repairer.sigma_stale).float()
    repairer.set_normalization(sigma, sigma * 2)
    with torch.no_grad():
        repairer.v_head.bias.copy_(torch.arange(repairer.v_head.out_features).float())
        prediction = repairer(stale, tokens, chunks)
        expected = torch.arange(repairer.v_head.out_features).reshape(repairer.target_config.kv_shape).float()
        assert torch.equal(prediction[0, 0], expected)
        torch.testing.assert_close(repairer.repair(stale, tokens, chunks), stale + expected * sigma * 2)
    with pytest.raises(ValueError, match="finite positive"):
        repairer.set_normalization(sigma, sigma * 0)


def hybrid_cache(geometry):
    layers = []
    for index in range(geometry.num_hidden_layers):
        if index in geometry.full_attention_layer_indices:
            # Unique constants identify each layer, head, type and dimension.
            key = torch.arange(2 * 3 * geometry.head_dim).reshape(1, 2, 3, -1).float() + 1000 * index
            layers.append(SimpleNamespace(keys=key, values=key + 500))
        else:
            layers.append(SimpleNamespace(conv_states=torch.randn(1, 4, 2), recurrent_states=torch.randn(1, 2, 4, 4)))
    return SimpleNamespace(layers=layers, seen_tokens=3)


def test_cache_mapping_order_and_gdn_untouched(geometry):
    cache = hybrid_cache(geometry)
    packed = pack_target_kv(cache, geometry)
    assert packed.shape == (1, 3, *geometry.kv_shape)
    for index, native_index in enumerate(geometry.full_attention_layer_indices):
        for head in range(geometry.n_kv_heads):
            assert torch.equal(packed[:, :, index, head, 0], cache.layers[native_index].keys[:, head])
            assert torch.equal(packed[:, :, index, head, 1], cache.layers[native_index].values[:, head])
    gdn = [(layer, layer.conv_states, layer.recurrent_states, layer.recurrent_states.clone())
           for layer in cache.layers if hasattr(layer, "recurrent_states")]
    first_key = cache.layers[1].keys
    replace_target_kv(cache, packed + 1, geometry)
    assert cache.layers[1].keys is first_key and cache.seen_tokens == 3
    assert torch.equal(pack_target_kv(cache, geometry), packed + 1)
    for layer, conv, recurrent, original in gdn:
        assert layer.conv_states is conv and layer.recurrent_states is recurrent
        assert torch.equal(recurrent, original)
    with pytest.raises(ValueError, match="mapping"):
        pack_target_kv(cache, replace(geometry, full_attention_layer_indices=(0, 4)))
    with pytest.raises(ValueError, match="length"):
        replace_target_kv(cache, packed[:, :2], geometry)
    assert torch.equal(pack_target_kv(cache, geometry), packed + 1)


def test_checkpoint_roundtrip_and_rejections(tmp_path, repairer, monkeypatch):
    import methods.CacheRepair.qwen38.model as module

    activate(repairer)
    save_repairer(tmp_path, repairer)
    assert not any("embed" in key for key in load_file(str(tmp_path / "model.safetensors")))
    settings = json.loads((tmp_path / "config.json").read_text())
    assert settings["target_config"]["full_attention_layer_indices"] == [1, 4]
    target = nn.Module()
    target.get_input_embeddings = lambda: repairer.tok_embed
    monkeypatch.setattr(TargetGeometry, "from_target", classmethod(lambda cls, *a, **kw: repairer.target_config))
    loaded = load_repairer(tmp_path, target)
    stale, tokens, chunks = inputs(repairer)
    with torch.no_grad():
        torch.testing.assert_close(loaded(stale, tokens, chunks), repairer(stale, tokens, chunks), rtol=0, atol=0)
    for field, value in [
        ("target_model_id", "other-qwen"), ("config_sha256", "different-config"),
        ("full_attention_layer_indices", (0, 4)), ("head_dim", 16),
        ("target_rope_spec", {"rotary_dim": 8}),
    ]:
        different = replace(repairer.target_config, **{field: value})
        monkeypatch.setattr(TargetGeometry, "from_target", classmethod(lambda cls, *a, **kw: different))
        with pytest.raises(ValueError, match="differs"):
            load_repairer(tmp_path, target)
    settings["segment_order"] = "K-then-V-before-head"
    (tmp_path / "config.json").write_text(json.dumps(settings))
    with pytest.raises(ValueError, match="coordinate convention"):
        module.load_repairer(tmp_path, target)


@pytest.mark.parametrize("kwargs", [{"d_b": 15}, {"n_heads": 0}, {"n_blocks": 0}, {"rope_theta": 0}])
def test_invalid_repair_config(kwargs):
    with pytest.raises(ValueError):
        RepairConfig(**kwargs)
