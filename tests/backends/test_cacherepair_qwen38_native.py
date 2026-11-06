"""Native Qwen hybrid cache and MRoPE checks; require Transformers with Qwen3.5.

These run a tiny random native target, not repairer training or a benchmark.
"""

import pytest
import torch

from methods.CacheRepair.qwen38 import (
    QwenCacheRepair, RepairConfig, TargetGeometry, canonicalize_target_kv,
    materialize_target_kv, pack_target_kv, replace_target_kv, reposition_target_kv,
)


@pytest.fixture
def native(monkeypatch):
    try:
        from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
        from transformers.models.qwen3_5 import modeling_qwen3_5 as native_module
    except ImportError:
        pytest.skip("Native Qwen3.5 support requires a separate Transformers 5.x environment")
    # These are CPU reference tests even when the environment also has CUDA
    # FLA/conv kernels installed. Prevent native initialization/forward from
    # choosing those GPU-only operators; production loading is untouched.
    for name in ('FusedRMSNormGated','causal_conv1d_fn','causal_conv1d_update',
                 'chunk_gated_delta_rule','fused_recurrent_gated_delta_rule'):
        monkeypatch.setattr(native_module,name,None)
    monkeypatch.setattr(native_module,'is_fast_path_available',False)
    config = Qwen3_5TextConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=5,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, vocab_size=40,
        layer_types=["linear_attention", "full_attention", "linear_attention", "linear_attention", "full_attention"],
        linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=8, linear_value_head_dim=8, linear_conv_kernel_dim=4,
        rope_parameters={"rope_type": "default", "rope_theta": 10_000_000.,
                         "partial_rotary_factor": 0.5, "mrope_section": [2, 1, 1], "mrope_interleaved": True},
    )
    target = Qwen3_5ForCausalLM(config).eval()
    target.config._attn_implementation = "eager"
    return target, apply_rotary_pos_emb


def test_native_geometry_and_real_hybrid_cache(native):
    target, _ = native
    geometry = TargetGeometry.from_target(target, target_model_id="tiny-native-qwen")
    assert geometry.full_attention_layer_indices == (1, 4)
    assert geometry.target_rope_spec["rotary_dim"] == 8
    tokens = torch.tensor([[1, 3, 5, 7]])
    with torch.no_grad():
        past = target(tokens, use_cache=True).past_key_values
    packed = pack_target_kv(past, geometry)
    assert packed.shape == (1, 4, 2, 2, 2, 16)
    gdn = [(past.layers[i], past.layers[i].conv_states.clone(), past.layers[i].recurrent_states.clone())
           for i in (0, 2, 3)]
    length = past.get_seq_length()
    replace_target_kv(past, packed + 0.1, geometry)
    assert past.get_seq_length() == length == 4
    for layer, conv, recurrent in gdn:
        assert torch.equal(conv, layer.conv_states) and torch.equal(recurrent, layer.recurrent_states)
    # The preserved native cache remains consumable by normal GDN + attention decode.
    with torch.no_grad():
        out = target(torch.tensor([[9]]), past_key_values=past, use_cache=True)
    assert torch.isfinite(out.logits).all() and past.get_seq_length() == 5


@pytest.mark.parametrize("multimodal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_partial_rope_roundtrip(native, multimodal, dtype):
    target, apply_rotary = native
    rotary = target.model.rotary_emb
    canonical = torch.randn(2, 5, 2, 2, 2, 16)
    positions = torch.arange(23, 28).expand(2, -1).clone()
    positions[1] += 11
    if multimodal:
        positions = torch.stack((positions, positions + 3, positions + 7))
    cos, sin = rotary(torch.empty(2, 5, 32), positions)
    expected_layers = []
    for index in range(2):
        keys = canonical[:, :, index, :, 0].transpose(1, 2)
        _, expected = apply_rotary(keys, keys, cos, sin)
        expected_layers.append(expected.transpose(1, 2))
    native_kv = materialize_target_kv(canonical, positions, rotary, dtype=dtype)
    assert torch.equal(native_kv[..., 0, :], torch.stack(expected_layers, dim=2).to(dtype))
    assert torch.equal(native_kv[..., 0, 8:], canonical[..., 0, 8:].to(dtype))
    assert torch.equal(native_kv[..., 1, :], canonical[..., 1, :].to(dtype))
    restored = canonicalize_target_kv(native_kv, positions, rotary)
    assert restored.dtype == torch.float32
    tolerance = 0.02 if dtype == torch.bfloat16 else 1e-6
    torch.testing.assert_close(restored, canonical, rtol=tolerance, atol=tolerance)
    rematerialized = materialize_target_kv(restored, positions, rotary, dtype=dtype)
    torch.testing.assert_close(rematerialized, native_kv, rtol=tolerance, atol=tolerance)


def test_scaled_rope_inverse_and_chunk_reposition(native):
    target, _ = native
    rotary = target.model.rotary_emb
    rotary.attention_scaling = 1.7
    canonical = torch.randn(1, 6, 2, 2, 2, 16)
    local = torch.arange(6).view(1, -1)
    global_pos = local + 1003
    original = materialize_target_kv(canonical, local, rotary, dtype=torch.float32)
    moved = reposition_target_kv(original, local, global_pos, rotary)
    expected = materialize_target_kv(canonical, global_pos, rotary, dtype=torch.float32)
    torch.testing.assert_close(moved, expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(canonicalize_target_kv(moved, global_pos, rotary), canonical)
    rounded = reposition_target_kv(original.bfloat16(), local, global_pos, rotary)
    assert rounded.dtype == torch.bfloat16
    assert torch.equal(rounded[..., 1, :], original.bfloat16()[..., 1, :])
    assert torch.equal(rounded[..., 0, 8:], original.bfloat16()[..., 0, 8:])


def test_native_checkpoint_embedding_and_runtime_mismatch(native, tmp_path):
    from methods.CacheRepair.qwen38 import load_repairer, save_repairer

    target, _ = native
    model = QwenCacheRepair.from_target(
        target, RepairConfig(d_b=16, n_heads=2, n_blocks=1), target_model_id="tiny-qwen",
    )
    assert not target.training and not any(p.requires_grad for p in target.parameters())
    save_repairer(tmp_path, model)
    loaded = load_repairer(tmp_path, target, target_model_id="tiny-qwen")
    assert loaded.tok_embed is target.get_input_embeddings()
    with torch.no_grad():
        target.model.rotary_emb.inv_freq.mul_(0.9)
    with pytest.raises(ValueError, match="differs"):
        load_repairer(tmp_path, target, target_model_id="tiny-qwen")


def test_runtime_geometry_and_embedding_are_checked(native, monkeypatch):
    target, _ = native
    target.model.layers[1].self_attn.head_dim = 8
    with pytest.raises(ValueError, match="runtime KV geometry"):
        TargetGeometry.from_target(target, target_model_id="tiny-qwen")
    target.model.layers[1].self_attn.head_dim = 16
    other = torch.nn.Embedding(40, 32)
    monkeypatch.setattr(target, "get_input_embeddings", lambda: other)
    with pytest.raises(ValueError, match="text backbone"):
        TargetGeometry.from_target(target, target_model_id="tiny-qwen")
