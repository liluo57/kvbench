"""Dense Qwen3 projections and QK norms integrated into the bundled A^3 model."""

from __future__ import annotations

from typing import Any


def _install_normed_projections(attention: Any, config: Any) -> None:
    """Replace Qwen2 projections with Qwen3-compatible projections.

    ``_NormedLinear`` keeps the parameter names ``q_proj.weight`` and
    ``k_proj.weight`` unchanged while applying Qwen3's per-head RMSNorm before
    ragkv's existing RoPE/A3 code runs.  The external norm reference is not a
    registered child, so checkpoint keys remain ``q_norm.weight`` and
    ``k_norm.weight`` as expected by Qwen3 checkpoints.
    """
    import torch.nn as nn

    hidden = int(config.hidden_size)
    heads = int(config.num_attention_heads)
    kv_heads = int(getattr(config, "num_key_value_heads", heads))
    head_dim = int(getattr(config, "head_dim", None) or hidden // heads)
    if head_dim != hidden // heads:
        raise ValueError(
            "The bundled Qwen3 A^3 runtime requires head_dim == hidden_size / "
            f"num_attention_heads; got {head_dim} vs {hidden // heads}"
        )
    bias = bool(getattr(config, "attention_bias", False))

    attention.q_norm = _rms_norm(head_dim, config)
    attention.k_norm = _rms_norm(head_dim, config)
    attention.q_proj = _NormedLinear(
        hidden, heads * head_dim, attention.q_norm, heads, head_dim, bias=bias
    )
    attention.k_proj = _NormedLinear(
        hidden, kv_heads * head_dim, attention.k_norm, kv_heads, head_dim, bias=bias
    )
    attention.v_proj = nn.Linear(hidden, kv_heads * head_dim, bias=bias)
    attention.o_proj = nn.Linear(heads * head_dim, hidden, bias=False)


def _rms_norm(size: int, config: Any):
    from .qwen.qwen_precompute import Qwen2RMSNorm

    return Qwen2RMSNorm(size, eps=float(getattr(config, "rms_norm_eps", 1e-6)))


class _NormedLinear:
    """Factory mixin; the concrete class is created lazily with torch.nn."""

    def __new__(cls, in_features, out_features, norm, heads, head_dim, bias=False):
        import torch.nn as nn
        import torch.nn.functional as F

        class NormedLinear(nn.Linear):
            def __init__(self):
                super().__init__(in_features, out_features, bias=bias)
                # Avoid registering the norm below q_proj/k_proj in the state
                # dict; Qwen3 checkpoints store it as self_attn.q_norm/k_norm.
                object.__setattr__(self, "_external_norm", norm)
                self._norm_heads = heads
                self._norm_head_dim = head_dim

            def forward(self, hidden_states):
                projected = F.linear(hidden_states, self.weight, self.bias)
                shape = projected.shape
                projected = projected.view(*shape[:-1], self._norm_heads, self._norm_head_dim)
                projected = self._external_norm(projected)
                return projected.reshape(*shape)

        return NormedLinear()


from transformers import Qwen3Config
from .qwen.qwen import Qwen2ForCausalLM
from .qwen.qwen_precompute import Qwen2ForCausalLM_Precompute


def _patch_layers(model):
    for layer in model.model.layers:
        _install_normed_projections(layer.self_attn, model.config)


class Qwen3ForCausalLM(Qwen2ForCausalLM):
    config_class = Qwen3Config

    def __init__(self, config):
        super().__init__(config)
        _patch_layers(self)
        self.post_init()


class Qwen3ForCausalLM_Precompute(Qwen2ForCausalLM_Precompute):
    config_class = Qwen3Config

    def __init__(self, config):
        super().__init__(config)
        _patch_layers(self)
        self.post_init()
