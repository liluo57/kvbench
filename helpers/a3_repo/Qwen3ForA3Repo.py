"""Out-of-tree Qwen3 model bridge for the official ragkv A3 implementation.

This module deliberately does *not* implement A3.  It reuses ragkv's existing
Qwen2 model classes and A3 forward functions, replacing only the attention
projection modules that differ in Qwen3 (QK-Norm and projection bias).  The
bridge is imported by :mod:`A3RepoHelper` before the official loader is used;
the ragkv checkout itself remains untouched.

The first target is dense Qwen3-8B.  Qwen3-32B/MoE and sliding-window variants
are rejected explicitly until their head layout and checkpoint loading have
been validated.
"""

from __future__ import annotations

from typing import Any, Tuple


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
    head_dim = int(getattr(config, "head_dim", hidden // heads))
    if head_dim != hidden // heads:
        raise ValueError(
            "Qwen3ForA3Repo currently requires head_dim == hidden_size / "
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
    from models.qwen.qwen_precompute import Qwen2RMSNorm

    return Qwen2RMSNorm(size, eps=float(getattr(config, "rms_norm_eps", 1e-6)))


class _NormedLinear:
    """Factory mixin; the concrete class is created lazily with torch.nn."""

    def __new__(cls, in_features, out_features, norm, heads, head_dim, bias=False):
        import torch
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


def _make_runtime_attention(config, layer_idx):
    from transformers.models.qwen2.modeling_qwen2 import Qwen2SdpaAttention
    from models.qwen.qwen import Qwen2SdpaAttention_Forward

    class Qwen3RuntimeAttention(Qwen2SdpaAttention):
        def __init__(self, cfg, idx=None):
            super().__init__(cfg, idx)
            _install_normed_projections(self, cfg)

    # This is ragkv's official attention/recompute state machine.  The only
    # Qwen3-specific behavior is injected by the projection wrappers above.
    Qwen3RuntimeAttention.forward = Qwen2SdpaAttention_Forward
    return Qwen3RuntimeAttention(config, layer_idx)


def _make_precompute_attention(config, layer_idx):
    from models.qwen.qwen_precompute import Qwen2SdpaAttention

    class Qwen3PrecomputeAttention(Qwen2SdpaAttention):
        def __init__(self, cfg, idx=None):
            super().__init__(cfg, idx)
            _install_normed_projections(self, cfg)

    # Qwen2SdpaAttention.forward already captures ``hack_kv`` immediately
    # after projection and before RoPE, which is exactly where Qwen3 must
    # capture its post-QK-Norm, pre-RoPE K/V.
    return Qwen3PrecomputeAttention(config, layer_idx)


def _patch_layers(model, precompute: bool) -> None:
    for idx, layer in enumerate(model.model.layers):
        layer.self_attn = (
            _make_precompute_attention(model.config, idx)
            if precompute
            else _make_runtime_attention(model.config, idx)
        )


def _build_classes():
    import transformers
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2Model
    from models.qwen.qwen_precompute import Qwen2ForCausalLM_Precompute
    from models.qwen.qwen import (
        Qwen2DecoderLayer_Forward,
        Qwen2ForCausalLM_Forward,
        Qwen2Model_Forward,
    )

    Qwen2Config = transformers.Qwen2Config

    class Qwen3Config(Qwen2Config):
        model_type = "qwen3"

    class Qwen3ForCausalLM(transformers.Qwen2ForCausalLM):
        config_class = Qwen3Config

        def __init__(self, config):
            super().__init__(config)
            _patch_layers(self, precompute=False)

    class Qwen3ForCausalLM_Precompute(Qwen2ForCausalLM_Precompute):
        config_class = Qwen3Config

        def __init__(self, config):
            super().__init__(config)
            _patch_layers(self, precompute=True)

    # Install ragkv's official model-level state machine on the Qwen2 base
    # classes used by Qwen3ForCausalLM.  This is process-local monkeypatching;
    # no file in the official checkout is edited.
    Qwen2Model.forward = Qwen2Model_Forward
    Qwen2DecoderLayer.forward = Qwen2DecoderLayer_Forward
    Qwen3ForCausalLM.forward = Qwen2ForCausalLM_Forward
    return Qwen3Config, Qwen3ForCausalLM, Qwen3ForCausalLM_Precompute


def install_qwen3(repo_root: str):
    """Install the bridge and return ragkv-compatible loader functions."""
    import os
    import sys

    # The worker's cwd is the official repo, so make KVBench's package root
    # importable without changing the official checkout.
    kvbench_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if kvbench_root not in sys.path:
        sys.path.insert(0, kvbench_root)

    import torch
    import transformers
    from transformers import AutoConfig
    from transformers import AutoTokenizer

    Qwen3Config, Qwen3ForCausalLM, Qwen3ForCausalLM_Precompute = _build_classes()
    # Older ragkv Qwen code calls create_flashinfer_mask with three
    # arguments, while the shared utility now requires the causal-mode
    # argument.  Keep this compatibility shim process-local and leave the
    # official checkout untouched.
    from models.qwen import qwen as ragkv_qwen
    from models.reuse_utils import create_flashinfer_mask as _create_mask

    def _qwen3_create_mask(query, key, indices, mode=True):
        return _create_mask(query, key, indices, mode)

    ragkv_qwen.create_flashinfer_mask = _qwen3_create_mask
    # Transformers 4.46 predates Qwen3.  Register the compatible config so
    # AutoTokenizer and any downstream config lookup can resolve model_type.
    try:
        AutoConfig.register("qwen3", Qwen3Config)
    except ValueError:
        # A second worker-side install is harmless.
        pass
    transformers.Qwen3Config = Qwen3Config
    transformers.Qwen3ForCausalLM = Qwen3ForCausalLM
    transformers.Qwen3ForCausalLM_Precompute = Qwen3ForCausalLM_Precompute

    def tokenizer(path):
        try:
            return AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        except Exception:
            return AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

    def load_model(args):
        config = Qwen3Config.from_pretrained(args.model)
        tok = tokenizer(args.model)
        model = Qwen3ForCausalLM.from_pretrained(
            args.model,
            config=config,
            cache_dir=os.path.join(repo_root, "cache"),
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        return model, tok

    def load_model_precompute(args):
        config = Qwen3Config.from_pretrained(args.model)
        tok = tokenizer(args.model)
        model = Qwen3ForCausalLM_Precompute.from_pretrained(
            args.model,
            config=config,
            cache_dir=os.path.join(repo_root, "cache"),
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        return model, tok

    return load_model, load_model_precompute
