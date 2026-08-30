"""Out-of-tree Qwen3 (dense) model for the CacheBlend fork (vLLM 0.4.1).

The original CacheBlend repo's ``vllm_blend`` implements its collect / check /
fusion machinery (``cache_fuse_metadata``, ``hack_kv``, ``old_kvs``, the check
state machine) only on ``LlamaForCausalLM`` — Mistral maps there, Qwen3 does
not exist in the fork's vLLM 0.4.1 registry or its transformers 4.44.2 at all.
This module plugs Qwen3 in **without touching the official repo**:

* ``AutoConfig.register("qwen3", Qwen3Config)`` makes the fork's config loader
  (``AutoConfig.from_pretrained``) accept a Qwen3 ``config.json``;
* ``ModelRegistry.register_model("Qwen3ForCausalLM", Qwen3ForCausalLM)`` is the
  vLLM-designed out-of-tree extension point the fork's ``get_model_architecture``
  already honors;
* the model class *subclasses* the fork's patched ``LlamaForCausalLM`` so the
  whole CacheBlend algorithm (``cache_fuse_metadata`` / ``hack_kv`` /
  ``old_kvs``, the check state machine, weight stacking) is **inherited, not
  copied**.

The only structural differences between a dense Qwen3 model and Llama/Mistral:

* QK-Norm: ``q_norm`` / ``k_norm`` RMSNorms on the query/key **before** RoPE.
  The collect capture point must sit *after* the QK-Norm and *before* the RoPE,
  so the check phase's ``org_pos`` re-rotation reproduces exactly the key the
  fresh path computes (norm then rotate) — this is what makes fuse == full.
* head_dim: Qwen3 may declare an explicit ``config.head_dim`` (e.g. 32B:
  hidden 5120 / 64 heads = 80, declared head_dim 128) that differs from
  ``hidden_size // num_attention_heads``; the attention reads it from config.

Qwen3-8B / Qwen3-32B are both dense (no sliding window), bf16, ``silu``, GQA,
``tie_word_embeddings=false``, vocab 151936; their tokenizer is the plain
``Qwen2Tokenizer`` already known to transformers 4.44.2.

──────────────────────────────────────────────────────────────────────────────
Import / registration contract — READ BEFORE REMOVING "DEAD" CODE
──────────────────────────────────────────────────────────────────────────────
This module has zero static importers in the repo (in-degree = 0). It is NOT
registered on import — registration is **explicit**:

  • Importer (the only one): :func:`helpers.CacheblendRepoHelper
    .CacheblendRepoHelper._registerOutOfTreeModel`
  • Trigger: only when the loaded model's ``config.json`` lists
    ``Qwen3ForCausalLM`` in its ``architectures`` field.
  • Action: a single call to :func:`register_qwen3` after a dynamic
    ``import Qwen3ForCacheBlendRepo`` (see CacheblendRepoHelper.py:342-343).
  • Why explicit and not side-effect: Mistral is the fork's native model and
    must reach its patched ``LlamaForCausalLM`` untouched — registering Qwen3
    unconditionally would pollute the registry for non-Qwen3 runs.

If you delete this file because it looks unused: you will break the only path
that runs Qwen3 through the upstream CacheBlend repo. The 8B model regression
suite (``fuse == full == stock``) will start failing immediately.
"""

import torch
from torch import nn
from transformers import AutoConfig, LlamaConfig

from vllm.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models import ModelRegistry
# Importing ``vllm.model_executor.models.llama`` cold triggers its own import
# of ``model_loader.weight_utils``, whose ``loader.py`` imports ``llava``,
# which imports ``llama`` back — a cycle. vLLM's own flow loads ``model_loader``
# first, so warm it before touching ``llama`` (matches the fork's import order).
import vllm.model_executor.model_loader  # noqa: E402  (pre-load; see above)
from vllm.model_executor.models.llama import (LlamaAttention, LlamaForCausalLM,
                                              LlamaModel)


class Qwen3Config(LlamaConfig):
    """Qwen3 config for transformers 4.44.2 (which predates Qwen3).

    Subclasses ``LlamaConfig`` so every field Qwen3 shares with Llama loads
    unchanged; ``head_dim`` (explicit in Qwen3, may differ from
    ``hidden_size // num_attention_heads``) is kept as an extra attribute.
    """

    model_type = "qwen3"

    def __init__(self, head_dim: int = None, **kwargs):
        super().__init__(**kwargs)
        self.head_dim = head_dim


class Qwen3Attention(LlamaAttention):
    """Qwen3 attention = the patched ``LlamaAttention`` + QK-Norm.

    ``__init__`` builds the base attention, overrides ``head_dim`` from config
    when it differs from ``hidden_size // num_heads`` (rebuilding the
    head_dim-dependent pieces), and adds ``q_norm`` / ``k_norm``. ``forward``
    is the base forward with one change: the QK-Norm is applied between the
    QKV split and the RoPE, and the CacheBlend collect capture (``hack_kv``)
    moves after it so the stored K is post-norm / pre-rotary — exactly what the
    check phase's ``org_pos`` re-rotation expects.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling=None,
        max_position_embeddings: int = 8192,
        linear_method=None,
        bias: bool = False,
        sliding_window=None,
        head_dim: int = None,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__(
            hidden_size,
            num_heads,
            num_kv_heads,
            rope_theta,
            rope_scaling,
            max_position_embeddings,
            linear_method,
            bias,
            sliding_window,
        )
        # Qwen3-32B declares head_dim 128 while hidden // heads is 80; the base
        # derived head_dim (and built qkv_proj / rotary / attn from it), so
        # rebuild the head_dim-dependent pieces when the config disagrees.
        if head_dim is not None and head_dim != self.head_dim:
            self.head_dim = head_dim
            self.q_size = self.num_heads * head_dim
            self.kv_size = self.num_kv_heads * head_dim
            self.scaling = head_dim**-0.5
            self.qkv_proj = QKVParallelLinear(
                hidden_size,
                head_dim,
                self.total_num_heads,
                self.total_num_kv_heads,
                bias=bias,
                linear_method=linear_method,
            )
            self.rotary_emb = get_rope(
                head_dim,
                rotary_dim=head_dim,
                max_position=max_position_embeddings,
                base=rope_theta,
                rope_scaling=rope_scaling,
            )
            self.attn = Attention(self.num_heads,
                                  head_dim,
                                  self.scaling,
                                  num_kv_heads=self.num_kv_heads,
                                  sliding_window=sliding_window)
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        # The fork's ``RMSNorm.forward`` calls the custom ``ops.rms_norm`` CUDA
        # kernel, which is written for the 2-D ``[tokens, hidden]`` layout of the
        # layer norms. QK-Norm normalizes per-head over ``head_dim``, i.e. a 3-D
        # ``[tokens, heads, head_dim]`` tensor, which the kernel mis-handles (it
        # normalizes across heads — output diverges from a plain torch RMSNorm).
        # Use the torch-native ``_forward`` implementation instead: verified
        # bit-identical to ``Qwen3RMSNorm`` on the same input. Cost is negligible
        # (q/k are small); the layer norms keep the fast kernel.
        self.q_norm.forward = self.q_norm._forward  # type: ignore[method-assign]
        self.k_norm.forward = self.k_norm._forward  # type: ignore[method-assign]

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata,
        status,
        cache_fuse_metadata,
        old_kv,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # HACK(Jiayi): rotate the old K — same as the patched llama attention.
        # ``old_kv[0]`` was captured post-QK-norm / pre-RoPE, so re-rotating it
        # with ``org_pos`` reproduces the fresh key exactly.
        if status in [1, 2]:
            if cache_fuse_metadata["fake_q"] is None:
                cache_fuse_metadata["fake_q"] = torch.rand_like(q)
            _, old_kv[0] = self.rotary_emb(cache_fuse_metadata["org_pos"],
                                           cache_fuse_metadata["fake_q"],
                                           old_kv[0])

        # Qwen3: QK-Norm before RoPE (and before the collect capture, so the
        # stored K/V is post-norm / pre-rotary — what the check phase expects).
        # The norm is per-head over ``head_dim``, so q/k are viewed as
        # [tokens, heads, head_dim], normalized, and flattened back — applying
        # it to the flat [tokens, q_size] tensor would normalize across heads
        # (and read the weight out of bounds).
        q = self.q_norm(q.view(-1, self.num_heads,
                               self.head_dim)).view(-1, self.q_size)
        k = self.k_norm(k.view(-1, self.num_kv_heads,
                               self.head_dim)).view(-1, self.kv_size)
        if cache_fuse_metadata["collect"]:
            self.hack_kv = [k.clone(), v.clone()]
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata, status,
                                cache_fuse_metadata, old_kv, self.kv_scale)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen3Model(LlamaModel):
    """Qwen3 stack = the patched ``LlamaModel`` with Qwen3 attentions.

    ``__init__`` delegates to the fork's ``LlamaModel`` (so ``cache_fuse_metadata``
    and ``old_kvs`` initialization is inherited unchanged), then swaps each
    layer's attention for :class:`Qwen3Attention` in place. ``forward`` — the
    check-phase state machine — is inherited. The transient ``LlamaAttention``
    built by each base layer is dropped before weight loading, so it is never
    materialized in memory.
    """

    def __init__(self, config, linear_method=None, lora_config=None):
        super().__init__(config, linear_method, lora_config=lora_config)
        head_dim = getattr(config, "head_dim", None)
        eps = getattr(config, "rms_norm_eps", 1e-6)
        for layer in self.layers:
            layer.self_attn = Qwen3Attention(
                hidden_size=layer.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=getattr(config, "num_key_value_heads",
                                     config.num_attention_heads),
                rope_theta=getattr(config, "rope_theta", 10000),
                rope_scaling=getattr(config, "rope_scaling", None),
                max_position_embeddings=getattr(config,
                                                "max_position_embeddings",
                                                8192),
                linear_method=linear_method,
                bias=getattr(config, "attention_bias", False) or getattr(
                    config, "bias", False),
                sliding_window=getattr(config, "sliding_window", None),
                head_dim=head_dim,
                rms_norm_eps=eps,
            )


class Qwen3ForCausalLM(LlamaForCausalLM):
    """Qwen3 LM head — the patched ``LlamaForCausalLM`` with a Qwen3 stack.

    ``load_weights`` (stacks q/k/v and routes the extra ``q_norm`` /
    ``k_norm`` weights), ``compute_logits`` and ``sample`` are all inherited.
    """

    def __init__(self, config, linear_method=None, lora_config=None):
        super().__init__(config, linear_method, lora_config=lora_config)
        self.model = Qwen3Model(config, linear_method, lora_config=lora_config)


def register_qwen3() -> None:
    """Register Qwen3 with transformers + the vLLM 0.4.1 registry.

    Call before constructing ``vllm.LLM`` when the model's ``config.json``
    declares ``Qwen3ForCausalLM``. Idempotent within a process.
    """
    AutoConfig.register("qwen3", Qwen3Config)
    ModelRegistry.register_model("Qwen3ForCausalLM", Qwen3ForCausalLM)
