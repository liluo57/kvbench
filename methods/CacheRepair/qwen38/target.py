"""Runtime geometry of Qwen3.8's Qwen3.5 text backbone (no GDN targets)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math

import torch


KEY_COORDINATES = "canonical_target_k_fp32_v_fp32"
SEGMENT_ORDER = "attention_layer,kv_head,kv_type(K,V),head_dimension"


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def text_backbone(target):
    """Find the text model in either the causal or multimodal native wrapper."""
    candidates = [
        module for module in target.modules()
        if getattr(getattr(module, "config", None), "model_type", None) == "qwen3_5_text"
        and all(hasattr(module, name) for name in ("layers", "rotary_emb", "embed_tokens"))
    ]
    if len(candidates) != 1:
        raise ValueError("Expected one native Qwen3.5/Qwen3.8 text backbone")
    return candidates[0]


@dataclass(frozen=True)
class TargetGeometry:
    target_model_id: str
    config_sha256: str
    full_attention_layer_indices: tuple[int, ...]
    num_hidden_layers: int
    n_kv_heads: int
    head_dim: int
    hidden_size: int
    n_attention_heads: int
    vocab_size: int
    max_position_embeddings: int
    target_rope_spec: dict

    def __post_init__(self):
        indices = self.full_attention_layer_indices
        if not indices or tuple(sorted(set(indices))) != indices:
            raise ValueError("Full-attention indices must be nonempty, unique and ordered")
        if indices[0] < 0 or indices[-1] >= self.num_hidden_layers:
            raise ValueError("Full-attention indices are outside the text backbone")
        if not self.target_model_id or min(
            self.n_kv_heads, self.head_dim, self.hidden_size,
            self.n_attention_heads, self.vocab_size, self.max_position_embeddings,
        ) <= 0:
            raise ValueError("Invalid target identity or geometry")

    @property
    def n_layers(self):
        return len(self.full_attention_layer_indices)

    @property
    def n_segments(self):
        return self.n_layers * self.n_kv_heads * 2

    @property
    def kv_shape(self):
        return self.n_layers, self.n_kv_heads, 2, self.head_dim

    def to_dict(self):
        values = asdict(self)
        values["full_attention_layer_indices"] = list(self.full_attention_layer_indices)
        return values

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        values["full_attention_layer_indices"] = tuple(values["full_attention_layer_indices"])
        return cls(**values)

    @classmethod
    def from_target(cls, target, *, target_model_id=None):
        if getattr(target, "_cr_vllm_target", False):
            geometry = cls.from_text_config(target.config.text_config, target.rotary_emb,
                target_model_id=target_model_id or target.model_id)
            if tuple(target.get_input_embeddings().weight.shape) != (geometry.vocab_size, geometry.hidden_size):
                raise ValueError("Frozen embedding differs from target configuration")
            return geometry
        backbone = text_backbone(target)
        config = backbone.config
        if len(backbone.layers) != config.num_hidden_layers:
            raise ValueError("Configured and runtime layer counts differ")
        indices = []
        for index, (kind, layer) in enumerate(zip(config.layer_types, backbone.layers, strict=True)):
            if kind != layer.layer_type or kind not in {"full_attention", "linear_attention"}:
                raise ValueError(f"Unsupported or mismatched layer type at layer {index}")
            if kind == "linear_attention":
                if not hasattr(layer, "linear_attn") or hasattr(layer, "self_attn"):
                    raise ValueError(f"Expected a GDN module at layer {index}")
                continue
            attn = layer.self_attn
            if (attn.layer_idx != index or attn.head_dim != config.head_dim
                or attn.k_proj.out_features != config.num_key_value_heads * config.head_dim
                or attn.v_proj.out_features != config.num_key_value_heads * config.head_dim):
                raise ValueError(f"Configured and runtime KV geometry differ at layer {index}")
            indices.append(index)

        embedding = target.get_input_embeddings()
        if embedding is not backbone.embed_tokens or tuple(embedding.weight.shape) != (
            config.vocab_size, config.hidden_size,
        ):
            raise ValueError("Target input embedding does not belong to its text backbone")

        return cls.from_text_config(config, backbone.rotary_emb,
            target_model_id=target_model_id or getattr(target.config, "_name_or_path", ""))

    @classmethod
    def from_text_config(cls, config, rotary, *, target_model_id):
        if config.model_type != "qwen3_5_text" or any(
            kind not in {"full_attention", "linear_attention"} for kind in config.layer_types):
            raise ValueError("Unsupported text architecture")
        indices = [i for i, kind in enumerate(config.layer_types) if kind == "full_attention"]
        if len(config.layer_types) != config.num_hidden_layers:
            raise ValueError("Layer type count differs from text configuration")
        parameters = dict(config.rope_parameters)
        # Dynamic frequency updates need a different cache-coordinate contract.
        if parameters.get("rope_type") != "default":
            raise ValueError("This baseline supports the target's native default RoPE only")
        frequency = rotary.inv_freq.detach().float().cpu().contiguous()
        rotary_dim = 2 * frequency.numel()
        expected = int(config.head_dim * parameters.get("partial_rotary_factor", 1.0))
        scaling = float(rotary.attention_scaling)
        if (rotary_dim != expected or rotary_dim > config.head_dim or rotary_dim % 2
            or rotary_dim == 0 or not math.isfinite(scaling) or scaling <= 0
            or not torch.isfinite(frequency).all() or not (frequency > 0).all()):
            raise ValueError("Configured and runtime target RoPE differ")
        values = config.to_dict()
        for key in list(values):
            if key.startswith("_") or key in {"transformers_version", "dtype", "torch_dtype"}:
                values.pop(key)
        return cls(
            target_model_id=target_model_id,
            config_sha256=fingerprint(values),
            full_attention_layer_indices=tuple(indices),
            num_hidden_layers=config.num_hidden_layers,
            n_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            hidden_size=config.hidden_size,
            n_attention_heads=config.num_attention_heads,
            vocab_size=config.vocab_size,
            max_position_embeddings=config.max_position_embeddings,
            target_rope_spec={
                "convention": "qwen3_5_partial_half_split_native_mrope",
                "parameters": parameters,
                "mrope_section": list(rotary.mrope_section),
                "rotary_dim": rotary_dim,
                "attention_scaling": scaling,
                "inv_freq_fp32_sha256": hashlib.sha256(frequency.numpy().tobytes()).hexdigest(),
            },
        )

    def validate_kv(self, kv):
        if kv.ndim != 6 or tuple(kv.shape[2:]) != self.kv_shape:
            raise ValueError(f"Expected [B,T,{self.kv_shape}], got {tuple(kv.shape)}")
        if not kv.is_floating_point():
            raise ValueError("KV coordinates must be floating point")
