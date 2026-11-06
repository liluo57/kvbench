"""CacheRepair-M body for full-attention KV in the Qwen3.8 text backbone.

The segment encoder and AdaLN blocks are the unmodified official components.
Only target geometry, shared-embedding ownership, and padding handling differ.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import load_file, save_file

from ..backend.cacherepair.model import RepairBlock, RepairConfig as OfficialRepairConfig, StaleEncoder
from .target import KEY_COORDINATES, SEGMENT_ORDER, TargetGeometry


@dataclass
class RepairConfig(OfficialRepairConfig):
    d_b: int = 640
    n_heads: int = 10

    def __post_init__(self):
        if min(self.d_b, self.n_heads, self.n_blocks, self.d_seg) <= 0:
            raise ValueError("Repairer dimensions must be positive")
        if self.d_b % self.n_heads or (self.d_b // self.n_heads) % 2:
            raise ValueError("Repairer width must give each head an even integer dimension")
        if not math.isfinite(self.mlp_ratio) or int(self.d_b * self.mlp_ratio) <= 0:
            raise ValueError("Invalid MLP ratio")
        if not math.isfinite(self.rope_theta) or self.rope_theta <= 0:
            raise ValueError("Invalid repairer RoPE base")


def chunk_causal_mask(chunk_ids):
    """True = visible for SDPA; -1 denotes padding, whose queries see self only."""
    if chunk_ids.ndim != 2 or chunk_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("chunk_ids must be an integer [B,T] tensor")
    if (chunk_ids < -1).any():
        raise ValueError("chunk_ids must be nonnegative, or -1 for padding")
    real = chunk_ids >= 0
    visible = (chunk_ids.unsqueeze(-1) >= chunk_ids.unsqueeze(-2))
    visible &= real.unsqueeze(-1) & real.unsqueeze(-2)
    diagonal = torch.eye(chunk_ids.shape[1], dtype=torch.bool, device=chunk_ids.device)
    return (visible | (~real).unsqueeze(-1) & diagonal).unsqueeze(1)


class QwenCacheRepair(nn.Module):
    """Predict normalized residuals for canonical [B,T,L_attn,H_kv,2,D] KV.

    The target owns the frozen embedding. It is deliberately not a registered
    child: repairer.to()/state_dict()/parameters() cannot move, cast, or copy it.
    Move the target separately before using this module.
    """

    def __init__(self, target_config: TargetGeometry, repair_config: RepairConfig, embedding):
        super().__init__()
        t, c = target_config, repair_config
        if tuple(embedding.weight.shape) != (t.vocab_size, t.hidden_size):
            raise ValueError("Embedding shape differs from target geometry")
        self.target_config, self.config = t, c
        embedding.requires_grad_(False)
        object.__setattr__(self, "_embedding", embedding)
        self.stale_enc = StaleEncoder(t, c.d_seg, c.d_b)
        self.tok_proj = nn.Linear(t.hidden_size, c.d_b)
        self.fuse = nn.Linear(2 * c.d_b, c.d_b)
        self.blocks = nn.ModuleList([RepairBlock(c) for _ in range(c.n_blocks)])
        with torch.random.fork_rng(devices=[]):
            self.stale_reinject = nn.ModuleList([nn.Linear(c.d_b, c.d_b) for _ in self.blocks])
        for projection in self.stale_reinject:
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)
        self.ln_f = nn.LayerNorm(c.d_b)
        self.v_head = nn.Linear(c.d_b, t.n_segments * t.head_dim)
        nn.init.zeros_(self.v_head.weight)
        nn.init.zeros_(self.v_head.bias)
        self.register_buffer("sigma_stale", torch.ones(t.kv_shape))
        self.register_buffer("sigma_delta", torch.ones(t.kv_shape))

    @classmethod
    def from_target(cls, target, repair_config=None, *, target_model_id=None):
        geometry = TargetGeometry.from_target(target, target_model_id=target_model_id)
        target.requires_grad_(False).eval()
        return cls(geometry, repair_config or RepairConfig(), target.get_input_embeddings())

    @property
    def tok_embed(self):
        return self._embedding

    @torch.no_grad()
    def set_normalization(self, sigma_stale, sigma_delta):
        """Install externally estimated token-weighted, per-coordinate RMS values."""
        for name, value in (("sigma_stale", sigma_stale), ("sigma_delta", sigma_delta)):
            if tuple(value.shape) != self.target_config.kv_shape:
                raise ValueError(f"{name} shape differs from target KV geometry")
            if not torch.isfinite(value).all() or not (value > 0).all():
                raise ValueError(f"{name} must contain finite positive RMS values")
        self.sigma_stale.copy_(sigma_stale)
        self.sigma_delta.copy_(sigma_delta)

    def forward(self, stale_kv, token_ids, chunk_ids):
        """Return delta_norm; positions are absolute indices in this input prefix."""
        self.target_config.validate_kv(stale_kv)
        if (token_ids.ndim != 2 or token_ids.dtype != torch.long
            or token_ids.shape != chunk_ids.shape or token_ids.shape != stale_kv.shape[:2]
            or token_ids.shape[1] == 0):
            raise ValueError("KV, token_ids and chunk_ids must have matching nonempty [B,T] axes")
        device = self.tok_proj.weight.device
        if any(x.device != device for x in (stale_kv, token_ids, chunk_ids, self.tok_embed.weight)):
            raise ValueError("Place the repairer, target embedding, and inputs on the same device")
        if token_ids.shape[1] > self.target_config.max_position_embeddings:
            raise ValueError("Repair prefix exceeds target position limit")
        real = chunk_ids >= 0
        # Clear padding before projections: masked attention cannot remove NaN V.
        normalized = (stale_kv.float() / self.sigma_stale.float()).masked_fill(
            ~real[..., None, None, None, None], 0,
        ).to(self.tok_proj.weight.dtype)
        batch, length = token_ids.shape
        with torch.no_grad():
            embedded = self.tok_embed(token_ids.masked_fill(~real, 0)).to(self.tok_proj.weight.dtype)
        t, c = self.target_config, self.config
        stale_features = self.stale_enc(normalized.reshape(batch, length, t.n_segments, t.head_dim))
        hidden = self.hidden_from_features(stale_features, self.tok_proj(embedded), chunk_ids)
        delta_norm = self.v_head(hidden).reshape(batch, length, *t.kv_shape)
        return delta_norm.masked_fill(~real[..., None, None, None, None], 0)

    def hidden_from_features(self, stale_features, token_features, chunk_ids, *, attention_mask=None):
        """The unchanged repair blocks, shared by dense and tiled inference.

        A serving attention adapter may supply its equivalent precomputed mask
        with aligned row stride for memory-efficient FP32 SDPA on long prefixes.
        """
        c = self.config
        length, device = chunk_ids.shape[1], stale_features.device
        mask = chunk_causal_mask(chunk_ids) if attention_mask is None else attention_mask
        hidden = self.fuse(torch.cat((stale_features, token_features), dim=-1))
        head_dim = c.d_b // c.n_heads
        frequency = c.rope_theta ** (-torch.arange(0, head_dim, 2, device=device).float() / head_dim)
        angles = torch.arange(length, device=device).float()[:, None] * frequency[None, :]
        cos, sin = [table.to(hidden.dtype).view(1, 1, length, -1) for table in (angles.cos(), angles.sin())]
        for projection, block in zip(self.stale_reinject, self.blocks):
            hidden = block(hidden + projection(stale_features), mask, cos, sin, None)
        return self.ln_f(hidden)

    @torch.no_grad()
    def repair(self, stale_kv, token_ids, chunk_ids):
        return stale_kv.float() + self(stale_kv, token_ids, chunk_ids).float() * self.sigma_delta.float()


def save_repairer(directory, model: QwenCacheRepair):
    """Save only repairer parameters, RMS buffers, and compatibility metadata."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.set_normalization(model.sigma_stale, model.sigma_delta)
    settings = {
        "format": "kvbench-qwen-cacherepair-v1",
        "target_config": model.target_config.to_dict(),
        "repair_config": asdict(model.config),
        "key_coordinates": KEY_COORDINATES,
        "segment_order": SEGMENT_ORDER,
        "repair_attention": "chunk_causal_sdpa",
    }
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
              str(directory / "model.safetensors"))
    (directory / "config.json").write_text(json.dumps(settings, indent=2) + "\n")


def load_repairer(directory, target, *, target_model_id=None):
    """Bind a local repairer checkpoint to a compatible live target embedding."""
    directory = Path(directory)
    settings = json.loads((directory / "config.json").read_text())
    if any(settings.get(key) != expected for key, expected in {
        "format": "kvbench-qwen-cacherepair-v1", "key_coordinates": KEY_COORDINATES,
        "segment_order": SEGMENT_ORDER, "repair_attention": "chunk_causal_sdpa",
    }.items()):
        raise ValueError("Unsupported checkpoint format or cache coordinate convention")
    geometry = TargetGeometry.from_target(target, target_model_id=target_model_id)
    if geometry != TargetGeometry.from_dict(settings["target_config"]):
        raise ValueError("Checkpoint target identity, configuration, layer mapping or RoPE differs")
    target.requires_grad_(False).eval()
    model = QwenCacheRepair(geometry, RepairConfig(**settings["repair_config"]), target.get_input_embeddings())
    model.load_state_dict(load_file(str(directory / "model.safetensors")), strict=True)
    model.set_normalization(model.sigma_stale, model.sigma_delta)
    return model.to(device=target.get_input_embeddings().weight.device).eval()
