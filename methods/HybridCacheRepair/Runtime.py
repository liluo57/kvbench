"""Scheduler-local attention repair; native Hypic owns every GDN operation."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time
from types import MethodType, SimpleNamespace

import torch
from torch import nn
from safetensors.torch import load_file

from methods.CacheRepair.qwen38.model import QwenCacheRepair, RepairConfig
from methods.CacheRepair.qwen38.target import TargetGeometry, KEY_COORDINATES, SEGMENT_ORDER
from methods.CacheRepair.qwen38.cache import canonicalize_target_kv, materialize_target_kv, reposition_target_kv
from methods.HypicAblation import _AblationPrefill, _ContiguousBatch, _SeedTrrSuffix
from .Method import _RID


def checked_checkpoint(directory, model_path):
    """Bind the native runtime to the training snapshot and complete geometry.

    Config/index hashes and the ModelScope ancestry sidecar bridge different HF
    config implementations. They are not claimed to hash all weight bytes.
    Actual loaded projections, cache mapping, embedding and RoPE are checked below.
    """
    directory, model_path = Path(directory), Path(model_path)
    if not (directory / "model.safetensors").is_file():
        raise FileNotFoundError(f"Missing trained repairer weights in {directory}")
    settings = json.loads((directory / "config.json").read_text())
    expected = {"format": "kvbench-qwen-cacherepair-v1", "key_coordinates": KEY_COORDINATES,
                "segment_order": SEGMENT_ORDER, "repair_attention": "chunk_causal_sdpa"}
    if any(settings.get(k) != v for k, v in expected.items()):
        raise ValueError("Incompatible repairer format or coordinates")
    provenance = json.loads((directory / "target-provenance.json").read_text())
    for name, digest in provenance["files_sha256"].items():
        if hashlib.sha256((model_path / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Target snapshot differs from repairer training: {name}")
    if "modelscope_cache_metadata_sha256" in provenance:
        if hashlib.sha256((model_path / ".msc").read_bytes()).hexdigest() != provenance["modelscope_cache_metadata_sha256"]:
            raise ValueError("Target download ancestry differs from repairer training")
    geometry = TargetGeometry.from_dict(settings["target_config"])
    config = json.loads((model_path / "config.json").read_text())["text_config"]
    names = {"num_hidden_layers": "num_hidden_layers", "num_key_value_heads": "n_kv_heads",
             "head_dim": "head_dim", "hidden_size": "hidden_size", "num_attention_heads": "n_attention_heads",
             "vocab_size": "vocab_size", "max_position_embeddings": "max_position_embeddings"}
    if any(config[k] != getattr(geometry, v) for k, v in names.items()):
        raise ValueError("Target configuration differs from repairer geometry")
    indices = tuple(i for i, kind in enumerate(config["layer_types"]) if kind == "full_attention")
    if indices != geometry.full_attention_layer_indices or config["rope_parameters"] != geometry.target_rope_spec["parameters"]:
        raise ValueError("Target attention mapping or rotary parameters differ")
    return settings, geometry


class LogicalEmbedding(nn.Module):
    """Frozen TP vocabulary shards, exposed as one logical shared embedding."""
    def __init__(self, native, geometry, reduce):
        super().__init__()
        native.requires_grad_(False)
        object.__setattr__(self, "native", native)
        object.__setattr__(self, "reduce", reduce)
        self.geometry = geometry
        if native.embedding_dim != geometry.hidden_size or native.org_vocab_size != geometry.vocab_size:
            raise ValueError("Runtime embedding geometry differs")
        if native.num_added_embeddings != 0:
            raise ValueError("Added vocabulary is unsupported")

    @property
    def weight(self):
        return SimpleNamespace(shape=(self.geometry.vocab_size, self.geometry.hidden_size),
                               device=self.native.weight.device)

    def forward(self, ids):
        indices = self.native.shard_indices
        mask = (ids >= indices.org_vocab_start_index) & (ids < indices.org_vocab_end_index)
        local = (ids - indices.org_vocab_start_index).masked_fill(~mask, 0)
        output = torch.nn.functional.embedding(local, self.native.weight).masked_fill(~mask[..., None], 0)
        # Explicit reduction avoids inheriting the model's prior scatter context.
        return self.reduce(output.contiguous())


class NativeRotaryView(nn.Module):
    """Use actual serving cos/sin tables with the existing canonical KV API."""
    rope_type = "default"

    def __init__(self, native, geometry):
        super().__init__()
        object.__setattr__(self, "native", native)
        spec = geometry.target_rope_spec
        dim = spec["rotary_dim"]
        if (native.rotary_dim != dim or native.head_size != geometry.head_dim
                or not native.is_neox_style or native.base != spec["parameters"]["rope_theta"]
                or list(native.mrope_section) != spec["mrope_section"]):
            raise ValueError("Runtime partial RoPE differs from the checkpoint")
        frequency = 1. / (native.base ** (torch.arange(0, dim, 2, device="cpu").float() / dim))
        if hashlib.sha256(frequency.numpy().tobytes()).hexdigest() != spec["inv_freq_fp32_sha256"]:
            raise ValueError("Runtime rotary frequency convention differs")
        self.register_buffer("inv_freq", frequency, persistent=False)

    def forward(self, probe, positions):
        if positions.ndim != 2:
            raise ValueError("HybridCacheRepair currently supports text positions only")
        table = self.native.cos_sin_cache[positions].float()
        cos, sin = table.chunk(2, dim=-1)
        return torch.cat((cos, cos), dim=-1), torch.cat((sin, sin), dim=-1)


def chunk_bias(chunk_ids, tile=256):
    """Exact additive chunk mask, aligned for FP32 memory-efficient SDPA.

    Construct in row tiles: no second T*T boolean allocation. Padding is not
    accepted here; the online prefix has no padding.
    """
    if chunk_ids.ndim != 2 or (chunk_ids < 0).any():
        raise ValueError("Online chunk IDs must be nonnegative [B,T]")
    b, length = chunk_ids.shape
    width = (length + 15) // 16 * 16
    mask = torch.empty((b, 1, length, width), dtype=torch.float32, device=chunk_ids.device)
    for start in range(0, length, tile):
        stop = min(length, start + tile)
        allowed = chunk_ids[:, start:stop, None] >= chunk_ids[:, None, :]
        mask[:, 0, start:stop, :length].zero_().masked_fill_(~allowed, -torch.inf)
    return mask[..., :length]


def _additive_attention(self, hidden, mask, cos, sin, block_mask):
    """Official attention math with a prebuilt aligned additive mask.

    The upstream wrapper rebuilds a dense additive tensor from a boolean mask
    in every block. Reuse its exact QKV/projection parameters and RoPE equations
    here, avoiding those copies and preserving efficient-SDPA row alignment.
    """
    batch, length, width = hidden.shape
    q, k, v = [x.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
               for x in self.qkv(hidden).chunk(3, -1)]
    def rotate(x):
        first, second = x.chunk(2, -1)
        return torch.cat((first*cos-second*sin, second*cos+first*sin), -1)
    result = torch.nn.functional.scaled_dot_product_attention(rotate(q), rotate(k), v, attn_mask=mask)
    return self.proj(result.transpose(1, 2).reshape(batch, length, width))


@torch.no_grad()
def streamed_repair(model, tokens, chunk_ids, read, write, *, tile=256, efficient=False, alpha=1.):
    """Same model as forward(), with bounded encoder and residual-head storage.

    read(start,end) supplies canonical full-head KV; write receives corrected
    canonical KV = stale + alpha * (delta_norm * sigma_delta). GDN is unchanged.
    The repair blocks attend over the complete prefix once, including at alpha=0
    so this ablation changes only the residual contribution.
    """
    from contextlib import nullcontext
    from torch.nn.attention import sdpa_kernel, SDPBackend
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be a finite nonnegative number")
    length = tokens.shape[1]
    stale_features, token_features = [], []
    for start in range(0, length, tile):
        stop = min(length, start + tile)
        stale = read(start, stop)
        normalized = stale.float() / model.sigma_stale.float()
        b, t = normalized.shape[:2]
        stale_features.append(model.stale_enc(normalized.reshape(b, t, model.target_config.n_segments, model.target_config.head_dim)))
        token_features.append(model.tok_proj(model.tok_embed(tokens[:, start:stop]).float()))
        del stale, normalized
    stale_features, token_features = torch.cat(stale_features, 1), torch.cat(token_features, 1)
    bias = chunk_bias(chunk_ids, tile)
    # FP32 math fallback materializes all attention scores at 32K. Require the
    # efficient exact backend rather than silently changing context or dtype.
    context = sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION) if efficient else nullcontext()
    original = [block.attn.forward for block in model.blocks]
    try:
        for block in model.blocks:
            block.attn.forward = MethodType(_additive_attention, block.attn)
        with context:
            hidden = model.hidden_from_features(stale_features, token_features, chunk_ids, attention_mask=bias)
    finally:
        for block, forward in zip(model.blocks, original):
            block.attn.forward = forward
    del stale_features, token_features, bias
    for start in range(0, length, tile):
        stop = min(length, start + tile)
        normalized_delta = model.v_head(hidden[:, start:stop]).reshape(1, stop-start, *model.target_config.kv_shape)
        residual = normalized_delta.float() * model.sigma_delta.float()
        # Keep the historical alpha=1 floating-point operations exactly intact.
        if alpha != 1.:
            residual = residual * alpha
        write(start, stop, read(start, stop).float() + residual)


class HybridPrefill(_AblationPrefill):
    def __init__(self, worker, tree_cache, checkpoint, tile, *, alpha=1.):
        # This installs only an instance wrapper on native TRR's GDN method.
        # Its capture path is reused; its full-prefill donor is never called.
        super().__init__(worker, "gdn_trr_kv_full", tree_cache)
        from methods.hypic.backend.kvbench_hypic.srt.distributed import (
            get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size,
            tensor_model_parallel_all_gather, tensor_model_parallel_all_reduce,
        )
        self.rank, self.world = get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size()
        self.gather = lambda x: tensor_model_parallel_all_gather(x, dim=3)
        reduce = tensor_model_parallel_all_reduce if self.world > 1 else lambda x: x
        self.tile, self.stale = tile, {}
        self.alpha = alpha
        self.tree_cache = tree_cache
        settings, self.geometry = checked_checkpoint(checkpoint, self.runner.server_args.model_path)
        g = self.geometry
        if g.n_kv_heads % self.world or self.world > g.n_kv_heads:
            raise ValueError("HybridCacheRepair requires unreplicated, evenly sharded KV heads")
        if tuple(sorted(self.kv.full_attention_layer_id_mapping)) != g.full_attention_layer_indices:
            raise ValueError("Actual runtime cache mapping differs from the checkpoint")
        candidates = [m for m in self.runner.model.modules()
                      if hasattr(m, "embed_tokens") and hasattr(m, "layers") and len(m.layers) == g.num_hidden_layers]
        if len(candidates) != 1:
            raise ValueError("Expected one actual Qwen3.8 text backbone")
        backbone = candidates[0]
        self.runner.model.requires_grad_(False).eval()
        for index in g.full_attention_layer_indices:
            layer = backbone.layers[index]
            if layer.head_dim != g.head_dim or layer.num_kv_heads != g.n_kv_heads // self.world:
                raise ValueError("Actual runtime attention head geometry differs")
        embedding = LogicalEmbedding(backbone.embed_tokens, g, reduce)
        self.rotary = NativeRotaryView(backbone.layers[g.full_attention_layer_indices[0]].rotary_emb, g)
        # Loading with assign=True preserves checkpoint FP32 even if the native
        # model loader left a different default parameter dtype in this process.
        with torch.device("meta"):
            self.model = QwenCacheRepair(g, RepairConfig(**settings["repair_config"]), embedding)
        state = load_file(str(Path(checkpoint) / "model.safetensors"))
        if any(value.dtype != torch.float32 for value in state.values()):
            raise ValueError("Expected an FP32 repairer checkpoint")
        self.model.load_state_dict(state, strict=True, assign=True)
        self.model.set_normalization(self.model.sigma_stale, self.model.sigma_delta)
        self.device = backbone.embed_tokens.weight.device
        self.model.to(device=self.device, dtype=torch.float32).requires_grad_(False).eval()
        reset, unfinished = tree_cache.reset, tree_cache.cache_unfinished_req

        def clear(cache, *args, **kw):
            self.stale.clear()
            return reset(*args, **kw)

        def cache_unfinished(cache, req, **kw):
            if not req.rid.startswith((_RID + "run:", _RID + "independent_prepare:")):
                return unfinished(req, **kw)
            from methods.hypic.backend.kvbench_hypic.srt.pic.policy import POLICIES
            req.pic_linearkv_skip_cache = True
            previous = cache.policy
            cache.policy = POLICIES["linearkv"]
            try:
                return unfinished(req, **kw)
            finally:
                cache.policy = previous

        tree_cache.reset = MethodType(clear, tree_cache)
        tree_cache.cache_unfinished_req = MethodType(cache_unfinished, tree_cache)

    def _slots(self, source, end):
        index = int(source.req_pool_indices[0].item())
        slots = self.pool.req_to_token[index, :end].long().clone()
        if slots.numel() != end or (slots <= 0).any() or slots.unique().numel() != end:
            raise RuntimeError("HybridCacheRepair requires request-private KV slots")
        return slots

    def _pack_local(self, slots):
        return torch.stack([torch.stack((self.kv.get_key_buffer(i)[slots], self.kv.get_value_buffer(i)[slots]), dim=-2)
                            for i in self.geometry.full_attention_layer_indices], dim=1).unsqueeze(0)

    def _independent(self, source, ids, slots):
        state_index = self.pool.get_mamba_indices(source.req_pool_indices)
        self.pool.mamba_pool.clear_slots(state_index)
        view = _ContiguousBatch(source, ids, slots, 0, len(ids))
        view.is_prefill_only = True
        donor = self.runner.forward(view)
        del donor, view
        return self._pack_local(slots[:len(ids)]).cpu()

    def _forward_generation(self, batch=None, forward_batch=None, pp_proxy_tensors=None,
                            is_verify=False, skip_attn_backend_init=None):
        from methods.hypic.backend.kvbench_hypic.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
        ordinary = dict(forward_batch=forward_batch, pp_proxy_tensors=pp_proxy_tensors,
                        is_verify=is_verify, skip_attn_backend_init=skip_attn_backend_init)
        if batch is None or batch.forward_mode != ForwardMode.EXTEND or not any(r.rid.startswith(_RID) for r in batch.reqs):
            return self._worker_forward(batch, **ordinary)
        req, = batch.reqs
        if req.rid.startswith(_RID + "trr_prepare:"):
            return self._worker_forward(batch, **ordinary)
        if (forward_batch is not None or is_verify or pp_proxy_tensors is not None or skip_attn_backend_init
                or batch.return_logprob or batch.return_hidden_states or req.output_ids or req.is_retracted):
            raise ValueError("HybridCacheRepair requires one ordinary, unretracted text request")
        req.pic_linearkv_skip_cache = True
        self.worker.set_hicache_consumer(batch.hicache_consumer_index)
        source = ForwardBatch.init_new(batch, self.runner)
        slots = self._slots(source, len(req.origin_input_ids))
        if req.rid.startswith(_RID + "independent_prepare:"):
            boundary = req.pic_segments[-1][0]
            ids = req.origin_input_ids[:boundary]
            self.stale[tuple(ids)] = self._independent(source, ids, slots)
            # Generate the one offline warmup token against the same native
            # independent prefix. The original TRR public entries stay intact.
            suffix = _ContiguousBatch(source, req.origin_input_ids, slots, boundary, len(req.origin_input_ids))
            return self._worker_forward(None, forward_batch=suffix)
        if not req.pic_hit_segments:
            # No prepared hits: ordinary full prefill, with no repairer call.
            self.pool.mamba_pool.clear_slots(self.pool.get_mamba_indices(source.req_pool_indices))
            return self._worker_forward(None, forward_batch=_ContiguousBatch(source, req.origin_input_ids, slots, 0, len(req.origin_input_ids)))
        boundary, end = req.pic_segments[-1]
        if boundary <= 0:
            raise RuntimeError("Matched history must precede a nonempty final query segment")
        snapshots = {}
        self._capture = snapshots
        try:
            donor = self.runner.forward(source)
            del donor
        finally:
            self._capture = None
        if set(snapshots) != set(self.pool.mamba_map):
            raise RuntimeError("Missing a native TRR prefix state")
        chunks = []
        for start, stop in req.pic_segments[:-1]:
            ids = req.origin_input_ids[start:stop]
            cached = self.stale.get(tuple(ids))
            if cached is None:
                cached = self._independent(source, ids, slots)
            chunks.append((start, stop, cached))
        self._repair(req.origin_input_ids[:boundary], chunks, slots)
        state_index = int(self.pool.get_mamba_indices(source.req_pool_indices)[0].item())
        for layer_id, (temporal, conv) in snapshots.items():
            cache = self.pool.mamba2_layer_cache(layer_id)
            cache.temporal[state_index].copy_(temporal)
            cache.conv[0][state_index].copy_(conv)
            cache.conv_tails[0][state_index].copy_(conv)
            cache.transition[state_index].zero_()
            cache.transition[state_index].diagonal(dim1=-2, dim2=-1).fill_(1)
        suffix = _ContiguousBatch(source, req.origin_input_ids, slots, boundary, end)
        _SeedTrrSuffix(suffix, slots, boundary, end, state_index)
        return self._worker_forward(None, forward_batch=suffix)

    @torch.no_grad()
    def _repair(self, ids, chunks, slots):
        started = time.perf_counter()
        g = self.geometry
        chunk_ids = torch.empty((1, len(ids)), dtype=torch.long, device=self.device)
        for i, (start, stop, _) in enumerate(chunks):
            chunk_ids[:, start:stop] = i

        def read(start, stop):
            pieces = []
            for chunk_start, chunk_stop, cache in chunks:
                left, right = max(start, chunk_start), min(stop, chunk_stop)
                if left >= right:
                    continue
                native = cache[:, left-chunk_start:right-chunk_start].to(self.device)
                if self.world > 1:
                    native = self.gather(native)
                local = torch.arange(left-chunk_start, right-chunk_start, device=self.device)[None]
                global_ = torch.arange(left, right, device=self.device)[None]
                moved = reposition_target_kv(native, local, global_, self.rotary)
                pieces.append(canonicalize_target_kv(moved, global_, self.rotary))
            if sum(piece.shape[1] for piece in pieces) != stop-start:
                raise RuntimeError("Stale chunks do not cover the repair prefix")
            return torch.cat(pieces, dim=1)

        def write(start, stop, canonical):
            positions = torch.arange(start, stop, device=self.device)[None]
            fixed = materialize_target_kv(canonical, positions, self.rotary, dtype=self.kv.dtype)
            heads = g.n_kv_heads // self.world
            fixed = fixed[:, :, :, self.rank*heads:(self.rank+1)*heads]
            destination = slots[start:stop]
            for i, layer in enumerate(g.full_attention_layer_indices):
                self.kv.get_key_buffer(layer)[destination] = fixed[0, :, i, :, 0]
                self.kv.get_value_buffer(layer)[destination] = fixed[0, :, i, :, 1]

        streamed_repair(self.model, torch.tensor([ids], device=self.device), chunk_ids,
                        read, write, tile=self.tile, efficient=True, alpha=self.alpha)
        torch.cuda.synchronize(self.device)
        if self.rank == 0:
            print(json.dumps({"event": "hybrid_repair", "tokens": len(ids), "chunks": len(chunks),
                              "residual_alpha": self.alpha,
                              "seconds": time.perf_counter()-started}), flush=True)
