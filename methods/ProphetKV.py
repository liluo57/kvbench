"""ProphetKV: paper-faithful query-driven KV-cache recomputation.

This module is intentionally self-contained.  ProphetKV has no public
reference implementation to import, so the method owns the small dense
decoder runtime needed by the algorithm while the KVBench core/workloads stay
unchanged.

The implementation follows the paper's two-stage selector:

* Stage I runs the query as an independent sequence and scores every decoder
  layer against the *frozen* isolated document keys ``K'``.
* Stage II fuses the per-layer query-attention scores and uses one global
  selected set.  During recomputation only selected cached-document
  positions, genuinely fresh spans, and query positions execute K/V
  projections; unselected cached positions use ``K'`` / ``V'`` directly.

The prompt adapter accepts the same interleaved segment shape as CacheBlend:
fresh prefix, cached chunks, fresh gaps, and a trailing fresh query can all
appear in their original order.  Fresh spans are mandatory-active positions,
not fallback conditions.

The first runtime targets the standard Llama/Mistral-style dense decoder
layout (``model.model.layers`` with ``q_proj/k_proj/v_proj/o_proj``).  A model
with a different or hybrid layout fails explicitly instead of silently
falling back to a non-ProphetKV algorithm.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from core.Config import ModelPath as DefaultModelPath
from core.Method import Method
from core.Result import NumOutputTokensKey, Result, TotalTimeKey, TtftKey
from core.Sampling import IsGreedy, ResolveSamplingConfig

from helpers.backends.Prompt import ComposeInterleavedReuse


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch_cat((-x[..., half:], x[..., :half]), dim=-1)


def torch_cat(values, *, dim: int):
    """Small indirection so importing this module remains CPU/lightweight."""
    import torch

    return torch.cat(values, dim=dim)


def _stable_topk(scores, k: int) -> List[int]:
    """Return stable descending top-k indices for a one-dimensional tensor."""
    values = scores.detach().float().cpu().tolist()
    order = sorted(range(len(values)), key=lambda i: (-values[i], i))
    return order[: max(0, min(k, len(order)))]


def _aggregate_attention(attention):
    """ProphetKV Stage-I reduction: mean over heads, sum over query rows."""
    # [batch, heads, query_tokens, context_tokens]
    return attention.mean(dim=1).sum(dim=1)


def _fuse_layer_scores(layer_scores):
    """Eq. 7: uniform mean over all decoder layers."""
    if not layer_scores:
        raise ValueError("ProphetKV requires at least one layer score")
    import torch

    return torch.stack(layer_scores, dim=0).mean(dim=0)


@dataclass
class _ChunkCache:
    text: str
    ids: List[int]
    # One (unrotated K, V) pair per decoder layer.
    layers: List[Tuple[Any, Any]]


@dataclass
class _CaseState:
    chunks: List[str]
    caches: List[_ChunkCache]


class _DenseProphetRuntime:
    """Minimal dense-decoder runtime with explicit token-position control."""

    def __init__(
        self,
        model_path: str,
        gpu_id: int,
        *,
        max_new_tokens: int,
        max_model_len: int,
        dtype: str,
        sampling_config: Dict[str, Any],
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        # The runtime launches many medium-sized CUDA kernels.  In the
        # default server environment PyTorch may create ~100 OpenMP threads,
        # which adds substantial host-side scheduling overhead for this
        # workload.  Keep a small bounded pool inside each isolated worker.
        try:
            torch.set_num_threads(min(4, torch.get_num_threads()))
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # Another component may already have initialized the pools.
            pass
        self.max_new_tokens = int(max_new_tokens)
        self.max_model_len = int(max_model_len)
        self.sampling_config = dict(sampling_config)
        self.device = f"cuda:{int(gpu_id)}"
        torch.cuda.set_device(int(gpu_id))
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        kwargs = {
            "torch_dtype": getattr(torch, dtype),
            "device_map": self.device,
            "low_cpu_mem_usage": True,
        }
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        self.model.eval()
        try:
            import flashinfer

            self.flashinfer = flashinfer
        except Exception:
            # The correctness path remains usable in a plain Transformers
            # environment; KVBench installations with ragkv provide the
            # fused FlashInfer kernel used by the fast path below.
            self.flashinfer = None

        base = getattr(self.model, "model", None)
        self.layers = getattr(base, "layers", None)
        self.embed_tokens = getattr(base, "embed_tokens", None)
        self.final_norm = getattr(base, "norm", None)
        self.lm_head = getattr(self.model, "lm_head", None)
        if self.layers is None or self.embed_tokens is None or self.final_norm is None:
            raise RuntimeError(
                "ProphetKV currently supports dense Llama/Mistral-style models "
                "with model.model.layers/embed_tokens/norm"
            )
        for index, layer in enumerate(self.layers):
            attn = getattr(layer, "self_attn", None)
            required = ("q_proj", "k_proj", "v_proj", "o_proj")
            if attn is None or any(not hasattr(attn, name) for name in required):
                raise RuntimeError(
                    f"ProphetKV unsupported attention layout at layer {index}"
                )

        eos = self.sampling_config.get("eos_token_id", self.tokenizer.eos_token_id)
        if isinstance(eos, (list, tuple)):
            self.eos_ids = {int(x) for x in eos}
        elif eos is None:
            self.eos_ids = set()
        else:
            self.eos_ids = {int(eos)}

    # -------------------------------------------------------------- encoding
    def encode(self, text: str, *, add_special_tokens: bool = False) -> List[int]:
        try:
            return list(
                self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
            )
        except TypeError:
            return list(
                self.tokenizer(text, add_special_tokens=add_special_tokens)[
                    "input_ids"
                ]
            )

    def decode(self, ids: Iterable[int]) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=True)

    def _tensor_ids(self, ids: Sequence[int]):
        return self.torch.tensor([list(ids)], dtype=self.torch.long, device=self.device)

    # -------------------------------------------------------------- projections
    def _rope(self, attn, x, position: int):
        """Apply the model's rotary embedding at exactly one absolute position."""
        torch = self.torch
        pos = torch.tensor([[int(position)]], dtype=torch.long, device=x.device)
        rotary = getattr(attn, "rotary_emb", None)
        if rotary is None:
            rotary = getattr(getattr(self.model, "model", None), "rotary_emb", None)
        if rotary is None:
            return x
        try:
            cos, sin = rotary(x, pos)
        except TypeError:
            cos, sin = rotary(x)
        while cos.ndim < x.ndim:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        cos = cos[..., : x.shape[-1]]
        sin = sin[..., : x.shape[-1]]
        return (x * cos) + (_rotate_half(x) * sin)

    def _project(self, attn, hidden):
        """Project one [1, 1, hidden] state into Q/K/V head tensors."""
        torch = self.torch
        q = attn.q_proj(hidden)
        k = attn.k_proj(hidden)
        v = attn.v_proj(hidden)

        q_heads = int(getattr(attn, "num_heads", q.shape[-1] // attn.head_dim))
        kv_heads = int(
            getattr(attn, "num_key_value_heads", k.shape[-1] // attn.head_dim)
        )
        q = q.view(1, 1, q_heads, attn.head_dim).transpose(1, 2)
        k = k.view(1, 1, kv_heads, attn.head_dim).transpose(1, 2)
        v = v.view(1, 1, kv_heads, attn.head_dim).transpose(1, 2)
        q_norm = getattr(attn, "q_norm", None)
        k_norm = getattr(attn, "k_norm", None)
        if q_norm is not None:
            q = q_norm(q)
        if k_norm is not None:
            k = k_norm(k)
        return q, k, v

    def _project_batch(self, attn, hidden):
        """Project a whole ``[batch, seq, hidden]`` tensor into Q/K/V.

        The original implementation used ``_project`` once per token.  That
        is faithful but prohibitively expensive for isolated-cache build: it
        launches thousands of tiny CUDA operations.  This batched variant is
        algebraically identical and is used only where every position in the
        chunk is active (the isolated prefill stage).
        """
        q = attn.q_proj(hidden)
        k = attn.k_proj(hidden)
        v = attn.v_proj(hidden)
        q_heads = int(getattr(attn, "num_heads", q.shape[-1] // attn.head_dim))
        kv_heads = int(
            getattr(attn, "num_key_value_heads", k.shape[-1] // attn.head_dim)
        )
        batch, seq = hidden.shape[:2]
        q = q.view(batch, seq, q_heads, attn.head_dim).transpose(1, 2)
        k = k.view(batch, seq, kv_heads, attn.head_dim).transpose(1, 2)
        v = v.view(batch, seq, kv_heads, attn.head_dim).transpose(1, 2)
        q_norm = getattr(attn, "q_norm", None)
        k_norm = getattr(attn, "k_norm", None)
        if q_norm is not None:
            q = q_norm(q)
        if k_norm is not None:
            k = k_norm(k)
        return q, k, v

    @staticmethod
    def _expand_kv(k, v, q_heads: int):
        groups = q_heads // k.shape[1]
        if groups <= 1:
            return k, v
        return k.repeat_interleave(groups, dim=1), v.repeat_interleave(groups, dim=1)

    def _attention(self, attn, q_rot, key_items):
        """Attend to ``[(position, k_raw, v), ...]`` in causal order."""
        torch = self.torch
        if not key_items:
            raise RuntimeError("ProphetKV attention received an empty key set")
        key_items = sorted(key_items, key=lambda item: item[0])
        keys = []
        values = []
        for position, k_raw, value in key_items:
            keys.append(self._rope(attn, k_raw, position))
            values.append(value)
        k = torch.cat(keys, dim=2)
        v = torch.cat(values, dim=2)
        k, v = self._expand_kv(k, v, q_rot.shape[1])
        logits = torch.matmul(q_rot.float(), k.float().transpose(-1, -2))
        logits = logits / math.sqrt(float(attn.head_dim))
        probs = torch.softmax(logits, dim=-1).to(q_rot.dtype)
        out = torch.matmul(probs, v)
        out = out.transpose(1, 2).reshape(1, 1, -1)
        return attn.o_proj(out), probs

    def _rope_batch(self, attn, x, positions):
        """Apply rotary embeddings to ``[batch, heads, seq, dim]`` in bulk."""
        torch = self.torch
        rotary = getattr(attn, "rotary_emb", None)
        if rotary is None:
            rotary = getattr(getattr(self.model, "model", None), "rotary_emb", None)
        if rotary is None:
            return x
        position_ids = positions.to(device=x.device, dtype=torch.long).view(1, -1)
        try:
            cos, sin = rotary(x, position_ids)
        except TypeError:
            cos, sin = rotary(x)
        # HF rotary implementations commonly return [B, S, D]; attention
        # tensors use [B, H, S, D].  Normalize both that layout and an already
        # broadcast [B, 1, S, D] layout.
        if cos.ndim == 3 and cos.shape[1] == x.shape[2]:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        while cos.ndim < x.ndim:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        cos = cos[..., : x.shape[-1]]
        sin = sin[..., : x.shape[-1]]
        return (x * cos) + (_rotate_half(x) * sin)

    def _attention_batch(self, attn, q_rot, k_rot, value):
        """Causal self-attention for a complete active sequence."""
        torch = self.torch
        k_rot, value = self._expand_kv(k_rot, value, q_rot.shape[1])
        logits = torch.matmul(q_rot.float(), k_rot.float().transpose(-1, -2))
        logits = logits / math.sqrt(float(attn.head_dim))
        seq = q_rot.shape[2]
        causal = torch.triu(
            torch.ones(seq, seq, dtype=torch.bool, device=q_rot.device), diagonal=1
        )
        logits = logits.masked_fill(causal.view(1, 1, seq, seq), float("-inf"))
        probs = torch.softmax(logits, dim=-1).to(q_rot.dtype)
        out = torch.matmul(probs, value)
        out = out.transpose(1, 2).reshape(q_rot.shape[0], seq, -1)
        return attn.o_proj(out)

    def _process_layer(
        self,
        layer,
        hidden_by_position: Dict[int, Any],
        *,
        context_len: int,
        base_k,
        base_v,
        selected: set[int],
        projection_counter: Optional[Dict[str, int]] = None,
    ):
        """Run one layer over active positions only.

        ``base_k/base_v`` hold the isolated context cache.  Positions not in
        ``selected`` are never projected; they are read directly from these
        tensors.  The returned dictionaries contain only active hidden states
        and newly projected K/V.
        """
        torch = self.torch
        attn = layer.self_attn
        new_hidden: Dict[int, Any] = {}
        new_k: Dict[int, Any] = {}
        new_v: Dict[int, Any] = {}
        active = sorted(hidden_by_position)
        for position in active:
            hidden = hidden_by_position[position]
            normed = layer.input_layernorm(hidden)
            q_raw, k_raw, value = self._project(attn, normed)
            q_rot = self._rope(attn, q_raw, position)
            # All active positions, including query positions, need fresh K/V.
            new_k[position] = k_raw
            new_v[position] = value
            if projection_counter is not None:
                projection_counter["q"] = projection_counter.get("q", 0) + 1
                projection_counter["k"] = projection_counter.get("k", 0) + 1
                projection_counter["v"] = projection_counter.get("v", 0) + 1

            key_items = []
            # Isolated context entries are available at every position.  A
            # selected entry is replaced only after its active projection.
            upto = min(position, context_len - 1)
            for context_position in range(upto + 1):
                if context_position in new_k:
                    key_items.append(
                        (
                            context_position,
                            new_k[context_position],
                            new_v[context_position],
                        )
                    )
                else:
                    key_items.append(
                        (
                            context_position,
                            base_k[:, :, context_position : context_position + 1],
                            base_v[:, :, context_position : context_position + 1],
                        )
                    )
            # Previously processed active query positions are causal keys.
            for query_position in sorted(new_k):
                if query_position >= context_len and query_position <= position:
                    key_items.append(
                        (query_position, new_k[query_position], new_v[query_position])
                    )

            attn_out, _ = self._attention(attn, q_rot, key_items)
            residual = hidden + attn_out
            mlp_in = layer.post_attention_layernorm(residual)
            output = residual + layer.mlp(mlp_in)
            new_hidden[position] = output

        return new_hidden, new_k, new_v

    def _process_layer_batched(
        self,
        layer,
        hidden,
        positions,
        *,
        context_len: int,
        total_len: Optional[int] = None,
        base_k,
        base_v,
        base_positions=None,
        projection_counter: Optional[Dict[str, int]] = None,
        attention_batch_size: int = 2048,
        allowed_mask=None,
        profile: Optional[Dict[str, float]] = None,
    ):
        """Vectorized partial recomputation for selected/query positions.

        ``hidden`` is ``[1, K, hidden_size]`` and ``positions`` is one sorted
        absolute-position tensor of length ``K``.  We project all active
        positions in one batch, splice their fresh K/V into a full cache with
        ``index_copy_``, and evaluate causal attention in bounded query
        batches.  This is the same tensor-level organization used by sparse
        HuggingFace implementations: there is no Python loop over token
        positions and no per-position cache concatenation.
        """
        torch = self.torch
        profiling = profile is not None and self.device.startswith("cuda")

        def sync():
            if profiling:
                torch.cuda.synchronize()

        def add_time(name: str, start: float):
            if profiling:
                sync()
                profile[name] = profile.get(name, 0.0) + (time.perf_counter() - start)

        if hidden.shape[1] == 0:
            return hidden, base_k, base_v
        attn = layer.self_attn
        phase_start = time.perf_counter()
        normed = layer.input_layernorm(hidden)
        q_raw, k_raw, value = self._project_batch(attn, normed)
        q_rot = self._rope_batch(attn, q_raw, positions)
        add_time("projection_rope_sec", phase_start)

        phase_start = time.perf_counter()
        total_len = int(total_len or max(context_len, int(positions[-1].item()) + 1))
        kv_heads = k_raw.shape[1]
        head_dim = k_raw.shape[-1]
        if base_positions is not None:
            full_k = torch.zeros(
                (base_k.shape[0], kv_heads, total_len, head_dim),
                device=base_k.device,
                dtype=base_k.dtype,
            )
            full_v = torch.zeros_like(full_k)
            full_k.index_copy_(2, base_positions, base_k)
            full_v.index_copy_(2, base_positions, base_v)
        elif total_len > context_len:
            full_k = torch.empty(
                (base_k.shape[0], kv_heads, total_len, head_dim),
                device=base_k.device,
                dtype=base_k.dtype,
            )
            full_v = torch.empty_like(full_k)
            full_k[:, :, :context_len, :].copy_(base_k)
            full_v[:, :, :context_len, :].copy_(base_v)
            # Query positions are active by construction, so these tail rows
            # are overwritten by index_copy_ below.  Initializing them keeps
            # the buffer well-defined even if a caller supplies a sparse set.
            full_k[:, :, context_len:, :].zero_()
            full_v[:, :, context_len:, :].zero_()
        else:
            full_k = base_k.clone()
            full_v = base_v.clone()
        # Every active position (selected document or query) gets a fresh K/V.
        full_k.index_copy_(2, positions, k_raw)
        full_v.index_copy_(2, positions, value)
        add_time("kv_buffer_sec", phase_start)
        if projection_counter is not None:
            count = int(positions.numel())
            projection_counter["q"] = projection_counter.get("q", 0) + count
            projection_counter["k"] = projection_counter.get("k", 0) + count
            projection_counter["v"] = projection_counter.get("v", 0) + count

        all_positions = torch.arange(total_len, device=hidden.device, dtype=torch.long)
        batch_size = max(1, int(attention_batch_size))
        # Attention logits are memory-heavy, so calculate them in bounded
        # query blocks.  Keep the MLP outside that loop: invoking the large
        # feed-forward module once per block was the second major runtime
        # bottleneck after the original token-by-token implementation.
        attention_outputs = torch.empty_like(hidden)
        # The isolated cache stores raw K.  Rotate the complete assembled K
        # buffer exactly once for attention; never mix rotated and raw rows.
        phase_start = time.perf_counter()
        full_k_rot = self._rope_batch(attn, full_k, all_positions)
        add_time("full_k_rope_sec", phase_start)
        # Use PyTorch's fused scaled-dot-product attention for the expensive
        # Stage-II query blocks.  The explicit logits/probability path is
        # mathematically equivalent but needlessly materializes a
        # [heads, active_queries, context] tensor and is much slower than the
        # kernels used by the other KVBench backends.
        allowed = (
            allowed_mask
            if allowed_mask is not None
            else all_positions.view(1, -1) <= positions.view(-1, 1)
        )
        if self.flashinfer is not None:
            # FlashInfer accepts the compact GQA layout directly and supports
            # an arbitrary causal mask for non-contiguous selected positions.
            # Unlike SDPA, it does not materialize/expand the KV heads and can
            # process the complete selected set in one fused call.
            q_f = q_rot[0].permute(1, 0, 2).contiguous()
            k_f = full_k_rot[0].permute(1, 0, 2).contiguous()
            v_f = full_v[0].permute(1, 0, 2).contiguous()
            out = self.flashinfer.single_prefill_with_kv_cache(
                q_f,
                k_f,
                v_f,
                custom_mask=allowed.contiguous(),
                causal=False,
                kv_layout="NHD",
                pos_encoding_mode="NONE",
                backend="auto",
            )
            out = out.reshape(1, int(positions.numel()), -1)
            attention_outputs = attn.o_proj(out)
        else:
            import torch.nn.functional as F

            # Portable fallback for environments without FlashInfer.
            # Older PyTorch releases do not expose SDPA's ``enable_gqa``
            # keyword, so expand grouped KV heads explicitly once here.
            sdpa_k, sdpa_v = self._expand_kv(
                full_k_rot, full_v, q_rot.shape[1]
            )
            for start in range(0, int(positions.numel()), batch_size):
                stop = min(start + batch_size, int(positions.numel()))
                q_batch = q_rot[:, :, start:stop]
                out = F.scaled_dot_product_attention(
                    q_batch,
                    sdpa_k,
                    sdpa_v,
                    attn_mask=allowed[start:stop].view(1, 1, stop - start, total_len),
                    dropout_p=0.0,
                    is_causal=False,
                )
                out = out.transpose(1, 2).reshape(1, stop - start, -1)
                attention_outputs[:, start:stop] = attn.o_proj(out)
        add_time("attention_sec", phase_start)
        phase_start = time.perf_counter()
        residual = hidden + attention_outputs
        output_hidden = residual + layer.mlp(layer.post_attention_layernorm(residual))
        add_time("mlp_residual_sec", phase_start)
        return output_hidden, full_k, full_v

    # ------------------------------------------------------------- cache build
    def prefill_isolated(self, ids: Sequence[int], text: str) -> _ChunkCache:
        if not ids:
            return _ChunkCache(text=text, ids=[], layers=[])
        torch = self.torch
        # All positions are active in an isolated chunk, so use one batched
        # tensor per layer instead of the quadratic Python token loop.
        hidden = self.embed_tokens(self._tensor_ids(ids))
        positions = torch.arange(len(ids), device=self.device, dtype=torch.long)
        layer_caches: List[Tuple[Any, Any]] = []
        with torch.inference_mode():
            for layer in self.layers:
                normed = layer.input_layernorm(hidden)
                q_raw, k_raw, value = self._project_batch(layer.self_attn, normed)
                q_rot = self._rope_batch(layer.self_attn, q_raw, positions)
                k_rot = self._rope_batch(layer.self_attn, k_raw, positions)
                attn_out = self._attention_batch(
                    layer.self_attn, q_rot, k_rot, value
                )
                residual = hidden + attn_out
                hidden = residual + layer.mlp(layer.post_attention_layernorm(residual))
                # Store unrotated K: runtime position is applied at scoring and
                # recomputation time, which makes chunks position-independent.
                layer_caches.append((k_raw.detach(), value.detach()))
        return _ChunkCache(text=text, ids=list(ids), layers=layer_caches)

    # -------------------------------------------------------------- Stage I
    def independent_query_q(self, query_ids: Sequence[int], offset: int):
        """Independent query-only forward, returning per-layer rotated Q."""
        torch = self.torch
        hidden = self.embed_tokens(self._tensor_ids(query_ids))
        positions = torch.arange(
            offset, offset + len(query_ids), device=self.device, dtype=torch.long
        )
        per_layer_q = []
        with torch.inference_mode():
            for layer in self.layers:
                attn = layer.self_attn
                normed = layer.input_layernorm(hidden)
                q_raw, k_raw, value = self._project_batch(attn, normed)
                q_rot = self._rope_batch(attn, q_raw, positions)
                k_rot = self._rope_batch(attn, k_raw, positions)
                per_layer_q.append(q_rot)
                attn_out = self._attention_batch(
                    attn, q_rot, k_rot, value
                )
                residual = hidden + attn_out
                hidden = residual + layer.mlp(layer.post_attention_layernorm(residual))
        return per_layer_q

    def score_all_layers(
        self,
        query_ids: Sequence[int],
        context_k_layers,
        *,
        context_positions: Optional[Sequence[int]] = None,
        query_offset: Optional[int] = None,
    ):
        """Compute frozen-K' scores for every layer.

        ``context_positions`` contains the positions of the cached tokens in
        the *complete* prompt.  It is optional for the original contiguous
        context path, but required when fresh prefix/gap spans are present so
        RoPE uses the same absolute positions as Stage II.
        """
        torch = self.torch
        if not query_ids:
            raise ValueError("ProphetKV query span tokenized to zero tokens")
        context_len = int(context_k_layers[0].shape[2])
        if context_positions is None:
            context_positions = list(range(context_len))
        if len(context_positions) != context_len:
            raise ValueError("context_positions must align with cached K'")
        if query_offset is None:
            query_offset = context_len
        q_layers = self.independent_query_q(query_ids, int(query_offset))
        layer_scores = []
        for q, base_k in zip(q_layers, context_k_layers):
            # Context K is stored unrotated; rotate the complete sequence in
            # one call at its assembled positions.
            attn = self.layers[len(layer_scores)].self_attn
            positions = torch.tensor(
                list(context_positions), device=base_k.device, dtype=torch.long
            )
            k = self._rope_batch(attn, base_k, positions)
            k, _ = self._expand_kv(k, k, q.shape[1])
            logits = torch.matmul(q.float(), k.float().transpose(-1, -2))
            logits = logits / math.sqrt(float(attn.head_dim))
            attention = torch.softmax(logits, dim=-1).to(q.dtype)
            layer_scores.append(_aggregate_attention(attention)[0])
        return layer_scores

    # ------------------------------------------------------------- Stage II
    def recompute(
        self,
        context_k_layers,
        context_v_layers,
        context_ids: Sequence[int],
        query_ids: Sequence[int],
        selected: Sequence[int],
    ):
        """Recompute only selected context positions plus query positions."""
        torch = self.torch
        context_len = len(context_ids)
        query_positions = list(range(context_len, context_len + len(query_ids)))
        selected_set = set(int(i) for i in selected)
        active_positions = sorted(selected_set | set(query_positions))
        all_ids = list(context_ids) + list(query_ids)
        embeddings = self.embed_tokens(self._tensor_ids(all_ids))
        active_tensor = torch.tensor(
            active_positions, dtype=torch.long, device=embeddings.device
        )
        # Keep the active sequence compact throughout Stage II.  Absolute
        # positions remain in ``active_tensor`` and are used for the causal
        # mask and RoPE, so non-contiguous selections are fully supported.
        hidden = embeddings.index_select(1, active_tensor)
        total_len = context_len + len(query_ids)
        all_positions = torch.arange(
            total_len, dtype=torch.long, device=embeddings.device
        )
        # The absolute-position causal mask is identical at every decoder
        # layer; construct it once rather than allocating an O(KN) mask per
        # layer.
        allowed_mask = all_positions.view(1, -1) <= active_tensor.view(-1, 1)
        updated_layers = []
        counters = []
        profile = {} if os.environ.get("PROPHETKV_PROFILE", "") == "1" else None
        with torch.inference_mode():
            for layer_index, layer in enumerate(self.layers):
                base_k = context_k_layers[layer_index]
                base_v = context_v_layers[layer_index]
                counter: Dict[str, int] = {}
                hidden, full_k, full_v = self._process_layer_batched(
                    layer,
                    hidden,
                    active_tensor,
                    context_len=context_len,
                    base_k=base_k,
                    base_v=base_v,
                    projection_counter=counter,
                    allowed_mask=allowed_mask,
                    profile=profile,
                )
                counters.append(counter)
                # ``full_k/full_v`` already contain reused rows and freshly
                # projected active rows.  Returning these buffers directly
                # avoids O(N) Python slicing and O(N) tiny concatenations per
                # layer.
                updated_layers.append((full_k, full_v))
        expected = len(selected_set) + len(query_ids)
        for counter in counters:
            if counter.get("k", 0) != expected or counter.get("v", 0) != expected:
                raise AssertionError(
                    "ProphetKV projection invariant violated: "
                    f"expected {expected}, got K={counter.get('k', 0)} "
                    f"V={counter.get('v', 0)}"
                )
        last_query_offset = active_positions.index(query_positions[-1])
        # ``hidden`` was produced under inference_mode; keep the final head in
        # the same context so PyTorch does not try to attach autograd metadata
        # to an inference tensor.
        with torch.inference_mode():
            last_query = hidden[:, last_query_offset : last_query_offset + 1]
            logits = self.lm_head(self.final_norm(last_query))[0, -1]
        if profile is not None:
            profile["total_stage2_sec"] = sum(
                profile.get(name, 0.0)
                for name in (
                    "projection_rope_sec",
                    "kv_buffer_sec",
                    "full_k_rope_sec",
                    "attention_sec",
                    "mlp_residual_sec",
                )
            )
            print(f"[ProphetKV profile] {profile}", flush=True)
        return updated_layers, logits

    def recompute_segments(
        self,
        context_k_layers,
        context_v_layers,
        full_ids: Sequence[int],
        cached_positions: Sequence[int],
        selected_cached_indices: Sequence[int],
        fresh_positions: Sequence[int],
        query_positions: Sequence[int],
    ):
        """Stage II for an interleaved cached/fresh prompt.

        ``context_*_layers`` contain only the independently-prefilled cached
        tokens.  ``cached_positions`` maps those rows into the reconstructed
        full prompt.  Every fresh position is mandatory-active; selected
        cached rows are added to that active set.  Unselected cached rows are
        copied directly into the assembled K/V buffer and are never projected.
        """
        torch = self.torch
        total_len = len(full_ids)
        cached_len = len(cached_positions)
        if cached_len == 0 or total_len == 0:
            raise ValueError("interleaved recompute requires non-empty prompt/cache")
        if any(int(p) < 0 or int(p) >= total_len for p in cached_positions):
            raise ValueError("cached position outside reconstructed prompt")
        if len(set(int(p) for p in cached_positions)) != cached_len:
            raise ValueError("cached positions must be unique")

        selected_cached = {
            int(selected_cached_indices[i])
            for i in range(len(selected_cached_indices))
        }
        if any(i < 0 or i >= cached_len for i in selected_cached):
            raise ValueError("selected cached index outside K' rows")
        fresh = {int(p) for p in fresh_positions}
        query = [int(p) for p in query_positions]
        if any(p < 0 or p >= total_len for p in fresh):
            raise ValueError("fresh position outside reconstructed prompt")
        if any(p < 0 or p >= total_len for p in query):
            raise ValueError("query position outside reconstructed prompt")
        if not query:
            raise ValueError("interleaved recompute requires a query suffix")

        cached_global = {
            int(cached_positions[i]) for i in selected_cached
        }
        active_positions = sorted(fresh | cached_global)
        active_positions.extend(p for p in query if p not in fresh)
        active_positions = sorted(set(active_positions))
        if not active_positions:
            raise ValueError("interleaved recompute produced no active positions")

        embeddings = self.embed_tokens(self._tensor_ids(full_ids))
        active_tensor = torch.tensor(
            active_positions, dtype=torch.long, device=embeddings.device
        )
        hidden = embeddings.index_select(1, active_tensor)
        cached_position_tensor = torch.tensor(
            list(cached_positions), dtype=torch.long, device=embeddings.device
        )
        counters = []
        updated_layers = []
        profile = {} if os.environ.get("PROPHETKV_PROFILE", "") == "1" else None
        with torch.inference_mode():
            for layer_index, layer in enumerate(self.layers):
                base_k = context_k_layers[layer_index]
                base_v = context_v_layers[layer_index]
                counter: Dict[str, int] = {}
                hidden, full_k, full_v = self._process_layer_batched(
                    layer,
                    hidden,
                    active_tensor,
                    context_len=cached_len,
                    total_len=total_len,
                    base_k=base_k,
                    base_v=base_v,
                    base_positions=cached_position_tensor,
                    projection_counter=counter,
                    profile=profile,
                )
                counters.append(counter)
                updated_layers.append((full_k, full_v))

        expected = len(active_positions)
        for counter in counters:
            if counter.get("k", 0) != expected or counter.get("v", 0) != expected:
                raise AssertionError(
                    "ProphetKV interleaved projection invariant violated: "
                    f"expected {expected}, got K={counter.get('k', 0)} "
                    f"V={counter.get('v', 0)}"
                )
        query_last = query[-1]
        try:
            query_offset = active_positions.index(query_last)
        except ValueError as exc:
            raise AssertionError("last query position was not active") from exc
        with torch.inference_mode():
            last_query = hidden[:, query_offset : query_offset + 1]
            logits = self.lm_head(self.final_norm(last_query))[0, -1]
        if profile is not None:
            print(f"[ProphetKV profile] {profile}", flush=True)
        return updated_layers, logits

    # ------------------------------------------------------------- generation
    def _sample(self, logits, history: Sequence[int]) -> int:
        torch = self.torch
        if IsGreedy(self.sampling_config):
            return int(torch.argmax(logits).item())
        temperature = float(self.sampling_config.get("temperature", 1.0))
        scores = logits.float() / max(temperature, 1e-6)
        top_k = int(self.sampling_config.get("top_k", -1))
        if top_k > 0 and top_k < scores.numel():
            threshold = torch.topk(scores, top_k).values[-1]
            scores = scores.masked_fill(scores < threshold, float("-inf"))
        top_p = float(self.sampling_config.get("top_p", 1.0))
        if 0 < top_p < 1:
            ordered, indices = torch.sort(scores, descending=True)
            probs = torch.softmax(ordered, dim=-1)
            cumulative = torch.cumsum(probs, dim=-1)
            remove = cumulative > top_p
            remove[1:] = remove[:-1].clone()
            remove[0] = False
            scores = scores.masked_fill(
                torch.zeros_like(remove).scatter(0, indices, remove),
                float("-inf"),
            )
        probs = torch.softmax(scores, dim=-1)
        return int(torch.multinomial(probs, 1).item())

    def decode_from_cache(self, layers_cache, last_hidden_logits, history: List[int]):
        """Decode after partial recomputation using the assembled layer cache."""
        torch = self.torch
        generated: List[int] = []
        logits = last_hidden_logits
        t0 = time.perf_counter()
        ttft = None
        for _ in range(max(1, self.max_new_tokens)):
            token = self._sample(logits, history)
            if ttft is None:
                ttft = time.perf_counter() - t0
            generated.append(token)
            history.append(token)
            if token in self.eos_ids:
                break

            position = int(layers_cache[0][0].shape[2])
            hidden = self.embed_tokens(
                torch.tensor([[token]], dtype=torch.long, device=self.device)
            )
            new_layers = []
            with torch.inference_mode():
                for layer, (old_k, old_v) in zip(self.layers, layers_cache):
                    normed = layer.input_layernorm(hidden)
                    q_raw, k_raw, value = self._project(layer.self_attn, normed)
                    q_rot = self._rope(layer.self_attn, q_raw, position)
                    # The decode query is a single token, but the cached
                    # context can contain thousands of entries.  Rotate and
                    # attend to the complete raw cache in one tensor call;
                    # constructing one Python ``key_items`` entry per token
                    # made generation dominate the selective-recompute path.
                    raw_k = torch.cat([old_k, k_raw], dim=2)
                    raw_v = torch.cat([old_v, value], dim=2)
                    positions = torch.arange(
                        raw_k.shape[2], device=raw_k.device, dtype=torch.long
                    )
                    rot_k = self._rope_batch(layer.self_attn, raw_k, positions)
                    sdpa_k, sdpa_v = self._expand_kv(
                        rot_k, raw_v, q_rot.shape[1]
                    )
                    attn_vec = torch.nn.functional.scaled_dot_product_attention(
                        q_rot,
                        sdpa_k,
                        sdpa_v,
                        dropout_p=0.0,
                        is_causal=False,
                    )
                    attn_vec = attn_vec.transpose(1, 2).reshape(1, 1, -1)
                    attn_out = layer.self_attn.o_proj(attn_vec)
                    residual = hidden + attn_out
                    hidden = residual + layer.mlp(
                        layer.post_attention_layernorm(residual)
                    )
                    new_layers.append(
                        (torch.cat([old_k, k_raw], dim=2),
                         torch.cat([old_v, value], dim=2))
                    )
            layers_cache = new_layers
            logits = self.lm_head(self.final_norm(hidden))[0, -1]
        return self.decode(generated), float(ttft or 0.0), time.perf_counter() - t0, len(generated)

    def full_generate(self, ids: Sequence[int]):
        """Reference full-prefill path used when reuse cannot be established."""
        if not ids:
            return "", 0.0, 0.0, 0
        full_start = time.perf_counter()
        context_ids = list(ids)
        empty = []
        # Treat every input position as selected and use the same explicit
        # layer loop, which makes this fallback a correctness reference.
        torch = self.torch
        base_k_layers = []
        base_v_layers = []
        for layer in self.layers:
            heads = int(getattr(layer.self_attn, "num_key_value_heads", 1))
            dim = int(layer.self_attn.head_dim)
            base_k_layers.append(torch.zeros(1, heads, 0, dim, device=self.device, dtype=self.embed_tokens.weight.dtype))
            base_v_layers.append(base_k_layers[-1].clone())
        cache, logits = self.recompute(
            base_k_layers,
            base_v_layers,
            [],
            context_ids,
            [],
        )
        prefill_ttft = time.perf_counter() - full_start
        text, decode_ttft, decode_total, n_tokens = self.decode_from_cache(
            cache, logits, list(context_ids)
        )
        return (
            text,
            prefill_ttft + decode_ttft,
            decode_total,
            n_tokens,
        )


class ProphetKV(Method):
    """Paper-based ProphetKV implementation for dense decoder-only models."""

    name = "prophetkv"
    maxCaseBatchSize = 1
    method_metrics = ("reuse_ratio", "recompute_ratio", "stage1_score_time", "fallback_rate")

    def __init__(
        self,
        gpuNums: int = 1,
        perfWeight: float = 1.0,
        *,
        maxNewTokens: int = 64,
        maxModelLen: int = 32768,
        recomputeRatio: float = 0.20,
        dtype: str = "bfloat16",
        tag: Optional[str] = None,
    ):
        super().__init__(
            gpuNums=gpuNums,
            perfWeight=perfWeight,
            maxGpuNums=1,
            tag=tag,
        )
        if not 0.0 <= float(recomputeRatio) <= 1.0:
            raise ValueError("recomputeRatio must be in [0, 1]")
        self.modelPath = DefaultModelPath()
        self.samplingConfig = ResolveSamplingConfig(self.modelPath)
        self.maxNewTokens = int(maxNewTokens)
        self.maxModelLen = int(maxModelLen)
        self.recomputeRatio = float(recomputeRatio)
        self.dtype = dtype
        self.debug = os.environ.get("PROPHETKV_DEBUG", "") not in {"", "0", "false"}
        self._runtime: Optional[_DenseProphetRuntime] = None
        self._states: List[_CaseState] = []

    def Initialize(self, gpuIds: Sequence[int]) -> None:
        super().Initialize(gpuIds)
        self._runtime = _DenseProphetRuntime(
            self.modelPath,
            self.gpuIds[0],
            max_new_tokens=self.maxNewTokens,
            max_model_len=self.maxModelLen,
            dtype=self.dtype,
            sampling_config=self.samplingConfig,
        )

    def Prepare(self, data: List[List[str]]) -> None:
        if self._runtime is None:
            raise RuntimeError("ProphetKV.Initialize must run before Prepare")
        self._states = []
        for case_index, chunks in enumerate(data):
            texts = list(chunks or [])
            caches = []
            for chunk_index, text in enumerate(texts):
                ids = self._runtime.encode(text, add_special_tokens=False)
                t_chunk = time.perf_counter()
                cache = self._runtime.prefill_isolated(ids, text)
                caches.append(cache)
                if self.debug:
                    print(
                        f"[ProphetKV] prepare case={case_index} chunk={chunk_index} "
                        f"tokens={len(ids)} seconds={time.perf_counter() - t_chunk:.3f}",
                        flush=True,
                    )
            self._states.append(_CaseState(chunks=texts, caches=caches))

    def Run(
        self,
        data: List[str],
        retainOutput: Optional[List[bool]] = None,
    ) -> List[Result]:
        request_start = time.perf_counter()
        if self._runtime is None:
            raise RuntimeError("ProphetKV.Initialize must run before Run")
        if len(self._states) != len(data):
            self._states = [_CaseState([], []) for _ in data]
        results = []
        # Prepared document caches are already available, but matching, prompt
        # reconstruction, scoring, stitching, and recomputation are query-time
        # work and are all measured from the Run boundary above.
        for index, prompt in enumerate(data):
            state = self._states[index]
            t_match = time.perf_counter()
            parts = ComposeInterleavedReuse(state.chunks, prompt)
            reused_positions = [
                i for i, (chunk_index, _) in enumerate(parts) if chunk_index is not None
            ]
            if self.debug:
                print(
                    f"[ProphetKV] match case={index} prompt_chars={len(prompt)} "
                    f"parts={len(parts)} reused_parts={len(reused_positions)} "
                    f"seconds={time.perf_counter() - t_match:.3f}",
                    flush=True,
                )
            if not reused_positions:
                t0 = time.perf_counter()
                ids = self._runtime.encode(prompt)
                generation_start = time.perf_counter()
                text, backend_ttft, total, n_tokens = self._runtime.full_generate(ids)
                ttft = generation_start - request_start + backend_ttft
                results.append(
                    self._result(
                        text,
                        ttft,
                        total,
                        n_tokens,
                        {
                            "reuse_ratio": 0.0,
                            "recompute_ratio": 1.0,
                            "selected_token_count": 0,
                            "context_token_count": 0,
                            "query_token_count": 0,
                            "fallback_reason": "no_prepared_chunk_match",
                            "stage1_score_time": 0.0,
                            "stage2_select_time": 0.0,
                            "stage2_recompute_time": time.perf_counter() - t0,
                        },
                    )
                )
                continue

            last_reused = reused_positions[-1]

            # Build the same ordered segment plan used by CacheBlend: cached
            # spans retain their independently-collected K/V, while every
            # fresh span remains in the reconstructed prompt and is marked
            # mandatory-active for Stage II.  ``full_ids`` is assembled from
            # the segment tokenization so cached row boundaries and absolute
            # positions stay aligned.
            full_ids: List[int] = []
            cached_ids: List[int] = []
            cached_positions: List[int] = []
            ordered_indices: List[int] = []
            fresh_positions: List[int] = []
            query_ids: List[int] = []
            query_positions: List[int] = []
            segment_debug = []
            last_cached_part = last_reused

            for part_index, (chunk_index, span) in enumerate(parts):
                if chunk_index is None and span:
                    ids = self._runtime.encode(span, add_special_tokens=False)
                    start = len(full_ids)
                    full_ids.extend(ids)
                    positions = list(range(start, start + len(ids)))
                    fresh_positions.extend(positions)
                    segment_debug.append({
                        "kind": "fresh",
                        "part_index": part_index,
                        "tokens": len(ids),
                    })
                    # Only fresh text after the final cached span is the
                    # query suffix used for ProphetKV Stage-I scoring.
                    if part_index > last_cached_part:
                        query_ids.extend(ids)
                        query_positions.extend(positions)
                    continue

                if chunk_index is None:
                    continue
                chunk_index = int(chunk_index)
                cache = state.caches[chunk_index]
                ids = list(cache.ids)
                start = len(full_ids)
                full_ids.extend(ids)
                positions = list(range(start, start + len(ids)))
                cached_ids.extend(ids)
                cached_positions.extend(positions)
                ordered_indices.append(chunk_index)
                segment_debug.append({
                    "kind": "cached",
                    "part_index": part_index,
                    "chunk_index": chunk_index,
                    "tokens": len(ids),
                    "start": start,
                    "end": start + len(ids),
                })

            if not query_ids:
                results.append(
                    self._fallback(prompt, "empty_query_span", request_start)
                )
                continue

            context_k_layers: List[Any] = []
            context_v_layers: List[Any] = []
            t_stitch = time.perf_counter()
            if self.debug:
                print(
                    f"[ProphetKV] run case={index} parts={len(parts)} "
                    f"reused_chunks={len(ordered_indices)} context_tokens={len(cached_ids)} "
                    f"fresh_tokens={len(fresh_positions)} "
                    f"query_tokens={len(query_ids)}",
                    flush=True,
                )
            if not cached_ids or len(full_ids) > self.maxModelLen:
                results.append(self._fallback(prompt, "length_limit", request_start))
                continue
            for layer_index in range(len(self._runtime.layers)):
                context_k_layers.append(
                    self._runtime.torch.cat(
                        [state.caches[i].layers[layer_index][0] for i in ordered_indices],
                        dim=2,
                    )
                )
                context_v_layers.append(
                    self._runtime.torch.cat(
                        [state.caches[i].layers[layer_index][1] for i in ordered_indices],
                        dim=2,
                    )
                )
            stitch_time = time.perf_counter() - t_stitch
            if self.debug:
                print(
                    f"[ProphetKV] cache_stitch seconds={stitch_time:.3f}",
                    flush=True,
                )

            t_score = time.perf_counter()
            layer_scores = self._runtime.score_all_layers(
                query_ids,
                context_k_layers,
                context_positions=cached_positions,
                query_offset=query_positions[0],
            )
            fused = _fuse_layer_scores(layer_scores)
            score_time = time.perf_counter() - t_score
            k = int(math.floor(self.recomputeRatio * len(cached_ids)))
            if self.recomputeRatio > 0 and k == 0:
                k = 1
            t_select = time.perf_counter()
            selected = _stable_topk(fused, k)
            if self.debug:
                print(
                    f"[ProphetKV] score seconds={time.perf_counter() - t_score:.3f} "
                    f"selected={len(selected)}",
                    flush=True,
                )
            select_time = time.perf_counter() - t_select

            t_recompute = time.perf_counter()
            updated, logits = self._runtime.recompute_segments(
                context_k_layers,
                context_v_layers,
                full_ids,
                cached_positions,
                selected,
                fresh_positions,
                query_positions,
            )
            recompute_only_time = time.perf_counter() - t_recompute
            history = list(full_ids)
            t_decode = time.perf_counter()
            text, decode_ttft, total, n_tokens = self._runtime.decode_from_cache(
                updated, logits, history
            )
            decode_wall_time = time.perf_counter() - t_decode
            recompute_time = time.perf_counter() - t_recompute
            if self.debug:
                print(
                    f"[ProphetKV] recompute seconds={recompute_only_time:.3f} "
                    f"decode seconds={decode_wall_time:.3f} generated={n_tokens}",
                    flush=True,
                )
            n_input = len(full_ids)
            results.append(
                self._result(
                    text,
                    t_decode - request_start + decode_ttft,
                    score_time + select_time + recompute_time + total,
                    n_tokens,
                    {
                        "reuse_ratio": len(cached_ids) / n_input if n_input else 0.0,
                        "recompute_ratio": len(selected) / len(cached_ids) if cached_ids else 0.0,
                        "selected_token_count": len(selected),
                        "context_token_count": len(cached_ids),
                        "query_token_count": len(query_ids),
                        "fresh_token_count": len(fresh_positions),
                        "fresh_prefix_token_count": sum(
                            1 for p in fresh_positions if p < cached_positions[0]
                        ),
                        "fresh_gap_token_count": sum(
                            1 for p in fresh_positions
                            if cached_positions[0] <= p < query_positions[0]
                        ),
                        "segment_plan": segment_debug,
                        "selected_token_indices": selected,
                        "fusion_rule": "mean_all_layers",
                        "layer_count_used": len(layer_scores),
                        "stage1_score_time": score_time,
                        "stage2_select_time": select_time,
                        "stage2_recompute_time": recompute_time,
                        "fallback_reason": None,
                        "fallback_rate": 0.0,
                    },
                )
            )
        return results

    def _fallback(
        self, prompt: str, reason: str, request_start: Optional[float] = None
    ) -> Result:
        if self._runtime is None:
            raise RuntimeError("ProphetKV.Initialize must run before fallback")
        if request_start is None:
            request_start = time.perf_counter()
        ids = self._runtime.encode(prompt)
        generation_start = time.perf_counter()
        text, backend_ttft, total, n_tokens = self._runtime.full_generate(ids)
        ttft = generation_start - request_start + backend_ttft
        return self._result(
            text,
            ttft,
            total,
            n_tokens,
            {
                "reuse_ratio": 0.0,
                "recompute_ratio": 1.0,
                "selected_token_count": 0,
                "context_token_count": 0,
                "query_token_count": 0,
                "fallback_reason": reason,
                "stage1_score_time": 0.0,
                "stage2_select_time": 0.0,
                "stage2_recompute_time": 0.0,
                "fallback_rate": 1.0,
            },
        )

    def _result(self, text, ttft, total, n_tokens, metadata) -> Result:
        return Result(
            output=text,
            performance={
                TtftKey: float(ttft),
                NumOutputTokensKey: int(n_tokens),
                TotalTimeKey: float(total),
            },
            metadata={"backend": "prophetkv", **metadata},
        )

    def Reset(self) -> None:
        self._states = []

    def Close(self) -> None:
        self._states = []
        self._runtime = None
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
