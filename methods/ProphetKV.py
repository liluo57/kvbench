"""ProphetKV: paper-faithful query-driven KV-cache recomputation.

This module is intentionally self-contained.  ProphetKV has no public
reference implementation to import, so the method owns the small dense
decoder runtime needed by the algorithm while the KVBench core/workloads stay
unchanged.

The implementation follows the paper's two-stage selector:

* Stage I runs the query as an independent sequence and scores every decoder
  layer against the *frozen* isolated document keys ``K'``.
* Stage II fuses the per-layer query-attention scores and uses one global
  selected set.  During recomputation only selected document positions and
  query positions execute K/V projections; unselected positions use ``K'`` /
  ``V'`` directly.

The first runtime targets the standard Llama/Mistral-style dense decoder
layout (``model.model.layers`` with ``q_proj/k_proj/v_proj/o_proj``).  A model
with a different or hybrid layout fails explicitly instead of silently
falling back to a non-ProphetKV algorithm.
"""

from __future__ import annotations

import math
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

    # ------------------------------------------------------------- cache build
    def prefill_isolated(self, ids: Sequence[int], text: str) -> _ChunkCache:
        if not ids:
            return _ChunkCache(text=text, ids=[], layers=[])
        torch = self.torch
        hidden = self.embed_tokens(self._tensor_ids(ids))[0]
        # Keep [1, 1, H] states in a position-indexed dictionary.
        hidden_by_position = {
            i: hidden[i : i + 1].unsqueeze(0) for i in range(len(ids))
        }
        layer_caches: List[Tuple[Any, Any]] = []
        with torch.inference_mode():
            for layer in self.layers:
                base_k = torch.zeros(
                    1,
                    int(getattr(layer.self_attn, "num_key_value_heads", 1)),
                    0,
                    int(layer.self_attn.head_dim),
                    device=self.device,
                    dtype=hidden.dtype,
                )
                base_v = base_k.clone()
                # For isolated prefill all positions are active and there is no
                # document base cache.  Their positions are local (0..N-1).
                hidden_by_position, new_k, new_v = self._process_layer(
                    layer,
                    hidden_by_position,
                    context_len=0,
                    base_k=base_k,
                    base_v=base_v,
                    selected=set(),
                )
                k = torch.cat([new_k[i] for i in range(len(ids))], dim=2)
                v = torch.cat([new_v[i] for i in range(len(ids))], dim=2)
                # Store unrotated K: runtime position is applied at scoring and
                # recomputation time, which makes chunks position-independent.
                layer_caches.append((k.detach(), v.detach()))
        return _ChunkCache(text=text, ids=list(ids), layers=layer_caches)

    # -------------------------------------------------------------- Stage I
    def independent_query_q(self, query_ids: Sequence[int], offset: int):
        """Independent query-only forward, returning per-layer rotated Q."""
        torch = self.torch
        hidden = self.embed_tokens(self._tensor_ids(query_ids))[0]
        hidden_by_position = {
            offset + i: hidden[i : i + 1].unsqueeze(0)
            for i in range(len(query_ids))
        }
        per_layer_q = []
        with torch.inference_mode():
            for layer in self.layers:
                attn = layer.self_attn
                layer_q = []
                for position in sorted(hidden_by_position):
                    normed = layer.input_layernorm(hidden_by_position[position])
                    q_raw, _, _ = self._project(attn, normed)
                    layer_q.append(self._rope(attn, q_raw, position))
                per_layer_q.append(torch.cat(layer_q, dim=2))
                empty_k = torch.zeros(
                    1,
                    int(getattr(attn, "num_key_value_heads", 1)),
                    0,
                    int(attn.head_dim),
                    device=self.device,
                    dtype=hidden.dtype,
                )
                hidden_by_position, _, _ = self._process_layer(
                    layer,
                    hidden_by_position,
                    context_len=0,
                    base_k=empty_k,
                    base_v=empty_k.clone(),
                    selected=set(),
                )
        return per_layer_q

    def score_all_layers(self, query_ids: Sequence[int], context_k_layers):
        """Compute frozen-K' scores for every layer."""
        torch = self.torch
        if not query_ids:
            raise ValueError("ProphetKV query span tokenized to zero tokens")
        context_len = int(context_k_layers[0].shape[2])
        q_layers = self.independent_query_q(query_ids, context_len)
        layer_scores = []
        for q, base_k in zip(q_layers, context_k_layers):
            # Context K is stored unrotated; rotate it at its assembled position.
            attn = self.layers[len(layer_scores)].self_attn
            keys = [
                self._rope(
                    attn,
                    base_k[:, :, i : i + 1],
                    i,
                )
                for i in range(base_k.shape[2])
            ]
            k = torch.cat(keys, dim=2)
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
        embeddings = self.embed_tokens(self._tensor_ids(all_ids))[0]
        hidden_by_position = {
            position: embeddings[position : position + 1].unsqueeze(0)
            for position in active_positions
        }
        updated_layers = []
        counters = []
        with torch.inference_mode():
            for layer_index, layer in enumerate(self.layers):
                base_k = context_k_layers[layer_index]
                base_v = context_v_layers[layer_index]
                counter: Dict[str, int] = {}
                hidden_by_position, new_k, new_v = self._process_layer(
                    layer,
                    hidden_by_position,
                    context_len=context_len,
                    base_k=base_k,
                    base_v=base_v,
                    selected=selected_set,
                    projection_counter=counter,
                )
                counters.append(counter)
                keys = []
                values = []
                for position in range(context_len + len(query_ids)):
                    if position in new_k:
                        # Keep raw K in the runtime cache.  RoPE is applied by
                        # _attention exactly once at the position where the
                        # key is consumed.
                        keys.append(new_k[position])
                        values.append(new_v[position])
                    else:
                        keys.append(base_k[:, :, position : position + 1])
                        values.append(base_v[:, :, position : position + 1])
                updated_layers.append((torch.cat(keys, dim=2), torch.cat(values, dim=2)))
        expected = len(selected_set) + len(query_ids)
        for counter in counters:
            if counter.get("k", 0) != expected or counter.get("v", 0) != expected:
                raise AssertionError(
                    "ProphetKV projection invariant violated: "
                    f"expected {expected}, got K={counter.get('k', 0)} "
                    f"V={counter.get('v', 0)}"
                )
        last_query = hidden_by_position[query_positions[-1]]
        logits = self.lm_head(self.final_norm(last_query))[0, -1]
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
                    key_items = [(i, old_k[:, :, i : i + 1], old_v[:, :, i : i + 1])
                                 for i in range(old_k.shape[2])]
                    key_items.append((position, k_raw, value))
                    attn_out, _ = self._attention(layer.self_attn, q_rot, key_items)
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
        return self.decode_from_cache(cache, logits, list(context_ids))


class ProphetKV(Method):
    """Paper-based ProphetKV implementation for dense decoder-only models."""

    name = "prophetkv"
    maxCaseBatchSize = 1
    method_metrics = ("reuse_ratio", "recompute_ratio", "stage1_score_time")

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
        for chunks in data:
            texts = list(chunks or [])
            caches = [
                self._runtime.prefill_isolated(
                    self._runtime.encode(text, add_special_tokens=False), text
                )
                for text in texts
            ]
            self._states.append(_CaseState(chunks=texts, caches=caches))

    def Run(
        self,
        data: List[str],
        retainOutput: Optional[List[bool]] = None,
    ) -> List[Result]:
        if self._runtime is None:
            raise RuntimeError("ProphetKV.Initialize must run before Run")
        if len(self._states) != len(data):
            self._states = [_CaseState([], []) for _ in data]
        results = []
        for index, prompt in enumerate(data):
            state = self._states[index]
            parts = ComposeInterleavedReuse(state.chunks, prompt)
            reused_positions = [
                i for i, (chunk_index, _) in enumerate(parts) if chunk_index is not None
            ]
            if not reused_positions:
                t0 = time.perf_counter()
                text, ttft, total, n_tokens = self._runtime.full_generate(
                    self._runtime.encode(prompt)
                )
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

            first_reused = reused_positions[0]
            last_reused = reused_positions[-1]
            # The first implementation requires cached chunks to form the
            # document context and the trailing fresh span to be the query.
            if any(chunk_index is None and span for chunk_index, span in parts[:first_reused]):
                results.append(self._fallback(prompt, "fresh_prefix_not_supported"))
                continue
            if any(chunk_index is None and span for chunk_index, span in parts[first_reused:last_reused + 1]):
                results.append(self._fallback(prompt, "fresh_interleaving_not_supported"))
                continue

            ordered_indices = [
                int(chunk_index)
                for chunk_index, _ in parts[first_reused:last_reused + 1]
                if chunk_index is not None
            ]
            query_text = "".join(
                span for chunk_index, span in parts[last_reused + 1:] if chunk_index is None
            )
            query_ids = self._runtime.encode(query_text, add_special_tokens=False)
            if not query_ids:
                results.append(self._fallback(prompt, "empty_query_span"))
                continue

            context_ids: List[int] = []
            context_k_layers: List[Any] = []
            context_v_layers: List[Any] = []
            for chunk_index in ordered_indices:
                cache = state.caches[chunk_index]
                context_ids.extend(cache.ids)
            if not context_ids or len(context_ids) + len(query_ids) > self.maxModelLen:
                results.append(self._fallback(prompt, "length_limit"))
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

            t_score = time.perf_counter()
            layer_scores = self._runtime.score_all_layers(query_ids, context_k_layers)
            fused = _fuse_layer_scores(layer_scores)
            score_time = time.perf_counter() - t_score
            k = int(math.floor(self.recomputeRatio * len(context_ids)))
            if self.recomputeRatio > 0 and k == 0:
                k = 1
            t_select = time.perf_counter()
            selected = _stable_topk(fused, k)
            select_time = time.perf_counter() - t_select

            t_recompute = time.perf_counter()
            updated, logits = self._runtime.recompute(
                context_k_layers,
                context_v_layers,
                context_ids,
                query_ids,
                selected,
            )
            history = list(context_ids) + list(query_ids)
            text, decode_ttft, total, n_tokens = self._runtime.decode_from_cache(
                updated, logits, history
            )
            recompute_time = time.perf_counter() - t_recompute
            n_input = len(context_ids) + len(query_ids)
            results.append(
                self._result(
                    text,
                    score_time + select_time + recompute_time + decode_ttft,
                    score_time + select_time + recompute_time + total,
                    n_tokens,
                    {
                        "reuse_ratio": len(context_ids) / n_input if n_input else 0.0,
                        "recompute_ratio": len(selected) / len(context_ids) if context_ids else 0.0,
                        "selected_token_count": len(selected),
                        "context_token_count": len(context_ids),
                        "query_token_count": len(query_ids),
                        "selected_token_indices": selected,
                        "fusion_rule": "mean_all_layers",
                        "layer_count_used": len(layer_scores),
                        "stage1_score_time": score_time,
                        "stage2_select_time": select_time,
                        "stage2_recompute_time": recompute_time,
                        "fallback_reason": None,
                    },
                )
            )
        return results

    def _fallback(self, prompt: str, reason: str) -> Result:
        if self._runtime is None:
            raise RuntimeError("ProphetKV.Initialize must run before fallback")
        text, ttft, total, n_tokens = self._runtime.full_generate(
            self._runtime.encode(prompt)
        )
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
