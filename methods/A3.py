"""A³ method — attention-guided selective KV recomputation.

This method follows KVBench's normal ``Method`` lifecycle and keeps the
model-side implementation in a persistent helper process.  KVBench remains
responsible for GPU assignment, task/workload scheduling, timing and result
aggregation; the helper is responsible for tokenizer/model execution, cache
assembly, A³ scoring and selected-token recomputation.

The engine calls, once per case lifecycle::

    Prepare(chunks)  -> record the independently reusable document chunks.
    Run(prompt)      -> locate those chunks in the complete prompt, build the
                        stitched document KV, compute the fixed layer-1
                        query-to-document attention score, select a stable
                        top-k set, recompute that set, and decode.
    Reset()          -> discard the prepared chunk state for the case.
    Close()          -> stop the helper and release its model resources.

The model-side primitives are self-contained in this adapter; importing this
module does not load a model or initialize CUDA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import threading
import time
from collections import deque
from contextlib import suppress
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

# Direct worker execution (``python methods/A3.py``) does not automatically
# place the repository root on sys.path.
if __package__ in (None, ""):
    _ROOT = str(Path(__file__).resolve().parent.parent)
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

from core.Config import ModelPath as DefaultModelPath
from core.Method import Method
from core.Result import NumOutputTokensKey, Result, TotalTimeKey, TtftKey
from core.Sampling import ResolveSamplingConfig


class _Span:
    def __init__(self, start: int, end: int):
        if start < 0 or end <= start:
            raise ValueError(f"invalid span [{start}, {end})")
        self.start, self.end = int(start), int(end)

    def indices(self):
        return range(self.start, self.end)


class _A3Layout:
    def __init__(self, input_ids, offsets, documents, chunks, query, context):
        self.input_ids = tuple(int(x) for x in input_ids)
        self.token_char_offsets = tuple((int(a), int(b)) for a, b in offsets)
        self.document_indices = tuple(int(x) for x in documents)
        self.cache_chunk_spans = tuple(chunks)
        self.query_span = query
        self.attention_context_span = context
        self.protocol_id = "kvbench_a3_512_v1"
        self.chunk_size = 512
        self.chunk_overlap = 0

    @property
    def fresh_indices(self):
        docs = set(self.document_indices)
        return tuple(i for i in range(len(self.input_ids)) if i not in docs)


class _A3Cache:
    def __init__(self, keys, values, layer_inputs, layer_outputs, local_positions):
        self.keys = tuple(keys)
        self.values = tuple(values)
        self.layer_inputs = tuple(layer_inputs)
        self.layer_outputs = tuple(layer_outputs)
        self.local_positions = local_positions


class _A3State:
    def __init__(self, query_states, key_states, query_positions, key_positions):
        self.query_states = query_states
        self.key_states = key_states
        self.query_positions = tuple(int(x) for x in query_positions)
        self.key_positions = tuple(int(x) for x in key_positions)


class _A3Record:
    def __init__(self, score):
        self.score = score


class _A3RecomputeOutput:
    def __init__(self, keys, values, first_token_logits):
        self.keys = tuple(keys)
        self.values = tuple(values)
        self.first_token_logits = first_token_logits


def _base(model: Any) -> Any:
    return getattr(model, "model", model)


def _layer_kv(past: Any, index: int):
    if hasattr(past, "key_cache"):
        return past.key_cache[index], past.value_cache[index]
    if hasattr(past, "layers"):
        layer = past.layers[index]
        key = getattr(layer, "keys", getattr(layer, "key", None))
        value = getattr(layer, "values", getattr(layer, "value", None))
        if key is not None and value is not None:
            return key, value
    item = past[index]
    if isinstance(item, dict):
        return item["key"], item["value"]
    return item[0], item[1]


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _cos_sin(base: Any, hidden: Any, positions: Any):
    rotary = base.rotary_emb
    pos = positions if positions.ndim == 2 else positions.unsqueeze(0)
    result = rotary(hidden, pos)
    if isinstance(result, tuple):
        return result[0].to(hidden.device), result[1].to(hidden.device)
    if hasattr(result, "cos") and hasattr(result, "sin"):
        return result.cos.to(hidden.device), result.sin.to(hidden.device)
    raise TypeError("unsupported rotary embedding return type")


def _apply_rope(x: Any, cos: Any, sin: Any):
    cos = cos[:, None, :, :].to(dtype=x.dtype)
    sin = sin[:, None, :, :].to(dtype=x.dtype)
    return (x * cos) + (_rotate_half(x) * sin)


def _repeat_kv(key: Any, value: Any, heads: int):
    if key.shape[1] == heads:
        return key, value
    if heads % key.shape[1]:
        raise ValueError("query heads must be an integer multiple of KV heads")
    repeats = heads // key.shape[1]
    return key.repeat_interleave(repeats, dim=1), value.repeat_interleave(repeats, dim=1)


def _project_qkv(layer: Any, hidden: Any, base: Any):
    attn = layer.self_attn
    heads = int(getattr(attn, "num_heads", base.config.num_attention_heads))
    kv_heads = int(getattr(attn, "num_key_value_heads", getattr(base.config, "num_key_value_heads", heads)))
    dim = int(getattr(attn, "head_dim", base.config.hidden_size // heads))
    q = attn.q_proj(hidden).view(hidden.shape[0], hidden.shape[1], heads, dim).transpose(1, 2)
    k = attn.k_proj(hidden).view(hidden.shape[0], hidden.shape[1], kv_heads, dim).transpose(1, 2)
    v = attn.v_proj(hidden).view(hidden.shape[0], hidden.shape[1], kv_heads, dim).transpose(1, 2)
    q_norm, k_norm = getattr(attn, "q_norm", None), getattr(attn, "k_norm", None)
    if q_norm is not None:
        q = q_norm(q.transpose(1, 2)).transpose(1, 2)
    if k_norm is not None:
        k = k_norm(k.transpose(1, 2)).transpose(1, 2)
    return q, k, v, dim


def _layout_checksum(layout: _A3Layout) -> str:
    payload = repr((layout.input_ids, layout.document_indices, layout.query_span.start, layout.query_span.end))
    return hashlib.sha256(payload.encode()).hexdigest()


@torch.inference_mode()
def _build_stitched_cache(model: Any, layout: _A3Layout) -> _A3Cache:
    base = _base(model)
    device = next(model.parameters()).device
    n = len(layout.input_ids)
    n_layers = len(base.layers)
    keys, values = [None] * n_layers, [None] * n_layers
    layer_inputs, layer_outputs = [None] * n_layers, [None] * n_layers
    local_positions = torch.full((1, n), -1, device=device, dtype=torch.long)
    ids_all = torch.tensor(layout.input_ids, device=device, dtype=torch.long).unsqueeze(0)
    for span in layout.cache_chunk_spans:
        indices = list(span.indices())
        ids = ids_all[:, indices]
        local = torch.arange(len(indices), device=device, dtype=torch.long).unsqueeze(0)
        hook = base.norm.register_forward_hook(lambda _m, inputs, _out: inputs[0])
        try:
            out = base(input_ids=ids, position_ids=local, use_cache=True, output_hidden_states=True, return_dict=True)
        finally:
            hook.remove()
        hidden_states = out.hidden_states
        if hidden_states is None or len(hidden_states) != n_layers + 1:
            raise RuntimeError("independent prefill did not return all hidden states")
        past = out.past_key_values
        local_positions[:, indices] = local
        for layer_index in range(n_layers):
            key, value = (x.detach() for x in _layer_kv(past, layer_index))
            if keys[layer_index] is None:
                keys[layer_index] = torch.zeros((1, key.shape[1], n, key.shape[-1]), device=device, dtype=key.dtype)
                values[layer_index] = torch.zeros((1, value.shape[1], n, value.shape[-1]), device=device, dtype=value.dtype)
                layer_inputs[layer_index] = torch.zeros((1, n, hidden_states[layer_index].shape[-1]), device=device, dtype=hidden_states[layer_index].dtype)
                layer_outputs[layer_index] = torch.zeros_like(layer_inputs[layer_index])
            keys[layer_index][:, :, indices, :] = key
            values[layer_index][:, :, indices, :] = value
            layer_inputs[layer_index][:, indices, :] = hidden_states[layer_index]
            layer_outputs[layer_index][:, indices, :] = hidden_states[layer_index + 1]
    if any(x is None for x in keys + values + layer_inputs + layer_outputs):
        raise RuntimeError("failed to assemble independent document KV")
    return _A3Cache(keys, values, layer_inputs, layer_outputs, local_positions)


@torch.inference_mode()
def _build_a3_state(model: Any, cache: _A3Cache, layout: _A3Layout) -> _A3State:
    base = _base(model)
    device = next(model.parameters()).device
    n = len(layout.input_ids)
    ids = torch.tensor(layout.input_ids, device=device, dtype=torch.long).unsqueeze(0)
    pos = torch.arange(n, device=device, dtype=torch.long).unsqueeze(0)
    hidden = base.embed_tokens(ids)
    layer0 = base.layers[0]
    normed = layer0.input_layernorm(hidden)
    q0, k0, v0, dim = _project_qkv(layer0, normed, base)
    cos, sin = _cos_sin(base, normed, pos)
    q0, k0 = _apply_rope(q0, cos, sin), _apply_rope(k0, cos, sin)
    k0, v0 = _repeat_kv(k0, v0, q0.shape[1])
    attended = torch.nn.functional.scaled_dot_product_attention(q0, k0, v0, dropout_p=0.0, is_causal=True, scale=1.0 / math.sqrt(dim))
    attended = attended.transpose(1, 2).contiguous().view(1, n, -1)
    hidden = hidden + layer0.self_attn.o_proj(attended)
    hidden = hidden + layer0.mlp(layer0.post_attention_layernorm(hidden))
    layer = base.layers[1]
    normed = layer.input_layernorm(hidden)
    q, k, _v, _dim = _project_qkv(layer, normed, base)
    cos, sin = _cos_sin(base, normed, pos)
    q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
    start, end = layout.attention_context_span.start, layout.query_span.end
    return _A3State(q[:, :, layout.query_span.start:end, :], k[:, :, start:end, :], range(layout.query_span.start, end), range(start, end))


@torch.inference_mode()
def _a3_score(state: _A3State, layout: _A3Layout) -> _A3Record:
    q, k = state.query_states, state.key_states
    k, _ = _repeat_kv(k, k, q.shape[1])
    qf = q.permute(0, 2, 1, 3).reshape(1, q.shape[2], -1)
    kf = k.permute(0, 2, 1, 3).reshape(1, k.shape[2], -1)
    logits = torch.matmul(qf, kf.transpose(1, 2)) / math.sqrt(qf.shape[-1])
    qp = torch.tensor(state.query_positions, device=q.device)
    kp = torch.tensor(state.key_positions, device=q.device)
    logits = logits.masked_fill(kp.view(1, 1, -1) > qp.view(1, -1, 1), torch.finfo(logits.dtype).min)
    weights = torch.nn.functional.softmax(logits, dim=-1, dtype=torch.float32).squeeze(0)
    context_count = len(state.key_positions) - len(state.query_positions)
    raw = weights[:, :context_count].sum(dim=0)
    pooled = torch.nn.functional.avg_pool1d(raw.view(1, 1, -1), kernel_size=5, padding=2, stride=1).view(-1)
    key_start = state.key_positions[0]
    offsets = torch.tensor([int(x) - int(key_start) for x in layout.document_indices], device=pooled.device)
    return _A3Record(pooled.index_select(0, offsets).detach().cpu().float())


def _rectangular_attention(query, key, value, rows, sliding_window=None, row_tile_size=512):
    key, value = _repeat_kv(key, value, query.shape[1])
    outputs = []
    columns = torch.arange(key.shape[-2], device=key.device).view(1, 1, 1, -1)
    for start in range(0, query.shape[-2], max(1, int(row_tile_size))):
        stop = min(query.shape[-2], start + max(1, int(row_tile_size)))
        rows_abs = rows[start:stop].view(1, 1, -1, 1)
        allowed = columns <= rows_abs
        if sliding_window is not None:
            allowed &= columns > rows_abs - int(sliding_window)
        outputs.append(torch.nn.functional.scaled_dot_product_attention(query[:, :, start:stop, :], key, value, attn_mask=allowed, dropout_p=0.0, is_causal=False, scale=1.0 / math.sqrt(query.shape[-1])))
    return torch.cat(outputs, dim=-2)


@torch.inference_mode()
def _recompute(model: Any, cache: _A3Cache, layout: _A3Layout, selected):
    base = _base(model)
    device = next(model.parameters()).device
    n = len(layout.input_ids)
    selected = selected.to(device=device, dtype=torch.long)
    fresh = torch.tensor(layout.fresh_indices, device=device, dtype=torch.long)
    update = torch.cat((selected, fresh)).unique(sorted=True)
    selected_set = set(int(x) for x in selected.detach().cpu().tolist())
    unselected = torch.tensor([i for i in layout.document_indices if i not in selected_set], device=device, dtype=torch.long)
    absolute = torch.arange(n, device=device, dtype=torch.long)
    ids = torch.tensor(layout.input_ids, device=device, dtype=torch.long).unsqueeze(0)
    hidden = base.embed_tokens(ids)
    recovered = []
    for layer_index, key in enumerate(cache.keys):
        recovered_key = key.to(device).clone()
        doc = torch.tensor(layout.document_indices, device=device, dtype=torch.long)
        old = cache.local_positions[:, doc]
        new = absolute[doc].unsqueeze(0)
        zero = torch.zeros((1, doc.numel(), base.config.hidden_size), device=device, dtype=recovered_key.dtype)
        old_cos, old_sin = _cos_sin(base, zero, old)
        new_cos, new_sin = _cos_sin(base, zero, new)
        unrotated = _apply_rope(recovered_key[:, :, doc, :], old_cos, -old_sin)
        recovered_key[:, :, doc, :] = _apply_rope(unrotated, new_cos, new_sin)
        recovered.append(recovered_key)
    keys, values = [], []
    for layer_index, layer in enumerate(base.layers):
        if unselected.numel():
            hidden[:, unselected, :] = cache.layer_inputs[layer_index][:, unselected, :].to(device)
        residual = hidden[:, update, :]
        normed = layer.input_layernorm(residual)
        q, k, v, dim = _project_qkv(layer, normed, base)
        pos = absolute[update].unsqueeze(0)
        cos, sin = _cos_sin(base, normed, pos)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        key = recovered[layer_index].clone()
        value = cache.values[layer_index].to(device).clone()
        key[:, :, update, :] = k
        value[:, :, update, :] = v
        attn = _rectangular_attention(q, key, value, absolute[update], getattr(base.config, "sliding_window", None))
        updated = residual + layer.self_attn.o_proj(attn.transpose(1, 2).contiguous().view(1, update.numel(), -1))
        updated = updated + layer.mlp(layer.post_attention_layernorm(updated))
        next_hidden = hidden.clone()
        if unselected.numel():
            next_hidden[:, unselected, :] = cache.layer_outputs[layer_index][:, unselected, :].to(device)
        next_hidden[:, update, :] = updated
        hidden = next_hidden
        keys.append(key.detach().clone())
        values.append(value.detach().clone())
    logits = model.lm_head(base.norm(hidden[:, -1, :])).squeeze(0).detach().clone()
    return _A3RecomputeOutput(keys, values, logits)

_READY = "[a3-helper] ready"


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _batch(value: Any, nested: bool) -> List[Any]:
    if not isinstance(value, list):
        raise TypeError("A3 data must be a list")
    for i, item in enumerate(value):
        if nested:
            if not isinstance(item, list) or any(not isinstance(x, str) for x in item):
                raise TypeError(f"A3 Prepare data[{i}] must be a list of strings")
        elif not isinstance(item, str):
            raise TypeError(f"A3 Run data[{i}] must be a string")
    return list(value)


class A3(Method):
    """CacheBlend-style Method wrapper around the A3 worker backend."""

    name = "a3"
    maxCaseBatchSize = 1
    method_metrics = ("reuse_ratio", "recompute_ratio")

    def __init__(
        self,
        gpuNums: int = 1,
        perfWeight: float = 1.0,
        *,
        maxNewTokens: int = 64,
        dtype: str = "bfloat16",
        maxModelLen: int = 32768,
        recompRatio: float = 0.15,
        workerPython: Optional[str] = None,
        startTimeout: float = 1800.0,
        requestTimeout: float = 3600.0,
        tag: Optional[str] = None,
    ):
        super().__init__(gpuNums=gpuNums, perfWeight=perfWeight, maxGpuNums=1, tag=tag)
        self.maxNewTokens = _positive_int(maxNewTokens, "maxNewTokens")
        if dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("A3 dtype must be float16, bfloat16, or float32")
        self.dtype = dtype
        self.maxModelLen = _positive_int(maxModelLen, "maxModelLen")
        if isinstance(recompRatio, bool) or not 0 <= float(recompRatio) <= 1:
            raise ValueError("recompRatio must be between 0 and 1")
        if startTimeout <= 0 or requestTimeout <= 0:
            raise ValueError("A3 timeouts must be positive")
        self.recompRatio = float(recompRatio)
        self.startTimeout = float(startTimeout)
        self.requestTimeout = float(requestTimeout)
        self.modelPath = DefaultModelPath()
        self.samplingConfig = ResolveSamplingConfig(self.modelPath)
        self.workerPython = Path(workerPython).expanduser() if workerPython else Path(sys.executable)
        if not self.workerPython.is_file():
            raise FileNotFoundError(f"A3 worker python not found: {self.workerPython}")
        self._proc: Optional[subprocess.Popen[str]] = None
        self._stderr = deque(maxlen=200)
        self._drain: Optional[threading.Thread] = None
        self._prepared: Optional[List[List[str]]] = None

    def Initialize(self, gpuIds: Sequence[int]) -> None:
        super().Initialize(gpuIds)
        env = dict(os.environ)
        env.pop("LD_PRELOAD", None)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in self.gpuIds)
        config = {
            "model_path": self.modelPath,
            "gpu_ids": list(self.gpuIds),
            "max_new_tokens": self.maxNewTokens,
            "dtype": self.dtype,
            "max_model_len": self.maxModelLen,
            "recomp_ratio": self.recompRatio,
            "sampling_config": self.samplingConfig,
        }
        self._proc = subprocess.Popen(
            [str(self.workerPython), str(Path(__file__).resolve()), "--a3-worker", "--config-json", json.dumps(config)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=env, bufsize=1,
        )
        self._drain = threading.Thread(target=self._drain_stderr, daemon=True, name="a3-stderr")
        self._drain.start()
        deadline = time.monotonic() + self.startTimeout
        while time.monotonic() < deadline:
            assert self._proc.stdout is not None
            line = self._proc.stdout.readline()
            if not line:
                if self._proc.poll() is not None:
                    raise RuntimeError(f"A3 worker exited: {''.join(self._stderr)}")
                continue
            if line.strip() == _READY:
                return
            print(line.rstrip(), flush=True)
        self._kill()
        raise TimeoutError("A3 worker did not become ready")

    def _drain_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        try:
            for line in proc.stderr:
                self._stderr.append(line)
                sys.stderr.write(f"[a3-helper] {line}")
                sys.stderr.flush()
        except Exception:
            pass

    def _kill(self) -> None:
        proc = self._proc
        if proc is None:
            return
        with suppress(Exception):
            proc.kill()
        with suppress(Exception):
            proc.wait(timeout=2)

    def _request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.stdout is None:
            raise RuntimeError("A3 worker is not initialized")
        proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        proc.stdin.flush()
        box: List[str] = []
        done = threading.Event()

        def read() -> None:
            try:
                box.append(proc.stdout.readline())
            finally:
                done.set()

        threading.Thread(target=read, daemon=True).start()
        if not done.wait(self.requestTimeout):
            self._kill()
            self._proc = None
            raise TimeoutError(f"A3 worker timeout after {self.requestTimeout:.1f}s")
        if not box or not box[0]:
            raise RuntimeError(f"A3 worker closed stdout: {''.join(self._stderr)}")
        try:
            response = json.loads(box[0])
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"A3 worker returned invalid JSON: {box[0]!r}") from exc
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "A3 worker request failed"))
        return response

    def Prepare(self, data: List[List[str]]) -> None:
        chunks = _batch(data, nested=True)
        response = self._request({"op": "prepare", "chunks": chunks})
        if response.get("cases") != len(chunks):
            raise RuntimeError("invalid A3 Prepare acknowledgement")
        self._prepared = [list(x) for x in chunks]

    def Run(self, data: List[str], retainOutput: Optional[List[bool]] = None) -> List[Result]:
        prompts = _batch(data, nested=False)
        if self._prepared is None:
            raise RuntimeError("A3 Run requires Prepare")
        if len(self._prepared) != len(prompts):
            raise ValueError("A3 Prepare/Run case count mismatch")
        response = self._request({"op": "run", "prompts": prompts, "chunks": self._prepared})
        raw = response.get("results")
        if not isinstance(raw, list) or len(raw) != len(prompts):
            raise RuntimeError("invalid A3 result list")
        return [self._result(item) for item in raw]

    @staticmethod
    def _result(item: Dict[str, Any]) -> Result:
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            raise RuntimeError("A3 result is missing text")
        try:
            performance = {
                TtftKey: float(item["ttft"]),
                TotalTimeKey: float(item["total_time"]),
                NumOutputTokensKey: int(item["num_output_tokens"]),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("A3 result has invalid timing fields") from exc
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        return Result(output=item["text"], performance=performance, metadata=metadata)

    def Reset(self) -> None:
        self._prepared = None
        if self._proc is not None:
            with suppress(Exception):
                self._request({"op": "reset"})

    def Close(self) -> None:
        proc = self._proc
        self._proc = None
        self._prepared = None
        if proc is None:
            return
        with suppress(Exception):
            if proc.poll() is None and proc.stdin is not None:
                proc.stdin.write('{"op":"close"}\n')
                proc.stdin.flush()
                proc.wait(timeout=5)
        if proc.poll() is None:
            with suppress(Exception):
                proc.kill()
                proc.wait(timeout=2)
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            with suppress(Exception):
                if stream is not None:
                    stream.close()
        if self._drain is not None:
            self._drain.join(timeout=1)
            self._drain = None

    def __del__(self) -> None:
        with suppress(Exception):
            self.Close()


class _Worker:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.chunks: Optional[List[List[str]]] = None
        self._load()

    def _load(self) -> None:
        try:
            import torch
            import torch.nn as nn
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            raise RuntimeError("A3 backend requires torch and transformers") from exc
        model_path = str(self.config["model_path"])
        if not model_path:
            raise RuntimeError("A3 ModelPath is empty")
        self.torch = torch
        self.nn = nn
        self.build_cache = _build_stitched_cache
        self.build_state = _build_a3_state
        self.recompute = _recompute
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=(
                getattr(self.torch, self.config["dtype"])
                if self.device.type == "cuda"
                else self.torch.float32
            ),
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device).eval()

        # A3's reviewed HF primitive is valid for dense Llama/Mistral.  Qwen3
        # is also dense but applies per-head Q/K RMSNorm; wrappers below make
        # the same primitive see the normalized projections without changing
        # the model's ordinary forward path.
        cfg = getattr(self.model, "config", None)
        descriptors = " ".join(
            str(x).lower()
            for x in (
                getattr(cfg, "model_type", ""),
                *(getattr(cfg, "architectures", None) or ()),
            )
        )
        if any(x in descriptors for x in ("qwen3.5", "qwen3_5", "hybrid", "moe", "mixtureofexperts")):
            raise RuntimeError("A3 supports dense Llama/Mistral/Qwen3 only; hybrid/MoE models are unsupported")
        if not any(x in descriptors for x in ("llama", "mistral", "qwen3")):
            raise RuntimeError(f"A3 unsupported model architecture: {descriptors}")
        self.family = "qwen3" if "qwen3" in descriptors else ("mistral" if "mistral" in descriptors else "llama")

    def _qk_norm(self):
        """Temporarily normalize Q/K projections for Qwen3-style attention."""
        if self.family != "qwen3":
            return []
        patches = []
        for layer in getattr(self.model.model, "layers", ()):
            attn = layer.self_attn
            q_norm, k_norm = getattr(attn, "q_norm", None), getattr(attn, "k_norm", None)
            if q_norm is None or k_norm is None:
                continue
            heads = int(getattr(attn, "num_heads"))
            kv_heads = int(getattr(attn, "num_key_value_heads", heads))
            dim = int(getattr(attn, "head_dim", self.model.config.hidden_size // heads))

            class _Project(self.nn.Module):
                def __init__(self, original, norm, count, head_dim):
                    super().__init__()
                    self.original, self.norm, self.count, self.head_dim = original, norm, count, head_dim

                def forward(self, hidden):
                    value = self.original(hidden)
                    shape = value.shape[:-1] + (self.count, self.head_dim)
                    return self.norm(value.view(shape)).view(value.shape)

            old_q, old_k = attn.q_proj, attn.k_proj
            attn.q_proj = _Project(old_q, q_norm, heads, dim)
            attn.k_proj = _Project(old_k, k_norm, kv_heads, dim)
            patches.append((attn, old_q, old_k))
        return patches

    @staticmethod
    def _restore_qk_norm(patches):
        for attn, q_proj, k_proj in patches:
            attn.q_proj, attn.k_proj = q_proj, k_proj

    def _layout(self, prompt: str, chunks: Sequence[str]):
        encoded = self.tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
        ids = encoded["input_ids"]
        offsets = tuple((int(a), int(b)) for a, b in encoded["offset_mapping"])
        ranges = []
        cursor = 0
        for chunk in chunks:
            if not chunk:
                continue
            start = prompt.find(chunk, cursor)
            if start < 0:
                raise ValueError("A3 prepared chunk is not present in the Run prompt")
            ranges.append((start, start + len(chunk)))
            cursor = start + len(chunk)
        if not ranges:
            raise ValueError("A3 requires at least one non-empty document chunk")
        query_start = max(right for _, right in ranges)
        documents = []
        for index, (start, end) in enumerate(offsets):
            if end <= start:
                continue
            if any(start >= left and end <= right for left, right in ranges):
                documents.append(index)
        if not documents:
            raise ValueError("A3 requires at least one document token")
        chunks_spans = []
        for left, right in ranges:
            unit = [i for i, (start, end) in enumerate(offsets) if end > start and start >= left and end <= right]
            for begin in range(0, len(unit), 512):
                part = unit[begin:begin + 512]
                if part:
                    chunks_spans.append(_Span(part[0], part[-1] + 1))
        query_indices = [i for i, (start, end) in enumerate(offsets) if end > start and start >= query_start and end <= len(prompt)]
        if not query_indices:
            raise ValueError("A3 requires a non-empty query suffix")
        return _A3Layout(
            ids,
            offsets,
            documents,
            chunks_spans,
            _Span(query_indices[0], query_indices[-1] + 1),
            _Span(0, query_start),
        )

    def _decode(self, result: Any, prompt_len: int) -> tuple[str, int, float]:
        token = int(self.torch.argmax(result.first_token_logits).item())
        generated = [token]
        decode_start = time.perf_counter()
        legacy_past = tuple((k.detach(), v.detach()) for k, v in zip(result.keys, result.values))
        try:
            from transformers import DynamicCache
            past = DynamicCache(ddp_cache_data=legacy_past)
        except Exception:
            past = legacy_past
        stop = {x for x in (self.tokenizer.eos_token_id,) if x is not None}
        for _ in range(max(0, int(self.config["max_new_tokens"]) - 1)):
            if token in stop:
                break
            ids = self.torch.tensor([[token]], device=self.device, dtype=self.torch.long)
            pos = self.torch.tensor([[prompt_len + len(generated) - 1]], device=self.device, dtype=self.torch.long)
            with self.torch.inference_mode():
                out = self.model(input_ids=ids, past_key_values=past, position_ids=pos, use_cache=True, return_dict=True)
            token = int(self.torch.argmax(out.logits[:, -1, :], dim=-1).item())
            generated.append(token)
            past = out.past_key_values
        return self.tokenizer.decode(generated, skip_special_tokens=True), len(generated), time.perf_counter() - decode_start

    def request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        op = payload.get("op")
        if op == "prepare":
            self.chunks = [list(x) for x in _batch(payload.get("chunks"), nested=True)]
            return {"ok": True, "cases": len(self.chunks)}
        if op == "reset":
            self.chunks = None
            return {"ok": True}
        if op == "close":
            return {"ok": True, "stop": True}
        if op != "run":
            raise ValueError(f"unknown A3 op: {op!r}")
        prompts = _batch(payload.get("prompts"), nested=False)
        chunks = _batch(payload.get("chunks"), nested=True)
        if len(prompts) != len(chunks):
            raise ValueError("A3 prompt/chunk case count mismatch")
        results = []
        for prompt, case_chunks in zip(prompts, chunks):
            started = time.perf_counter()
            layout = self._layout(prompt, case_chunks)
            cache = self.build_cache(self.model, layout)
            patches = self._qk_norm()
            try:
                state = self.build_state(self.model, cache, layout)
                record = _a3_score(state, layout)
                candidates = list(layout.document_indices)
                count = min(len(candidates), int(len(candidates) * float(self.config["recomp_ratio"])))
                order = record.score.detach().cpu().argsort(descending=True).tolist()
                selected = [candidates[i] for i in order[:count]]
                recomputed = self.recompute(self.model, cache, layout, self.torch.tensor(selected, dtype=self.torch.long, device=self.device))
            finally:
                self._restore_qk_norm(patches)
            text, num_tokens, decode_time = self._decode(recomputed, len(layout.input_ids))
            total = time.perf_counter() - started
            ratio = len(selected) / max(1, len(candidates))
            results.append({
                "text": text,
                "ttft": total - decode_time,
                "total_time": total,
                "num_output_tokens": num_tokens,
                "metadata": {"reuse_ratio": 1.0 - ratio, "recompute_ratio": ratio, "selected_tokens": len(selected), "candidate_tokens": len(candidates)},
            })
        return {"ok": True, "results": results}


def _worker_main(config_json: str) -> int:
    worker = _Worker(json.loads(config_json))
    print(_READY, flush=True)
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            response = worker.request(json.loads(line))
        except Exception as exc:
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(response, ensure_ascii=False), flush=True)
        if response.get("stop"):
            return 0
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a3-worker", action="store_true")
    parser.add_argument("--config-json")
    args = parser.parse_args()
    if not args.a3_worker or not args.config_json:
        parser.error("worker mode requires --a3-worker --config-json")
    return _worker_main(args.config_json)


if __name__ == "__main__":
    raise SystemExit(main())

__all__ = ["A3"]
