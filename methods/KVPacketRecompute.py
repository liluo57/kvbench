from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence

from core.Config import DefaultConfigPath, Get, ModelPath as DefaultModelPath
from core.Method import Method, ResolveMaxNewTokens
from core.Result import NumOutputTokensKey, Result, TotalTimeKey, TtftKey
from core.Sampling import ResolveSamplingConfig, TransformersGenerationKwargs
from helpers.backends.Prompt import ComposeInterleavedReuse


def _import_kvpacket_modules():
    config = Get("KVPacket", {}) or {}
    if not isinstance(config, dict):
        raise TypeError("KVPacket in config.yaml must be a mapping")

    configured_root = config.get("RecomputeRepoPath")
    root = (
        Path(str(configured_root)).expanduser()
        if configured_root
        else Path(__file__).resolve().parents[2] / "kvpacket"
    )
    if not root.is_absolute():
        root = DefaultConfigPath.parent / root
    root = root.resolve()
    # Accept either the artifact checkout root or the package directory.
    if root.name == "kv_packet" and root.is_dir():
        root = root.parent
    if not (root / "kv_packet").is_dir():
        raise RuntimeError(
            "KVPacket recomputation artifact is missing: "
            f"{root} (set KVPacket.RecomputeRepoPath in config.yaml)"
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return importlib.import_module("kv_packet")


def _contiguous_reuse_plan(texts: list[str], prompt: str):
    """Map a prompt to KVPacket's ``preamble + documents + task_prompt`` API.

    The artifact can correctly handle a fresh preamble followed by one
    contiguous run of independently cached documents and a fresh suffix.  It
    cannot represent a fresh span between two cached documents.  Such prompts
    must use full recomputation instead of silently treating later cached text
    as fresh query tokens.
    """
    parts = ComposeInterleavedReuse(texts, prompt)
    cached = [i for i, (index, _) in enumerate(parts) if index is not None]
    if not cached:
        return None

    first, last = cached[0], cached[-1]
    if any(index is None for index, _ in parts[first:last + 1]):
        return None

    preamble = "".join(text for _, text in parts[:first])
    document_indices = [index for index, _ in parts[first:last + 1]]
    assert all(index is not None for index in document_indices)
    task_prompt = "".join(text for _, text in parts[last + 1:])
    return preamble, [int(index) for index in document_indices], task_prompt


class _KVPacketRecompute(Method):
    """Shared lifecycle for KVPacket cache-combination methods."""

    backend = "kvpacket"
    method_metrics = ("reuse_ratio",)
    maxCaseBatchSize = 1
    _method_name: str

    def __init__(self, gpuNums=1, perfWeight=1.0, *, maxNewTokens=64,
                 dtype="bfloat16", tag: Optional[str] = None):
        super().__init__(gpuNums=gpuNums, perfWeight=perfWeight, maxGpuNums=1, tag=tag)
        self.modelPath = DefaultModelPath()
        self.samplingConfig = ResolveSamplingConfig(self.modelPath)
        self.maxNewTokens = ResolveMaxNewTokens(maxNewTokens)
        self.dtype = dtype
        self._torch = None
        self._model = None
        self._tokenizer = None
        self._get_kv_caches = None
        self._evaluate: Callable[..., dict[str, Any]] | None = None
        self._full_evaluate: Callable[..., dict[str, Any]] | None = None
        self._states: list[dict[str, Any]] = []

    def Initialize(self, gpuIds: Sequence[int]) -> None:
        super().Initialize(gpuIds)
        _import_kvpacket_modules()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from kv_packet.cache import get_kv_caches
        from kv_packet.cache_comb.methods import get_cache_comb_func

        self._torch = torch
        gpu = self.gpuIds[0]
        if not torch.cuda.is_available():
            raise RuntimeError("KVPacket recomputation methods require CUDA")
        torch.cuda.set_device(gpu)
        device = f"cuda:{gpu}"
        self._tokenizer = AutoTokenizer.from_pretrained(self.modelPath)
        load_kwargs = {"device_map": device, "low_cpu_mem_usage": True}
        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.modelPath, dtype=getattr(torch, self.dtype), **load_kwargs
            )
        except TypeError:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.modelPath, torch_dtype=getattr(torch, self.dtype), **load_kwargs
            )
        self._model.eval()
        # KVPacket currently supplies recompute kernels for these architectures.
        if self._model.__class__.__name__ not in {"LlamaForCausalLM", "MistralForCausalLM", "Qwen3ForCausalLM"}:
            raise RuntimeError(
                "KVPacket-backed methods support LlamaForCausalLM and "
                "MistralForCausalLM, and "
                f"Qwen3ForCausalLM; got {self._model.__class__.__name__}"
            )
        self._get_kv_caches = get_kv_caches
        self._evaluate = get_cache_comb_func(self._method_name)
        self._full_evaluate = get_cache_comb_func("full_recompute")

    def Prepare(self, data: List[List[str]]) -> None:
        assert self._tokenizer is not None and self._get_kv_caches is not None
        self._states = []
        for chunks in data:
            texts = [str(chunk) for chunk in chunks or [] if str(chunk)]
            caches = []
            for text in texts:
                ids = self._tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
                caches.append(self._get_kv_caches(self._model, input_ids=ids.to(self._model.device))[0])
            self._states.append({"texts": texts, "caches": caches})

    def _kwargs(self) -> dict[str, Any]:
        return {}

    def Run(
        self,
        data: List[str],
        retainOutput: Optional[List[bool]] = None,
        maxNewTokens: Optional[int] = None,
    ) -> List[Result]:
        assert self._evaluate is not None and self._full_evaluate is not None and self._tokenizer is not None and self._torch is not None
        if len(self._states) != len(data):
            self._states = [{"texts": [], "caches": []} for _ in data]
        from transformers import GenerationConfig

        generation_limit = (
            self.maxNewTokens
            if maxNewTokens is None
            else ResolveMaxNewTokens(maxNewTokens)
        )
        generation_kwargs = TransformersGenerationKwargs(
            self.samplingConfig, generation_limit
        )
        generation_kwargs["use_cache"] = True
        generation_config = GenerationConfig(**generation_kwargs)
        results = []
        for prompt, state in zip(data, self._states, strict=True):
            request_started = time.perf_counter()
            plan = _contiguous_reuse_plan(state["texts"], prompt)
            if plan is None:
                preamble, documents, document_kvs, task_prompt = "", [], [], prompt
                reuse_mode = (
                    "full_recompute"
                    if self._method_name == "full_recompute"
                    else "full_recompute_fallback"
                )
            else:
                preamble, indices, task_prompt = plan
                documents = [state["texts"][i] for i in indices]
                document_kvs = [state["caches"][i].copy(clone_tensor=False) for i in indices]
                reuse_mode = "contiguous"

            evaluate_started = time.perf_counter()
            if not documents and self._method_name != "full_recompute":
                result = self._full_evaluate(
                    self._model, self._tokenizer, generation_config, "", [], task_prompt,
                    [], "", kwargs=self._kwargs()
                )
            else:
                result = self._evaluate(
                    self._model, self._tokenizer, generation_config, preamble, documents,
                    task_prompt, document_kvs, "", kwargs=self._kwargs()
                )
            evaluate_ttft = float(result["ttft"])
            output = str(result.get("output", ""))
            n_output = len(self._tokenizer(output, add_special_tokens=False)["input_ids"])
            n_input = len(self._tokenizer(prompt, add_special_tokens=False)["input_ids"])
            total = time.perf_counter() - request_started
            results.append(Result(
                output=output,
                performance={
                    # Prompt matching and cache selection happen in this
                    # adapter before the artifact starts its TTFT clock.
                    TtftKey: float(evaluate_started - request_started) + evaluate_ttft,
                    NumOutputTokensKey: n_output,
                    TotalTimeKey: total,
                },
                metadata={
                    "backend": self.backend,
                    "reuse_mode": reuse_mode,
                    "n_input": n_input,
                    "cached_tokens": sum(len(self._tokenizer(text, add_special_tokens=False)["input_ids"]) for text in documents),
                    "reuse_ratio": (sum(len(self._tokenizer(text, add_special_tokens=False)["input_ids"]) for text in documents) / n_input if n_input else 0.0),
                    "flops": result["flops"],
                },
            ))
        return results

    def Reset(self) -> None:
        self._states = []

    def Close(self) -> None:
        self._states = []
        self._evaluate = None
        self._full_evaluate = None
        self._get_kv_caches = None
        self._tokenizer = None
        self._model = None
        if self._torch is not None:
            self._torch.cuda.empty_cache()


class EPIC(_KVPacketRecompute):
    name = "epic"
    _method_name = "epic"

    def __init__(self, *args, recomputeTokens=32, recompute_tokens=None, **kwargs):
        self.recomputeTokens = int(recomputeTokens if recompute_tokens is None else recompute_tokens)
        if self.recomputeTokens < 0:
            raise ValueError("recompute_tokens must be non-negative")
        super().__init__(*args, **kwargs)

    def _kwargs(self):
        return {"recompute_tokens": self.recomputeTokens}


class RandRecomputeTransformer(_KVPacketRecompute):
    name = "rand_recompute"
    _method_name = "rand_recompute"

    def __init__(self, *args, recomputeRatio=0.2, recompute_ratio=None, seed=None, **kwargs):
        self.recomputeRatio = float(recomputeRatio if recompute_ratio is None else recompute_ratio)
        if self.recomputeRatio < 0:
            raise ValueError("recompute_ratio must be non-negative")
        self.seed = seed
        super().__init__(*args, **kwargs)

    def _kwargs(self):
        return {"recompute_ratio": self.recomputeRatio, "seed": self.seed}


class FullRecomputeTransformer(_KVPacketRecompute):
    name = "full_recompute"
    _method_name = "full_recompute"

    def Prepare(self, data: List[List[str]]) -> None:
        # Match KVPacket's full_recompute: document caches are an ignored input.
        self._states = [{"texts": [], "caches": []} for _ in data]
