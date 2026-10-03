"""Load bundled A^3 model classes without changing Transformers globals."""
from __future__ import annotations

import torch
from transformers import AutoConfig, AutoTokenizer


def model_classes(config):
    """Return runtime/precompute classes for the checkpoint's architecture."""
    family = config.model_type
    if family == "llama":
        from .llama.llama import LlamaForCausalLM
        from .llama.llama_precompute import LlamaForCausalLM_Precompute
        return LlamaForCausalLM, LlamaForCausalLM_Precompute
    if family == "mistral":
        from .mistral.mistral import MistralForCausalLM
        from .mistral.mistral_precompute import MistralForCausalLM_Precompute
        return MistralForCausalLM, MistralForCausalLM_Precompute
    if family == "qwen2":
        from .qwen.qwen import Qwen2ForCausalLM
        from .qwen.qwen_precompute import Qwen2ForCausalLM_Precompute
        return Qwen2ForCausalLM, Qwen2ForCausalLM_Precompute
    if family == "qwen3":
        from .qwen3 import Qwen3ForCausalLM, Qwen3ForCausalLM_Precompute
        return Qwen3ForCausalLM, Qwen3ForCausalLM_Precompute
    raise ValueError(f"A^3 supports Llama, Mistral, Qwen2 and dense Qwen3; got {family!r}")


def load_config(path):
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    if getattr(config, "quantization_config", None):
        raise ValueError("A^3 requires an unquantized FP16/BF16 checkpoint")
    # The precompute path captures pre-RoPE KV in its SDPA attention class.
    config._attn_implementation = "sdpa"
    return config


def _load(args, precompute):
    config = load_config(args.model)
    runtime_cls, precompute_cls = model_classes(config)
    cls = precompute_cls if precompute else runtime_cls
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
    model = cls.from_pretrained(
        args.model, config=config, low_cpu_mem_usage=True,
        dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()
    return model, tokenizer


def load_model(args):
    return _load(args, precompute=False)


def load_model_precompute(args):
    return _load(args, precompute=True)
