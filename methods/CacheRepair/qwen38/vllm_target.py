"""Frozen checkpoint embedding for the repairer; the 27B forward lives in vLLM."""
from __future__ import annotations
import json
from pathlib import Path
import torch
from safetensors import safe_open
from .target import TargetGeometry


class VllmTarget(torch.nn.Module):
    _cr_vllm_target = True

    def __init__(self, path, device, model_id='Qwen/Qwen3.8-27B'):
        super().__init__()
        from transformers import Qwen3_5Config
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
        path = Path(path)
        self.model_id = model_id
        self.config = Qwen3_5Config.from_pretrained(path, local_files_only=True)
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(self.config.text_config).to(device)
        key = 'model.language_model.embed_tokens.weight'
        index = json.loads((path/'model.safetensors.index.json').read_text())
        with safe_open(path/index['weight_map'][key], framework='pt', device='cpu') as f:
            weight = f.get_tensor(key).to(device)
        self.embed_tokens = torch.nn.Embedding.from_pretrained(weight, freeze=True)
        TargetGeometry.from_target(self, target_model_id=model_id)
        self.requires_grad_(False).eval()

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(self, *args, **kwargs):
        raise RuntimeError('Use VllmPairPool for frozen-target forward')
