"""Experimental vLLM TP worker extension for auditing supervision K/V.

Capture the actual post-normalization/post-RoPE BF16 tensors passed to vLLM's
attention cache writer. Never read padded pages or Mamba recurrent state. This
is deliberately eager-only; a CUDA graph would bypass Python capture hooks.
It is NOT a validated replacement for the native training target.
"""
from __future__ import annotations

from pathlib import Path
import re

import torch


class CaptureWorkerExtension:
    def cr_capture_begin(self):
        from vllm.forward_context import get_forward_context

        self._cr_capture = {}
        self._cr_prompts = {}
        if not hasattr(self, '_cr_hooks'):
            self._cr_hooks = []
            for name, module in self.model_runner.model.named_modules():
                if not (name.endswith('.self_attn.attn') and hasattr(module, 'num_kv_heads')):
                    continue
                layer = int(re.search(r'\.layers\.(\d+)\.', '.' + name).group(1))

                def capture(module, args, layer=layer):
                    if self._cr_capture is None:
                        return
                    ctx = get_forward_context()
                    metadata = ctx.attn_metadata[module.layer_name]
                    starts = metadata.query_start_loc.cpu().tolist()
                    batch = self.model_runner.input_batch
                    req_ids = list(batch.req_ids)
                    # At most one complete prefill per request. Fail closed on
                    # cache hits, chunking, decode, or unsupported scheduling.
                    for row, request in enumerate(req_ids):
                        start, end = starts[row:row + 2]
                        if (int(batch.num_computed_tokens_cpu[row]) != 0
                                or end - start != int(batch.num_prompt_tokens[row])):
                            raise RuntimeError('Capture requires complete uncached prompt prefill')
                        layers = self._cr_capture.setdefault(request, {})
                        self._cr_prompts[request] = batch.token_ids_cpu[row, :end-start].tolist()
                        if layer in layers:
                            raise RuntimeError('Repeated layer/request capture')
                        k, v = args[1:3]
                        k = k.reshape(-1, module.num_kv_heads, module.head_size)
                        v = v.reshape_as(k)
                        layers[layer] = torch.stack((k[start:end], v[start:end]), dim=-2).detach().clone()

                self._cr_hooks.append(module.register_forward_pre_hook(capture))
            if len(self._cr_hooks) != 16:
                raise ValueError(f'Expected 16 full-attention hooks, got {len(self._cr_hooks)}')
        return {'hooks': len(self._cr_hooks)}

    def cr_capture_export(self, directory):
        from vllm.distributed import get_tensor_model_parallel_rank

        rank = get_tensor_model_parallel_rank()
        result = {request: {layer: kv.cpu() for layer, kv in layers.items()}
                  for request, layers in self._cr_capture.items()}
        path = Path(directory) / f'rank-{rank}.pt'
        torch.save(result, path)
        torch.save(self._cr_prompts, Path(directory)/f'prompts-rank-{rank}.pt')
        self._cr_capture = None
        return {'rank': rank, 'path': str(path), 'requests': list(result)}


def assemble_shards(shards, request_ids, *, layers, n_heads=4):
    """Restore TP rank/head order to [1,T,L,H,2,D] native BF16 cache."""
    if len(shards) != 2:
        raise ValueError('Missing TP rank captures')
    mapped=[]
    for shard in shards:
        selected={}
        for request in request_ids:
            matches=[key for key in shard if key==request or key.startswith(request+'-')]
            if len(matches)!=1:
                raise ValueError('Missing or ambiguous request capture')
            selected[request]=shard[matches[0]]
        if len(selected)!=len(shard):
            raise ValueError('Unexpected extra request captures')
        mapped.append(selected)
    shards=mapped
    outputs = []
    for request in request_ids:
        if any(set(s[request]) != set(layers) for s in shards):
            raise ValueError('Missing attention layer captures')
        kv = torch.stack([torch.cat([s[request][layer] for s in shards], dim=1)
                          for layer in layers], dim=1).unsqueeze(0)
        if kv.shape[3:] != (n_heads, 2, 256) or kv.dtype != torch.bfloat16:
            raise ValueError(f'Unexpected native cache geometry/dtype: {kv.shape}, {kv.dtype}')
        outputs.append(kv)
    return outputs
