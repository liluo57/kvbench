"""Inference-only instrumentation for native/vLLM differential diagnosis."""
from __future__ import annotations

from pathlib import Path
import re
import types

import torch

from .vllm_capture import CaptureWorkerExtension


class InvestigationWorker(CaptureWorkerExtension):
    def cr_probe_begin(self):
        self.cr_capture_begin()
        self._cr_probe_active = True
        self._cr_probe = {}
        if not hasattr(self, '_cr_probe_hooks'):
            self._cr_probe_hooks = []
            model = self.model_runner.model
            def save(name, value, full=False):
                if not self._cr_probe_active:
                    return
                if isinstance(value,tuple):value=value[0]
                if value is None:return
                value=value.detach()
                if value.ndim>=2 and not full:
                    value=value[:min(64,value.shape[0])]
                self._cr_probe[name]=value.clone()
            for name,module in model.named_modules():
                if name.endswith('.embed_tokens'):
                    self._cr_probe_hooks.append(module.register_forward_hook(
                        lambda m,a,o:save('embedding',o)))
                match=re.search(r'\.layers\.(\d+)\.(input_layernorm|post_attention_layernorm)$','.'+name)
                if match:
                    key=f'layer.{match[1]}.{match[2]}'
                    self._cr_probe_hooks.append(module.register_forward_hook(
                        lambda m,a,o,key=key:save(key,o)))
                if '.layers.0.linear_attn.' in '.'+name:
                    leaf=name.rsplit('.',1)[-1]
                    if leaf in ('in_proj_qkvz','in_proj_ba'):
                        self._cr_probe_hooks.append(module.register_forward_hook(
                            lambda m,a,o,leaf=leaf:save('gdn0.'+leaf,o,True)))
                    if leaf=='out_proj':
                        self._cr_probe_hooks.append(module.register_forward_pre_hook(
                            lambda m,a:save('gdn0.gated',a[0],True)))
                        self._cr_probe_hooks.append(module.register_forward_hook(
                            lambda m,a,o:save('gdn0.output',o,True)))
            original=model.compute_logits
            def logits(hidden_states,*args,**kwargs):
                result=original(hidden_states,*args,**kwargs)
                if self._cr_probe_active and result is not None:
                    self._cr_probe['logits']=result.detach().clone()
                return result
            model.compute_logits=logits
            def teacher_tail(module,args,output):
                if getattr(self,'_cr_teacher_count',0):
                    self._cr_teacher_hidden=output[-self._cr_teacher_count:].detach().clone()
            self._cr_probe_hooks.append(model.register_forward_hook(teacher_tail))
            # Observe the first GDN layer's post-convolution and preprocessing.
            from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn
            layer=next(m for n,m in model.named_modules() if n.endswith('.layers.0.linear_attn'))
            self._cr_probe_layer=layer
            original_prep=gdn.fused_post_conv_prep
            def prep(*args,**kwargs):
                result=original_prep(*args,**kwargs)
                if self._cr_probe_active and kwargs.get('A_log').data_ptr()==layer.A_log.data_ptr():
                    save('gdn0.conv',kwargs['conv_output'],True)
                    for key,value in zip(['q','k','v','g','beta'],result):save('gdn0.'+key,value,True)
                return result
            gdn.fused_post_conv_prep=prep
        return {'instrumented':True}

    def cr_probe_export(self,directory):
        from vllm.distributed import get_tensor_model_parallel_rank
        self.cr_capture_export(directory)
        rank=get_tensor_model_parallel_rank()
        result={k:v.cpu() for k,v in self._cr_probe.items()}
        layer=self._cr_probe_layer
        result['weights']={n:p.detach().cpu() for n,p in layer.named_parameters()}
        torch.save(result,Path(directory)/f'probe-rank-{rank}.pt')
        self._cr_probe_active=False
        self._cr_probe={}
        return {'rank':rank,'keys':list(result)}

    def cr_teacher_begin(self,count):
        self._cr_teacher_count=count
        return True

    def cr_teacher_export(self,directory):
        from vllm.distributed import get_tensor_model_parallel_rank
        logits=self.model_runner.model.compute_logits(self._cr_teacher_hidden)
        rank=get_tensor_model_parallel_rank()
        torch.save(logits.detach().cpu(),Path(directory)/f'logits-rank-{rank}.pt')
        self._cr_teacher_count=0
        del self._cr_teacher_hidden
        return True

    def cr_ablation(self,mode):
        """One-variable controls, confined to this disposable worker instance."""
        from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn
        self._cr_ablation_mode=mode
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=mode not in ('strict','aligned_strict','aligned_conv')
        if not hasattr(self,'_cr_ablation_original_norms'):
            self._cr_ablation_original_norms=[]
            self._cr_ablation_original_cores=[]
            for name,module in self.model_runner.model.named_modules():
                if name.endswith(('.input_layernorm','.post_attention_layernorm','.norm')) and type(module).__name__=='GemmaRMSNorm':
                    self._cr_ablation_original_norms.append((module,module.forward))
                if name.endswith('.linear_attn.chunk_gated_delta_rule'):
                    self._cr_ablation_original_cores.append((module,module._forward_method))
            original=gdn.fused_post_conv_prep
            def prep(*args,_original=original,**kwargs):
                result=_original(*args,**kwargs)
                if self._cr_ablation_mode in ('beta','both','both_fla','aligned_strict','aligned_conv'):
                    result=(*result[:4],result[4].to(torch.bfloat16).float())
                return result
            gdn.fused_post_conv_prep=prep
            original_conv=gdn.causal_conv1d_fn
            def conv(*args,_original=original_conv,**kwargs):
                if self._cr_ablation_mode!='aligned_conv':return _original(*args,**kwargs)
                from causal_conv1d import causal_conv1d_fn
                x,weight,bias=args[:3]
                if kwargs['has_initial_state'].any():raise ValueError('Ablation only supports fresh prefills')
                starts=kwargs['query_start_loc'].cpu().tolist()
                values=[causal_conv1d_fn(x[:,start:end].unsqueeze(0),weight,bias,activation=kwargs['activation']).squeeze(0)
                        for start,end in zip(starts[:-1],starts[1:])]
                # The fused post-conv prep assumes contiguous feature rows
                # after the caller transposes this [D,T] result back to [T,D].
                return torch.cat(values,dim=-1).T.contiguous().T
            gdn.causal_conv1d_fn=conv
        round_norm=mode in ('norm','both','both_fla','aligned_strict','aligned_conv')
        for module,original in self._cr_ablation_original_norms:
            if not round_norm:
                module.forward=original
                continue
            def hf_norm(module,x,residual=None):
                dtype=x.dtype
                combined=x if residual is None else x+residual
                value=combined.float()
                value=value*torch.rsqrt(value.square().mean(-1,keepdim=True)+module.variance_epsilon)
                value=(value*(1.+module.weight.float())).to(dtype)
                return value if residual is None else (value,combined)
            module.forward=types.MethodType(hf_norm,module)
        for module,original in self._cr_ablation_original_cores:
            if mode not in ('both_fla','aligned_strict','aligned_conv'):
                module._forward_method=original
                continue
            def external(q,k,v,g,beta,initial_state,output_final_state,cu_seqlens=None,
                         core_attn_out=None,**kwargs):
                from fla.ops.gated_delta_rule import chunk_gated_delta_rule
                repeats=v.shape[2]//q.shape[2]
                if repeats>1:q=q.repeat_interleave(repeats,dim=2);k=k.repeat_interleave(repeats,dim=2)
                result=chunk_gated_delta_rule(q,k,v,g,beta,initial_state=initial_state,
                    output_final_state=output_final_state,cu_seqlens=cu_seqlens,use_qk_l2norm_in_kernel=False)
                if core_attn_out is not None:core_attn_out.copy_(result[0].squeeze(0))
                return result
            module._forward_method=external
        return {'mode':mode,'norms':len(self._cr_ablation_original_norms),'cores':len(self._cr_ablation_original_cores)}
