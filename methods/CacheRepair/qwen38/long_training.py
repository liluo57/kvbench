"""Memory-bounded FP32 residual loss with the same model and MSE objective.

Only activation storage and the exact SDPA backend change. Encoder and output
head work in token tiles; non-reentrant checkpointing recomputes intermediates
for backward. Parameter names, checkpoint format and serving forward stay intact.
"""
from contextlib import nullcontext
from types import MethodType

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def aligned_chunk_bias(chunk_ids, tile=256):
    if chunk_ids.ndim!=2 or (chunk_ids < -1).any():
        raise ValueError('Expected [B,T] chunk IDs, with -1 denoting padding')
    batch,length=chunk_ids.shape;stride=(length+15)//16*16
    bias=torch.empty(batch,1,length,stride,device=chunk_ids.device,dtype=torch.float32)
    keys=chunk_ids[:,None,:];key_real=keys>=0
    for start in range(0,length,tile):
        stop=min(length,start+tile);queries=chunk_ids[:,start:stop,None]
        visible=(queries>=keys)&(queries>=0)&key_real
        diagonal=torch.arange(start,stop,device=chunk_ids.device)[:,None]==torch.arange(length,device=chunk_ids.device)[None,:]
        visible|=(queries<0)&diagonal
        bias[:,0,start:stop,:length].zero_().masked_fill_(~visible,-torch.inf)
    return bias[...,:length]


def _attention(self,hidden,mask,cos,sin,block_mask):
    batch,length,width=hidden.shape
    q,k,v=[x.view(batch,length,self.n_heads,self.head_dim).transpose(1,2)
        for x in self.qkv(hidden).chunk(3,-1)]
    def rotate(x):
        a,b=x.chunk(2,-1)
        return torch.cat((a*cos-b*sin,b*cos+a*sin),-1)
    from torch.nn.attention import sdpa_kernel,SDPBackend
    # PyTorch's SDPA backend switches are process-global, so the repairer runs
    # in the same process as the native teacher pipeline on GPU 0. Keep the
    # memory-efficient kernel first, but leave math enabled as a fallback for
    # teacher attention shapes (notably grouped-query attention with masks)
    # that Efficient Attention cannot serve while repairer backward overlaps.
    context=(sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH], set_priority=True)
             if self._require_efficient else nullcontext())
    with context:
        result=F.scaled_dot_product_attention(rotate(q),rotate(k),v,attn_mask=mask)
    return self.proj(result.transpose(1,2).reshape(batch,length,width))


def configure(model,*,tile=256,efficient=True,checkpoint_blocks=True):
    if tile<1:raise ValueError('Token tile must be positive')
    model._long_training={'tile':tile,'efficient':efficient,'checkpoint_blocks':checkpoint_blocks}
    for block in model.blocks:
        block.attn._require_efficient=efficient
        block.attn.forward=MethodType(_attention,block.attn)
    return model


def hidden(model,stale,tokens,chunk_ids):
    settings=model._long_training;tile=settings['tile'];length=tokens.shape[1]
    model.target_config.validate_kv(stale)
    if tokens.shape!=chunk_ids.shape or tokens.shape!=stale.shape[:2] or not length:
        raise ValueError('Tokens, chunk IDs and KV must have matching nonempty axes')
    features,token_features=[],[]
    def encode(s,ids,cids):
        real=cids>=0
        normalized=(s.float()/model.sigma_stale.float()).masked_fill(~real[...,None,None,None,None],0)
        b,t=ids.shape
        sf=model.stale_enc(normalized.reshape(b,t,model.target_config.n_segments,model.target_config.head_dim))
        with torch.no_grad():embedding=model.tok_embed(ids.masked_fill(~real,0)).float()
        return sf,model.tok_proj(embedding)
    use_checkpoint=torch.is_grad_enabled()
    for start in range(0,length,tile):
        end=min(length,start+tile);args=(stale[:,start:end],tokens[:,start:end],chunk_ids[:,start:end])
        sf,tf=checkpoint(encode,*args,use_reentrant=False) if use_checkpoint else encode(*args)
        features.append(sf);token_features.append(tf)
    sf,tf=torch.cat(features,1),torch.cat(token_features,1)
    bias=aligned_chunk_bias(chunk_ids,tile)
    h=model.fuse(torch.cat((sf,tf),-1))
    dim=model.config.d_b//model.config.n_heads
    freq=model.config.rope_theta**(-torch.arange(0,dim,2,device=h.device).float()/dim)
    angle=torch.arange(length,device=h.device).float()[:,None]*freq[None,:]
    cos,sin=[x.to(h.dtype).view(1,1,length,-1) for x in (angle.cos(),angle.sin())]
    for projection,block in zip(model.stale_reinject,model.blocks):
        # Bind each block explicitly: backward recomputation must not capture
        # the final loop iteration through a late-bound Python closure.
        def run(value,stale_features,projection=projection,block=block):
            return block(value+projection(stale_features),bias,cos,sin,None)
        h=checkpoint(run,h,sf,use_reentrant=False) if use_checkpoint and settings['checkpoint_blocks'] else run(h,sf)
    return model.ln_f(h)


def _head(model,value,cids):
    result=model.v_head(value).reshape(*value.shape[:2],*model.target_config.kv_shape)
    return result.masked_fill((cids<0)[...,None,None,None,None],0)


def forward(model,stale,tokens,chunk_ids):
    h=hidden(model,stale,tokens,chunk_ids);tile=model._long_training['tile']
    return torch.cat([_head(model,h[:,a:a+tile],chunk_ids[:,a:a+tile])
        for a in range(0,tokens.shape[1],tile)],1)


def loss(model,stale,joint,tokens,chunk_ids):
    """The upstream normalized residual MSE, with bounded saved activations."""
    h=hidden(model,stale,tokens,chunk_ids);tile=model._long_training['tile']
    total=stale.numel();value=h.new_zeros(())
    def part(hh,ss,jj,cids):
        prediction=_head(model,hh,cids).float()
        target=(jj-ss)/model.sigma_delta
        return (prediction-target.float()).square().sum()/total
    for a in range(0,tokens.shape[1],tile):
        args=(h[:,a:a+tile],stale[:,a:a+tile],joint[:,a:a+tile],chunk_ids[:,a:a+tile])
        value=value+(checkpoint(part,*args,use_reentrant=False) if torch.is_grad_enabled() else part(*args))
    return value


@torch.no_grad()
def validation_diagnostics(model,target,rows,pair_function):
    """Match legacy validation fields and sample/token weighting in token tiles."""
    model.eval();totals=torch.zeros(6,dtype=torch.float64,device=model.tok_proj.weight.device)
    layer_error=torch.zeros(model.target_config.n_layers,2,dtype=torch.float64,device=totals.device)
    layer_zero=torch.zeros_like(layer_error);tokens=0;tile=model._long_training['tile']
    for row in rows:
        pair=pair_function(target,row['chunks']);h=hidden(model,pair['stale'],pair['tokens'],pair['chunk_ids'])
        sums=torch.zeros_like(totals);elements=pair['stale'].numel()
        for a in range(0,pair['tokens'].shape[1],tile):
            s=pair['stale'][:,a:a+tile];truth=pair['joint'][:,a:a+tile]-s
            prediction=_head(model,h[:,a:a+tile],pair['chunk_ids'][:,a:a+tile]).float()
            normalized=truth/model.sigma_delta;error=(prediction-normalized).double().square()
            raw=prediction*model.sigma_delta
            sums+=torch.stack((error.sum(),normalized.double().square().sum(),(raw-truth).double().square().sum(),
                truth.double().square().sum(),raw.double().square().sum(),s.double().square().sum()))
            layer_error+=error.sum(dim=(0,1,3,5))
            layer_zero+=normalized.double().square().sum(dim=(0,1,3,5))
        sums[:2]/=elements;totals+=sums;tokens+=pair['tokens'].numel()
        del pair,h
    values=totals.cpu().tolist();count=len(rows)
    scale=tokens*model.target_config.n_kv_heads*model.target_config.head_dim
    return {'examples':count,'tokens':tokens,'normalized_residual_mse':values[0]/count,
        'zero_residual_baseline_mse':values[1]/count,
        'relative_residual_l2_error':(values[2]/max(values[3],1e-30))**.5,
        'predicted_to_true_residual_l2':(values[4]/max(values[3],1e-30))**.5,
        'predicted_to_stale_kv_l2':(values[4]/max(values[5],1e-30))**.5,
        'per_attention_layer_k_v_normalized_mse':(layer_error/scale).cpu().tolist(),
        'per_attention_layer_k_v_zero_residual_mse':(layer_zero/scale).cpu().tolist()}
