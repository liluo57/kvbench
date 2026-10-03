import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import time
from flashinfer.quantization import packbits

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def get_layer(reuse_method):
    if 'blend' in reuse_method:
        return 1 # compute imp indices at layer 1
    if 'debug' in reuse_method:
        return 1
    else:
        return 0

def get_topindices(reuse_config, other_config, query_states, key_states, value_states, value_old, kvgroups):
    if 'blend' in reuse_config['reuse']: 
        last_len = other_config['data_params']['last_len']
        total_len = value_states.shape[2]
        last_indices = [total_len-last_len+l for l in range(last_len)]
        
        topk_num = int((total_len-last_len)*reuse_config['recomp_ratio'])
        
        temp_diff = torch.sum((value_states[:,:,:-last_len,:]-value_old[:,:,:-last_len,:])**2, dim=[0,1,3])
        top_indices = torch.topk(temp_diff, k=topk_num).indices # (, topk_num)
        top_indices, _ = torch.sort(top_indices)
        top_indices = torch.cat([top_indices, torch.tensor(last_indices, device=top_indices.device)])

    elif 'attnlink' in reuse_config['reuse']: 
        reuse_config['imp_indices'] = other_config['data_params']['sink_pos']
        top_indices = reuse_config['imp_indices']

    elif 'full' in reuse_config['reuse']: 
        last_len = other_config['data_params']['last_len']
        total_len = value_states.shape[2]
        last_indices = [total_len-last_len+l for l in range(last_len)]

        top_indices = torch.tensor(last_indices, device=value_states.device)
    
    elif 'cat' in reuse_config['reuse']:
        last_len = other_config['data_params']['last_len']
        total_len = value_states.shape[2]
        last_indices = [total_len-1]

        top_indices = torch.tensor(last_indices, device=value_states.device)

    elif 'debug' in reuse_config['reuse']:
        kernel_size = 5
        last_len = other_config['data_params']['last_len']
        total_len = value_states.shape[2]
        
        attn_weights_sum = compute_attention_sum_wo_head(query_states, repeat_kv(key_states[:,:, other_config['data_params']['prefix_len']:,:], kvgroups), last_len)
        attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = kernel_size, padding=kernel_size//2, stride=1)
        topk_num = int((total_len-last_len)*reuse_config['recomp_ratio'])

        
        top_indices = attn_cache.topk(topk_num, dim=-1).indices.squeeze(0)+other_config['data_params']['prefix_len']
        # import pdb; pdb.set_trace()
        total_len = value_states.shape[2]
        last_indices = torch.arange(total_len - last_len, total_len, device=top_indices.device)

        top_indices = torch.cat([top_indices, last_indices])

    return top_indices

def compute_attention_sum_wo_head(query_states, key_states, last_len):
    query_states = query_states.permute(0, 2, 1, 3).reshape(1, query_states.shape[-2], -1)
    key_states = key_states.permute(0, 2, 1, 3).reshape(1, key_states.shape[-2], -1)

    dim=key_states.shape[-1]
    attn_weights = torch.matmul(query_states[:,-last_len:,:], key_states.transpose(1,2)) / math.sqrt(dim)
    mask = torch.full((last_len, last_len), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)

    mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(attn_weights.device)
    mask = mask[None, :, :]
    attn_weights[:, -last_len:, -last_len:] += mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights_sum = attn_weights[:,:,:-last_len].sum(dim = -2)

    return attn_weights_sum.float()

def create_flashinfer_mask(query_state, key_state, indices, mode):
    """
    为 flashinfer.single_prefill_with_kv_cache 生成布尔 mask 并进行打包。

    参数：
    - q_len: query 序列长度
    - k_len: key 序列长度
    - indices: list[int]，每个 query token 可关注的 key 长度
    - mode: 'causal' or 'rightbottom'

    返回：
    - custom_mask: torch.BoolTensor, shape [q_len, k_len]
    - packed_mask: torch.ByteTensor, shape [ceil(q_len * k_len / 8)]
    """
    q_len, k_len = query_state.shape[2], key_state.shape[2]

    device = 'cuda'
    custom_mask = torch.zeros((q_len, k_len), dtype=torch.bool, device=device)

    if mode:
        # 每个 token 只能看前 index 个 key
        for i, index in enumerate(indices):
            custom_mask[i, :index+1] = True

    else:
        # 对角右下遮蔽逻辑，例如 tril 结构
        for i in range(q_len):
            shift = i - q_len + 1
            if shift < 0:
                custom_mask[i, :k_len + shift] = True
            else:
                custom_mask[i, :] = True

    # 打包 mask 成 packed_custom_mask（1D uint8）
    packed_mask = packbits(custom_mask.view(-1), bitorder="little")
    return custom_mask, packed_mask
