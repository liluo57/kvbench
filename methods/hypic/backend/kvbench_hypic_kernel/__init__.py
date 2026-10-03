"""Upstream kernel wrappers compiled against the project PyTorch."""
from methods.hypic.Build import load_kernels
load_kernels()
from methods.hypic.backend.kvbench_hypic_kernel.version import __version__  # noqa: F401
from methods.hypic.backend.kvbench_hypic_kernel.allreduce import *
from methods.hypic.backend.kvbench_hypic_kernel.attention import (
    cutlass_mla_decode,
    cutlass_mla_get_workspace_size,
    merge_state_v2,
)
from methods.hypic.backend.kvbench_hypic_kernel.cutlass_moe import (
    cutlass_w4a8_moe_mm,
    get_cutlass_w4a8_moe_mm_data,
)
from methods.hypic.backend.kvbench_hypic_kernel.elementwise import (
    concat_mla_absorb_q,
    concat_mla_k,
    copy_to_gpu_no_ce,
    dsv4_fused_k_norm_rope_flashmla,
    dsv4_fused_q_indexer_rope_hadamard_quant,
    dsv4_fused_q_norm_rope,
    fused_add_rmsnorm,
    gelu_and_mul,
    gelu_tanh_and_mul,
    gemma_fused_add_rmsnorm,
    gemma_rmsnorm,
    rmsnorm,
    rotary_embedding,
    silu_and_mul,
)
from methods.hypic.backend.kvbench_hypic_kernel.expert_specialization import (
    es_fp8_blockwise_scaled_grouped_mm,
    es_sm100_mxfp8_blockscaled_grouped_mm,
    es_sm100_mxfp8_blockscaled_grouped_quant,
)
from methods.hypic.backend.kvbench_hypic_kernel.gemm import (
    awq_dequantize,
    bmm_fp8,
    dsv3_fused_a_gemm,
    dsv3_router_gemm,
    fp8_blockwise_scaled_mm,
    fp8_scaled_mm,
    gptq_gemm,
    gptq_shuffle,
    int8_scaled_mm,
    qserve_w4a8_per_chn_gemm,
    qserve_w4a8_per_group_gemm,
    sgl_per_token_group_quant_8bit,
    sgl_per_token_group_quant_fp8,
    sgl_per_token_group_quant_int8,
    sgl_per_token_quant_fp8,
    shuffle_rows,
)
from methods.hypic.backend.kvbench_hypic_kernel.grammar import apply_token_bitmask_inplace_cuda
from methods.hypic.backend.kvbench_hypic_kernel.kvcacheio import (
    transfer_kv_all_layer,
    transfer_kv_all_layer_mla,
    transfer_kv_per_layer,
    transfer_kv_per_layer_mla,
)
from methods.hypic.backend.kvbench_hypic_kernel.mamba import (
    causal_conv1d_fn_cpu,
    causal_conv1d_fwd,
    causal_conv1d_update,
    causal_conv1d_update_cpu,
    chunk_gated_delta_rule_cpu,
)
from methods.hypic.backend.kvbench_hypic_kernel.memory import weak_ref_tensor
from methods.hypic.backend.kvbench_hypic_kernel.moe import (
    apply_shuffle_mul_sum,
    fp8_blockwise_scaled_grouped_mm,
    fused_qk_norm_rope,
    kimi_k2_moe_fused_gate,
    moe_align_block_size,
    moe_fused_gate,
    moe_sum,
    moe_sum_reduce,
    prepare_moe_input,
    topk_sigmoid,
    topk_softmax,
)
from methods.hypic.backend.kvbench_hypic_kernel.quantization import (
    ggml_dequantize,
    ggml_moe_a8,
    ggml_moe_a8_vec,
    ggml_moe_get_block_size,
    ggml_mul_mat_a8,
    ggml_mul_mat_vec_a8,
)
from methods.hypic.backend.kvbench_hypic_kernel.sampling import (
    top_k_renorm_prob,
    top_p_renorm_prob,
)
from methods.hypic.backend.kvbench_hypic_kernel.speculative import (
    build_tree_kernel_efficient,
    reconstruct_indices_from_tree_mask,
    segment_packbits,
    tree_speculative_sampling_target_only,
    verify_tree_greedy,
)
from methods.hypic.backend.kvbench_hypic_kernel.top_k import (
    fast_topk,
    fast_topk_transform_fused,
    fast_topk_transform_ragged_fused,
    fast_topk_v2,
)
from methods.hypic.backend.kvbench_hypic_kernel.version import __version__
from methods.hypic.backend.kvbench_hypic_kernel.spatial import create_greenctx_stream_by_value as _impl
from methods.hypic.backend.kvbench_hypic_kernel.spatial import get_sm_available as _impl
