#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/from_blob.h>

void paged_attention_v1(
  at::Tensor& out,
  at::Tensor& query,
  at::Tensor& key_cache,
  at::Tensor& value_cache,
  int num_kv_heads,
  float scale,
  at::Tensor& block_tables,
  at::Tensor& context_lens,
  int block_size,
  int max_context_len,
  const c10::optional<at::Tensor>& alibi_slopes,
  const std::string& kv_cache_dtype,
  float kv_scale);

void paged_attention_v2(
  at::Tensor& out,
  at::Tensor& exp_sums,
  at::Tensor& max_logits,
  at::Tensor& tmp_out,
  at::Tensor& query,
  at::Tensor& key_cache,
  at::Tensor& value_cache,
  int num_kv_heads,
  float scale,
  at::Tensor& block_tables,
  at::Tensor& context_lens,
  int block_size,
  int max_context_len,
  const c10::optional<at::Tensor>& alibi_slopes,
  const std::string& kv_cache_dtype,
  float kv_scale);

void rms_norm(
  at::Tensor& out,
  at::Tensor& input,
  at::Tensor& weight,
  float epsilon);

void fused_add_rms_norm(
  at::Tensor& input,
  at::Tensor& residual,
  at::Tensor& weight,
  float epsilon);

void rotary_embedding(
  at::Tensor& positions,
  at::Tensor& query,
  at::Tensor& key,
  int head_size,
  at::Tensor& cos_sin_cache,
  bool is_neox);

void batched_rotary_embedding(
  at::Tensor& positions,
  at::Tensor& query,
  at::Tensor& key,
  int head_size,
  at::Tensor& cos_sin_cache,
  bool is_neox,
  int rot_dim,
  at::Tensor& cos_sin_cache_offsets);

void silu_and_mul(
  at::Tensor& out,
  at::Tensor& input);

void gelu_and_mul(
  at::Tensor& out,
  at::Tensor& input);

void gelu_tanh_and_mul(
  at::Tensor& out,
  at::Tensor& input);

void gelu_new(
  at::Tensor& out,
  at::Tensor& input);

void gelu_fast(
  at::Tensor& out,
  at::Tensor& input);

#ifndef USE_ROCM
at::Tensor aqlm_gemm(
  const at::Tensor& input,
  const at::Tensor& codes,
  const at::Tensor& codebooks,
  const at::Tensor& scales,
  const at::Tensor& codebook_partition_sizes,
  const std::optional<at::Tensor>& bias
);

at::Tensor aqlm_dequant(
  const at::Tensor& codes,
  const at::Tensor& codebooks,
  const at::Tensor& codebook_partition_sizes
);

at::Tensor awq_gemm(
  at::Tensor _in_feats,
  at::Tensor _kernel,
  at::Tensor _scaling_factors,
  at::Tensor _zeros,
  int split_k_iters);

at::Tensor awq_dequantize(
    at::Tensor _kernel,
    at::Tensor _scaling_factors,
    at::Tensor _zeros,
    int split_k_iters,
    int thx,
    int thy);

at::Tensor marlin_gemm(
    at::Tensor& a, 
    at::Tensor& b_q_weight,
    at::Tensor& b_scales, 
    at::Tensor& workspace,
    int64_t size_m, 
    int64_t size_n, 
    int64_t size_k);
#endif

void squeezellm_gemm(
  at::Tensor vec,
  at::Tensor mat,
  at::Tensor mul,
  at::Tensor lookup_table);

at::Tensor gptq_gemm(
  at::Tensor a,
  at::Tensor b_q_weight,
  at::Tensor b_gptq_qzeros,
  at::Tensor b_gptq_scales,
  at::Tensor b_g_idx,
  bool use_exllama,
  int bit);

void gptq_shuffle(
  at::Tensor q_weight,
  at::Tensor q_perm,
  int bit);

void scaled_fp8_quant(
  at::Tensor& out,
  at::Tensor& input,
  at::Tensor& scale);

void moe_align_block_size(
  at::Tensor topk_ids,
  int num_experts,
  int block_size,
  at::Tensor sorted_token_ids,
  at::Tensor experts_ids,
  at::Tensor num_tokens_post_pad);

#ifndef USE_ROCM
using fptr_t = uint64_t;
fptr_t init_custom_ar(at::Tensor &meta, at::Tensor &rank_data,
                    const std::vector<std::string> &handles,
                    const std::vector<int64_t> &offsets, int rank,
                    bool full_nvlink);
bool should_custom_ar(at::Tensor &inp, int max_size, int world_size,
                      bool full_nvlink);
void all_reduce_reg(fptr_t _fa, at::Tensor &inp, at::Tensor &out);
void all_reduce_unreg(fptr_t _fa, at::Tensor &inp, at::Tensor &reg_buffer,
                      at::Tensor &out);
void dispose(fptr_t _fa);
int meta_size();
void register_buffer(fptr_t _fa, at::Tensor &t,
                     const std::vector<std::string> &handles,
                     const std::vector<int64_t> &offsets);
std::pair<std::vector<uint8_t>, std::vector<int64_t>> get_graph_buffer_ipc_meta(fptr_t _fa);
void register_graph_buffers(fptr_t _fa, const std::vector<std::string> &handles,
                            const std::vector<std::vector<int64_t>> &offsets);
#endif
