/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/all.h>
#include <torch/library.h>

#include "sgl_kernel_ops.h"

TORCH_LIBRARY_FRAGMENT(kvbench_hypic_kernel, m) {
m.def("get_graph_buffer_ipc_meta", &get_graph_buffer_ipc_meta);
m.def("register_graph_buffers", &register_graph_buffers);
m.def("dispose", &dispose);
m.def("meta_size", &meta_size);
m.def("register_buffer", &register_buffer);
m.def(
      "init_custom_ar(int[] ipc_tensors, Tensor rank_data, "
      "int rank, bool full_nvlink) -> int");
m.impl("init_custom_ar", torch::kCUDA, &init_custom_ar);
m.def(
      "all_reduce(int fa, Tensor inp, Tensor! out, int reg_buffer, "
      "int reg_buffer_sz_bytes) -> ()");
m.impl("all_reduce", torch::kCUDA, &all_reduce);
m.def("rmsnorm(Tensor! output, Tensor input, Tensor weight, float eps, bool enable_pdl) -> ()");
m.impl("rmsnorm", torch::kCUDA, &rmsnorm);
m.def("fused_add_rmsnorm(Tensor! input, Tensor! residual, Tensor weight, float eps, bool enable_pdl) -> ()");
m.impl("fused_add_rmsnorm", torch::kCUDA, &sgl_fused_add_rmsnorm);
m.def("silu_and_mul(Tensor! out, Tensor input) -> ()");
m.impl("silu_and_mul", torch::kCUDA, &silu_and_mul);
m.def("gelu_tanh_and_mul(Tensor! out, Tensor input) -> ()");
m.impl("gelu_tanh_and_mul", torch::kCUDA, &gelu_tanh_and_mul);
m.def("gelu_and_mul(Tensor! out, Tensor input) -> ()");
m.impl("gelu_and_mul", torch::kCUDA, &gelu_and_mul);
m.def(
      "rotary_embedding(Tensor positions, Tensor! query,"
      "                 Tensor!? key, int head_size,"
      "                 Tensor cos_sin_cache, bool is_neox) -> ()");
m.impl("rotary_embedding", torch::kCUDA, &rotary_embedding);
m.def(
      "moe_align_block_size(Tensor topk_ids, int num_experts, int block_size, Tensor! sorted_token_ids, Tensor! "
      "experts_ids, Tensor! num_tokens_post_pad, Tensor! cumsum_buffer, bool "
      "pad_sorted_token_ids) -> ()");
m.impl("moe_align_block_size", torch::kCUDA, &moe_align_block_size);
m.def(
      "topk_softmax(Tensor! topk_weights, Tensor! topk_indices, Tensor gating_output, bool renormalize, float "
      "moe_softcapping, Tensor? correction_bias) -> ()");
m.impl("topk_softmax", torch::kCUDA, &topk_softmax);
m.def(
      "topk_sigmoid(Tensor! topk_weights, Tensor! topk_indices, Tensor gating_output, bool renormalize, Tensor? "
      "correction_bias) -> ()");
m.impl("topk_sigmoid", torch::kCUDA, &topk_sigmoid);
m.def("moe_sum_reduce(Tensor input, Tensor output, float routed_scaling_factor) -> ()");
m.impl("moe_sum_reduce", torch::kCUDA, &moe_sum_reduce);
m.def("moe_sum(Tensor input, Tensor! output) -> ()");
m.impl("moe_sum", torch::kCUDA, &moe_sum);
m.def("weak_ref_tensor(Tensor tensor) -> Tensor");
m.impl("weak_ref_tensor", torch::kCUDA, &weak_ref_tensor);
m.def(
      "causal_conv1d_update(Tensor! x,"
      "Tensor! conv_state,"
      "Tensor! weight,"
      "Tensor? bias_,"
      "bool silu_activation,"
      "Tensor? cache_seqlens_,"
      "Tensor? conv_state_indices,"
      "int pad_slot_id) -> ()");
m.impl("causal_conv1d_update", torch::kCUDA, &causal_conv1d_update);
m.def(
      "causal_conv1d_fwd(Tensor! x, Tensor! weight,"
      "Tensor? bias_,"
      "Tensor!? conv_states,"
      "Tensor? query_start_loc,"
      "Tensor? cache_indices,"
      "Tensor? has_initial_state,"
      "bool silu_activation,"
      "int pad_slot_id) -> ()");
m.impl("causal_conv1d_fwd", torch::kCUDA, &causal_conv1d_fwd);
}
REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
