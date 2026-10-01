#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/from_blob.h>

void topk_softmax(
  at::Tensor& topk_weights,
  at::Tensor& topk_indices,
  at::Tensor& token_expert_indices,
  at::Tensor& gating_output);
