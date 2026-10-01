#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/from_blob.h>

#include <map>
#include <vector>

void swap_blocks(
  at::Tensor& src,
  at::Tensor& dst,
  const std::map<int64_t, int64_t>& block_mapping);

void copy_blocks(
  std::vector<at::Tensor>& key_caches,
  std::vector<at::Tensor>& value_caches,
  const std::map<int64_t, std::vector<int64_t>>& block_mapping);

void reshape_and_cache(
  at::Tensor& key,
  at::Tensor& value,
  at::Tensor& key_cache,
  at::Tensor& value_cache,
  at::Tensor& slot_mapping,
  const std::string& kv_cache_dtype,
  const float kv_scale);

// Just for unittest
void convert_fp8(
  at::Tensor& src_cache,
  at::Tensor& dst_cache);
