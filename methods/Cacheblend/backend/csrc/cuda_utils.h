#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/from_blob.h>

int get_device_attribute(
    int attribute,
    int device_id);

int get_max_shared_memory_per_block_device_attribute(
    int device_id);
