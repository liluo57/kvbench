# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://raw.githubusercontent.com/vllm-project/vllm/v0.5.5/vllm/model_executor/layers/quantization/__init__.py
"""This bundled runtime serves unquantized hybrid checkpoints."""
from methods.hypic.backend.kvbench_hypic.srt.layers.quantization.base_config import QuantizationConfig

QUANTIZATION_METHODS = {}


def get_quantization_config(quantization):
    raise ValueError(
        f"The bundled HYPIC backend supports unquantized FP16/BF16 checkpoints, got {quantization!r}"
    )
