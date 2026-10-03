from __future__ import annotations

import functools

from methods.hypic.backend.kvbench_hypic.srt.environ import envs
from methods.hypic.backend.kvbench_hypic.srt.utils import is_hip


@functools.lru_cache(maxsize=1)
def is_unified_kv_triton() -> bool:
    # unified_kv_triton is only implemented on HIP (ROCm)
    return is_hip() and envs.SGLANG_HACK_FLASHMLA_BACKEND.get() == "unified_kv_triton"
