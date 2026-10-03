# [Deprecated] Re-export shim for backward compatibility. Use dsa.triton_kernel instead.
import warnings

warnings.warn(
    "methods.hypic.backend.kvbench_hypic.srt.layers.attention.nsa.triton_kernel is deprecated; "
    "use methods.hypic.backend.kvbench_hypic.srt.layers.attention.dsa.triton_kernel instead.",
    DeprecationWarning,
    stacklevel=2,
)
from methods.hypic.backend.kvbench_hypic.srt.layers.attention.dsa.triton_kernel import *  # noqa: F401, F403
