# [Deprecated] attention/nsa/ is a thin re-export shim for backward compatibility.
# Use attention/dsa/ instead. This directory will be removed in a future release.
import warnings

warnings.warn(
    "methods.hypic.backend.kvbench_hypic.srt.layers.attention.nsa is deprecated; "
    "use methods.hypic.backend.kvbench_hypic.srt.layers.attention.dsa instead.",
    DeprecationWarning,
    stacklevel=2,
)
from methods.hypic.backend.kvbench_hypic.srt.layers.attention.dsa import *  # noqa: F401, F403
