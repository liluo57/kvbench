"""The Engine package: split into ``Engine`` / ``Scheduler`` / ``GpuGovernor``
/ ``Reporter`` plus shared ``State``. The public surface (used by
:mod:`core` and external callers) is just :class:`Engine` and the two
    benchmark-error classes.
"""

from .Engine import Engine
from .State import (
    BenchmarkInitializationError,
    BenchmarkResourceReleaseError,
)


__all__ = [
    "BenchmarkInitializationError",
    "BenchmarkResourceReleaseError",
    "Engine",
]