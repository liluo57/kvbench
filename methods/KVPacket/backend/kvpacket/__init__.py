"""Vendored KVPacket runtime and training APIs used by KVBench.

Upstream's optional service and KVPress integrations are excluded. See the
adjacent LICENSE and UPSTREAM files for attribution.
"""

from .backend import HFBackend
from .packet import PacketWrapper, RawText, TextSourceEncoder, WrapperEntry, WrapperRegistry
from .preprocessing import PacketPreprocessor
from .session import GenerationResult, Mode, PacketSession, PacketStore
from .training import PacketTrainer

__all__ = [
    "HFBackend",
    "PacketWrapper",
    "RawText",
    "TextSourceEncoder",
    "WrapperEntry",
    "WrapperRegistry",
    "PacketPreprocessor",
    "PacketStore",
    "GenerationResult",
    "Mode",
    "PacketSession",
    "PacketTrainer",
]

__version__ = "0.1.0+kvbench"
